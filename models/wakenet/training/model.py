"""PyTorch graph neural network architectures for WakeNet training rungs.

All models take padded farm graphs:
  * node_x     : (B, N, 17)
  * edge_x     : (B, N, N, 12)   where edge_x[b, j, i] is upstream j -> downstream i
  * edge_mask  : (B, N, N)       bool mask of active wake edges j -> i
  * online     : (B, N)          bool mask of online generating turbines
  * is_free    : (B, N)          bool mask of geometrically unwaked front-row turbines
  * eta_anchor : (B, N)          steady 3-D Gaussian physics anchor ratio P_anc / mean(P_anc_front)

Outputs:
  * eta_pred       : (B, N)      predicted turbine power ratio P_i / mean(P_front), 0 when offline
  * log_sig_eta    : (B, N)      predicted log sigma on eta_i
  * u_def_pred     : (B, N)      predicted rotor wind-speed deficit 1 - U_i / U_free_i
  * farm_wake_pred : (B,)        predicted farm wake loss 1 - mean_{online}(eta_pred)
  * log_sig_farm   : (B,)        predicted log sigma on farm_wake_pred

Inference in models/wakenet/wakenet.py and site/src/instruments/wakenet/net.ts ports EdgeWakeGNN
directly to pure numpy and TypeScript using exact Abramowitz & Stegun erf GELU (parity ~ 1e-6).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODE_FEAT = 17
N_EDGE_FEAT = 12


def exact_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


class _GNNLayer(nn.Module):
    """Directed edge-conditioned message passing layer with sigmoid wake-superposition gate."""

    def __init__(self, hidden: int, agg: str = "gated_sum"):
        super().__init__()
        self.hidden = hidden
        self.agg = agg  # "gated_sum" | "softmax_attn"
        self.msg_fc1 = nn.Linear(3 * hidden, hidden)
        self.msg_fc2 = nn.Linear(hidden, hidden)
        self.gate_fc = nn.Linear(hidden, 1)
        self.upd_fc1 = nn.Linear(3 * hidden, hidden)
        self.upd_fc2 = nn.Linear(hidden, hidden)
        # Small residual init for stable message-passing steps
        nn.init.zeros_(self.upd_fc2.bias)
        with torch.no_grad():
            self.upd_fc2.weight.mul_(0.35)

    def forward(
        self,
        h: torch.Tensor,
        e_emb: torch.Tensor,
        edge_mask: torch.Tensor,
        online: torch.Tensor,
    ) -> torch.Tensor:
        # h: (B, N, H), e_emb: (B, N, N, H), edge_mask: (B, N, N) [j -> i]
        B, N, H = h.shape
        hj = h.unsqueeze(2).expand(B, N, N, H)  # upstream j
        hi = h.unsqueeze(1).expand(B, N, N, H)  # downstream i
        z = torch.cat([hj, hi, e_emb], dim=-1)  # (B, N, N, 3H)
        m_hid = exact_gelu(self.msg_fc1(z))
        msg = self.msg_fc2(m_hid)               # (B, N, N, H)
        logit = self.gate_fc(m_hid).squeeze(-1) # (B, N, N)

        mask_f = edge_mask.to(h.dtype)
        if self.agg == "softmax_attn":
            # Softmax averages incoming wakes (wrong symmetry when multiple wakes overlap!)
            has_any = edge_mask.any(dim=1, keepdim=True)  # (B, 1, N)
            neg_inf = torch.full_like(logit, -1e9)
            scores = torch.where(edge_mask, logit, neg_inf)
            weights = torch.softmax(scores, dim=1) * has_any.to(h.dtype)
        else:
            # Gated sum: 0 when unwaked, additive across multiple upstream wakes
            weights = torch.sigmoid(logit) * mask_f

        # Sum over upstream index j (dim=1) -> incoming wake embedding at downstream i: (B, N, H)
        agg_in = torch.einsum("bji,bjih->bih", weights, msg)

        # Farm-wide context over online turbines
        on_f = online.to(h.dtype).unsqueeze(-1)
        denom = on_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        h_farm = (h * on_f).sum(dim=1, keepdim=True) / denom  # (B, 1, H)
        h_farm_exp = h_farm.expand(B, N, H)

        u_in = torch.cat([h, agg_in, h_farm_exp], dim=-1)
        dh = self.upd_fc2(exact_gelu(self.upd_fc1(u_in)))
        return (h + dh) * on_f


class EdgeWakeGNN(nn.Module):
    """Physics-anchored directed Graph Neural Network for wind-farm wake inference."""

    def __init__(
        self,
        hidden: int = 96,
        n_layers: int = 3,
        head_hidden: int = 64,
        anchored: bool = True,
        use_edges: bool = True,
        agg: str = "gated_sum",
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.head_hidden = head_hidden
        self.anchored = anchored
        self.use_edges = use_edges
        self.agg = agg

        self.node_enc = nn.Linear(N_NODE_FEAT, hidden)
        if use_edges and n_layers > 0:
            self.edge_enc = nn.Linear(N_EDGE_FEAT, hidden)
            self.layers = nn.ModuleList([_GNNLayer(hidden, agg=agg) for _ in range(n_layers)])
        else:
            self.edge_enc = None
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden, hidden), nn.Linear(hidden, hidden))
                for _ in range(n_layers)
            ])

        self.node_head1 = nn.Linear(hidden + N_NODE_FEAT, head_hidden)
        self.node_head2 = nn.Linear(head_hidden, 3)  # (d_eta, log_sig_eta, d_u_def)

        self.farm_head1 = nn.Linear(2 * hidden + 4, head_hidden)
        self.farm_head2 = nn.Linear(head_hidden, 1)  # (log_sig_farm)

        # Start residual correction near 0 so initial predictions sit close to physics anchor
        nn.init.zeros_(self.node_head2.bias)
        with torch.no_grad():
            self.node_head2.weight.mul_(0.1)
            self.node_head2.bias[1] = -2.4  # initial sigma_eta ~ 0.09
        nn.init.zeros_(self.farm_head2.bias)
        with torch.no_grad():
            self.farm_head2.weight.mul_(0.1)
            self.farm_head2.bias[0] = -3.4  # initial sigma_farm ~ 0.033

    def forward(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        edge_mask: torch.Tensor,
        online: torch.Tensor,
        is_free: torch.Tensor,
        eta_anchor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        on_f = online.to(node_x.dtype)
        free_f = is_free.to(node_x.dtype)
        udef_anc = node_x[..., 13]
        if not self.anchored:
            node_x = node_x.clone()
            node_x[..., 12:17] = 0.0
            edge_x = edge_x.clone()
            edge_x[..., [3, 6, 7, 8]] = 0.0

        h = exact_gelu(self.node_enc(node_x)) * on_f.unsqueeze(-1)
        if self.use_edges and self.edge_enc is not None:
            e_emb = exact_gelu(self.edge_enc(edge_x)) * edge_mask.to(node_x.dtype).unsqueeze(-1)
            for layer in self.layers:
                h = layer(h, e_emb, edge_mask, online)
        else:
            for mlp in self.layers:
                fc1, fc2 = mlp[0], mlp[1]
                h = (h + 0.35 * fc2(exact_gelu(fc1(h)))) * on_f.unsqueeze(-1)

        nh = exact_gelu(self.node_head1(torch.cat([h, node_x], dim=-1)))
        nout = self.node_head2(nh)  # (B, N, 3)

        raw_eta = nout[..., 0]
        log_sig_eta = nout[..., 1].clamp(-5.0, -0.2)
        d_udef = nout[..., 2]

        if self.anchored:
            eta_unnorm = (eta_anchor + 0.5 * torch.tanh(raw_eta)) * on_f
        else:
            eta_unnorm = torch.sigmoid(raw_eta + 1.2) * 1.25 * on_f

        # Enforce exact physical front-row normalisation: mean_{k in F}(eta_k) == 1.0
        f_count = free_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        f_mean = ((eta_unnorm * free_f).sum(dim=1, keepdim=True) / f_count).clamp_min(0.25)
        eta_pred = (eta_unnorm / f_mean).clamp(0.0, 1.35) * on_f

        if self.anchored:
            u_def_pred = (udef_anc + 0.35 * torch.tanh(d_udef)).clamp(-0.15, 0.95) * on_f
        else:
            u_def_pred = torch.sigmoid(d_udef - 1.0).clamp(0.0, 0.95) * on_f

        on_count = on_f.sum(dim=1).clamp_min(1.0)
        farm_wake_pred = 1.0 - eta_pred.sum(dim=1) / on_count

        # Farm-level uncertainty head
        h_mean = (h * on_f.unsqueeze(-1)).sum(dim=1) / on_count.unsqueeze(-1)
        waked_f = (online & (~is_free)).to(node_x.dtype)
        w_count = waked_f.sum(dim=1).clamp_min(1.0)
        h_waked = (h * waked_f.unsqueeze(-1)).sum(dim=1) / w_count.unsqueeze(-1)
        f_scalars = torch.stack([
            farm_wake_pred,
            waked_f.sum(dim=1) / on_count,
            node_x[:, 0, 2],  # normalized u_inf
            node_x[:, 0, 6],  # normalized sigma_wd
        ], dim=-1)
        fh = exact_gelu(self.farm_head1(torch.cat([h_mean, h_waked, f_scalars], dim=-1)))
        log_sig_farm = self.farm_head2(fh).squeeze(-1).clamp(-5.5, -0.5)

        return {
            "eta_pred": eta_pred,
            "log_sig_eta": log_sig_eta,
            "u_def_pred": u_def_pred,
            "farm_wake_pred": farm_wake_pred,
            "log_sig_farm": log_sig_farm,
        }


def beta_nll(pred: torch.Tensor, target: torch.Tensor, log_sig: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Seitzer et al. (2022) beta-NLL loss so variance gradients don't starve mean accuracy."""
    var = torch.exp(2.0 * log_sig).clamp_min(1e-8)
    nll = 0.5 * ((target - pred) ** 2 / var + 2.0 * log_sig)
    if beta > 0.0:
        nll = nll * (var.detach() ** beta)
    return nll


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
