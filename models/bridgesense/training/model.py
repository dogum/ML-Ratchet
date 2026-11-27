"""BridgeSense Transformer (attn) — Mode-Set Self-Attention + Thermal Cross-Attention.

ML Ratchet's first Attention/Transformer model. Uses only LayerNorm, Linear, GELU, and
scaled dot-product multi-head attention (zero BatchNorm) so pure-numpy and TypeScript
inference match PyTorch to ~1e-6.

Outputs (predicting corrections to the classical OLS thermal physics anchor):
  - logit_change: P(structural change) = sigmoid(logit_change)
  - dk_mu, dk_logsig: effective global stiffness shift dK/K (%) ± 68% sigma
  - env_mu, env_logsig: pure environmental/thermal frequency shift (%) ± 68% sigma
  - mode_dk: per-mode implied stiffness shift dK_m/K_m (%) for interpretability
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttn(nn.Module):
    """Explicit multi-head attention with separate q/k/v/out projections for clean numpy/TS export."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        # q_in: [B, Lq, D], kv_in: [B, Lk, D], kv_mask: [B, Lk] (1=valid, 0=pad)
        B, Lq, D = q_in.shape
        Lk = kv_in.shape[1]
        H, dh = self.n_heads, self.d_head
        Q = self.q_proj(q_in).view(B, Lq, H, dh).transpose(1, 2)   # [B, H, Lq, dh]
        K = self.k_proj(kv_in).view(B, Lk, H, dh).transpose(1, 2)  # [B, H, Lk, dh]
        V = self.v_proj(kv_in).view(B, Lk, H, dh).transpose(1, 2)  # [B, H, Lk, dh]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dh)  # [B, H, Lq, Lk]
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask[:, None, None, :] < 0.5, -1e9)
        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.o_proj(out)


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention block: Query snapshots attend to Reference baseline snapshots."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.05):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttn(d_model, n_heads, dropout=dropout)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.ln_q(q), self.ln_kv(kv))
        if self.training and self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout)
        x = q + h
        ff = self.ff2(F.gelu(self.ff1(self.ln_ff(x))))
        if self.training and self.dropout > 0.0:
            ff = F.dropout(ff, p=self.dropout)
        return x + ff


class SelfAttnBlock(nn.Module):
    """Pre-norm masked self-attention block over variable-length bridge mode sets."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttn(d_model, n_heads, dropout=dropout)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h_in = self.ln1(x)
        h = self.attn(h_in, h_in, kv_mask=mask)
        if self.training and self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout)
        x = x + h * mask[:, :, None]
        ff = self.ff2(F.gelu(self.ff1(self.ln_ff(x))))
        if self.training and self.dropout > 0.0:
            ff = F.dropout(ff, p=self.dropout)
        return x + ff * mask[:, :, None]


class BridgeSenseNet(nn.Module):
    """Dual-axis Transformer: Temporal Thermal Cross-Attention + Mode-Set Self-Attention."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_t_layers: int = 2,
        n_m_layers: int = 2,
        dropout: float = 0.05,
        use_cross_attn: bool = True,
        use_anchor: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_t_layers = n_t_layers
        self.n_m_layers = n_m_layers
        self.use_cross_attn = use_cross_attn
        self.use_anchor = use_anchor

        # Temporal encoders (shared projection for ref & query, plus segment embedding)
        self.t_in = nn.Linear(10, d_model)
        self.t_seg_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.t_blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_t_layers)
        ])
        self.t_ln = nn.LayerNorm(d_model)
        self.t_pool = nn.Linear(d_model, 1)

        # Mode-set encoder
        self.m_in = nn.Linear(14, d_model)
        self.m_blocks = nn.ModuleList([
            SelfAttnBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_m_layers)
        ])
        self.m_ln = nn.LayerNorm(d_model)
        self.m_pool = nn.Linear(d_model, 1)
        self.mode_head = nn.Linear(d_model, 1)

        # Global fusion MLP
        self.g_in = nn.Linear(16, d_model)
        self.fuse1 = nn.Linear(d_model * 4, d_model * 2)
        self.fuse_ln = nn.LayerNorm(d_model * 2)
        self.fuse2 = nn.Linear(d_model * 2, d_model)
        # Outputs: [logit_change, dk_corr, dk_logsig, env_corr, env_logsig]
        self.head = nn.Linear(d_model, 5)

    def forward(
        self,
        x_ref: torch.Tensor,
        x_q: torch.Tensor,
        x_mode: torch.Tensor,
        m_mask: torch.Tensor,
        g_feat: torch.Tensor,
        anchor: torch.Tensor,
        mode_anchor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # 1. Temporal branch: query snapshots cross-attend to reference baseline snapshots
        h_ref = F.gelu(self.t_in(x_ref))
        h_q = F.gelu(self.t_in(x_q)) + self.t_seg_q
        if self.use_cross_attn:
            for blk in self.t_blocks:
                h_q = blk(h_q, h_ref)
        else:
            # Ablation mode: self-attention on query only (for ratchet ablation rung)
            for blk in self.t_blocks:
                h_q = blk(h_q, h_q)
        h_q = self.t_ln(h_q)
        w_t = F.softmax(self.t_pool(h_q), dim=1)     # [B, N_q, 1]
        z_t = torch.sum(w_t * h_q, dim=1)            # [B, D]

        # 2. Mode-set branch: masked self-attention across variable active modes
        h_m = F.gelu(self.m_in(x_mode)) * m_mask[:, :, None]
        for blk in self.m_blocks:
            h_m = blk(h_m, m_mask)
        h_m = self.m_ln(h_m) * m_mask[:, :, None]

        # Masked attention pooling + masked mean pooling
        scores_m = self.m_pool(h_m).squeeze(-1)      # [B, M]
        scores_m = scores_m.masked_fill(m_mask < 0.5, -1e9)
        w_m = F.softmax(scores_m, dim=1)             # [B, M]
        z_m_attn = torch.sum(w_m[:, :, None] * h_m, dim=1)
        m_cnt = torch.clamp(torch.sum(m_mask, dim=1, keepdim=True), min=1.0)
        z_m_mean = torch.sum(h_m, dim=1) / m_cnt

        # Per-mode structural shift prediction
        mode_corr = self.mode_head(h_m).squeeze(-1)
        mode_dk = (mode_anchor + mode_corr) if self.use_anchor else mode_corr
        mode_dk = mode_dk * m_mask

        # 3. Fuse temporal, mode-set (attn + mean), and global physics-anchor features
        z_g = F.gelu(self.g_in(g_feat))
        z = torch.cat([z_t, z_m_attn, z_m_mean, z_g], dim=-1)
        z = F.gelu(self.fuse_ln(self.fuse1(z)))
        z = F.gelu(self.fuse2(z))
        out = self.head(z)

        logit_change = out[:, 0]
        dk_mu = (anchor[:, 0] + out[:, 1]) if self.use_anchor else out[:, 1]
        dk_logsig = torch.clamp(out[:, 2], -2.5, 3.5)
        env_mu = (anchor[:, 1] + out[:, 3]) if self.use_anchor else out[:, 3]
        env_logsig = torch.clamp(out[:, 4], -3.0, 3.0)

        return {
            "logit_change": logit_change,
            "dk_mu": dk_mu,
            "dk_logsig": dk_logsig,
            "env_mu": env_mu,
            "env_logsig": env_logsig,
            "mode_dk": mode_dk,
        }


def beta_nll_loss(mu: torch.Tensor, logsig: torch.Tensor, target: torch.Tensor,
                  beta: float = 0.5) -> torch.Tensor:
    """Seitzer et al. (2022) beta-NLL heteroscedastic Gaussian loss."""
    var = torch.exp(2.0 * logsig)
    nll = 0.5 * ((target - mu) ** 2 / var + 2.0 * logsig)
    weight = var.detach() ** beta
    return torch.mean(weight * nll)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
