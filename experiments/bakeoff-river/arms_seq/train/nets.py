"""Torch versions of the four arms (training only; inference is numpy in common/seqlib.py).
Same input (29 channels/day), same head (FlowCast's mean + monotone quantiles), same loss (loss_fn). Only the core differs.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

QS = [0.05, 0.25, 0.5, 0.75, 0.95]
N_DYN, N_ST = 11, 18


class Base(nn.Module):
    def __init__(self, H, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, 1 + len(QS))
        self.register_buffer('st_mu', torch.zeros(N_ST))
        self.register_buffer('st_sd', torch.ones(N_ST))

    def inputs(self, x, st):
        s = (st - self.st_mu) / self.st_sd
        return torch.cat([x, s[:, None, :].expand(-1, x.shape[1], -1)], -1)

    def forward(self, x, st):
        h = self.core(self.inputs(x, st))
        o = self.head(self.drop(h))
        mean = Fn.softplus(o[..., 0])
        q = torch.cumsum(torch.cat([Fn.softplus(o[..., 1:2]) * 0.2, Fn.softplus(o[..., 2:])], -1), -1)
        return mean, q


class LSTMNet(Base):
    """FlowCast's network: one LSTM layer over the 29 channels."""
    def __init__(self, cfg):
        super().__init__(cfg['H'], cfg.get('dropout', 0.1))
        self.lstm = nn.LSTM(N_DYN + N_ST, cfg['H'], batch_first=True)

    def core(self, z):
        return self.lstm(z)[0]


def sinusoid(T, d):
    pos = torch.arange(T, dtype=torch.float64)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float64)
    ang = pos / (10000.0 ** (i / d))
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang[:, :d // 2])
    return pe.float()


class TBlock(nn.Module):
    def __init__(self, d, nh, ff, drop):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.ff1 = nn.Linear(d, ff); self.ff2 = nn.Linear(ff, d)
        self.drop = drop

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        sh = lambda t: t.view(B, T, self.nh, d // self.nh).transpose(1, 2)  # noqa: E731
        a = Fn.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True,
                                            dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        x = x + Fn.dropout(self.ff2(Fn.gelu(self.ff1(self.ln2(x)), approximate='tanh')), self.drop, self.training)
        return x


class TransformerNet(Base):
    """Causal pre-LN transformer encoder over daily tokens, sinusoidal position encoding (positions 0..729)."""
    def __init__(self, cfg):
        d = cfg['d']
        super().__init__(d, cfg.get('dropout', 0.1))
        self.inp = nn.Linear(N_DYN + N_ST, d)
        self.blocks = nn.ModuleList([TBlock(d, cfg['heads'], cfg['ff'], cfg.get('inner_dropout', 0.0))
                                     for _ in range(cfg['layers'])])
        self.lnf = nn.LayerNorm(d)
        self.register_buffer('pe', sinusoid(730, d), persistent=False)
        self.patch = cfg.get('patch', 1)
        if self.patch > 1:      # causal patching (see core)
            p = self.patch
            self.pemb = nn.Linear(p * d, d)
            self.c0 = nn.Parameter(torch.zeros(d))
            self.local = nn.Conv1d(d, d, p, groups=d)

    def core(self, z):
        if self.patch == 1:
            x = self.inp(z) + self.pe[:z.shape[1]]
            for b in self.blocks:
                x = b(x)
            return self.lnf(x)
        # Causal patched tokens: token k = days [k*p, (k+1)*p) (window left-padded with zeros to a multiple of p).
        # Day t in patch k reads the transformer output of token k-1 (everything before its own patch) plus a
        # depthwise causal conv over its last p days (the days of its own patch up to t). No day sees the future.
        e = self.inp(z)
        B, T, d = e.shape
        p = self.patch
        padn = (-T) % p
        K = (T + padn) // p
        tok = self.pemb(Fn.pad(e, (0, 0, padn, 0)).reshape(B, K, p * d)) + self.pe[:K]
        for b in self.blocks:
            tok = b(tok)
        ctx = torch.cat([self.c0.expand(B, 1, d), tok[:, :-1]], 1)
        ctx = ctx.repeat_interleave(p, 1)[:, padn:]
        loc = self.local(Fn.pad(e.transpose(1, 2), (p - 1, 0))).transpose(1, 2)
        return self.lnf(e + ctx + loc)


class S4DBlock(nn.Module):
    """Diagonal state-space layer (S4D-Lin init, ZOH), FFT convolution, then GELU + GLU mixing, residual."""
    def __init__(self, H, N2, dt_min, dt_max, drop):
        super().__init__()
        self.ln = nn.LayerNorm(H)
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_A_re = nn.Parameter(torch.log(0.5 * torch.ones(H, N2)))
        self.A_im = nn.Parameter(math.pi * torch.arange(N2).float()[None].repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, N2, 2) * math.sqrt(0.5))
        self.D = nn.Parameter(torch.ones(H))
        self.out = nn.Linear(H, 2 * H)
        self.drop = drop

    def kernel(self, T):
        dt = torch.exp(self.log_dt)
        A = torch.complex(-torch.exp(self.log_A_re), self.A_im)
        C = torch.view_as_complex(self.C.contiguous())
        dtA = dt[:, None] * A
        Cb = C * (torch.exp(dtA) - 1.0) / A
        ar = torch.arange(T, dtype=torch.float32)
        return 2.0 * torch.einsum('hn,hnl->hl', Cb, torch.exp(dtA[:, :, None] * ar[None, None])).real

    def forward(self, x):
        B, T, H = x.shape
        y = self.ln(x)
        K = self.kernel(T)
        n = 1 << (2 * T - 1).bit_length()      # >= 2T-1: exact linear convolution; power of 2 is faster
        u = torch.fft.rfft(y, n=n, dim=1)
        k = torch.fft.rfft(K, n=n, dim=1).T[None]
        y = torch.fft.irfft(u * k, n=n, dim=1)[:, :T] + y * self.D
        y = Fn.gelu(y, approximate='tanh')
        y = Fn.glu(self.out(y), -1)
        return x + Fn.dropout(y, self.drop, self.training)


def segsum(x):
    """x (..., T) -> (..., T, T): out[i, j] = sum_{k=j+1..i} x[k] for i >= j, -inf above the diagonal."""
    T = x.shape[-1]
    xe = x[..., None].expand(*x.shape, T)                      # (..., T(i), T(j)) holding x[i]
    low = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
    xe = xe.masked_fill(~low, 0.0)
    out = torch.cumsum(xe, -2)
    return out.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool), 0), float('-inf'))


class MambaBlock(nn.Module):
    """'Mamba-lite': a selective (input-dependent) diagonal state space with Mamba-2's scalar decay per head, computed
    exactly with the chunked SSD algorithm (no approximation; chunks only organise the sum). Per head h:
      state_t = exp(dt_t A_h) state_{t-1} + dt_t B_t x_t^T,   y_t = C_t . state_t + D_h x_t
    with dt_t = softplus(w.u_t + bias), B_t, C_t linear in the input: the decay (memory length) and what is written /
    read change with the day's input, unlike S4D whose filter is fixed. Gated output y * silu(z), residual."""
    def __init__(self, d, nh, N, chunk, drop):
        super().__init__()
        self.d, self.nh, self.N, self.chunk, self.drop = d, nh, N, chunk, drop
        self.ln = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * d + nh + 2 * N)
        self.out_proj = nn.Linear(d, d)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, nh)))
        dt = torch.exp(torch.rand(nh) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))          # inverse softplus
        self.D = nn.Parameter(torch.ones(nh))

    def forward(self, x):
        Bn, T, d = x.shape
        nh, N, Q = self.nh, self.N, self.chunk
        P = d // nh
        zxbcdt = self.in_proj(self.ln(x))
        z, xs, Bm, Cm, dt = torch.split(zxbcdt, [d, d, N, N, nh], -1)
        dt = Fn.softplus(dt + self.dt_bias)                                      # (B, T, nh)
        A = -torch.exp(self.A_log)                                               # (nh,)
        pad = (-T) % Q
        X = xs.view(Bn, T, nh, P) * dt[..., None]
        a = dt * A                                                               # (B, T, nh) log-decay <= 0
        if pad:
            X = Fn.pad(X, (0, 0, 0, 0, 0, pad)); a = Fn.pad(a, (0, 0, 0, pad))
            Bm = Fn.pad(Bm, (0, 0, 0, pad)); Cm = Fn.pad(Cm, (0, 0, 0, pad))
        c = (T + pad) // Q
        X = X.view(Bn, c, Q, nh, P); Bc = Bm.view(Bn, c, Q, N); Cc = Cm.view(Bn, c, Q, N)
        a = a.view(Bn, c, Q, nh).permute(0, 3, 1, 2)                             # (B, nh, c, Q)
        acs = torch.cumsum(a, -1)
        L = torch.exp(segsum(a))                                                 # (B, nh, c, Q, Q)
        G = torch.einsum('bcln,bcsn->bcls', Cc, Bc)
        Yd = torch.einsum('bhcls,bcshp->bclhp', G[:, None] * L, X)
        dstates = torch.exp(acs[..., -1:] - acs)                                 # (B, nh, c, Q)
        states = torch.einsum('bcln,bhcl,bclhp->bchpn', Bc, dstates, X)
        states = torch.cat([torch.zeros_like(states[:, :1]), states], 1)
        dchunk = torch.exp(segsum(Fn.pad(acs[..., -1], (1, 0))))                 # (B, nh, c+1, c+1)
        states = torch.einsum('bhzc,bchpn->bzhpn', dchunk, states)[:, :-1]
        Yo = torch.einsum('bcln,bchpn,bhcl->bclhp', Cc, states, torch.exp(acs))
        Y = (Yd + Yo).reshape(Bn, c * Q, nh, P)[:, :T]
        Y = Y + xs.view(Bn, T, nh, P) * self.D[:, None]
        y = self.out_proj(Y.reshape(Bn, T, d) * Fn.silu(z))
        return x + Fn.dropout(y, self.drop, self.training)


class SSMNet(Base):
    def __init__(self, cfg):
        d = cfg['d']
        super().__init__(d, cfg.get('dropout', 0.1))
        self.cfg = cfg
        self.inp = nn.Linear(N_DYN + N_ST, d)
        if cfg.get('pre_mlp'):
            self.pre1 = nn.Linear(d, cfg['pre_mlp']); self.pre2 = nn.Linear(cfg['pre_mlp'], d)
        if cfg.get('mixer', 's4d') == 'mamba':
            self.blocks = nn.ModuleList([MambaBlock(d, cfg['heads'], cfg['N'], cfg.get('chunk', 73),
                                                    cfg.get('inner_dropout', 0.0)) for _ in range(cfg['layers'])])
        else:
            self.blocks = nn.ModuleList([S4DBlock(d, cfg['N2'], cfg.get('dt_min', 1e-3), cfg.get('dt_max', 1e-1),
                                                  cfg.get('inner_dropout', 0.0)) for _ in range(cfg['layers'])])
        self.lnf = nn.LayerNorm(d)

    def core(self, z):
        x = self.inp(z)
        if self.cfg.get('pre_mlp'):
            x = x + self.pre2(Fn.gelu(self.pre1(x), approximate='tanh'))
        for b in self.blocks:
            x = b(x)
        return self.lnf(x)


class TCNBlock(nn.Module):
    """Pre-LN residual block: causal dilated conv -> GELU (or, gated=True, WaveNet's tanh x sigmoid gate) -> 1x1 mix."""
    def __init__(self, c, k, dil, drop, gated=False):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.gated = gated
        self.conv = nn.Conv1d(c, 2 * c if gated else c, k, dilation=dil)
        self.mix = nn.Linear(c, c)
        self.pad = (k - 1) * dil
        self.drop = drop

    def forward(self, x):
        y = self.ln(x).transpose(1, 2)
        y = self.conv(Fn.pad(y, (self.pad, 0))).transpose(1, 2)
        if self.gated:
            a, b = y.chunk(2, -1)
            y = self.mix(torch.tanh(a) * torch.sigmoid(b))
        else:
            y = self.mix(Fn.gelu(y, approximate='tanh'))
        return x + Fn.dropout(y, self.drop, self.training)


class TCNNet(Base):
    """Dilated causal 1-D CNN (residual blocks, dilations 1..2^k)."""
    def __init__(self, cfg):
        c = cfg['c']
        super().__init__(c, cfg.get('dropout', 0.1))
        self.inp = nn.Linear(N_DYN + N_ST, c)
        self.blocks = nn.ModuleList([TCNBlock(c, cfg['k'], d, cfg.get('inner_dropout', 0.0), cfg.get('gated', False))
                                     for d in cfg['dilations']])
        self.lnf = nn.LayerNorm(c)

    def core(self, z):
        x = self.inp(z)
        for b in self.blocks:
            x = b(x)
        return self.lnf(x)


NETS = {'lstm': LSTMNet, 'transformer': TransformerNet, 'ssm': SSMNet, 'tcn': TCNNet}


def build(cfg):
    return NETS[cfg['arch']](cfg)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def loss_fn(mean, q, y, sig):
    """FlowCast's loss: NSE* scaled by an input-only scale (Fu runoff ratio of the forcing aridity, 'budyko') on the
    mean + the same-scaled pinball loss on the quantiles."""
    w = 1.0 / (sig + 0.1)
    n = y.numel()
    nse = (((mean - y) * w[:, None]) ** 2).sum() / n
    taus = torch.tensor(QS, dtype=y.dtype)
    e = y[..., None] - q
    pin = torch.maximum(taus * e, (taus - 1) * e).mean(-1) * w[:, None]
    pin = pin.sum() / n
    return nse + pin, nse.detach(), pin.detach()


def export(net, cfg, path):
    import json
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items() if k != 'pe'}
    np.savez(path, cfg_json=np.array(json.dumps(cfg)), **sd)
