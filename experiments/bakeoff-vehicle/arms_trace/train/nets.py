"""Torch versions of the four trace arms (training only; inference is numpy in common/tracelib.py).
Same input (12 block channels + 57 statics), same stem, same pooling, same head, same loss. Only the core differs.
Batches are right-padded to the longest trip in the batch; `mask` (B, T) marks the valid blocks. Every core keeps the
valid blocks' outputs independent of the padding (checked against the unpadded numpy inference)."""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

QS = [0.05, 0.25, 0.5, 0.75, 0.95]
N_DYN, N_ST = 12, 57


def gelu(x):
    return Fn.gelu(x, approximate='tanh')


class CNNCore(nn.Module):
    """non-causal dilated residual 1-D CNN: LN -> conv (k, dilation, zero 'same' padding) -> GELU -> 1x1 mix."""
    def __init__(self, cfg):
        super().__init__()
        d, k = cfg['d'], cfg['k']
        self.blocks = nn.ModuleList()
        for dil in cfg['dilations']:
            b = nn.Module()
            b.ln = nn.LayerNorm(d)
            b.conv = nn.Conv1d(d, d, k, dilation=dil, padding=(k - 1) // 2 * dil)
            b.mix = nn.Linear(d, d)
            self.blocks.append(b)

    def forward(self, x, mask, lens):
        m = mask[..., None]
        for b in self.blocks:
            y = b.ln(x) * m
            y = gelu(b.conv(y.transpose(1, 2)).transpose(1, 2))
            x = x + b.mix(y)
        return x


class GRUCore(nn.Module):
    """bidirectional GRU (one layer per direction) + linear mix, residual. The backward GRU reads each trip reversed in
    place (padding stays at the end), so neither direction ever sees a padded block before a valid one."""
    def __init__(self, cfg):
        super().__init__()
        d, H = cfg['d'], cfg['H']
        self.gru = nn.GRU(d, H, batch_first=True)
        self.grub = nn.GRU(d, H, batch_first=True)
        self.gmix = nn.Linear(2 * H, d)

    def forward(self, x, mask, lens):
        B, T, d = x.shape
        H = self.gru.hidden_size
        t = torch.arange(T)[None, :].expand(B, T)
        r = torch.where(t < lens[:, None], lens[:, None] - 1 - t, t)[..., None]
        f, _ = self.gru(x)
        b, _ = self.grub(torch.gather(x, 1, r.expand(B, T, d)))
        b = torch.gather(b, 1, r.expand(B, T, H))
        return x + self.gmix(torch.cat([f, b], -1))


def sinusoid(T, d):
    pos = torch.arange(T, dtype=torch.float64)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float64)
    ang = pos / (10000.0 ** (i / d))
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang[:, :d // 2])
    return pe.float()


class TBlock(nn.Module):
    def __init__(self, d, nh, ff):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.ff1 = nn.Linear(d, ff); self.ff2 = nn.Linear(ff, d)

    def forward(self, x, am):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        sh = lambda t: t.view(B, T, self.nh, d // self.nh).transpose(1, 2)  # noqa: E731
        a = Fn.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=am)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ff2(gelu(self.ff1(self.ln2(x))))


class TransformerCore(nn.Module):
    """patch tokens of p blocks (p x 5 s), sinusoidal positions, non-causal pre-LN encoder; tokens repeated back to
    their blocks and added to the stem output (residual)."""
    def __init__(self, cfg):
        super().__init__()
        d, p = cfg['d'], cfg['patch']
        self.p, self.d = p, d
        self.pemb = nn.Linear(p * d, d)
        self.tblocks = nn.ModuleList([TBlock(d, cfg['heads'], cfg['ff']) for _ in range(cfg['layers'])])
        self.register_buffer('pe', sinusoid(4096, d), persistent=False)

    def forward(self, x, mask, lens):
        B, T, d = x.shape
        p = self.p
        padn = (-T) % p
        K = (T + padn) // p
        xp = Fn.pad(x * mask[..., None], (0, 0, 0, padn)).reshape(B, K, p * d)
        t = self.pemb(xp) + self.pe[:K]
        ntok = (lens + p - 1) // p
        valid = torch.arange(K)[None, :] < ntok[:, None]          # (B, K)
        am = valid[:, None, None, :]                               # attend to valid tokens only
        for b in self.tblocks:
            t = b(t, am)
        return x + t.repeat_interleave(p, 1)[:, :T]


class S4DBlock(nn.Module):
    """bidirectional diagonal state space layer (S4D-Lin init, ZOH, FFT convolution): forward + backward kernels,
    D skip, GELU, GLU mix, residual."""
    def __init__(self, d, N2, dt_min, dt_max):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_A_re = nn.Parameter(torch.log(0.5 * torch.ones(d, N2)))
        self.A_im = nn.Parameter(math.pi * torch.arange(N2).float()[None].repeat(d, 1))
        self.C = nn.Parameter(torch.randn(2, d, N2, 2) * math.sqrt(0.5))
        self.D = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, 2 * d)

    def kernel(self, T, j):
        dt = torch.exp(self.log_dt)
        A = torch.complex(-torch.exp(self.log_A_re), self.A_im)
        C = torch.view_as_complex(self.C[j].contiguous())
        dtA = dt[:, None] * A
        Cb = C * (torch.exp(dtA) - 1.0) / A
        ar = torch.arange(T, dtype=torch.float32)
        return 2.0 * torch.einsum('hn,hnl->hl', Cb, torch.exp(dtA[:, :, None] * ar[None, None])).real

    @staticmethod
    def conv(y, K, n):
        u = torch.fft.rfft(y, n=n, dim=1)
        k = torch.fft.rfft(K, n=n, dim=1).T[None]
        return torch.fft.irfft(u * k, n=n, dim=1)[:, :y.shape[1]]

    def forward(self, x, mask, rev):
        B, T, d = x.shape
        y = self.ln(x) * mask[..., None]
        n = 1 << (2 * T - 1).bit_length()
        yf = self.conv(y, self.kernel(T, 0), n)
        yr = torch.gather(y, 1, rev)                       # each trip reversed in place (padding stays at the end)
        yb = torch.gather(self.conv(yr, self.kernel(T, 1), n), 1, rev)
        y = gelu(yf + yb + y * self.D)
        return x + Fn.glu(self.out(y), -1)


class S4DCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([S4DBlock(cfg['d'], cfg['N2'], cfg.get('dt_min', 1e-3), cfg.get('dt_max', 1e-1))
                                     for _ in range(cfg['layers'])])

    def forward(self, x, mask, lens):
        B, T, d = x.shape
        t = torch.arange(T)[None, :].expand(B, T)
        r = torch.where(t < lens[:, None], lens[:, None] - 1 - t, t)
        rev = r[..., None].expand(B, T, d)
        for b in self.blocks:
            x = b(x, mask, rev)
        return x


CORES = {'cnn': CNNCore, 'gru': GRUCore, 'transformer': TransformerCore, 's4d': S4DCore}


class TraceNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d']
        self.inp = nn.Linear(N_DYN + N_ST, d)
        self.stem1 = nn.Linear(d, 2 * d); self.stem2 = nn.Linear(2 * d, d)
        self.core = CORES[cfg['arch']](cfg)
        self.lnf = nn.LayerNorm(d)
        self.hd1 = nn.Linear(d + N_ST, cfg.get('head', 64)); self.hd2 = nn.Linear(cfg.get('head', 64), 10)
        self.drop = cfg.get('dropout', 0.0)

    def encode(self, dyn, st, mask, lens):
        """-> pooled (B, d): stem -> core -> final LN -> mean over the valid blocks."""
        z = torch.cat([dyn, st[:, None, :].expand(-1, dyn.shape[1], -1)], -1)
        x = self.inp(z)
        x = x + self.stem2(gelu(self.stem1(x)))
        h = self.lnf(self.core(x, mask, lens))
        m = mask[..., None]
        return (h * m).sum(1) / m.sum(1)

    def head(self, pooled, st):
        o = self.hd2(gelu(self.hd1(torch.cat([pooled, st], 1))))
        q50 = o[:, 0]
        q25 = q50 - Fn.softplus(o[:, 1]); q05 = q25 - Fn.softplus(o[:, 2])
        q75 = q50 + Fn.softplus(o[:, 3]); q95 = q75 + Fn.softplus(o[:, 4])
        return torch.stack([q05, q25, q50, q75, q95], 1), o[:, 5:10]

    def forward(self, dyn, st, mask, lens):
        pooled = Fn.dropout(self.encode(dyn, st, mask, lens), self.drop, self.training)
        return self.head(pooled, st)


def build(cfg):
    return TraceNet(cfg)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def loss_fn(q, c, t, ct, w, wc):
    """pinball loss of the 5 log-quantiles of log(E / anchor) + wc x MSE of the log component ratios; w = trip mask."""
    taus = torch.tensor(QS, dtype=q.dtype)
    e = t[:, None] - q
    pin = torch.maximum(taus * e, (taus - 1) * e).mean(1)
    pin = (pin * w).sum() / w.sum()
    cm = (((c - ct) ** 2).mean(1) * w).sum() / w.sum()
    return pin + wc * cm, pin.detach(), cm.detach()


def export(net, cfg, extra, path):
    sd = {(k[5:] if k.startswith('core.') else k): v.detach().cpu().numpy().astype(np.float32)
          for k, v in net.state_dict().items()}
    sd.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez(path, cfg_json=np.array(json.dumps(cfg)), **sd)
