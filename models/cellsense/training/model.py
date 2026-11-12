"""CellSense network: a 1-D residual CNN along the voltage axis of a 4C discharge, read together with the same cell's
curve from when it was new (optional), returning each label with its own error bar."""
import torch, torch.nn as nn, numpy as np
LABELS = ['q_total', 'q_cc', 'ir', 'lli', 'lam_ne', 'lam_pe', 'r_growth', 'fade']
N_IN = 8

def gsmooth(a, sigma):
    """Gaussian smoothing along the voltage axis, edges held (scipy's gaussian_filter1d, mode='nearest', truncate=4)"""
    if not sigma: return a
    r = int(4.0 * sigma + 0.5); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.concatenate([np.repeat(a[:, :1], r, 1), a, np.repeat(a[:, -1:], r, 1)], 1)
    return np.stack([np.convolve(row, k[::-1], 'valid') for row in p]).astype(a.dtype)

def features(Q1, T1, Q0=None, T0=None, smooth=0):
    """(B, 250) arrays on the voltage grid -> (B, 8, 250) input channels. Missing values (NaN) are filled and masked.
    smooth: Gaussian width in grid points (6 mV each) applied to the filled curves; the lifetime network uses 4 (24 mV), because
    the simulator's 4C curves carry sharper graphite-staging features than real cells do."""
    B, L = Q1.shape; has = np.zeros(B, np.float32) if Q0 is None else np.isfinite(Q0).any(1).astype(np.float32)
    if Q0 is None: Q0 = np.zeros_like(Q1); T0 = np.full_like(T1, 30.0)
    def fill(a, default):
        m = np.isfinite(a); a = np.where(m, a, np.nan); out = a.copy()
        for b in range(len(a)):                      # carry the last value forward (curve ended), first value backward
            r = out[b]; ok = np.isfinite(r)
            if not ok.any(): r[:] = default; continue
            idx = np.where(ok, np.arange(L), 0); np.maximum.accumulate(idx, out=idx); r[:] = r[idx]; r[:np.argmax(ok)] = r[np.argmax(ok)]
        return out, m.astype(np.float32)
    q1, m1 = fill(Q1, 0.0); t1, _ = fill(T1, 30.0); q0, m0 = fill(Q0, 0.0); t0, _ = fill(T0, 30.0)
    if smooth: q1, t1, q0, t0 = (gsmooth(a, smooth) for a in (q1, t1, q0, t0))
    h = has[:, None] * np.ones((1, L), np.float32)
    X = np.stack([q1, q0 * h, (q1 - q0) * 10 * h, (t1 - 30) / 10, (t0 - 30) / 10 * h, m1, m0 * h, h], 1)
    return np.nan_to_num(X).astype(np.float32)

class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 5, stride, 2); self.b1 = nn.BatchNorm1d(co); self.c2 = nn.Conv1d(co, co, 5, 1, 2); self.b2 = nn.BatchNorm1d(co)
        self.sk = nn.Conv1d(ci, co, 1, stride); self.bs = nn.BatchNorm1d(co); self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + self.bs(self.sk(x)))

class CellSense(nn.Module):
    def __init__(self, widths=(48, 64, 96, 128, 160), n_out=len(LABELS), extra=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(N_IN, widths[0], 7, 1, 3), nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks, c = [], widths[0]
        for w in widths: blocks.append(Block(c, w, 2)); c = w
        self.blocks = nn.Sequential(*blocks); self.extra = extra
        self.head = nn.Sequential(nn.Linear(2 * c + extra, 256), nn.GELU(), nn.Linear(256, 2 * n_out))
    def forward(self, x, s=None):
        h = self.blocks(self.stem(x)); z = torch.cat([h.mean(-1), h.amax(-1)] + ([s] if self.extra else []), 1)
        return self.head(z)
