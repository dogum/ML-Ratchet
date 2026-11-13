"""CellSense NMC811 network: the same 1-D residual CNN as CellSense (model.Block), along the voltage axis of a C/20 discharge
from 4.15 V to 2.50 V, read with the same cell's first test when there is one. A curve that stops early (a partial
discharge) is masked below where it stopped. Six input channels:
  q1          capacity discharged at each voltage (Ah), last value carried forward past the end
  q0          the reference curve (zeros when there is none)
  10 (q1-q0)  the difference, where there is a reference
  m1, m0      where each curve has data
  h           1 if there is a reference
Each label comes out as a mean and a log standard deviation (beta-NLL), in units of its bank standard deviation."""
import numpy as np, torch, torch.nn as nn
import model as M
LABELS = ['q_total', 'fade', 'lli', 'lam_ne', 'lam_pe']
N_IN = 6


def fill(a):
    """carry the last finite value forward (curve ended) and the first one backward; mask of where data are"""
    a = np.asarray(a, np.float64); m = np.isfinite(a); out = np.where(m, a, 0.0); B, L = a.shape
    idx = np.where(m, np.arange(L)[None], 0); np.maximum.accumulate(idx, axis=1, out=idx); out = np.take_along_axis(out, idx, 1)
    first = np.argmax(m, 1); out = np.where(np.arange(L)[None] < first[:, None], out[np.arange(B), first][:, None], out)
    out[~m.any(1)] = 0.0; return out, m.astype(np.float32)


def q_last(Q1):
    """the last recorded capacity on the grid (where the curve stops; the end of a full curve)"""
    Q1 = np.atleast_2d(Q1); return np.nanmax(np.where(np.isfinite(Q1), Q1, -np.inf), 1)


def features(Q1, Q0=None, smooth=0):
    """smooth: Gaussian width in grid points (6.6 mV each) applied to the filled curves (model.gsmooth)"""
    Q1 = np.atleast_2d(Q1); B, L = Q1.shape
    has = np.zeros(B, np.float32) if Q0 is None else np.isfinite(np.atleast_2d(Q0)).any(1).astype(np.float32)
    q1, m1 = fill(Q1); q0, m0 = (np.zeros_like(q1), np.zeros_like(m1)) if Q0 is None else fill(np.atleast_2d(Q0)); h = has[:, None] * np.ones((1, L), np.float32)
    if smooth: q1, q0 = M.gsmooth(q1, smooth), M.gsmooth(q0, smooth)
    X = np.stack([q1, q0 * h, (q1 - q0) * 10 * h, m1, m0 * h, h], 1)
    return np.nan_to_num(X).astype(np.float32)


class Net(nn.Module):
    def __init__(self, widths=(32, 48, 64, 96, 128), n_out=len(LABELS)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(N_IN, widths[0], 7, 1, 3), nn.BatchNorm1d(widths[0]), nn.GELU())
        blocks, c = [], widths[0]
        for w in widths: blocks.append(M.Block(c, w, 2)); c = w
        self.blocks = nn.Sequential(*blocks); self.extra = 0
        self.head = nn.Sequential(nn.Linear(2 * c, 256), nn.GELU(), nn.Linear(256, 2 * n_out))

    def forward(self, x, s=None):
        h = self.blocks(self.stem(x)); return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))
