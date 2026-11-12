"""SwellNet: a small 2-D CNN over the buoy spectrogram (frequency x time) with five input channels.
Outputs: mean and log-sigma of log10 D, t0 (h, relative to window centre) and log10 storm Hs, plus a 2-vector whose
angle is the swell direction and length the von Mises concentration, plus a detection logit (is a storm swell arriving
near the window centre?)."""
import numpy as np, torch, torch.nn as nn
OUT = ['logD', 't0', 'loghs']; NIN = 5

def features(X):
    """X: (B, 3, 80, 34) raw [E m^2/Hz, r1 cos, r1 sin] with NaN -> (B, 5, 34, 80) float32: log energy, anomaly against
    the window's per-frequency median, r1 cos, r1 sin, data mask. Frequency is the row axis, time the column axis."""
    X = np.asarray(X, np.float32); E = X[:, 0]; m = np.isfinite(E)
    L = np.log10(np.maximum(np.where(m, E, 0.005), 0.005)); L = np.where(m, L, np.nan)
    med = np.nanmedian(np.where(np.isfinite(L), L, np.nan), axis=1, keepdims=True); med = np.where(np.isfinite(med), med, -2.3)
    Ln = np.where(m, (L + 0.5) / 1.2, 0.0); A = np.where(m, (L - med) / 0.6, 0.0)
    C = np.nan_to_num(X[:, 1]); S = np.nan_to_num(X[:, 2]); C = np.where(m, C, 0); S = np.where(m, S, 0)
    F = np.stack([Ln, A, C, S, m.astype(np.float32)], 1)
    return np.ascontiguousarray(F.transpose(0, 1, 3, 2)).astype(np.float32)

class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, stride, 1); self.b1 = nn.BatchNorm2d(co); self.c2 = nn.Conv2d(co, co, 3, 1, 1); self.b2 = nn.BatchNorm2d(co)
        self.sk = nn.Conv2d(ci, co, 1, stride, 0); self.bs = nn.BatchNorm2d(co); self.act = nn.GELU()
    def forward(self, x):
        y = self.act(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return self.act(y + self.bs(self.sk(x)))

class SwellNet(nn.Module):
    def __init__(self, widths=(24, 32, 48, 64), red=24, hidden=128):
        super().__init__()
        self.stem = nn.Conv2d(NIN, widths[0], 3, 1, 1); self.bstem = nn.BatchNorm2d(widths[0]); self.act = nn.GELU()
        self.body = nn.ModuleList([Block(widths[i], widths[i + 1], 2) for i in range(len(widths) - 1)])
        self.red = nn.Conv2d(widths[-1], red, 1); self.bred = nn.BatchNorm2d(red)
        self.head = nn.Sequential(nn.Linear(red * 5 * 10, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * len(OUT) + 3))
        self.register_buffer('mu_y', torch.zeros(len(OUT))); self.register_buffer('sd_y', torch.ones(len(OUT)))
    def forward(self, x):
        h = self.act(self.bstem(self.stem(x)))
        for b in self.body: h = b(h)
        h = self.act(self.bred(self.red(h)))
        return self.head(h.flatten(1))

def targets(y):
    """bank label rows -> (n, 3) [log10 D_km, t0_rel_h, log10 hs] and direction (deg)"""
    return np.stack([np.log10(y[:, 0]), y[:, 1], np.log10(y[:, 3])], 1), y[:, 2]
