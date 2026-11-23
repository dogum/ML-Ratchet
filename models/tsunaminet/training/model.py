"""TsunamiNet: a per-station 1-D CNN encoder, a permutation-invariant set pooling over up to 3 DARTs, and a
conditional RealNVP normalizing flow (ChirpNet-PE pattern) over theta = (Mw, along-strike centroid [100 km],
deep-row moment fraction, log10 rupture length [km])."""
import math, torch, torch.nn as nn
from features import NCH, NSC
MU = torch.tensor([8.3, 0.0, 0.5, 2.4]); SD = torch.tensor([0.58, 1.2, 0.3, 0.35])

class Coupling(nn.Module):
    def __init__(self, dim, mask, ctx, hidden=128):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = nn.Sequential(nn.Linear(dim + ctx, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def _st(self, xm, c):
        s, t = self.net(torch.cat([xm, c], 1)).chunk(2, 1)
        s = 2.5 * torch.tanh(s / 2.5)
        return s * (1 - self.mask), t * (1 - self.mask)
    def forward(self, x, c):
        s, t = self._st(x * self.mask, c)
        return x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t), s.sum(1)
    def inverse(self, z, c):
        s, t = self._st(z * self.mask, c)
        return z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))

class Flow(nn.Module):
    def __init__(self, dim=4, ctx=128, n=8, hidden=128):
        super().__init__()
        masks = []
        for i in range(n):
            m = torch.zeros(dim); idx = torch.randperm(dim, generator=torch.Generator().manual_seed(i))[: dim // 2]; m[idx] = 1; masks.append(m)
        self.layers = nn.ModuleList([Coupling(dim, m, ctx, hidden) for m in masks]); self.dim = dim
    def log_prob(self, x, c):
        ld = 0
        for L in self.layers:
            x, l = L(x, c); ld = ld + l
        return -0.5 * (x ** 2).sum(1) - 0.5 * self.dim * math.log(2 * math.pi) + ld
    def sample(self, c, n, z=None):
        z = torch.randn(n, self.dim) if z is None else z
        c = c.expand(n, -1)
        for L in reversed(self.layers): z = L.inverse(z, c)
        return z

class TsunamiNet(nn.Module):
    def __init__(self, w=(48, 64, 96, 128), ctx=128, flow_hidden=128, n_flow=8, extra=0):
        super().__init__()
        self.extra = extra
        c0 = NCH; L = []
        for i, c in enumerate(w):
            L += [nn.Conv1d(c0, c, 5 if i < 3 else 3, stride=1 if i == 0 else 2, padding=2 if i < 3 else 1), nn.GELU()]; c0 = c
        self.cnn = nn.Sequential(*L)
        self.st = nn.Sequential(nn.Linear(2 * w[-1] + NSC, 128), nn.GELU(), nn.Linear(128, 128))
        self.ctx = nn.Sequential(nn.Linear(256 + 1 + extra, 256), nn.GELU(), nn.Linear(256, ctx))
        self.flow = Flow(4, ctx, n_flow, flow_hidden)
    def context(self, X, S, M, E=None):
        B, K = X.shape[:2]
        h = self.cnn(X.reshape(B * K, *X.shape[2:]))
        e = self.st(torch.cat([h.mean(-1), h.amax(-1), S.reshape(B * K, -1)], 1)).reshape(B, K, -1)
        m = M.float().unsqueeze(-1)
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1); mx = (e - 1e4 * (1 - m)).amax(1)
        parts = [mean, mx, m.sum(1) / K] + ([E] if self.extra else [])
        return self.ctx(torch.cat(parts, 1))
    def log_prob(self, theta, X, S, M, E=None):
        return self.flow.log_prob((theta - MU) / SD, self.context(X, S, M, E)) - torch.log(SD).sum()
    @torch.no_grad()
    def sample(self, X, S, M, n=1000, z=None, E=None):
        c = self.context(X, S, M, E)
        return self.flow.sample(c, n, z) * SD + MU
