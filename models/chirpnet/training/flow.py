"""ChirpNet-PE: neural posterior estimation in the spirit of DINGO.
A CNN embeds two-detector whitened strain (plus a summary of each detector's noise spectrum); a conditional
RealNVP flow turns that embedding into a full posterior over source parameters."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from model import ChirpNet

PARAMS = ['log_mc', 'q', 'chi_eff', 'log10_dL', 'cos_iota', 't_c']
MU = torch.tensor([2.88, 0.59, 0.0, 3.07, 0.0, 1.33]); SD = torch.tensor([0.59, 0.22, 0.35, 0.36, 0.58, 0.31])

class Coupling(nn.Module):
    def __init__(self, dim, mask, ctx, hidden=192):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = nn.Sequential(nn.Linear(dim + ctx, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def _st(self, xm, c):
        s, t = self.net(torch.cat([xm, c], 1)).chunk(2, 1)
        s = 2.5 * torch.tanh(s / 2.5)
        return s * (1 - self.mask), t * (1 - self.mask)
    def forward(self, x, c):          # data -> latent
        s, t = self._st(x * self.mask, c)
        return x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t), s.sum(1)
    def inverse(self, z, c):          # latent -> data
        s, t = self._st(z * self.mask, c)
        return z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))

class Flow(nn.Module):
    def __init__(self, dim=6, ctx=128, n=8):
        super().__init__()
        masks = []
        for i in range(n):
            m = torch.zeros(dim); idx = torch.randperm(dim, generator=torch.Generator().manual_seed(i))[: dim // 2]; m[idx] = 1; masks.append(m)
        self.layers = nn.ModuleList([Coupling(dim, m, ctx) for m in masks])
        self.dim = dim
    def log_prob(self, x, c):
        ld = 0
        for L in self.layers:
            x, l = L(x, c); ld = ld + l
        return -0.5 * (x ** 2).sum(1) - 0.5 * self.dim * math.log(2 * math.pi) + ld
    def sample(self, c, n):
        z = torch.randn(n, self.dim)
        c = c.expand(n, -1)
        for L in reversed(self.layers): z = L.inverse(z, c)
        return z

class ChirpPE(nn.Module):
    def __init__(self, widths=(24, 32, 48, 64, 96, 128), n_asd=20):
        super().__init__()
        self.cnn = ChirpNet(widths=widths)      # we use its trunk only
        c = widths[-1]
        self.ctx = nn.Sequential(nn.Linear(2 * c + n_asd, 256), nn.GELU(), nn.Linear(256, 128))
        self.flow = Flow()
    def context(self, x, asd):
        h = self.cnn.blocks(self.cnn.stem(x))
        return self.ctx(torch.cat([h.mean(-1), h.amax(-1), asd], 1))
    def log_prob(self, theta, x, asd):
        return self.flow.log_prob((theta - MU) / SD, self.context(x, asd)) - torch.log(SD).sum()
    @torch.no_grad()
    def sample(self, x, asd, n=4000):
        c = self.context(x, asd)
        return self.flow.sample(c, n) * SD + MU
