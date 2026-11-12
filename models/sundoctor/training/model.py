"""SunDoctor network: reads D days x 96 solar-time steps of a PV system as an image (rows = days, columns = time of day)."""
import torch, torch.nn as nn, torch.nn.functional as F

REG = ['pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'dfrac']      # per-day regressions (mu, log sigma)
SCALE = [0.1, 0.05, 0.1, 0.03, 0.03, 0.1, 0.1]                         # typical size of each quantity
GLOB = ['eta0', 'ac_limit']; GSCALE = [0.05, 0.1]
N_IN = 10                                                               # 8 data channels + reference level + reference flag

class Block2(nn.Module):
    def __init__(self, cin, cout, stride=(1, 2), k=(3, 3)):
        super().__init__(); p = (k[0] // 2, k[1] // 2)
        self.c1 = nn.Conv2d(cin, cout, k, stride, p, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, k, 1, p, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sk = nn.Conv2d(cin, cout, 1, stride, bias=False); self.bs = nn.BatchNorm2d(cout)
    def forward(self, x):
        return F.gelu(self.b2(self.c2(F.gelu(self.b1(self.c1(x))))) + self.bs(self.sk(x)))

class TBlock(nn.Module):
    """residual dilated convolution along the day axis"""
    def __init__(self, c, dil):
        super().__init__(); self.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil, bias=False); self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, 1, bias=False); self.b2 = nn.BatchNorm1d(c)
    def forward(self, x):
        return F.gelu(x + self.b2(self.c2(F.gelu(self.b1(self.c1(x))))))

class SunDoctor(nn.Module):
    def __init__(self, widths=(24, 32, 48, 64, 96), tw=128, hidden=128):
        super().__init__(); w = widths
        self.stem = nn.Sequential(nn.Conv2d(N_IN, w[0], (1, 7), padding=(0, 3), bias=False), nn.BatchNorm2d(w[0]), nn.GELU())
        self.blocks = nn.Sequential(*[Block2(w[i], w[i + 1]) for i in range(len(w) - 1)])
        c = w[-1]
        self.proj = nn.Sequential(nn.Conv1d(2 * c, tw, 1, bias=False), nn.BatchNorm1d(tw), nn.GELU())
        self.tblocks = nn.Sequential(*[TBlock(tw, d) for d in (1, 2, 4, 1)])
        self.day = nn.Conv1d(tw, 2 * len(REG) + 1, 1)
        self.glob = nn.Sequential(nn.Linear(2 * tw, hidden), nn.GELU(), nn.Linear(hidden, 2 * len(GLOB)))
    def forward(self, x):                       # x: (B, 10, D, 96)
        h = self.blocks(self.stem(x))           # (B, C, D, 6)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        z = self.tblocks(self.proj(h))          # (B, tw, D)
        day = self.day(z)                       # (B, 15, D): mu x7, log sigma x7, step logit
        g = self.glob(torch.cat([z.mean(-1), z.amax(-1)], 1))
        return day, g
