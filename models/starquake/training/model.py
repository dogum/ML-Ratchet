"""StarQuake network: two small CNN branches and a head.
- 1-D branch over the log power spectrum on 1024 log-spaced bins (3-283 uHz): where is the hump (numax)?
- 2-D branch over the 48 x 110 comb map (autocorrelation of the spectrum around 48 trial centres, lags in units of
  the expected large separation): how far apart are the teeth of the comb (dnu)?
- Head: flattened features (position matters, so no global pooling) + 2 scalars (log observing length, duty cycle)
  -> mean and log-sigma of ln numax and ln dnu (and, with det=True, a logit: are oscillations in the band at all?)."""
import torch, torch.nn as nn
OUT = ['ln_numax', 'ln_dnu']

class B1(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.c1 = nn.Conv1d(ci, co, 7, 2, 3); self.b1 = nn.BatchNorm1d(co); self.c2 = nn.Conv1d(co, co, 7, 1, 3); self.b2 = nn.BatchNorm1d(co)
        self.sk = nn.Conv1d(ci, co, 1, 2); self.bs = nn.BatchNorm1d(co); self.a = nn.GELU()
    def forward(self, x): return self.a(self.b2(self.c2(self.a(self.b1(self.c1(x))))) + self.bs(self.sk(x)))

class B2(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.c1 = nn.Conv2d(ci, co, 3, 2, 1); self.b1 = nn.BatchNorm2d(co); self.c2 = nn.Conv2d(co, co, 3, 1, 1); self.b2 = nn.BatchNorm2d(co)
        self.sk = nn.Conv2d(ci, co, 1, 2); self.bs = nn.BatchNorm2d(co); self.a = nn.GELU()
    def forward(self, x): return self.a(self.b2(self.c2(self.a(self.b1(self.c1(x))))) + self.bs(self.sk(x)))

class StarQuake(nn.Module):
    def __init__(self, w1=(24, 32, 48, 48, 32), w2=(24, 32, 48, 48), hid=160, det=False):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.GELU()); L = []; c = 16
        for w in w1: L.append(B1(c, w)); c = w
        self.r1 = nn.Sequential(*L); n1 = c * (1024 // 2 ** (len(w1) + 1))
        self.s2 = nn.Sequential(nn.Conv2d(1, 12, 3, (1, 2), 1), nn.BatchNorm2d(12), nn.GELU()); L = []; c = 12
        for w in w2: L.append(B2(c, w)); c = w
        self.r2 = nn.Sequential(*L); n2 = c * 3 * 4
        self.head = nn.Sequential(nn.Linear(n1 + n2 + 2, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 2 * len(OUT) + int(det)))
    def forward(self, x1, x2, s):
        h1 = self.r1(self.s1(((x1 - 2.5) / 2).unsqueeze(1))).flatten(1); h2 = self.r2(self.s2((x2 * 4).unsqueeze(1))).flatten(1)
        return self.head(torch.cat([h1, h2, s], 1))
