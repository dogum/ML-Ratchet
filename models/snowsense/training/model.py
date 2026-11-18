"""SnowSense network: a small 2-layer GRU over a 427-day water-year window of station weather.

Works in z = sqrt(SWE mm)/10 space. Two heads:
  gauss (R1-R3): mu, sigma = softplus + 0.01; Gaussian NLL.
  quant (R4+)  : five quantiles q05, q16, q50, q84, q95 built monotone (q50 plus cumulative softplus offsets);
                 pinball loss. Handles the many zero-SWE days (a point mass) that a Gaussian smears.
quantiles(x) returns (B, T, 5) for either head. SWE = 100 max(q, 0)^2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import NCH

SIG_MIN = 0.01
TAUS = (0.05, 0.16, 0.5, 0.84, 0.95)
ZQ = (-1.6449, -0.9945, 0.0, 0.9945, 1.6449)


class SnowNet(nn.Module):
    def __init__(self, hidden=64, layers=2, nin=NCH, head='gauss'):
        super().__init__()
        self.inp = nn.Linear(nin, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.kind = head
        self.head = nn.Linear(hidden, 2 if head == 'gauss' else 5)
        self.hidden, self.layers = hidden, layers

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        h, _ = self.gru(h)
        o = self.head(h)
        if self.kind == 'gauss':
            return o[..., 0], F.softplus(o[..., 1]) + SIG_MIN
        q50 = o[..., 0]
        d = F.softplus(o[..., 1:]) + 1e-3
        return torch.stack([q50 - d[..., 0] - d[..., 1], q50 - d[..., 0], q50, q50 + d[..., 2],
                            q50 + d[..., 2] + d[..., 3]], -1)

    def quantiles(self, x):
        if self.kind == 'gauss':
            mu, sg = self(x)
            return torch.stack([mu + z * sg for z in ZQ], -1)
        return self(x)


def to_swe(mu):
    return 100.0 * torch.clamp(mu, min=0) ** 2


def nll(mu, sig, y_mm, w):
    z = torch.sqrt(torch.clamp(y_mm, min=0)) / 10.0
    l = torch.log(sig) + 0.5 * ((z - mu) / sig) ** 2
    return (l * w).sum() / w.sum()


def pinball(Q, y_mm, w):
    z = (torch.sqrt(torch.clamp(y_mm, min=0)) / 10.0).unsqueeze(-1)
    tau = torch.tensor(TAUS, dtype=Q.dtype)
    e = z - Q
    l = torch.maximum(tau * e, (tau - 1) * e).sum(-1)
    return (l * w).sum() / w.sum()
