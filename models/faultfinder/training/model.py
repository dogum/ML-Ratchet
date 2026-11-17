"""FaultFinder network: a small 1-D CNN over one day (96 x 15 min) of trend-log features.
Heads: class logits (none + 18 fault classes), per-class fault parameter (severity / position / bias),
energy-waste fraction of the day."""
import torch
import torch.nn as nn

from features import NF
from common import NC


class FFNet(nn.Module):
    def __init__(self, nf=NF, nc=NC, w=48, drop=0.2):
        super().__init__()

        def blk(a, b, k):
            return nn.Sequential(nn.Conv1d(a, b, k, padding=k // 2), nn.BatchNorm1d(b), nn.ReLU())
        self.c1 = blk(nf, w, 5)
        self.c2 = blk(w, w, 5)
        self.c3 = blk(w, 2 * w, 5)
        self.c4 = blk(2 * w, 2 * w, 3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Sequential(nn.Linear(4 * w, 128), nn.ReLU(), nn.Dropout(drop))
        self.cls = nn.Linear(128, nc)
        self.par = nn.Linear(128, nc)
        self.waste = nn.Linear(128, 1)

    def forward(self, x):
        h = self.c1(x)
        h = self.pool(self.c2(h))
        h = self.pool(self.c3(h))
        h = self.c4(h)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        h = self.fc(h)
        return self.cls(h), self.par(h), self.waste(h)[:, 0]
