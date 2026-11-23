"""QuakeNet: a 1-D convolutional network over 41 minutes of three-component ground velocity (from 1 minute before P),
plus the record's amplitude scale. Returns magnitude Mw, epicentral distance, depth (log10 km) each with an error bar, and
the back-azimuth as a von Mises direction with a concentration."""
import torch, torch.nn as nn, numpy as np
OUT = ['mw', 'dist', 'logz']

def features(x):
    """x: (B, 3, L) normalized velocity -> (B, 6, L): the traces and their log envelopes (8 s smoothing)"""
    env = torch.nn.functional.avg_pool1d(x.abs(), 9, 1, 4); return torch.cat([x, torch.log10(env + 1e-3)], 1)

class Block(nn.Module):
    def __init__(self, ci, co, s):
        super().__init__(); self.c1 = nn.Conv1d(ci, co, 7, s, 3); self.b1 = nn.BatchNorm1d(co); self.c2 = nn.Conv1d(co, co, 7, 1, 3); self.b2 = nn.BatchNorm1d(co)
        self.sk = nn.Conv1d(ci, co, 1, s); self.bs = nn.BatchNorm1d(co); self.a = nn.GELU()
    def forward(self, x): return self.a(self.b2(self.c2(self.a(self.b1(self.c1(x))))) + self.bs(self.sk(x)))

class QuakeNet(nn.Module):
    def __init__(self, widths=(32, 48, 64, 96, 128, 160, 192)):
        super().__init__(); L = []; c = 6
        for w in widths: L.append(Block(c, w, 2)); c = w
        self.body = nn.Sequential(*L); self.head = nn.Sequential(nn.Linear(2 * c + 2, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2 * len(OUT) + 2))
    def forward(self, x, s):
        h = self.body(features(x)); return self.head(torch.cat([h.mean(-1), h.amax(-1), s], 1))
