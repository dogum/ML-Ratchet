"""BeatNet: a small 1-D CNN over a 2.5 s ECG window centred on the beat (plus, optionally, the record's median
neighbour window as a second channel), joined with six rhythm features. Outputs: 4 class logits (N, S, V, F) and
the QRS width (log ms) with a log-sigma; from rung 2 on, a 7th output: the logit that the beat sits in atrial fibrillation."""
import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(self, ci, co, s):
        super().__init__(); self.c1 = nn.Conv1d(ci, co, 7, s, 3); self.b1 = nn.BatchNorm1d(co); self.c2 = nn.Conv1d(co, co, 5, 1, 2); self.b2 = nn.BatchNorm1d(co)
        self.sk = nn.Conv1d(ci, co, 1, s); self.bs = nn.BatchNorm1d(co); self.a = nn.GELU()
    def forward(self, x): return self.a(self.b2(self.c2(self.a(self.b1(self.c1(x))))) + self.bs(self.sk(x)))

class BeatNet(nn.Module):
    def __init__(self, cin=1, widths=(16, 24, 32, 48, 64), nr=6, ctr=7, nout=6):
        super().__init__(); L = []; c = cin
        for i, w in enumerate(widths): L.append(Block(c, w, 1 if i == 0 else 2)); c = w
        self.body = nn.Sequential(*L); self.ctr = ctr; self.cin = cin
        self.head = nn.Sequential(nn.Linear(c * (ctr + 2) + nr, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, nout))
    def forward(self, x, r):
        h = self.body(x); m = h.shape[-1] // 2; k = self.ctr // 2
        z = torch.cat([h[:, :, m - k:m + k + 1].flatten(1), h.mean(-1), h.amax(-1), r], 1); return self.head(z)
