import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    """Strided conv -> conv, with a residual shortcut."""
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False); self.b2 = nn.BatchNorm1d(cout)
        self.sk = nn.Conv1d(cin, cout, 1, stride=stride, bias=False); self.bs = nn.BatchNorm1d(cout)
    def forward(self, x):
        y = F.gelu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.gelu(y + self.bs(self.sk(x)))

class ChirpNet(nn.Module):
    """Two-detector whitened strain (2 x 4096 @ 2048 Hz) -> detection logit + source properties.
    pos=True adds a small position-aware branch (1x1 conv -> flatten) so the head knows *where* the merger is."""
    OUT = 7   # logit, mu_logMc, logsig_logMc, mu_q, logsig_q, t_merge, logsig_t
    def __init__(self, widths=(24, 32, 48, 64, 96, 128), kernels=(15, 9, 7, 5, 3), stem_k=32, pos=False, hidden=128):
        super().__init__()
        w = widths
        self.pos = pos
        self.stem = nn.Sequential(nn.Conv1d(2, w[0], stem_k, stride=4, padding=stem_k // 2 - 2, bias=False), nn.BatchNorm1d(w[0]), nn.GELU())
        self.blocks = nn.Sequential(*[Block(w[i], w[i + 1], kernels[i], 2) for i in range(len(kernels))])
        c = w[len(kernels)]
        L = 4096 // 4 // (2 ** len(kernels))
        extra = 0
        if pos:
            self.red = nn.Conv1d(c, 8, 1); extra = 8 * L
        self.head = nn.Sequential(nn.Linear(2 * c + extra, hidden), nn.GELU(), nn.Linear(hidden, self.OUT))
    def forward(self, x):
        h = self.blocks(self.stem(x))
        f = [h.mean(-1), h.amax(-1)]
        if self.pos: f.append(F.gelu(self.red(h)).flatten(1))
        return self.head(torch.cat(f, 1))


class ChirpNet2S(nn.Module):
    """Two-stream ChirpNet: one shared encoder reads each detector separately; an interaction stage compares them
    position by position ([h, l, h*l, |h-l|]), so 'seen in both' is a first-class feature. Two auxiliary heads say
    whether each detector, on its own, holds a chirp."""
    OUT = 9   # logit, mu_logMc, logsig, mu_q, logsig_q, t, logsig_t, aux_H, aux_L
    def __init__(self, widths=(24, 32, 48, 64, 96, 128), kernels=(15, 9, 7, 5, 3), stem_k=32, hidden=128):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(nn.Conv1d(1, w[0], stem_k, stride=4, padding=stem_k // 2 - 2, bias=False), nn.BatchNorm1d(w[0]), nn.GELU())
        self.blocks = nn.Sequential(*[Block(w[i], w[i + 1], kernels[i], 2) for i in range(len(kernels))])
        c = w[len(kernels)]
        self.mix = nn.Sequential(nn.Conv1d(4 * c, c, 1, bias=False), nn.BatchNorm1d(c), nn.GELU())
        self.post = Block(c, c, 3, 1)
        self.head = nn.Sequential(nn.Linear(4 * c, hidden), nn.GELU(), nn.Linear(hidden, 7))
        self.aux = nn.Linear(2 * c, 1)
    def encode(self, x):          # x: (B, 2, L) -> (B, C, T) per detector
        B = x.shape[0]
        e = self.blocks(self.stem(x.transpose(0, 1).reshape(2 * B, 1, x.shape[-1])))   # [all H1 ..., all L1 ...]
        return e[:B], e[B:]
    def forward(self, x):
        h, l = self.encode(x)
        z = self.post(self.mix(torch.cat([h, l, h * l, (h - l).abs()], 1)))
        g = torch.cat([z.mean(-1), z.amax(-1), h.amax(-1), l.amax(-1)], 1)
        out = self.head(g)
        aH = self.aux(torch.cat([h.mean(-1), h.amax(-1)], 1)); aL = self.aux(torch.cat([l.mean(-1), l.amax(-1)], 1))
        return torch.cat([out, aH, aL], 1)


class Cascade(nn.Module):
    """Two-stage search: a sensitive detector proposes, a coincidence-aware network confirms. A window keeps the first
    network's score only if the second scores it above `veto`; otherwise it is pushed down by `drop`."""
    def __init__(self, finder, checker, veto, drop=30.0):
        super().__init__(); self.finder, self.checker, self.veto, self.drop = finder, checker, float(veto), float(drop)
    def forward(self, x):
        o = self.finder(x).clone(); c = self.checker(x)[:, 0]
        o[:, 0] = o[:, 0] - self.drop * (c < self.veto).float()
        return o
