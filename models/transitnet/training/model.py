"""TransitNet: four 1-D convolutional branches (local transit, secondary eclipse, whole orbit, odd/even transits) plus the
scalars, merged into a head that returns class logits (planet, eclipsing binary, blended binary, nothing) and, for a
planet, four parameters each with its own error bar: log10 radius ratio, impact parameter, log10 a/R*, log10 duration."""
import torch, torch.nn as nn, numpy as np
PARAMS = ['log_k', 'b', 'log_aR', 'log_T14']; CLASSES = ['planet', 'eb', 'blend', 'nothing']; N_S = 10

def base(s):
    """what the folded transit says without a model: k ~ sqrt(depth), duration ~ the candidate's, a/R* ~ P / (pi T14);
    networks with residual=True predict the parameters as corrections to these (s: raw scalars, numpy)"""
    s = np.asarray(s, np.float64); z = np.zeros((len(s), 4))
    z[:, 0] = 0.5 * s[:, 2]; z[:, 2] = s[:, 0] - np.log10(np.pi) - s[:, 1] + np.log10(24); z[:, 3] = s[:, 1]; return z

def branch(ci, widths, pools):
    L = []; c = ci
    for w, p in zip(widths, pools):
        L += [nn.Conv1d(c, w, 5, padding=2), nn.BatchNorm1d(w), nn.GELU(), nn.Conv1d(w, w, 5, padding=2), nn.BatchNorm1d(w), nn.GELU()]
        if p: L.append(nn.MaxPool1d(2))
        c = w
    return nn.Sequential(*L), c

class TransitNet(nn.Module):
    def __init__(self, width=1.0, cent=False):
        super().__init__(); W = lambda *x: [int(v * width) for v in x]; self.use_cent = cent
        self.loc, cl = branch(2, W(32, 64, 128, 128), [1, 1, 1, 0]); self.sec, cs = branch(2, W(16, 32, 64), [1, 1, 0])
        self.glo, cg = branch(2, W(16, 32, 64, 128), [1, 1, 1, 0]); self.oe, co = branch(4, W(32, 64, 64), [1, 1, 0])
        d = 2 * (cl + cs + cg + co) + N_S
        if cent: self.cen, cc = branch(4, W(32, 64), [1, 0]); d += 2 * cc + 1          # centroid motion view and its noise
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 4 + 2 * len(PARAMS)))
    def forward(self, loc, sec, glo, oe, s, cent=None, sc=None):
        pool = lambda h: torch.cat([h.mean(-1), h.amax(-1)], 1)
        parts = [pool(self.loc(loc)), pool(self.sec(sec)), pool(self.glo(glo)), pool(self.oe(oe)), s]
        if self.use_cent: parts += [pool(self.cen(cent.clamp(-20, 20))), sc[:, None]]
        z = torch.cat(parts, 1)
        return self.head(z)
