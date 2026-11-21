"""FireDetective network (torch for training) and target transforms. The numpy port lives in release/firedetective.py."""
import numpy as np
import torch
import torch.nn as nn

# regression heads: name -> (label column in the bank, forward transform, inverse transform)
REG = ['crown', 'wind', 'lfmc', 'ros', 'm1', 'mfac', 'regime', 'wf']
BIN = ['spot', 'ia']


def t_crown(c): return np.log((np.asarray(c) + 0.02) / (1.02 - np.asarray(c)))
def i_crown(y): e = np.exp(y); return (1.02 * e - 0.02) / (1 + e)


def targets(lab, labels):
    L = {k: lab[:, i] for i, k in enumerate(labels)}
    Y = np.stack([t_crown(L['crown']), np.log(np.maximum(L['wind'], 0.3)), L['lfmc'] / 100.0, L['log_ros_adj'],
                  np.log(np.maximum(L['m1_eff'], 1.0)), L['log_mfac'], L['wind_regime'], L['log_wf']], 1)
    B = np.stack([L['spot'], L['ia']], 1)
    return Y.astype(np.float32), B.astype(np.float32)


class Net(nn.Module):
    def __init__(self, n_seq, n_stat, h=48, c=64, d=128, k=3):
        super().__init__()
        self.emb = nn.Linear(n_seq, h)
        self.c1 = nn.Conv1d(h, c, k, padding=k // 2)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.st = nn.Linear(n_stat, 32)
        self.f1 = nn.Linear(2 * c + 32, d)
        self.f2 = nn.Linear(d, d)
        self.mu = nn.Linear(d, len(REG))
        self.ls = nn.Linear(d, len(REG))
        self.bn = nn.Linear(d, len(BIN))

    def forward(self, xs, xt, mask):
        # xs [B, T, F] normalised; mask [B, T]
        h = torch.relu(self.emb(xs)) * mask[..., None]
        h = h.transpose(1, 2)
        h = torch.relu(self.c1(h)) * mask[:, None, :]
        h = torch.relu(self.c2(h)) * mask[:, None, :]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        mean = h.sum(2) / n
        mx = (h - 1e4 * (1 - mask[:, None, :])).max(2).values
        s = torch.relu(self.st(xt))
        z = torch.relu(self.f1(torch.cat([mean, mx, s], 1)))
        z = torch.relu(self.f2(z))
        return self.mu(z), self.ls(z).clamp(-5, 3), self.bn(z)


class Ens(nn.Module):
    """Seed ensemble: mean of member means; variance = mean member variance + spread of member means."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, xs, xt, mask):
        outs = [m(xs, xt, mask) for m in self.members]
        mu = torch.stack([o[0] for o in outs])
        var = torch.stack([torch.exp(2 * o[1]) for o in outs]).mean(0) + mu.var(0, unbiased=False)
        p = torch.stack([torch.sigmoid(o[2]) for o in outs]).mean(0).clamp(1e-6, 1 - 1e-6)
        return mu.mean(0), 0.5 * torch.log(var), torch.log(p / (1 - p))
