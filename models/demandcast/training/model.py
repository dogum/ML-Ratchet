"""DemandCast networks: one small MLP per output (demand, wind, solar), each predicting 24 hourly means and sigmas.

demand: mu = log(y / prior) per hour (0 = the physics prior), sigma on the log scale
wind/solar: CF = prior CF + delta, sigma in CF units (solar forced to 0 at night)
Inputs are standardised with the training-bank mean/std stored in the checkpoint.
"""
import numpy as np, torch, torch.nn as nn

FLOOR = {'d': 0.004, 'w': 0.005, 's': 0.003}


class Net(nn.Module):
    def __init__(self, n_in, hidden=(256, 256), n_out=24, drop=0.0):
        super().__init__()
        layers, h0 = [], n_in
        for h in hidden:
            layers += [nn.Linear(h0, h), nn.SiLU()] + ([nn.Dropout(drop)] if drop else [])
            h0 = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(h0, n_out); self.ls = nn.Linear(h0, n_out)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.ls.weight); nn.init.constant_(self.ls.bias, -3.0)
        self.register_buffer('xm', torch.zeros(n_in)); self.register_buffer('xs', torch.ones(n_in))

    def forward(self, x):
        z = self.body((x - self.xm) / self.xs)
        return self.mu(z), self.ls(z)


def head(kind, net, x, prior=None):
    """(mu, sigma) in target units. kind 'd' -> log ratio; 'w'/'s' -> CF (prior added)."""
    m, ls = net(x)
    sig = nn.functional.softplus(ls) + FLOOR[kind]
    if kind != 'd':
        m = m + prior
    return m, sig


def nll(mu, sig, y, mask):
    z = (y - mu) / sig
    l = torch.log(sig) + 0.5 * z ** 2
    return (l * mask).sum() / mask.sum().clamp(min=1)


def export_numpy(net):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return sd


def np_forward(W, x, kind, prior=None):
    """Pure-numpy forward pass (release / JS reference)."""
    z = (x - W['xm']) / W['xs']
    i = 0
    while f'body.{i}.weight' in W:
        z = z @ W[f'body.{i}.weight'].T + W[f'body.{i}.bias']
        z = z / (1 + np.exp(-z))
        i += 2
    mu = z @ W['mu.weight'].T + W['mu.bias']
    ls = z @ W['ls.weight'].T + W['ls.bias']
    sig = np.logaddexp(0, ls) + FLOOR[kind]
    if kind != 'd':
        mu = mu + prior
    return mu, sig
