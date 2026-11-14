"""GridPulse network: a small MLP with Gaussian heads (mean, log sigma) per target. numpy forward for release."""
import numpy as np
try:
    import torch, torch.nn as nn
except Exception:          # numpy-only use
    torch = None

NIN = 81 + 3
NOUT = 7
HID = (256, 256, 128)

if torch is not None:
    class Net(nn.Module):
        def __init__(self, nin=NIN, nout=NOUT, hid=HID):
            super().__init__()
            layers, a = [], nin
            for h in hid:
                layers += [nn.Linear(a, h), nn.GELU()]; a = h
            self.body = nn.Sequential(*layers)
            self.mu = nn.Linear(a, nout); self.ls = nn.Linear(a, nout)

        def forward(self, x):
            h = self.body(x)
            return self.mu(h), self.ls(h).clamp(-6, 3)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def np_forward(W, x):
    """W: dict of numpy arrays from export (b{i}_w, b{i}_b, mu_w, mu_b, ls_w, ls_b, xm, xs, ym, ys)."""
    h = (x - W['xm']) / W['xs']
    i = 0
    while f'b{i}_w' in W:
        h = h @ W[f'b{i}_w'].T + W[f'b{i}_b']
        h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
        i += 1
    mu = h @ W['mu_w'].T + W['mu_b']; ls = np.clip(h @ W['ls_w'].T + W['ls_b'], -6, 3)
    return mu * W['ys'] + W['ym'], np.exp(ls) * W['ys']


def erf(x):
    # Abramowitz-Stegun 7.1.26 is too coarse for 1e-5 parity; use scipy if present
    try:
        from scipy.special import erf as e
        return e(x)
    except Exception:
        import math
        return np.vectorize(math.erf)(x)
