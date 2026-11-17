"""LoadCast network: a small MLP per hour with a heteroscedastic Gaussian head.

cold: mu = f(x)                      (units: A * IREF; x = metadata + weather + calendar)
warm: mu = towt_hat + f(x)           (units: history mean S; x adds the history encoding; the network learns a
                                      physics-informed correction to a TOWT model fitted on the building's own history)
sigma = exp(g(x)) in the same units.
"""
import numpy as np
import torch, torch.nn as nn


class MLP(nn.Module):
    def __init__(self, nin, width=192, depth=3, dropout=0.0):
        super().__init__()
        self.register_buffer('mu_x', torch.zeros(nin)); self.register_buffer('sd_x', torch.ones(nin))
        L = []; d = nin
        for _ in range(depth):
            L += [nn.Linear(d, width), nn.SiLU()] + ([nn.Dropout(dropout)] if dropout > 0 else []); d = width
        self.body = nn.Sequential(*L); self.head = nn.Linear(d, 2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.nin, self.width, self.depth = nin, width, depth

    def forward(self, x, base):
        o = self.head(self.body((x - self.mu_x) / self.sd_x))
        return base + o[:, 0], o[:, 1].clamp(-7, 4)


def to_numpy(net):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return sd


def np_forward(W, x, base):
    """Numpy inference (release + JS reference). W: dict from to_numpy. Returns mu, log_sigma."""
    h = (x - W['mu_x']) / W['sd_x']
    for i in sorted(int(k.split('.')[1]) for k in W if k.startswith('body.') and k.endswith('.weight')):
        h = h @ W[f'body.{i}.weight'].T + W[f'body.{i}.bias']
        h = h / (1.0 + np.exp(-h))          # SiLU
    o = h @ W['head.weight'].T + W['head.bias']
    return base + o[:, 0], np.clip(o[:, 1], -7, 4)


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = MLP(ck['nin'], ck['width'], ck['depth'], ck.get('dropout', 0.0)); net.load_state_dict(ck['state']); net.eval()
    return net, ck


def predict(net, X, base, bs=16384):
    out_m, out_s = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            m, s = net(torch.from_numpy(X[i:i + bs]), torch.from_numpy(base[i:i + bs]))
            out_m.append(m.numpy()); out_s.append(s.numpy())
    return np.concatenate(out_m), np.exp(np.concatenate(out_s))
