"""EnergyNet forward pass in numpy (the release and the web page use the same weights)."""
import json
import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class ENet:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z['meta']))
        self.mu, self.sd = z['x_mu'].astype(np.float64), z['x_sd'].astype(np.float64)
        self.L = [(z[f'body_{i}_weight'].astype(np.float64), z[f'body_{i}_bias'].astype(np.float64)) for i in (0, 2, 4, 6)]
        self.H = (z['head_weight'].astype(np.float64), z['head_bias'].astype(np.float64))
        self.mode = self.meta['mode']

    def forward(self, X):
        h = np.clip((np.asarray(X, float) - self.mu) / self.sd, -8, 8)
        for W, b in self.L:
            h = gelu(h @ W.T + b)
        o = h @ self.H[0].T + self.H[1]
        return o[:, 0], np.exp(np.clip(o[:, 1], -5, 2)), o[:, 2:]

    def predict(self, X, E_phys):
        """-> energy MJ, log-sigma (natural log units), components MJ (aero, rolling, brake, aux, losses)."""
        m, s, c = self.forward(X)
        lE = m + np.log(E_phys) if self.mode == 'corr' else m
        return np.exp(lE), s, np.exp(c) * np.asarray(E_phys)[:, None]


class Ensemble:
    """average of several ENet members in log space; sigma = sqrt(mean sigma^2 + spread of the means)."""
    def __init__(self, paths):
        self.m = [ENet(p) for p in paths]; self.meta = self.m[0].meta; self.mode = self.m[0].mode

    def predict(self, X, E_phys):
        P = [m.predict(X, E_phys) for m in self.m]
        L = np.stack([np.log(np.maximum(p[0], 1e-9)) for p in P]); S = np.stack([p[1] for p in P])
        C = np.exp(np.mean([np.log(np.maximum(p[2], 1e-9)) for p in P], axis=0))
        return np.exp(L.mean(0)), np.sqrt((S ** 2).mean(0) + L.var(0)), C
