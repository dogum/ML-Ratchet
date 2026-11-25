"""Shared inference pieces of the physics-in-the-loop arms (numpy): parameter bounds, the numpy MLP, and the
error bands. Bands: on held-out sim-TRAIN runs, the relative residual r = (qo - q_hat) / (q_hat + 0.05 p_mean) is
binned by log(q_hat / p_mean); per bin its 5, 25, 50, 75, 95 % quantiles are stored. Prediction: q_k = max(q_hat +
r_k (q_hat + 0.05 p_mean), 0); the mean stays q_hat (the WaterSim flow)."""
import math

import numpy as np

# name, lo, hi, log-scale (same table as hbv_torch.PARAMS)
PARAMS = [
    ('FC', 50.0, 700.0, True), ('LP', 0.3, 1.0, False), ('BETA', 1.0, 6.0, False), ('PERC', 0.0, 6.0, False),
    ('UZL', 0.0, 100.0, False), ('K0', 0.05, 0.5, True), ('K1', 0.01, 0.4, True), ('K2', 0.0005, 0.15, True),
    ('MAXBAS', 1.0, 7.0, False),
    ('TT', -2.5, 2.5, False), ('TTI', 0.0, 4.0, False), ('CFMAX', 1.0, 8.0, False), ('SFCF', 0.7, 1.4, False),
    ('TLAPSE', -9.0, -4.0, False), ('PGRAD', 0.0, 1.0, False),
    ('CH_VEL', 0.3, 3.0, True), ('PMULT', 0.5, 2.0, True),
]
NAMES = [p[0] for p in PARAMS]
KEYS = ['q05', 'q25', 'q50', 'q75', 'q95']


def to_params(z):
    z = np.atleast_2d(np.asarray(z, np.float64))
    s = 1.0 / (1.0 + np.exp(-z))
    out = {}
    for j, (n, lo, hi, lg) in enumerate(PARAMS):
        out[n] = np.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * s[:, j]) if lg else lo + (hi - lo) * s[:, j]
    return out


def mlp(x, Ws, bs):
    """SiLU MLP: hidden layers then a linear output (torch.nn.Linear convention: y = x W^T + b)."""
    h = np.atleast_2d(np.asarray(x, np.float64))
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = h @ W.T + b
        h = h / (1.0 + np.exp(-h))
    return h @ Ws[-1].T + bs[-1]


def fit_bands(qhat, qo, pm, nbins=10):
    """qhat, qo, pm: 1-D arrays over pooled days. Returns dict edges (nbins-1,), rq (nbins, 5)."""
    u = np.log(qhat / pm + 1e-3)
    edges = np.quantile(u, np.linspace(0, 1, nbins + 1)[1:-1])
    k = np.searchsorted(edges, u)
    r = (qo - qhat) / (qhat + 0.05 * pm)
    rq = np.array([np.quantile(r[k == j], [0.05, 0.25, 0.5, 0.75, 0.95]) for j in range(nbins)])
    return {'edges': edges, 'rq': rq}


def apply_bands(qhat, pm, edges, rq):
    u = np.log(qhat / pm + 1e-3)
    k = np.searchsorted(edges, u)
    Q = qhat[:, None] + rq[k] * (qhat + 0.05 * pm)[:, None]
    return np.sort(np.maximum(Q, 0.0), 1)
