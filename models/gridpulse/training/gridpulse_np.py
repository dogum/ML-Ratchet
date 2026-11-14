"""GridPulse numpy inference (vendored into the release as gridpulse.py's core)."""
import numpy as np
from scipy.special import erf

PRE, POST = 20, 60


def features(w, N, is_gb, f0, demand=None, ctx=False):
    w = np.asarray(w, np.float64) * (50.0 / np.asarray(f0, np.float64))[:, None] * 10
    tt = np.arange(-PRE, POST + 1)[None, :]
    x = np.where(tt <= np.asarray(N)[:, None], w, 0.0)
    N = np.asarray(N, np.float64)
    F = np.c_[x, np.asarray(is_gb, np.float64), N / 60, np.log(N) / 4]
    if ctx:
        d = np.asarray(demand if demand is not None else np.full(len(w), np.nan), np.float64); k = np.isfinite(d)
        F = np.c_[F, np.where(k, np.log(np.where(k, d, 30000.0) / 30000.0), 0.0), k.astype(np.float64)]
    return F


def forward(W, w, N, is_gb, f0, demand=None):
    """Returns (mu, sigma) in target space (log MW, log MW s, log MW/Hz, log MW, log(MW+100), nadir, settle scaled);
    sigma already includes the dev widening factors."""
    h = (features(w, N, is_gb, f0, demand, bool(W['ctx'])) - W['xm']) / W['xs']
    i = 0
    while f'b{i}_w' in W:
        h = h @ W[f'b{i}_w'].T.astype(np.float64) + W[f'b{i}_b']
        h = 0.5 * h * (1 + erf(h / np.sqrt(2))); i += 1
    mu = h @ W['mu_w'].T + W['mu_b']; ls = np.clip(h @ W['ls_w'].T + W['ls_b'], -6, 3)
    return mu * W['ys'] + W['ym'], np.exp(ls) * W['ys'] * W['widen']
