"""GP arm (sparse Gaussian process, SGPR with inducing points; numpy-only inference).

predict(batch): the scoped daily features (a subset of feats.py, listed in gp.npz) -> standardise -> ARD
squared-exponential kernel against the M inducing inputs -> Gaussian predictive mean and variance (latent + noise) of
the transformed target z = sqrt(qo / p_mean) (or log(qo / p_mean + 0.02)). Quantiles are the Gaussian quantiles
mapped back; the mean is the exact mean of the back-transformed Gaussian; everything x p_mean.
"""
import importlib.util
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('tab_gp_feats', os.path.join(HERE, 'feats.py'))
feats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feats)
G = dict(np.load(os.path.join(HERE, 'gp.npz')))
TRANSFORM = str(G['transform'])
ZQ = np.array([-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722])
KEYS = ['q05', 'q25', 'q50', 'q75', 'q95']


def _erf(x):
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
    s = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _latent(X):
    Zs = (X[:, G['feature_index']].astype(np.float64) - G['xm']) / G['xs']
    A = Zs / G['ell']; Bz = G['Z'] / G['ell']
    d2 = (A * A).sum(1)[:, None] + (Bz * Bz).sum(1)[None] - 2.0 * A @ Bz.T
    K = float(G['s2']) * np.exp(-0.5 * np.maximum(d2, 0.0))
    mu = K @ G['alpha'] + float(G['y0'])
    vf = float(G['s2']) - np.einsum('ij,ij->i', K @ G['W'], K)
    var = np.maximum(vf, 0.0) + float(G['n2'])
    return mu, np.sqrt(var)


def _outputs(X, pm):
    mu, sd = _latent(X)
    zq = mu[:, None] + sd[:, None] * ZQ[None]
    if TRANSFORM == 'sqrt':
        Q = np.maximum(zq, 0.0) ** 2
        r = mu / sd
        Phi = 0.5 * (1.0 + _erf(r / math.sqrt(2.0)))
        phi = np.exp(-0.5 * r * r) / math.sqrt(2 * math.pi)
        mean = (mu ** 2 + sd ** 2) * Phi + mu * sd * phi        # E[max(z, 0)^2], z ~ N(mu, sd^2)
    else:
        off = float(G['off'])
        Q = np.maximum(np.exp(zq) - off, 0.0)
        mean = np.maximum(np.exp(mu + 0.5 * sd ** 2) - off, 0.0)
    return np.maximum(mean, 0.0) * pm, Q * pm[:, None]


def predict_many(batches):
    Xs, pms, ns = [], [], []
    for b in batches:
        X, pm = feats.batch_features(b)
        Xs.append(X); pms.append(np.full(len(X), pm)); ns.append(len(X))
    X = np.concatenate(Xs); pm = np.concatenate(pms)
    mean = np.empty(len(X)); Q = np.empty((len(X), 5))
    for s in range(0, len(X), 20000):
        mean[s:s + 20000], Q[s:s + 20000] = _outputs(X[s:s + 20000], pm[s:s + 20000])
    out, s = [], 0
    for n in ns:
        o = {'mean': mean[s:s + n]}
        for j, k in enumerate(KEYS):
            o[k] = Q[s:s + n, j]
        out.append(o); s += n
    return out


def predict(batch):
    return predict_many([batch])[0]
