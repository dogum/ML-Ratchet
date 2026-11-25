"""GP arm: sparse Gaussian process (SGPR, Titsias 2009; M inducing points; ARD squared-exponential kernel), numpy-only.

Inputs: a scoped subset of the table track's 151 EnergyNet features ('feature_index' in each gp*.npz: the top features
of the GBM arm's gain ranking on the simulated train bank + the 14 one-hot architecture / class columns), standardised.
Target: z = log(E / E_phys). The GP's predictive distribution of z is Gaussian (latent variance + noise variance), so
the 5 quantiles are mu + sd x z_q and pred = E_phys x exp(mu) (the median).
One GP (gp.npz), or one GP per powertrain group (groups.json: file -> architecture indices; each row is routed by its
one-hot architecture input, X[:, 102:109]).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZQ = np.array([-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722])


def _load(f):
    G = dict(np.load(os.path.join(HERE, f)))
    G['FI'] = G['feature_index'].astype(int)
    G['BZ'] = G['Z'] / G['ell']
    G['BZ2'] = (G['BZ'] * G['BZ']).sum(1)
    return G


if os.path.exists(os.path.join(HERE, 'groups.json')):
    _GR = json.load(open(os.path.join(HERE, 'groups.json')))
    MODELS = [(_load(f), np.array(a)) for f, a in _GR.items()]
else:
    MODELS = [(_load('gp.npz'), np.arange(7))]


def _latent(G, X):
    A = ((X[:, G['FI']].astype(np.float64) - G['xm']) / G['xs']) / G['ell']
    d2 = (A * A).sum(1)[:, None] + G['BZ2'][None] - 2.0 * A @ G['BZ'].T
    K = float(G['s2']) * np.exp(-0.5 * np.maximum(d2, 0.0))
    mu = K @ G['alpha'] + float(G['y0'])
    vf = float(G['s2']) - np.einsum('ij,ij->i', K @ G['W'], K)
    return mu, np.sqrt(np.maximum(vf, 0.0) + float(G['n2']))


def predict(batch):
    X = np.asarray(batch['X'], np.float64)
    lE = np.log(np.asarray(batch['E_phys'], np.float64))
    arch = np.argmax(X[:, 102:109], 1)
    mu = np.zeros(len(X)); sd = np.ones(len(X))
    for G, archs in MODELS:
        ii = np.where(np.isin(arch, archs))[0]
        for s in range(0, len(ii), 4096):
            j = ii[s:s + 4096]
            mu[j], sd[j] = _latent(G, X[j])
    out = {'pred': np.exp(lE + mu)}
    for j, k in enumerate(['q05', 'q25', 'q50', 'q75', 'q95']):
        out[k] = np.exp(lE + mu + ZQ[j] * sd)
    return out
