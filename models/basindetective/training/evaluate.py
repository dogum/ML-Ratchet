"""Scores: frozen simulated validation skill, the real dev gate, and helpers shared with exam.py.

Sim skill (primary ratchet metric) = mean over the 11 regression targets of max(R2, 0) and of (2 AUC - 1) for
'regulated', on the frozen simulated set of the current physics version.
Dev composite (gate) = mean over the six out-of-period change targets of Spearman rho (dev basins, calibration
1999-2008 -> early window 1980-89, real Q truth) and (2 AUC - 1) for regulation (5 regulated of 52 dev basins).
"""
import common  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import D, DELTA_T
from net import REG_TARGETS, Detective


def r2(y, x):
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    return float(1 - ((y - x) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def auc(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], p[ok]))


def coverage(y, mu, sd, k=1.0):
    ok = np.isfinite(y) & np.isfinite(mu)
    return float((np.abs(y[ok] - mu[ok]) <= k * sd[ok]).mean())


def sim_scores(model, X, lab):
    mu, sd, p = model.forward(X)
    out = {}
    for j, t in enumerate(model.targets):
        y = lab[t].values.astype(float)
        out[t] = {'r2': r2(y, mu[:, j]), 'cov68': coverage(y, mu[:, j], sd[:, j])}
    out['reg'] = {'auc': auc(lab['reg'].values.astype(float), p)}
    sk = [max(out[t]['r2'], 0) for t in model.targets] + [2 * out['reg']['auc'] - 1]
    out['_skill'] = float(np.mean(sk))
    return out


def real_set(split):
    z = np.load(f'{D}/feats/real_{split}.npz', allow_pickle=False)
    meta = pd.read_parquet(f'{D}/feats/real_{split}_meta.parquet')
    return z['X'], meta, [str(n) for n in z['names']]


def real_scores(model, X, meta):
    mu, sd, p = model.forward(X)
    out = {}
    for t in DELTA_T:
        j = model.targets.index(t)
        y = meta[f'true_{t}'].values.astype(float)
        ok = np.isfinite(y)
        out[t] = {'rho': float(spearmanr(mu[ok, j], y[ok]).statistic), 'mae': float(np.median(np.abs(mu[ok, j] - y[ok]))),
                  'cov68': coverage(y, mu[:, j], sd[:, j]), 'n': int(ok.sum())}
    yreg = (meta['dor'].values > 0.1).astype(float)
    out['reg'] = {'auc': auc(yreg, p), 'n_pos': int(yreg.sum())}
    comp = [out[t]['rho'] for t in DELTA_T] + [2 * out['reg']['auc'] - 1]
    out['_composite'] = float(np.mean(comp))
    return out, (mu, sd, p)


def dev_gate(model_path):
    m = Detective(model_path)
    X, meta, names = real_set('dev')
    assert names == m.names
    return real_scores(m, X, meta)


def classical_dev():
    X, meta, _ = real_set('dev')
    out = {}
    for t in DELTA_T:
        y = meta[f'true_{t}'].values
        for c in ('gr4j', 'budyko'):
            col = f'{c}_{t}'
            if col in meta:
                x = meta[col].values
                ok = np.isfinite(y) & np.isfinite(x)
                out[f'{t}/{c}'] = float(spearmanr(x[ok], y[ok]).statistic)
    yreg = (meta['dor'].values > 0.1).astype(float)
    out['reg/gr4j_misfit'] = auc(yreg, 1 - meta['gr4j_nse_cal'].values)
    out['reg/seasdamp'] = auc(yreg, -meta['seasdamp'].values)
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(classical_dev(), indent=1))
