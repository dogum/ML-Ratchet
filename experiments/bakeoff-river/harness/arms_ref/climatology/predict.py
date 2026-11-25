"""Climatology arm (dumb ungauged baseline, harness test arm): a constant flow from the long-term water balance.

mean(t) = c x Fu(aridity) x p_mean for every day, with aridity and p_mean from the batch's static vector (the
forcing's own climate indices) and Fu's runoff ratio (w = 2.6). Quantiles = fixed multiples of Fu x p_mean.
c and the multiples were fitted on the simulated train bank only (fit.py). No seasonality, no weather response.
numpy only."""
import json
import os

import numpy as np

_P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clim_params.json')))


def _fu(aridity, w):
    return float(np.clip((1 + aridity ** w) ** (1 / w) - aridity, 0.0, 1.0))


def predict(batch):
    st = np.asarray(batch['static'], float)
    base = _fu(float(np.exp(st[13])), _P['w']) * float(np.exp(st[11]))
    n = int(batch['n_target'])
    out = {'mean': np.full(n, _P['ratio_mean'] * base)}
    for k, r in zip(('q05', 'q25', 'q50', 'q75', 'q95'), _P['ratio_quantiles']):
        out[k] = np.full(n, r * base)
    return out
