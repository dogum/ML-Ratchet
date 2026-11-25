"""Fit the climatology arm on the simulated TRAIN bank only (no real flow): the ratio between daily flow and the
Budyko-Fu water-balance mean flow, pooled over 2,000 random training runs (days 365-1460 of each run).
usage: python3 fit.py  -> clim_params.json"""
import json, os, sys
sys.path.insert(0, '/home/claude/bakeoff/river/harness')
import numpy as np
import bake
from features import fu_runoff_ratio

B = bake.load_bank('train', ['qo', 'st'])
rng = np.random.default_rng(20261123)
idx = np.sort(rng.choice(len(B['qo']), 2000, replace=False))
st = np.asarray(B['st'][idx], np.float64)
base = fu_runoff_ratio(np.exp(st[:, 13])) * np.exp(st[:, 11])          # mm/day
R = np.asarray(B['qo'][idx, 365:], np.float64) / np.maximum(base, 1e-3)[:, None]
ratio_mean = float(R.mean())
qs = [float(x) for x in np.quantile(R, [0.05, 0.25, 0.5, 0.75, 0.95])]
json.dump({'w': 2.6, 'ratio_mean': ratio_mean, 'ratio_quantiles': qs, 'n_runs': 2000,
           'what': 'mean = ratio_mean x Fu(aridity) x p_mean; quantile k = ratio_quantiles[k] x Fu x p_mean'},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clim_params.json'), 'w'), indent=1)
print(ratio_mean, qs)
