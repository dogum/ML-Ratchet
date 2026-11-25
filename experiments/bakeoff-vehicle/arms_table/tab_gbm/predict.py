"""GBM arm: gradient-boosted trees (trained with LightGBM, exported to numpy arrays; numpy-only inference).

Input: the table track's 151 EnergyNet features, unchanged. Target: z = log(E / E_phys), EnergyNet's correction to its
physics estimate. Six tree ensembles (gbm_trees.npz): 'mean' (L2 on z; pred = E_phys x exp(mean)) and the 5, 25, 50,
75, 95 % quantiles of z (LightGBM quantile objective). Quantiles are sorted (crossings counted by the harness).
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('tab_gbm_treeexport', os.path.join(HERE, 'treeexport.py'))
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)
_Z = np.load(os.path.join(HERE, 'gbm_trees.npz'))
KEYS = ['mean', 'q05', 'q25', 'q50', 'q75', 'q95']
MODELS = {k: {f: _Z[f'{k}__{f}'] for f in ('feat', 'thr', 'left', 'right', 'value', 'roots', 'depth')} for k in KEYS}


def predict(batch):
    X = np.asarray(batch['X'], np.float64)
    lE = np.log(np.asarray(batch['E_phys'], np.float64))
    z = {k: te.predict(MODELS[k], X) for k in KEYS}
    Q = np.sort(np.stack([z[k] for k in KEYS[1:]], 1), 1)
    out = {'pred': np.exp(lE + z['mean'])}
    for j, k in enumerate(KEYS[1:]):
        out[k] = np.exp(lE + Q[:, j])
    return out
