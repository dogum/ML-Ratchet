"""GBM arm (gradient-boosted trees, LightGBM-trained, numpy-only inference).

predict(batch): daily lag / antecedent features from the batch's forcing and static vector (feats.py), then six tree
ensembles exported to numpy arrays (gbm_trees.npz): the mean (L2 on qo / p_mean) and the 5, 25, 50, 75, 95 %
quantiles; everything x p_mean (the forcing's mean precipitation). Flows are clipped at 0; quantiles are sorted.
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location('tab_gbm_' + name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


feats = _load('feats')
te = _load('treeexport')
_Z = np.load(os.path.join(HERE, 'gbm_trees.npz'))
KEYS = ['mean', 'q05', 'q25', 'q50', 'q75', 'q95']
MODELS = {k: {f: _Z[f'{k}__{f}'] for f in ('feat', 'thr', 'left', 'right', 'value', 'roots', 'depth')} for k in KEYS}
FIDX = _Z['feature_index']


def _outputs(X, pm):
    X = X[:, FIDX]
    raw = {k: te.predict(MODELS[k], X) * pm for k in KEYS}
    mean = np.maximum(raw['mean'], 0.0)
    Q = np.sort(np.maximum(np.stack([raw[k] for k in KEYS[1:]], 1), 0.0), 1)
    return mean, Q


def predict_many(batches):
    Xs, pms, ns = [], [], []
    for b in batches:
        X, pm = feats.batch_features(b)
        Xs.append(X); pms.append(np.full(len(X), pm)); ns.append(len(X))
    mean, Q = _outputs(np.concatenate(Xs), np.concatenate(pms))
    out, s = [], 0
    for n in ns:
        o = {'mean': mean[s:s + n]}
        for j, k in enumerate(KEYS[1:]):
            o[k] = Q[s:s + n, j]
        out.append(o); s += n
    return out


def predict(batch):
    return predict_many([batch])[0]
