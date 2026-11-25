"""Scoped feature sets for the GP and TabPFN arms (feature engineering on the table track's 151 inputs only).

scoped(X, idx) -> columns X[:, idx] + arch index + class index (argmax of the one-hot blocks V0-6 / V7-13).
The index list comes from the GBM r0 'mean' model's gain ranking on the simulated train bank (no val, no dev):
top features, skipping the one-hot architecture / class columns (they enter as the two integer codes).
"""
import json
import numpy as np

V0 = 102


def ranked(train_json, k):
    g = json.load(open(train_json))['gain_top40']
    out = [i for i, _ in g if not (V0 <= i < V0 + 14)]
    return out[:k]


def scoped(X, idx):
    X = np.asarray(X)
    a = np.argmax(X[:, V0:V0 + 7], 1).astype(X.dtype)
    c = np.argmax(X[:, V0 + 7:V0 + 14], 1).astype(X.dtype)
    return np.concatenate([X[:, idx], a[:, None], c[:, None]], 1)
