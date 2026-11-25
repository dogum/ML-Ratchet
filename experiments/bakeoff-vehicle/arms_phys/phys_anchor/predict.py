"""Physics anchor (agent C, Vehicle Bake-off): EnergyNet's road-load physics estimate E_phys alone, with dev-free
defaults (EnergyNet's own constants PHYS_K, fitted by EnergyNet on simulations), and simulation-derived bands.

pred = E_phys (MJ). Table track: the input log E_phys (X[:, 90]). Trace track: E_phys recomputed from the 1 Hz trace,
spec and rl with EnergyNet's own features.phys_ints / e_phys (enet_features.py, a byte copy; lawvars.from_trace).
Quantiles: q_k = E_phys x exp(r_k), r_k = the 5/25/50/75/95 % quantiles of log(y / E_phys) on the simulated TRAIN bank
(two tier draws) per architecture x light/heavy duty (bands.json, fitted by ../src/fit_anchor.py). No real label is
used anywhere; no dev-fitted scale (both scale rungs were rejected by the gate, see ratchet.json). numpy only.
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location('phys_anchor_' + name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


F = _load('enet_features')
LV = _load('lawvars')
_B = json.load(open(os.path.join(HERE, 'bands.json')))
R = np.array(_B['logq'], float)            # (7 archs, 2 duty, 5)
QK = ['q05', 'q25', 'q50', 'q75', 'q95']


def e_phys(batch):
    if batch['track'] == 'table':
        return np.exp(np.asarray(batch['X'], np.float64)[:, 90])
    return LV.from_trace(batch, F)['E_phys']


def predict(batch):
    E = e_phys(batch)
    a = np.asarray(batch['arch'], int); hd = (np.asarray(batch['cls'], int) >= 4).astype(int)
    Q = E[:, None] * np.exp(R[a, hd])
    out = {'pred': E}
    out.update({k: Q[:, j] for j, k in enumerate(QK)})
    return out
