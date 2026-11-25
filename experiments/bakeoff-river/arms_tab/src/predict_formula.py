"""Formula arm (symbolic attribute -> parameter formulas, run inside WaterSim).

predict(batch): 17 closed-form formulas (formulas.json, found by a LawFinder-style symbolic search, lawsearch.py)
map the batch's 18-value static vector to WaterSim's HBV (9), degree-day snow (6), channel-velocity and
precipitation-multiplier parameters (through fixed sigmoid bounds); the harness's vendored WaterSim then simulates
the basin (wsim.py). Mean = WaterSim's flow; bands = residual quantiles by flow level fitted on held-out sim-TRAIN
runs (bands.npz, physarm.py).
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(os.path.basename(HERE) + '_' + name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


wsim = _load('wsim')
pa = _load('physarm')
fe = _load('formula_eval')
F = json.load(open(os.path.join(HERE, 'formulas.json')))
_BD = dict(np.load(os.path.join(HERE, 'bands.npz')))


def params_of(batch):
    out = fe.params_from_formulas(F, batch['static'])
    out.update({'CFR': wsim.FIXED['CFR'], 'CWH': wsim.FIXED['CWH']})
    return out


def predict(batch):
    q = np.maximum(wsim.run_batch(batch, params_of(batch)), 0.0)
    pm = float(np.exp(batch['static'][11]))
    Q = pa.apply_bands(q, np.full(len(q), pm), _BD['edges'], _BD['rq'])
    out = {'mean': q}
    for j, k in enumerate(pa.KEYS):
        out[k] = Q[:, j]
    return out
