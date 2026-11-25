"""Physics-in-the-loop arm ("decides through physics"; differentiable parameter learning, Tsai et al. 2021 style).

predict(batch): a small MLP (numpy) maps the batch's 18-value static vector to 17 parameters of WaterSim's HBV
(9), degree-day snow (6; CFR 0.05 and CWH 0.1 fixed), gamma channel UH velocity and a precipitation multiplier
PMULT; then the harness's vendored WaterSim simulates the basin (wsim.py: spec from the batch attributes, 365 warm-up
rows as spin-up, PET from the forcing). The mean is WaterSim's flow; the bands are residual quantiles by flow level
fitted on held-out sim-TRAIN runs (physarm.py). The MLP was trained through a torch copy of the same equations
(verified to 7e-14 mm/day against WaterSim) on the simulated train bank only.

params_of(batch) -> dict of the 17 parameters (+ fixed CFR, CWH): the attribute -> WaterSim parameter mapping, for
reuse (e.g. arm C's ensemble Kalman filter).
"""
import importlib.util
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
_W = dict(np.load(os.path.join(HERE, 'dpl_weights.npz')))
_NL = int(_W['n_layers'])
_Ws = [_W[f'W{i}'] for i in range(_NL)]
_bs = [_W[f'b{i}'] for i in range(_NL)]


def params_of(batch):
    x = (np.asarray(batch['static'], np.float64) - _W['mu']) / _W['sd']
    p = pa.to_params(pa.mlp(x, _Ws, _bs))
    out = {k: float(v[0]) for k, v in p.items()}
    out.update({'CFR': wsim.FIXED['CFR'], 'CWH': wsim.FIXED['CWH']})
    return out


def predict(batch):
    prm = params_of(batch)
    q = np.maximum(wsim.run_batch(batch, prm), 0.0)
    pm = float(np.exp(batch['static'][11]))
    Q = pa.apply_bands(q, np.full(len(q), pm), _W['edges'], _W['rq'])
    out = {'mean': q}
    for j, k in enumerate(pa.KEYS):
        out[k] = Q[:, j]
    return out
