"""Evaluate the formula arm's attribute -> WaterSim parameter formulas (numpy only).

formulas.json: {'vars': [18 static-vector names], 'params': {NAME: {'tree': lawsearch JSON tree, 'consts': [...],
'str': readable form, ...}}}. Each formula gives the parameter's unbounded value z, mapped into WaterSim's bounds by
the same sigmoid (log-scale where flagged) as the physics-in-the-loop arm (physarm.to_params). A non-finite z falls
back to the formula's stored fallback (the median teacher value)."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(os.path.basename(HERE) + '_fe_' + name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ls = _load('lawsearch')
pa = _load('physarm')


def z_from_formulas(F, static):
    X = np.atleast_2d(np.asarray(static, np.float64))
    z = np.empty((X.shape[0], len(pa.NAMES)))
    for j, n in enumerate(pa.NAMES):
        f = F['params'][n]
        v = ls.evaluate(ls.from_json(f['tree']), X, np.asarray(f['consts'], float))
        z[:, j] = np.clip(np.where(np.isfinite(v), v, f['fallback']), -12.0, 12.0)
    return z


def params_from_formulas(F, static):
    p = pa.to_params(z_from_formulas(F, static))
    return {k: float(v[0]) for k, v in p.items()}
