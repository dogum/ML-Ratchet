"""Formula arm (agent C, Vehicle Bake-off): a LawFinder-style trip-energy law, one short formula per powertrain group.

e = trip energy per km (MJ/km) = law_g(p, n, t, m, T, c, d), found by symbolic regression (lawsearch.py, vendored from
LawFinder) on the SIMULATED train bank only; laws.json holds the trees and constants. E = e x distance. Groups:
conventional -> ice law; the three hybrid systems -> hev law; battery-electric -> bev law; plug-ins ->
f x E_bev + (1 - f) x E_hev with EnergyNet's charge-depleting share f (as the road-load estimate does).
Variables (lawvars.py): p / n = positive / negative wheel work per km at the tier's road load (MJ/km), t = minutes per
km, m = mass (t), T = mean ambient temperature (C), c = start state (-1 warm, 0 unknown, +1 cold), d = distance (km).
Table track: read from EnergyNet's 151 inputs. Trace track: recomputed from the 1 Hz trace + spec + rl with EnergyNet's
own physics integrals (enet_features.py, a byte copy of features.py). Bands: q_k = E x exp(r_k), r_k = quantiles of
log(y / E) on the simulated train bank per architecture x duty (bands.json). A non-finite or non-positive law value
falls back to E_phys. numpy only.
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location('formula_arm_' + name, os.path.join(HERE, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LS = _load('lawsearch')
LV = _load('lawvars')
F = _load('enet_features')
LAWS = json.load(open(os.path.join(HERE, 'laws.json')))
_B = json.load(open(os.path.join(HERE, 'bands.json')))
R = np.array(_B['logq'], float)                      # (7 archs, 2 duty, 5)
GROUP_OF_ARCH = np.array(['ice', 'hev', 'hev', 'hev', 'phev', 'phev', 'bev'])
QK = ['q05', 'q25', 'q50', 'q75', 'q95']
_TREES = {g: (LS.from_json(L['tree']), np.array(L['consts'], float), L['vars']) for g, L in LAWS['laws'].items()}


def law_values(V, g):
    t, c, names = _TREES[g]
    d = dict(V); d['d'] = V['dist_km']
    X = np.stack([np.asarray(d[k], np.float64) for k in names], 1)
    return LS.evaluate(t, X, c)


def energy(V):
    """law variables -> (E MJ, number of E_phys fallbacks)."""
    g = GROUP_OF_ARCH[np.asarray(V['arch'], int)]
    E = np.full(len(g), np.nan)
    cache = {}

    def grp(name):
        if name not in cache:
            cache[name] = law_values(V, name) * V['dist_km']
        return cache[name]
    for name in ('ice', 'hev', 'bev'):
        m = g == name
        if m.any():
            E[m] = grp(name)[m]
    m = g == 'phev'
    if m.any():
        if 'phev' in _TREES:                 # a dedicated plug-in law (reads f itself)
            E[m] = grp('phev')[m]
        else:
            f = np.clip(V['f'], 0, 1)
            E[m] = (f * grp('bev') + (1 - f) * grp('hev'))[m]
    bad = ~np.isfinite(E) | (E <= 0)
    E[bad] = V['E_phys'][bad]
    return E, int(bad.sum())


def variables(batch):
    if batch['track'] == 'table':
        return LV.from_table(batch['X'])
    return LV.from_trace(batch, F)


def predict(batch):
    V = variables(batch)
    E, _ = energy(V)
    a = np.asarray(batch['arch'], int); hd = (np.asarray(batch['cls'], int) >= 4).astype(int)
    Q = E[:, None] * np.exp(R[a, hd])
    out = {'pred': E}
    out.update({k: Q[:, j] for j, k in enumerate(QK)})
    return out
