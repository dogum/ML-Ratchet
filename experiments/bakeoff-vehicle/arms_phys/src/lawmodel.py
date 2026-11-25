"""Evaluate a set of trip-energy laws (one per powertrain group) on law variables. numpy only.

laws = {'ice': law, 'hev': law, 'bev': law}; law = {'vars': [...], 'tree': [...], 'consts': [...]} (lawsearch JSON).
Conventional -> ice law; the three hybrid systems -> hev law; battery-electric -> bev law; plug-ins ->
f x E_bev + (1 - f) x E_hev with EnergyNet's charge-depleting share f (from start SOC and battery size; 0.5 if unknown),
the same mixing rule as the road-load estimate E_phys. E = e (MJ/km) x distance (km). A non-finite or non-positive law
value falls back to E_phys (counted).
"""
import numpy as np

import lawsearch as ls

GROUP_OF_ARCH = np.array(['ice', 'hev', 'hev', 'hev', 'phev', 'phev', 'bev'])


def law_eval(law, V):
    d = dict(V); d['d'] = V['dist_km']
    X = np.stack([np.asarray(d[k], np.float64) for k in law['vars']], 1)
    return ls.evaluate(ls.from_json(law['tree']), X, np.array(law['consts'], float))


def energy(laws, V, return_fallback=False):
    arch = np.asarray(V['arch'], int)
    g = GROUP_OF_ARCH[arch]
    E = np.full(len(arch), np.nan)
    cache = {}

    def grp(name):
        if name not in cache:
            cache[name] = law_eval(laws[name], V) * V['dist_km']
        return cache[name]
    for name in ('ice', 'hev', 'bev'):
        m = g == name
        if m.any():
            E[m] = grp(name)[m]
    m = g == 'phev'
    if m.any():
        if 'phev' in laws:                 # a dedicated plug-in law (reads f itself)
            E[m] = grp('phev')[m]
        else:
            f = np.clip(V['f'], 0, 1)
            E[m] = (f * grp('bev') + (1 - f) * grp('hev'))[m]
    bad = ~np.isfinite(E) | (E <= 0)
    E[bad] = V['E_phys'][bad]
    return (E, int(bad.sum())) if return_fallback else E
