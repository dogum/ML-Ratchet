"""River Bake-off round 2, ensemble arm: equal-weight mean of registered member arms. Ungauged. numpy only.

members.json lists the member arm folders (relative to this folder; all hashed through ratchet.json extra_files).
Per target day, weights 1/n each: mean = the mean of the members' means; each quantile (5, 25, 50, 75, 95 %) = the
mean of the members' same quantile (quantile averaging), then sorted. No re-weighting, no widening (PREREG section 5).
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QK = ['q05', 'q25', 'q50', 'q75', 'q95']
MEMBERS = json.load(open(os.path.join(HERE, 'members.json')))['members']
_MODS = []


def _load():
    if _MODS:
        return _MODS
    tag = os.path.basename(HERE)
    for i, rel in enumerate(MEMBERS):
        d = os.path.normpath(os.path.join(HERE, rel))
        if d not in sys.path:
            sys.path.insert(0, d)
        name = f'r2ens_{tag}_m{i}'
        spec = importlib.util.spec_from_file_location(name, os.path.join(d, 'predict.py'))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        _MODS.append(m)
    return _MODS


def member_outputs(batches):
    """List (one per member) of lists of raw outputs (one per batch)."""
    outs = []
    for m in _load():
        if hasattr(m, 'predict_many') and len(batches) > 1:
            outs.append(list(m.predict_many(batches)))
        else:
            outs.append([m.predict(b) for b in batches])
    return outs


def combine(per_member):
    """per_member: list of output dicts for ONE batch -> the ensemble output."""
    mean = np.mean([np.asarray(o['mean'], np.float64) for o in per_member], 0)
    Q = np.stack([np.mean([np.asarray(o[k], np.float64) for o in per_member], 0) for k in QK], 1)
    Q = np.sort(Q, 1)
    out = {'mean': mean}
    for j, k in enumerate(QK):
        out[k] = Q[:, j]
    return out


def predict_many(batches):
    outs = member_outputs(batches)
    return [combine([outs[m][i] for m in range(len(outs))]) for i in range(len(batches))]


def predict(batch):
    return predict_many([batch])[0]
