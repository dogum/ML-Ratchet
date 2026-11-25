"""Reference arm: FlowCast v1.0 exactly as released (models/flowcast/flowcast.py + weights, byte copies).

Ungauged batches use flowcast_weights.npz (rung r1b, bands widened 1.00 / 1.08 on dev, as released).
Gauged batches use flowcast_nowcast_weights.npz (the released nowcast: r1b fine-tuned to also read the observed
flow `lead` days earlier; raw pinball bands, k_widen 1/1). numpy only.

The only adaptation: the harness batch already holds the forcing rows (365 warm-up + target + 1) and the static
vector built exactly as flowcast.predict() builds them, so this calls FlowCast.predict_arrays directly.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flowcast as fc  # noqa: E402

_M = {}


def _model(kind):
    if kind not in _M:
        w = 'flowcast_weights.npz' if kind == 'ungauged' else 'flowcast_nowcast_weights.npz'
        _M[kind] = fc.FlowCast(os.path.join(HERE, w))
    return _M[kind]


def predict(batch):
    kind = batch['protocol']
    m = _model(kind)
    F = np.asarray(batch['forcing'], np.float32)
    st = np.asarray(batch['static'], np.float32)
    pm = float(np.exp(st[11]))                      # mean precipitation of the forcing (FlowCast's unit)
    qn, lead = None, None
    if kind == 'gauged':
        qn = np.asarray(batch['qobs'], float) / pm
        lead = int(batch['lead'])
    mean, q = m.predict_arrays(F, np.asarray(batch['doy']), st, int(batch['n_target']), qn, lead)
    q = m.widen(q) * pm
    out = {'mean': mean * pm}
    for j, k in enumerate(('q05', 'q25', 'q50', 'q75', 'q95')):
        out[k] = q[:, j]
    return out
