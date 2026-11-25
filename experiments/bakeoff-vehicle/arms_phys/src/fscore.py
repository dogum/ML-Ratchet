"""Fast in-process scoring of a point/quantile predictor on the frozen sim val set and the dev gate, with the
harness's own score.py functions (identical to evaluate.py's numbers; evaluate.py is still run for every rung that
enters ratchet.json). Table-track inputs (law variables are identical on the trace track to ~5e-4)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import config as C  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
import lawvars as LV  # noqa: E402

_C = {}


def simval():
    if 'sv' not in _C:
        Xs, ys = [], []
        for b, y in bake.simval_batches('table', chunk=4000):
            Xs.append(b['X']); ys.append(y)
        X = np.concatenate(Xs).astype(np.float64); y = np.concatenate(ys)
        _C['sv'] = (X, y, bake.simval_meta(), bake.dumb_rates())
    return _C['sv']


def dev():
    if 'dev' not in _C:
        Fr = bake.dev_frame()
        X = np.zeros((len(Fr), 151))
        off = {name: int(Fr.index[Fr.recset == name][0]) for name in bake.SETS}
        for name, s, b in bake._real_batches('dev', 'table'):
            i0 = off[name] + s.start
            X[i0:i0 + b['n']] = b['X']
        _C['dev'] = (X, Fr)
    return _C['dev']


def score_sim(pred, Q=None):
    X, y, meta, dumb = simval()
    Q = np.repeat(pred[:, None], 5, 1) if Q is None else Q
    return S.simval_summary(y, pred, Q, meta, dumb)


def score_dev(pred, Q=None):
    X, Fr = dev()
    Q = np.repeat(pred[:, None], 5, 1) if Q is None else Q
    d = Fr.assign(pred=pred, **{k: Q[:, j] for j, k in enumerate(C.QKEYS)})
    U = S.real_units(d)
    s = S.real_summary(U, with_conditions=False)
    return S.d16(s), s


def both(fn):
    """fn(X) -> pred (and optionally Q) on the table inputs."""
    X, y, meta, dumb = simval()
    o = fn(X)
    ss = score_sim(*o) if isinstance(o, tuple) else score_sim(o)
    Xd, Fr = dev()
    o = fn(Xd)
    d16, sd = score_dev(*o) if isinstance(o, tuple) else score_dev(o)
    return {'skill': ss['skill'], 'MdAPE': ss['MdAPE'], 'D16': d16,
            'dev': {k: round(v['MdAPE'], 4) for k, v in sd.items() if k in S.D16}, 'sim_by_arch': ss['by_arch']}
