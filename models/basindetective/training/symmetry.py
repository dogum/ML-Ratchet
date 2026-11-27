"""Forward-plus-reverse check on paired dev periods (27 Nov 2026, audit follow-up; dev basins only).

For every dev basin with gauged flow in both 1980-89 (EARLY) and 1999-2008 (CAL):
  forward  input = CAL forcing + CAL flow, target = EARLY forcing   -> predicted change f (EARLY minus CAL)
  reverse  input = EARLY forcing + EARLY flow, target = CAL forcing -> predicted change r (CAL minus EARLY)
Each direction uses only its own input period's flow. The observed changes are exact negatives of each other, so a
direction-consistent model has f + r = 0. Reported per change: median and mean of f + r, median |f + r|, and the
share of basins where f and r have the same sign (both predict a decline, or both a rise).

python symmetry.py <model npz> [<model npz> ...]   -> prints JSON; with --out <file> writes it.
"""
import common  # noqa: F401
import json
import sys

import numpy as np
import pandas as pd

import waterdata as wd
from common import CAL, EARLY, DELTA_T
from features import featurize, sigs, delta_labels
from net import Detective

_CACHE = {}


def paired_features():
    if 'X' in _CACHE:
        return _CACHE['X']
    ids = wd.basins('camels', 'dev')
    Xf, Xr, rows = [], [], []
    for b in ids:
        f = wd.forcing(b, EARLY[0], CAL[1])
        q = wd.streamflow(b, 'caravan').Q_mm
        A = f.loc[CAL[0]:CAL[1], ['P', 'T_mean', 'PET']]
        E = f.loc[EARLY[0]:EARLY[1], ['P', 'T_mean', 'PET']]
        qa, qe = q.reindex(A.index), q.reindex(E.index)
        if qa.notna().sum() < 3 * 365 or qe.notna().sum() < 3 * 365:
            continue
        try:
            xf, s_a, names = featurize(A, qa.values, E)
            xr, s_e, _ = featurize(E, qe.values, A)
        except ValueError:
            continue
        obs = delta_labels(s_a, sigs(qe, E.P, E.PET, E.T_mean))
        Xf.append(xf); Xr.append(xr)
        rows.append({'basin': b, **{f'obs_{k}': v for k, v in obs.items()}})
    _CACHE['X'] = (np.array(Xf), np.array(Xr), pd.DataFrame(rows), names)
    return _CACHE['X']


def check(path):
    Xf, Xr, meta, names = paired_features()
    m = Detective(path)
    assert names == m.names
    mf, _, _ = m.forward(Xf)
    mr, _, _ = m.forward(Xr)
    out = {'n_basins': int(len(meta))}
    for t in DELTA_T:
        j = m.targets.index(t)
        s = mf[:, j] + mr[:, j]
        out[t] = {'median_sum': float(np.median(s)), 'mean_sum': float(np.mean(s)),
                  'median_abs_sum': float(np.median(np.abs(s))),
                  'same_sign_share': float(np.mean(np.sign(mf[:, j]) == np.sign(mr[:, j])))}
    out['d_logq_both_decline_share'] = float(np.mean((mf[:, m.targets.index('d_logq')] < 0) &
                                                     (mr[:, m.targets.index('d_logq')] < 0)))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--out']
    outf = None
    if '--out' in sys.argv:
        outf = sys.argv[sys.argv.index('--out') + 1]
        args.remove(outf)
    res = {p: check(p) for p in args}
    s = json.dumps(res, indent=1)
    print(s)
    if outf:
        open(outf, 'w').write(s)
