"""Physics version p1c: the drag mismatch measured on the DEV twins, built into the simulator's labels.

For every dev drag point where the twin (mfoil, ncrit 9) answered, r = ln Cd_tunnel - ln Cd_solver.
Fit r = a + b (log10 Re - 5) by least squares (dev only). p1c labels = p1 labels with ln Cd += a + b (log10 Re - 5).
-> work/calib_sim.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
sys.path.insert(0, HERE)
import baselines as bl  # noqa

units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(os.path.join(ROOT, 'splits.json')))
S = bl.load_sweeps('p1_twin_dev')
X, R, A = [], [], []
for aid in splits['dev']:
    for run in units[aid]['runs']:
        if run['kind'] != 'drag':
            continue
        q = S.get((aid, run['model'], float(round(run['Re'] / 1e4) * 1e4)))
        if q is None or len(q) < 2:
            continue
        D = np.array(run['data'], float)
        lp = bl._interp_gap(D[:, 0], q[:, 0], np.log(q[:, 2]))
        ok = np.isfinite(lp) & (D[:, 2] > 0)
        X += [np.log10(run['Re']) - 5] * int(ok.sum()); R += list(np.log(D[ok, 2]) - lp[ok]); A += [aid] * int(ok.sum())
X = np.array(X); R = np.array(R)
# airfoil-weighted fit (each airfoil counts once)
w = np.array([1.0 / A.count(a) for a in A])
M = np.stack([np.ones_like(X), X], 1)
coef = np.linalg.solve(M.T @ (M * w[:, None]), M.T @ (w * R))
out = dict(a=float(coef[0]), b=float(coef[1]), n_points=len(R), n_airfoils=len(set(A)),
           resid_mean=float(np.mean(R)), note='ln Cd_p1c = ln Cd_p1 + a + b*(log10 Re - 5); fitted on dev twins only')
json.dump(out, open(os.path.join(WORK, 'calib_sim.json'), 'w'), indent=1)
print(out)
for lo, hi in ((4, 4.9), (4.9, 5.2), (5.2, 6)):
    k = (X + 5 >= lo) & (X + 5 < hi)
    if k.any():
        print(f'log10Re {lo}-{hi}: mean ln ratio {R[k].mean():+.3f} (n={k.sum()})')
