"""TabPFN arm context: simulated train-bank trips only (zero real labels).

usage: python3 build_tp_ctx.py OUT_NPZ N [--k 38] [--strat none|archcls] [--seed 5]
One tier draw of the train bank (default_rng(seed)); rows with y > 0.02 MJ; N trips drawn at random (strat none) or
balanced over the 49 architecture x class cells (archcls: equal quota per occupied cell, the remainder at random).
Features: featsel.scoped (top-k of the GBM r0 gain ranking + arch / class codes). Target z = log(y / E_phys).
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
sys.path.insert(0, '/home/claude/bakeoff/vehicle/arms_table/src')
import bake  # noqa: E402
import featsel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('n', type=int); ap.add_argument('--k', type=int, default=38)
ap.add_argument('--strat', default='none'); ap.add_argument('--seed', type=int, default=5)
a = ap.parse_args()
idx = np.array(featsel.ranked('/home/claude/bakeoff/vehicle/arms_table/tab_gbm/work/rungs/r0/train.json', a.k), np.int32)
B = bake.load_bank('train')
T = bake.draw_tiers(B, np.random.default_rng(a.seed))
D = bake.sim_table(B, T)
ok = np.where(D['y'] > 0.02)[0]
rng = np.random.default_rng(a.seed + 1)
if a.strat == 'none':
    pick = rng.choice(ok, a.n, replace=False)
else:
    cell = D['arch'][ok] * 10 + D['cls'][ok]
    cells = np.unique(cell)
    q = a.n // (2 * len(cells))          # half the context balanced, half at random
    pick = []
    for c in cells:
        ii = ok[cell == c]
        pick += rng.choice(ii, min(q, len(ii)), replace=False).tolist()
    rest = np.setdiff1d(ok, pick)
    pick = np.concatenate([pick, rng.choice(rest, a.n - len(pick), replace=False)])
pick = np.sort(pick)
Xc = featsel.scoped(D['X'][pick], idx).astype(np.float32)
zc = np.log(D['y'][pick] / D['E_phys'][pick])
np.savez(a.out, X=Xc, z=zc, feature_index=idx, n=a.n, strat=a.strat, seed=a.seed)
print('context', Xc.shape, 'arch counts', np.bincount(D['arch'][pick], minlength=7).tolist(),
      'class counts', np.bincount(D['cls'][pick], minlength=7).tolist())
