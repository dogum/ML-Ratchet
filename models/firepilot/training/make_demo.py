"""Copy a few exam fires' planner inputs into release/demo_cases (seeded draw, one per size tercile; no hand-picking).

  python3 make_demo.py
Uses only stored exam outputs (results/exam/*): the planner inputs at first detection and the scored truth rows.
Windows above 200k cells are skipped for zip size (recorded in demo_cases/README.txt).
"""
import os
import sys
import json
import numpy as np

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import firedata as fd

ROOT = _paths.WORK
EX = os.path.join(ROOT, 'results', 'exam')
OUT = os.path.join(ROOT, 'release', 'demo_cases')
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
ft = fd.fires('exam').set_index('Event_ID')
from exam import merged
recs = [merged(f[:-5]) for f in sorted(os.listdir(EX)) if f.endswith('.json') and not f.endswith('.B.json')]
ok = [r for r in recs if 'skip' not in r and r.get('stage_b')]   # demo needs the MPC / PF rows too
a0 = np.array([r['a0'] for r in ok])
q = np.quantile(a0, [1 / 3, 2 / 3])
rs = np.random.default_rng(20261123)
chosen, skipped = [], []
for t in range(3):
    pool = [r for r, a in zip(ok, a0) if (a > q[t - 1] if t else True) and (a <= q[t] if t < 2 else True)]
    for i in rs.permutation(len(pool)):
        r = pool[i]
        if r['shape'][0] * r['shape'][1] > 200000:
            skipped.append(r['fire_id'])
            continue
        chosen.append(r)
        break
SHOW = ['none', 'direct_head', 'indirect_1km', 'indirect_2km', 'anchor_flank', 'mpc_forecast', 'firepilot', 'pf_bound']
for r in chosen:
    fid = r['fire_id']
    z = np.load(os.path.join(EX, fid + '_ctx.npz'))
    meta = ft.loc[fid]
    name = f"{fid}"
    m300 = np.load(os.path.join(EX, f'{fid}_300_masks.npz'))
    arr = np.nan_to_num(z['arr_f'], nan=1e4).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, name + '.npz'), burned0=z['burned0'], arr_f=arr,
                        fl_f=z['fl_f'].astype(np.float32), cost=z['cost'].astype(np.float32),
                        clock_t=z['clock_t'].astype(np.float32), clock_w=m300['clock_w'].astype(np.float32))
    fin = np.isfinite(z['arr_f']) & (z['arr_f'] <= 72)
    rec = dict(fire_id=fid, fire_name=str(meta.Incid_Name).title(), period=str(meta.exam_period), k=r['k'],
               res=90.0, head=float(z['head']), a0_ha=r['a0'], forecast_none_ha=float(fin.sum() * 0.81), budgets={})
    for b, rr in r['budgets'].items():
        rec['budgets'][b] = dict(firepilot_plan=rr['rows']['firepilot']['plan'],
                                 truth_saved={'none': 0.0, **{k: round(rr['rows'][k]['saved'], 3) for k in SHOW[1:]}})
    json.dump(rec, open(os.path.join(OUT, name + '.json'), 'w'), indent=1)
open(os.path.join(OUT, 'README.txt'), 'w').write(
    'Seeded draw (seed 20261123), one exam fire per detected-area tercile; skipped for size: %s\n'
    'clock_w is the 300 m/h budget work clock; the 1000 m/h budget uses the same crew clock (rate differs).\n' % skipped)
print([c['fire_id'] for c in chosen], 'skipped', skipped)
