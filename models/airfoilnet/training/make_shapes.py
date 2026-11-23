"""Training/validation shape pool from the UIUC coordinate database, with every tested airfoil removed.

Rules: drop DB files whose name matches a tested airfoil (any dev or exam unit, any variant/smoothing);
drop DB shapes within sensor-RMS 0.002 c of any tested airfoil; CST-fit the rest (rms < 2e-3 c, valid);
hold out 12% (fixed seed) as the frozen simulated validation pool. Also CST-fits every dev/exam model.
-> work/shapes.npz, work/shapes_report.json
"""
import os, sys, json, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import xsolver as xs  # noqa
import shapes as sh  # noqa

ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(os.path.join(ROOT, 'splits.json')))

# tested shapes
tested = {}
for aid, u in units.items():
    for model, Y in u['coords'].items():
        p, rms = sh.cst_fit(np.array(Y))
        tested[f'{aid}|{model}'] = dict(p=p, rms=rms, split='dev' if aid in splits['dev'] else 'exam')
T = np.array([sh.sensors_from_p(v['p']) for v in tested.values()])

tested_ids = set(units.keys())
def is_tested_name(fn):
    b = re.sub(r'(sm|-il|-jf)?$', '', fn.lower().replace('.dat', ''))
    b2 = re.sub(r'[\s\-_]', '', b)
    for t in tested_ids:
        if b2 == t or b2.startswith(t) and len(b2) - len(t) <= 3:   # e.g. sd7037-092-88 -> sd7037, s1223rtl
            return True
    return False

pool, names, rmss, reasons = [], [], [], {}
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'raw', 'coord_seligFmt', 'coord_seligFmt', '*.dat'))):
    fn = os.path.basename(f)
    if is_tested_name(fn):
        reasons[fn] = 'tested-name'; continue
    try:
        Y = xs.normalize(xs.read_dat(f))
        if len(Y) < 20:
            reasons[fn] = 'few-points'; continue
        p, rms = sh.cst_fit(Y)
    except Exception as e:
        reasons[fn] = 'read-fail'; continue
    if not np.all(np.isfinite(p)) or rms > 2e-3:
        reasons[fn] = f'fit-rms {rms:.4f}'; continue
    if not sh.valid(p):
        reasons[fn] = 'invalid-geometry'; continue
    s = sh.sensors_from_p(p)
    d = np.sqrt(np.mean((T - s) ** 2, axis=1)).min()
    if d < 0.002:
        reasons[fn] = f'near-duplicate of a tested airfoil ({d:.4f})'; continue
    pool.append(p); names.append(fn[:-4]); rmss.append(rms)
pool = np.array(pool)
rng = np.random.default_rng(7)
isval = np.zeros(len(pool), bool); isval[rng.choice(len(pool), int(0.12 * len(pool)), replace=False)] = True
np.savez(os.path.join(WORK, 'shapes.npz'), pool=pool, names=np.array(names), isval=isval,
         tested_keys=np.array(list(tested.keys())), tested_p=np.array([v['p'] for v in tested.values()]),
         tested_rms=np.array([v['rms'] for v in tested.values()]))
from collections import Counter
rep = dict(n_pool=len(pool), n_val=int(isval.sum()), fit_rms_median=float(np.median(rmss)),
           tested_fit_rms={k: round(v['rms'], 5) for k, v in tested.items()},
           dropped=Counter(r.split(' ')[0] for r in reasons.values()), dropped_files=reasons)
json.dump(rep, open(os.path.join(WORK, 'shapes_report.json'), 'w'), indent=1)
print({k: v for k, v in rep.items() if k not in ('dropped_files', 'tested_fit_rms')})
print('tested fit rms max', max(v['rms'] for v in tested.values()),
      sorted(((round(v['rms'], 4), k) for k, v in tested.items()), reverse=True)[:6])
