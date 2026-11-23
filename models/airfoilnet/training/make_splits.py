"""Freeze the dev/exam split by airfoil FAMILY (written once, before any training). -> splits.json"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'splits.json')
SEED = 20261122

units = json.load(open(os.path.join(HERE, '..', 'data', 'units.json')))
fams = {}
for k, u in units.items():
    fams.setdefault(u['family'], []).append(k)
small = sorted(f for f, v in fams.items() if len(v) <= 3)
rng = np.random.default_rng(SEED)
order = list(rng.permutation(small))
dev = []
dev_fams = []
for f in order:
    if len(dev) >= 7:
        break
    dev_fams.append(f); dev += sorted(fams[f])
exam = sorted(k for k in units if k not in dev)
if os.path.exists(OUT):
    raise SystemExit('splits.json exists; the split is frozen')
json.dump(dict(seed=SEED, rule='by family; dev = whole families drawn by seed from families with <=3 airfoils '
                                 'until >=7 airfoils; exam = all other airfoils. Builds (A)-(E) of one airfoil '
                                 'stay together.',
               dev_families=dev_fams, dev=sorted(dev), exam=exam,
               family={k: units[k]['family'] for k in units}), open(OUT, 'w'), indent=1)
print('dev', dev_fams, sorted(dev)); print('exam', len(exam))
