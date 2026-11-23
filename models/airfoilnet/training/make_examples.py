"""Demo examples: a fixed random draw (seed 4) of exam runs within two Reynolds strata, 2 per stratum.
Each example = as-tested coordinates + one measured lift run and the drag run at the same Re (UIUC LSAT, GPL v2).
-> release/examples/*.json
"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import SPLITS   # the frozen split (training/_paths.py)
units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(SPLITS))
OUT = os.path.join(ROOT, 'release', 'examples'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(4)
for lab, lo, hi in (('lowRe', 0, 1.05e5), ('highRe', 1.05e5, 1e7)):
    cands = []
    for aid in splits['exam']:
        u = units[aid]
        for r in u['runs']:
            if r['kind'] == 'lift' and lo < r['Re'] <= hi:
                d = [x for x in u['runs'] if x['kind'] == 'drag' and x['model'] == r['model'] and abs(x['Re'] / r['Re'] - 1) < 0.05]
                if d:
                    cands.append((aid, r, d[0]))
    fams = sorted({units[c[0]]['family'] for c in cands})
    pick_f = rng.choice(fams, 2, replace=False)
    for f in pick_f:
        cs = [c for c in cands if units[c[0]]['family'] == f]
        aid, lr, dr = cs[int(rng.integers(len(cs)))]
        ex = dict(airfoil=lr['label'], family=units[aid]['family'], stratum=lab, Re=lr['Re'],
                  coords=units[aid]['coords'][lr['model']], coords_source=lr['coords_src'],
                  lift=dict(Re=lr['Re'], alpha_cl_cm=lr['data']), drag=dict(Re=dr['Re'], alpha_cl_cd=dr['data']),
                  source=f"UIUC Low-Speed Airfoil Tests, volume {lr['vol']} (Selig et al.), GPL v2 data licence")
        fn = f"{aid}_{int(round(lr['Re'], -3))}.json"
        json.dump(ex, open(os.path.join(OUT, fn), 'w'), indent=0)
        print(lab, fn, ex['airfoil'], ex['family'])
