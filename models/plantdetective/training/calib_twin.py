"""Dev twin calibration of the behaviour wrappers (per technology): start-cost multiplier (wear and tear on top of
start fuel) and offer slope kappa.  Grid over both; each technology picks the combination whose twins best match
the real dev units on starts/yr, median run, capacity factor and the load distribution (dev years only).

usage: python calib_twin.py <base_version>
"""
from common import *
import json, itertools, copy
import numpy as np
import calib, twin

base = sys.argv[1]
SM = [0.0, 1.0, 2.0]          # log-mean of the start-cost multiplier
KAPPA = [0.0, 10.0, 30.0, 60.0]
KEYS = dict(starts_yr='log', med_up='log', cf='diff', load_p10='diff', load_p50='diff', at_max='diff')
res = {}
for sm, ka in itertools.product(SM, KAPPA):
    cal = calib.get(base)
    for t in cal['start_mult']:
        cal['start_mult'][t] = (sm, 0.5)
        cal['kappa'][t] = (ka * 0.8, ka * 1.2)
    df = twin.run(base, f'_grid_sm{sm}_k{ka}', cal=cal, verbose=False)
    df = df[~df.cogen]
    for tech, g in df.groupby('tech'):
        err = 0.0
        for k, kind in KEYS.items():
            a, b = g[f'real_{k}'].astype(float), g[f'twin_{k}'].astype(float)
            m = np.isfinite(a) & np.isfinite(b)
            d = np.log((b[m] + 0.5) / (a[m] + 0.5)) if kind == 'log' else (b[m] - a[m]) * 3
            err += float(np.abs(np.median(d))) + 0.5 * float(np.median(np.abs(d)))
        res.setdefault(tech, []).append(dict(sm=sm, kappa=ka, err=err))
    print(sm, ka, {t: round(r[-1]['err'], 3) for t, r in res.items()}, flush=True)
best = {t: min(r, key=lambda x: x['err']) for t, r in res.items()}
print(json.dumps(best, indent=1))
json.dump(dict(grid=res, best=best), open(f'{ROOT}/out/calib_twin_{base}.json', 'w'), indent=1)
