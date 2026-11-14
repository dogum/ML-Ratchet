"""Demo: the Detective on a few real EXAM units (EPA CAMPD hourly gross output, all years 2018-2025), compared with
the CAMPD truth it never saw.  Runs in seconds.  python demo.py"""
import json, glob, os, time
import numpy as np
from plantdetective import Detective

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = dict(pmin_frac='min stable load', ramp_up='ramp up p99 (frac/h)', starts_yr='starts per year',
             med_up='median run (h)', hr_full='heat rate full load', hr_min='heat rate part load', cf='capacity factor')
t0 = time.time()
det = Detective()
for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*.npz'))):
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    r = det.infer(z['x'].astype(float), z['t'], z['price'], z['gas'], meta['context'])
    tech = max(r['tech'], key=r['tech'].get)
    print(f"\n{meta['name']} unit {meta['unit']}  ({meta['years']})   truth: {meta['tech']}   Detective: {tech} "
          f"({100 * r['tech'][tech]:.0f} %), output-only: {max(r['tech_output_only'], key=r['tech_output_only'].get)}")
    print(f"  {'quantity':24s} {'Detective':>10s} {'68 % interval':>21s} {'CAMPD truth':>12s}")
    for k, n in NAMES.items():
        e = r['estimates'][k]; tv = meta['truth'].get(k)
        print(f"  {n:24s} {e['value']:10.3f} {e['lo68']:10.3f} - {e['hi68']:8.3f} {tv if tv is None else round(tv, 3)!s:>12s}")
    print(f"  operated as: {r['flex_class']} (truth {meta['flex_true']})")
print(f'\n{time.time() - t0:.1f} s')
