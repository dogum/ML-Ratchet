"""Demo on a few REAL exam units (fixed random draw, not hand-picked): BDG2 buildings (hourly 2017 meters + site weather),
one HUE home (hourly electricity), and RECS / CBECS respondents (annual energy only). Runs in a few seconds.
python3 demo.py
The BDG2 series are in examples/bdg2/ (CC BY-SA 4.0, see examples/bdg2/LICENSE-DATA.md); the HUE home and the survey
respondents are in examples/examples.*."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from buildingdetective import Detective  # noqa: E402


def show(r, keys=('floor_area_m2', 'year_built', 'UA_W_K', 'tau_h', 'Tb_heat_C', 'op_hours')):
    s = []
    for k in keys:
        v, lo, hi = r[k]
        s.append(f'{k}={v:,.4g} [{lo:,.4g}-{hi:,.4g}]')
    top = lambda d: max(d, key=d.get) + f' {100 * max(d.values()):.0f}%'
    return '  ' + '\n  '.join(s) + f"\n  use: {top(r['use'])} | heat: {top(r['heat'])} | cool: {top(r['cool'])}"


def main():
    t0 = time.time()
    det = Detective(os.path.join(HERE, 'weights'))
    ex = json.load(open(os.path.join(HERE, 'examples', 'examples.json')))
    bdg2 = json.load(open(os.path.join(HERE, 'examples', 'bdg2', 'bdg2_hourly.json')))
    ex['hourly'] = bdg2['hourly'] + ex['hourly']
    z = dict(np.load(os.path.join(HERE, 'examples', 'examples.npz')))
    z.update(np.load(os.path.join(HERE, 'examples', 'bdg2', 'bdg2_hourly.npz')))
    for u in ex['hourly']:
        idx = pd.DatetimeIndex(z[u['key'] + '_utc'].astype('datetime64[s]'), tz='UTC').tz_convert(u['tz'])
        met = {c: z[u['key'] + '_' + c] for c in u['channels']}
        r = det.read_hourly(met, z[u['key'] + '_T'], z[u['key'] + '_GHI'], idx, u['lat'])
        print(f"\n{u['id']}  (truth: {u['truth']})")
        print(show(r))
    for u in ex['annual']:
        r = det.read_annual(u['kwh'], u['hdd'], u['cdd'])
        print(f"\n{u['id']}  annual {u['kwh']}  (truth: {u['truth']})")
        print(show(r, ('floor_area_m2', 'year_built', 'UA_W_m2K')))
    print(f'\n{time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
