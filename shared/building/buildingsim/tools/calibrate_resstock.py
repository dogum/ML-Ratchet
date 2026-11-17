"""Fit the residential infiltration factor (one scalar on the LBL flow) on the CALIBRATION half of the ResStock
twin sample (first dwelling of each stratum, half == 0), minimising |median log(heating-load ratio)|.
The other half is held out and reported by checks/check_resstock.py. Writes buildingsim/data/calibration.json."""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, f'{BLDG_CODE}/checks')
sys.path.insert(0, f'{BLDG_CODE}/buildingsim/tools')
import check_resstock as cr  # noqa: E402
from build_priors import RES_COLS  # noqa: E402
from buildingsim.stock import resstock_to_spec  # noqa: E402

CAL = f'{BLDG_CODE}/buildingsim/buildingsim/data/calibration.json'


def run(factor, rows, hourly):
    rat = []
    import buildingsim.stock as st
    orig = st.resstock_to_spec
    def patched(r, whole_building=False):
        s = orig(r, whole_building)
        s['infiltration_factor'] = factor
        return s
    cr.resstock_to_spec = patched
    for _, (row, off) in rows.iterrows() if False else enumerate(rows):
        ref = hourly.to_table(filter=ds.field('bldg_id') == int(row['bldg_id'])).to_pandas()
        res = cr.one(row, off, ref)
        if res and res['annual']['heat_load'][1] > 500:
            rat.append(res['annual']['heat_load'][0] / res['annual']['heat_load'][1])
    cr.resstock_to_spec = orig
    return np.array(rat)


def main():
    pick = pd.read_csv(f'{BLDG_ROOT}/buildingsim/tmp/res_pick.csv')
    pick = pick[pick.half == 0]
    meta = ds.dataset(f'{BLDG_ROOT}/data/proc/stock/resstock_meta.parquet').to_table(
        columns=RES_COLS, filter=ds.field('bldg_id').isin([int(x) for x in pick.bldg_id])).to_pandas()
    meta = meta.set_index('bldg_id', drop=False)
    rows = []
    for _, p in pick.iterrows():
        r = meta.loc[int(p.bldg_id)]
        if str(r['in.vacancy_status']) == 'Vacant' or not os.path.exists(os.path.join(cr.WX, f"{r['in.county']}_2018.csv")):
            continue
        rows.append((r, int(p.std_utc_offset_h)))
    hourly = ds.dataset(f'{BLDG_ROOT}/data/proc/stock/resstock_hourly.parquet')
    res = {}
    for f in (1.0, 0.75, 0.55, 0.4):
        rat = run(f, rows, hourly)
        res[f] = dict(n=len(rat), median=float(np.median(rat)), p25=float(np.percentile(rat, 25)),
                      p75=float(np.percentile(rat, 75)))
        print(f, res[f], flush=True)
    fs = np.array(sorted(res))
    med = np.array([np.log(res[f]['median']) for f in fs])
    best = float(np.interp(0.0, med[np.argsort(med)], fs[np.argsort(med)]))
    best = float(np.clip(best, 0.3, 1.0))
    cal = json.load(open(CAL)) if os.path.exists(CAL) else {}
    cal.update(infiltration_factor_residential=round(best, 3),
               infiltration_factor_note='scalar on the LBL flow for residential buildings, fitted on the calibration '
                                        'half (n=%d) of the ResStock twins (heating-load median ratio -> 1); '
                                        'ResStock/OpenStudio-HPXML uses AIM-2 with site shielding' % res[1.0]['n'],
               infiltration_factor_scan={str(k): v for k, v in res.items()})
    json.dump(cal, open(CAL, 'w'), indent=1)
    print('best', best)


if __name__ == '__main__':
    main()
