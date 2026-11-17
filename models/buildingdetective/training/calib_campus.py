"""W2: fit the campus district factor on the BDG2 DEV twins (dev only, 2017 local year).

Each dev building with a good district meter gets a twin from its metadata (Phase 1 spec_from_bdg2: use, area, year,
climate, meters -> system) plus the W1 intensity factor and the W2 campus physics (long AHU hours on, higher VAV
minimum and outdoor air) at district_eff = 1. Real/sim annual ratios give the distribution factor:
meter = delivered / eff, so eff = sim(eff=1) / real. eff_median and the log-sd go to calibration.json['campus'].
Also records the electricity twin ratio for every dev building (W1 effect on BDG2).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import common as C
from wrappers import load_cal, CAL, apply_intensity

from _paths import BLDG  # noqa: E402

sys.path.insert(0, os.path.join(BLDG, 'checks'))


def twin_spec(bid):
    import buildingdata as bd
    from check_bdg2 import spec_from_bdg2
    return spec_from_bdg2(bd.bdg2_spec(bid))


def run(write=True):
    import buildingdata as bd
    import buildingsim as bs
    cal = load_cal()
    meta = bd.bdg2_meta(split='dev')
    q = bd.bdg2_quality()
    q = q[q.building_id.isin(meta.id)]
    rows = []
    for _, b in meta.iterrows():
        bid, site = b['id'], b['site_id']
        good = {r.meter: r.annual_2017 for r in q[q.building_id == bid].itertuples() if r.q_ok_daily_2017}
        if not good:
            continue
        spec = twin_spec(bid)
        f = bs.fill_spec(spec)
        apply_intensity(f, cal['intensity'].get(f['primary_use'], 1.0))
        ms = set(str(b['meters']).split(','))
        if f['system'] == 'district_steam_chw' or f['hvac'].get('heat_source') == 'district':
            f['hvac']['vav_min_frac'] = 0.45
            f['ventilation']['L_s_m2'] = float(f['ventilation'].get('L_s_m2') or 0.3) * 1.5
        w, m = C.site_wx(site)
        start = pd.Timestamp('2017-01-01', tz=m['tz']).tz_convert('UTC')
        end = pd.Timestamp('2018-01-01', tz=m['tz']).tz_convert('UTC')
        out = bs.simulate(f, (w, m), start, end, dt='1h', record_trend=False)
        km = out['summary']['kWh_by_meter']
        hw = km.get('steam', 0.0) + km.get('hot_water', 0.0)
        simv = dict(electricity=km.get('electricity', 0.0), chilledwater=km.get('chilled_water', 0.0), steam=hw,
                    hotwater=hw, gas=km.get('gas', 0.0))
        for mt, real in good.items():
            if mt not in simv or not real or real <= 0:
                continue
            rows.append(dict(building=bid, use=b['primary_use'], meter=mt, system=f['system'], meters=','.join(sorted(ms)),
                             real_kWh=float(real), sim_kWh=float(simv[mt]), sim_over_real=float(simv[mt] / real)))
            print(bid, mt, f['system'], 'sim/real %.2f' % rows[-1]['sim_over_real'], flush=True)
    d = pd.DataFrame(rows)
    res = {}
    for mt, g in d.groupby('meter'):
        r = g.sim_over_real[g.sim_over_real > 0]
        res[mt] = dict(n=len(r), median=float(r.median()), p25=float(r.quantile(.25)), p75=float(r.quantile(.75)),
                       log_sd=float(np.std(np.log(r))))
    heat = d[d.meter.isin(['steam', 'hotwater'])].sim_over_real
    chw = d[d.meter == 'chilledwater'].sim_over_real
    camp = dict(eff_heat_median=float(np.clip(heat.median(), 0.05, 1.0)),
                eff_chw_median=float(np.clip(chw.median(), 0.05, 1.0)),
                eff_sd=float(np.clip(np.std(np.log(pd.concat([heat, chw]))) * 0.8, 0.2, 0.8)),
                p_campus=cal['campus'].get('p_campus', 0.3), n_heat=int(len(heat)), n_chw=int(len(chw)),
                note='eff = sim(eff=1)/real on BDG2 dev twins, 2017; eff_sd = 0.8 x log-sd of all district ratios '
                     '(part of the spread is metadata error, not plant)')
    if write:
        cal['campus'] = camp
        cal['campus_twins'] = dict(per_meter=res, rows=rows)
        json.dump(cal, open(CAL, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    print(json.dumps(camp, indent=1))
    return d


if __name__ == '__main__':
    run()
