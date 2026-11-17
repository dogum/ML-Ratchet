"""BuildingSim against REAL meters: the 52 BDG2 dev buildings of the frozen split (agent B), 2016, 1 h.

The simulator only knows what BDG2 metadata gives (use, floor area, year built for some, climate zone, location,
which meters exist); everything else comes from BuildingSim defaults. The system is inferred from the meters:
chilled water + steam/hot water -> district_steam_chw; steam or hot water only -> VAV with district heat and an
electric chiller; chilled water only -> district chilled water with district heat (unmetered); electricity only ->
the use-type default. This is the "generic twin from metadata" level that Phase 2 starts from, so large errors in
annual totals are expected; the hourly SHAPE (normalised by each series' mean) is the more meaningful comparison.
Only building-meters with q_ok (agent B's quality flag) are scored. Exam buildings are never touched.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, BLDG_CODE)
import buildingdata as bd  # noqa: E402
from _common import write  # noqa: E402
from buildingsim import simulate  # noqa: E402

METERS = {'electricity': 'meter_electricity_kW', 'chilledwater': 'meter_chilled_water_kW',
          'steam': 'meter_steam_kW', 'hotwater': 'meter_hot_water_kW', 'gas': 'meter_gas_kW'}


def spec_from_bdg2(s):
    s = {k: v for k, v in s.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    meters = set((s.get('meters') or {}).keys()) if isinstance(s.get('meters'), dict) else set(
        str(s.get('meters', '')).split(','))
    chw = 'chilledwater' in meters
    stm = 'steam' in meters
    hw = 'hotwater' in meters
    h = {}
    if chw and (stm or hw):
        system = 'district_steam_chw'
        if hw and not stm:
            h['district_heat'] = 'hot_water'
    elif (stm or hw) and not chw:
        system = 'vav_chiller_boiler'
        h['heat_source'] = 'district'
        if hw:
            h['district_heat'] = 'hot_water'
    elif chw:
        system = 'district_steam_chw'
    else:
        system = None
    out = dict(s)
    out.pop('meters', None)
    if system:
        out['system'] = system
    out['hvac'] = h
    if 'n_floors' not in out and out.get('floor_area_m2'):
        out['n_floors'] = max(1, int(round(np.sqrt(out['floor_area_m2']) / 30)))
    return out


def main():
    meta = bd.bdg2_meta(split='dev')
    rows = []
    wcache = {}
    for _, b in meta.iterrows():
        bid = b['id']
        site = b['site_id']
        spec = spec_from_bdg2(bd.bdg2_spec(bid))
        if site not in wcache:
            w = bd.site_weather(site).copy()
            if 'LW_down' in w:
                w['IR_horiz'] = w['LW_down']
            wm = bd.site_weather_meta(site)
            wcache[site] = (w, dict(lat=wm['lat'], lon=wm['lon'], elev=wm.get('elev', 0.0), tz=wm['tz']))
        w, wm = wcache[site]
        start = pd.Timestamp('2016-01-01', tz=wm['tz']).tz_convert('UTC')
        end = pd.Timestamp('2017-01-01', tz=wm['tz']).tz_convert('UTC')
        try:
            out = simulate(spec, (w, wm), start, end, dt='1h')
        except Exception as e:
            print('ERR', bid, e)
            continue
        ts = out['ts']
        for mtr, col in METERS.items():
            if not b.get(f'q_ok_{mtr}'):
                continue
            real = bd.bdg2(mtr, building=bid)
            real = real.iloc[:, 0].reindex(ts.index)
            sim = ts[col] if col in ts else None
            if sim is None or real.notna().sum() < 4000 or real.sum() <= 0:
                continue
            m = real.notna().values
            rv, sv = real.values[m], sim.values[m]
            ann_r, ann_s = rv.mean() * 8760, sv.mean() * 8760
            rn, sn = rv / max(rv.mean(), 1e-9), sv / max(sv.mean(), 1e-9)
            loc = ts.index[m].tz_convert(wm['tz'])
            wk = loc.dayofweek < 5
            prof_r = pd.Series(rn[wk]).groupby(loc.hour[wk]).mean().values
            prof_s = pd.Series(sn[wk]).groupby(loc.hour[wk]).mean().values
            dr = pd.Series(rv, index=loc).resample('D').mean()
            ds_ = pd.Series(sv, index=loc).resample('D').mean()
            dT = pd.Series(ts['T_out'].values[m], index=loc).resample('D').mean()
            ok = dr.notna() & ds_.notna()
            rows.append(dict(building=bid, use=b['primary_use'], cz=b['climate_zone'], meter=mtr, system=out['spec']['system'],
                             annual_real_kWh=float(ann_r), annual_sim_kWh=float(ann_s), ratio=float(ann_s / ann_r),
                             hourly_r_normalised=float(np.corrcoef(rn, sn)[0, 1]) if np.std(sn) > 0 else None,
                             cvrmse_hourly_normalised=float(np.sqrt(np.mean((rn - sn) ** 2))),
                             weekday_profile_r=float(np.corrcoef(prof_r, prof_s)[0, 1]) if np.std(prof_s) > 0 else None,
                             daily_r=float(np.corrcoef(dr[ok], ds_[ok])[0, 1]) if np.std(ds_[ok]) > 0 else None,
                             daily_vs_T_slope_real=float(np.polyfit(dT[ok], dr[ok] / dr[ok].mean(), 1)[0]),
                             daily_vs_T_slope_sim=float(np.polyfit(dT[ok], ds_[ok] / ds_[ok].mean(), 1)[0])))
            print(bid, mtr, out['spec']['system'], 'ratio %.2f hourly r %.2f profile r %.2f daily r %.2f' % (
                rows[-1]['ratio'], rows[-1]['hourly_r_normalised'] or 0, rows[-1]['weekday_profile_r'] or 0,
                rows[-1]['daily_r'] or 0), flush=True)
    d = pd.DataFrame(rows)
    summ = {}
    for mtr, g in d.groupby('meter'):
        summ[mtr] = dict(n=len(g), annual_ratio_median=float(g.ratio.median()),
                         annual_ratio_p25=float(g.ratio.quantile(0.25)), annual_ratio_p75=float(g.ratio.quantile(0.75)),
                         within_25pct=float((np.abs(g.ratio - 1) <= 0.25).mean()),
                         within_50pct=float((np.abs(g.ratio - 1) <= 0.5).mean()),
                         hourly_r_median=float(g.hourly_r_normalised.median()),
                         cvrmse_normalised_median=float(g.cvrmse_hourly_normalised.median()),
                         weekday_profile_r_median=float(g.weekday_profile_r.median()),
                         daily_r_median=float(g.daily_r.median()),
                         temp_slope_sign_agreement=float((np.sign(g.daily_vs_T_slope_real) ==
                                                          np.sign(g.daily_vs_T_slope_sim)).mean()))
    write('bdg2_dev_meters.json', dict(
        what='BuildingSim generic twins (metadata only) vs REAL BDG2 meters, 52 dev buildings, 2016, hourly',
        split='bldg/data/proc/splits.json bdg2.dev (frozen by agent B); exam buildings untouched',
        per_meter=summ, buildings=rows,
        notes=['envelope, schedules, setpoints, systems are BuildingSim defaults for the use type: annual errors '
               'measure the defaults + metadata gap, not the physics alone',
               'campus district meters (chilled water, steam, hot water) are compared with the district meters of the '
               'twin', 'shape metrics use each series divided by its own mean']))


if __name__ == '__main__':
    main()
