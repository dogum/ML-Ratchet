"""Simulator vs simulator: BuildingSim twins of ResStock 2025 R1 baseline dwelling models (EnergyPlus via
OpenStudio-HPXML), same AMY2018 county weather, same characteristics (mapped by buildingsim.stock), 2018, 1 h.
This is NOT a check against reality: both sides are models.

Sample: 2 dwellings per (building type, climate zone) stratum of agent B's frozen ResStock sample (seed 7),
vacant units skipped. Reference hourly series: shared/building/data/proc/stock/resstock_hourly.parquet (agent B).
Compared: delivered heating and cooling loads (independent of equipment), indoor temperature, electricity,
on-site fossil fuel, site energy; annual ratios and hourly/daily shape.
Resumable: one JSON per dwelling in checks/out/resstock/.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from _common import write
from buildingsim import simulate
from buildingsim.stock import resstock_to_spec, read_stock_weather

sys.path.insert(0, f'{BLDG_CODE}/buildingsim/tools')
from build_priors import RES_COLS  # noqa: E402

OUT = f'{BLDG_ROOT}/checks/out/resstock'
WX = f'{BLDG_ROOT}/buildingsim/refdata/stock_weather'
KBTU = 0.29307107


def one(row, std_off, ref):
    spec = resstock_to_spec(row, whole_building=False)
    f = os.path.join(WX, f"{row['in.county']}_2018.csv")
    if not os.path.exists(f):
        return None
    wx = read_stock_weather(f, float(row['in.weather_file_latitude']), float(row['in.weather_file_longitude']),
                            std_off)
    start = pd.Timestamp('2018-01-01') - pd.Timedelta(hours=std_off)
    out = simulate(spec, wx, start, start + pd.Timedelta(days=365), dt='1h')
    ts = out['ts']
    lst = ts.index.tz_localize(None) + pd.Timedelta(hours=std_off)
    ref = ref.set_index('time_lst').reindex(lst)
    sim_heat = ts['Q_heat_kW'].values
    sim_cool = (ts['Q_cool_sens_kW'] + ts['Q_cool_lat_kW']).values
    ref_heat = ref['load_heating_delivered_kBtu'].values * KBTU
    ref_cool = ref['load_cooling_delivered_kBtu'].values * KBTU
    sim_el = ts['meter_electricity_kW'].values + ts['pv_kW'].values
    ref_el = ref['electricity_total_kWh'].values
    sim_fuel = ts['meter_gas_kW'].values
    ref_fuel = (ref['natural_gas_total_kWh'].fillna(0) + ref['fuel_oil_total_kWh'].fillna(0) +
                ref['propane_total_kWh'].fillna(0)).values
    def r(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 and np.std(a[m]) > 0 and np.std(b[m]) > 0 \
            else None
    def daily(x):
        return pd.Series(x, index=lst).resample('D').sum().values
    res = dict(bldg_id=int(row['bldg_id']), type=str(row['in.geometry_building_type_recs']),
               cz=str(row['in.ashrae_iecc_climate_zone_2004']), system=spec['system'],
               heating=str(row['in.hvac_heating_type_and_fuel']), area_m2=spec['floor_area_m2'],
               annual=dict(heat_load=(float(np.nansum(sim_heat)), float(np.nansum(ref_heat))),
                           cool_load=(float(np.nansum(sim_cool)), float(np.nansum(ref_cool))),
                           electricity=(float(np.nansum(sim_el)), float(np.nansum(ref_el))),
                           fuel=(float(np.nansum(sim_fuel)), float(np.nansum(ref_fuel))),
                           site=(float(np.nansum(sim_el + sim_fuel)), float(np.nansum(ref['site_energy_total_kWh'])))),
               hourly_r=dict(heat_load=r(sim_heat, ref_heat), cool_load=r(sim_cool, ref_cool),
                             electricity=r(sim_el, ref_el), T_indoor=r(ts['T_zone'].values,
                                                                       ref['indoor_temperature_conditioned_space_C'].values)),
               daily_r=dict(electricity=r(daily(sim_el), daily(ref_el)), site=r(daily(sim_el + sim_fuel),
                                                                               daily(ref['site_energy_total_kWh'].values))),
               T_indoor_mean=(float(ts['T_zone'].mean()), float(ref['indoor_temperature_conditioned_space_C'].mean())),
               ledger_closure=out['ledger']['closure_rel'],
               mean_profile_el=dict(sim=pd.Series(sim_el, index=lst).groupby(lst.hour).mean().tolist(),
                                    ref=pd.Series(ref_el, index=lst).groupby(lst.hour).mean().tolist()))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    pick = pd.read_csv(f'{BLDG_ROOT}/buildingsim/tmp/res_pick.csv')
    ids = [int(x) for x in pick.bldg_id]
    meta = ds.dataset(f'{BLDG_ROOT}/data/proc/stock/resstock_meta.parquet').to_table(
        columns=RES_COLS, filter=ds.field('bldg_id').isin(ids)).to_pandas().set_index('bldg_id', drop=False)
    hourly = ds.dataset(f'{BLDG_ROOT}/data/proc/stock/resstock_hourly.parquet')
    for _, p in pick.iterrows():
        f = os.path.join(OUT, f'{p.bldg_id}.json')
        if os.path.exists(f):
            continue
        row = meta.loc[int(p.bldg_id)]
        if str(row['in.vacancy_status']) == 'Vacant':
            json.dump(dict(skipped='vacant'), open(f, 'w'))
            continue
        ref = hourly.to_table(filter=ds.field('bldg_id') == int(p.bldg_id)).to_pandas()
        try:
            res = one(row, int(p.std_utc_offset_h), ref)
        except Exception as e:
            res = dict(error=repr(e))
            print('ERR', p.bldg_id, e)
        if res is None:
            continue
        json.dump(res, open(f, 'w'))
        if 'annual' in res:
            a = res['annual']
            print(res['bldg_id'], res['type'][:14], res['cz'], res['system'], {k: round(v[0] / max(v[1], 1), 2)
                                                                              for k, v in a.items()})
    summarize()


def summarize():
    rows = []
    for fn in sorted(os.listdir(OUT)):
        d = json.load(open(os.path.join(OUT, fn)))
        if 'annual' in d:
            rows.append(d)
    def stats(key, min_ref=100.0):
        rat = np.array([r['annual'][key][0] / r['annual'][key][1] for r in rows if r['annual'][key][1] > min_ref])
        if len(rat) == 0:
            return None
        return dict(n=len(rat), median_ratio=float(np.median(rat)), p25=float(np.percentile(rat, 25)),
                    p75=float(np.percentile(rat, 75)), within_25pct=float(np.mean(np.abs(rat - 1) <= 0.25)),
                    sum_ratio=float(sum(r['annual'][key][0] for r in rows if r['annual'][key][1] > min_ref) /
                                    sum(r['annual'][key][1] for r in rows if r['annual'][key][1] > min_ref)))
    def rstat(sec, key):
        v = np.array([r[sec][key] for r in rows if r[sec].get(key) is not None])
        return dict(n=len(v), median=float(np.median(v)), p25=float(np.percentile(v, 25))) if len(v) else None
    by_sys = {}
    for r in rows:
        by_sys.setdefault(r['system'], []).append(r['annual']['site'][0] / max(r['annual']['site'][1], 1))
    summary = dict(
        what='BuildingSim twins of ResStock 2025 R1 dwellings (simulator vs simulator, NOT reality)',
        n_dwellings=len(rows),
        annual_ratio_sim_over_resstock={k: stats(k) for k in ('heat_load', 'cool_load', 'electricity', 'fuel',
                                                                'site')},
        hourly_correlation={k: rstat('hourly_r', k) for k in ('heat_load', 'cool_load', 'electricity', 'T_indoor')},
        daily_correlation={k: rstat('daily_r', k) for k in ('electricity', 'site')},
        indoor_T_mean_diff_K=float(np.median([r['T_indoor_mean'][0] - r['T_indoor_mean'][1] for r in rows])),
        site_ratio_by_system={k: dict(n=len(v), median=float(np.median(v))) for k, v in by_sys.items()},
        max_ledger_closure=float(max(r['ledger_closure'] for r in rows)),
        mean_hourly_electricity_profile=dict(
            sim=np.mean([r['mean_profile_el']['sim'] for r in rows], axis=0).tolist(),
            ref=np.mean([r['mean_profile_el']['ref'] for r in rows], axis=0).tolist()),
        notes=['weather: OEDI AMY2018 county files used by ResStock; elevation not in the files (sea-level pressure '
               'assumed)', 'multi-family units simulated as units with exposure factors (shared walls adiabatic)',
               'ResStock appliances (dryers, ranges, pools, EV...) are lumped into plug loads in BuildingSim'])
    write('resstock_twins.json', summary)
    return summary


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summarize()
    else:
        main()
