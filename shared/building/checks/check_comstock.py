"""Simulator vs simulator: BuildingSim twins of ComStock 2025 R3 baseline models (EnergyPlus via OpenStudio),
same AMY2018 county weather (agent B's copies), characteristics mapped by buildingsim.stock, 2018, 1 h.
NOT a check against reality. Sample: 3 models per ComStock building type, spread over climate zones (seed 11),
from agent B's frozen ComStock sample. Compared: annual end uses (heating, cooling, fans, lighting, equipment,
water heating), electricity, gas, district, site; hourly and daily shape. Resumable: checks/out/comstock/."""
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
from buildingsim.stock import comstock_to_spec, read_stock_weather_parquet

sys.path.insert(0, f'{BLDG_CODE}/buildingsim/tools')
from build_priors import COM_COLS  # noqa: E402

OUT = f'{BLDG_ROOT}/checks/out/comstock'
SW = f'{BLDG_ROOT}/data/proc/stock/weather'
# approximate county centroid latitude from the state (solar geometry only needs ~1 degree)
STATE_LATLON = {'AL': (32.8, -86.8), 'AZ': (34.2, -111.7), 'AR': (34.9, -92.4), 'CA': (36.8, -119.6), 'CO': (39.0, -105.5),
                'CT': (41.6, -72.7), 'DE': (39.0, -75.5), 'FL': (28.6, -82.4), 'GA': (32.7, -83.4), 'ID': (44.4, -114.6),
                'IL': (40.0, -89.2), 'IN': (39.9, -86.3), 'IA': (42.1, -93.5), 'KS': (38.5, -98.4), 'KY': (37.5, -85.3),
                'LA': (31.1, -92.0), 'ME': (45.4, -69.2), 'MD': (39.0, -76.8), 'MA': (42.3, -71.8), 'MI': (44.3, -85.4),
                'MN': (46.3, -94.3), 'MS': (32.7, -89.7), 'MO': (38.4, -92.5), 'MT': (47.0, -109.6), 'NE': (41.5, -99.8),
                'NV': (39.3, -116.6), 'NH': (43.7, -71.6), 'NJ': (40.2, -74.7), 'NM': (34.4, -106.1), 'NY': (42.9, -75.5),
                'NC': (35.6, -79.4), 'ND': (47.5, -100.5), 'OH': (40.3, -82.8), 'OK': (35.6, -97.5), 'OR': (43.9, -120.6),
                'PA': (40.9, -77.8), 'RI': (41.7, -71.5), 'SC': (33.9, -80.9), 'SD': (44.4, -100.2), 'TN': (35.9, -86.4),
                'TX': (31.5, -99.3), 'UT': (39.3, -111.7), 'VT': (44.1, -72.7), 'VA': (37.5, -78.9), 'WA': (47.4, -120.5),
                'WV': (38.6, -80.6), 'WI': (44.6, -89.9), 'WY': (43.0, -107.6), 'DC': (38.9, -77.0)}

ENDUSE = {  # comparison group: (sim end-use keys, ComStock columns)
    'heating': (['heating_electricity', 'heating_gas', 'heating_district'],
                ['electricity_heating_kWh', 'natural_gas_heating_kWh', 'district_heating_heating_kWh',
                 'propane_heating_kWh', 'fuel_oil_heating_kWh']),
    'cooling': (['cooling_electricity', 'cooling_district'], ['electricity_cooling_kWh', 'district_cooling_cooling_kWh']),
    'fans': (['fans_electricity'], ['electricity_fans_kWh']),
    'pumps_rejection': (['pumps_electricity', 'heat_rejection_electricity'], ['electricity_pumps_kWh',
                                                                           'electricity_heat_rejection_kWh']),
    'interior_lighting': (['interior_lighting_electricity'], ['electricity_interior_lighting_kWh']),
    'equipment_refrigeration': (['interior_equipment_electricity', 'cooking_gas'],
                                ['electricity_interior_equipment_kWh', 'electricity_refrigeration_kWh',
                                 'natural_gas_interior_equipment_kWh']),
    'water_heating': (['water_systems_electricity', 'water_systems_gas', 'water_systems_district'],
                      ['electricity_water_systems_kWh', 'natural_gas_water_systems_kWh',
                       'district_heating_water_systems_kWh', 'propane_water_systems_kWh', 'fuel_oil_water_systems_kWh']),
    'exterior_lighting': (['exterior_lighting_electricity'], ['electricity_exterior_lighting_kWh']),
    'electricity': (None, ['electricity_total_kWh']),
    'site': (None, ['site_energy_total_kWh']),
}


def one(row, smp, ref):
    spec = comstock_to_spec(row)
    f = os.path.join(SW, f"{smp['county']}.parquet")
    if not os.path.exists(f):
        return None
    lat, lon = STATE_LATLON.get(str(smp['state']), (38.0, -95.0))
    wx = read_stock_weather_parquet(f, lat, lon)
    off = wx[1]['std_offset']
    start = pd.Timestamp('2018-01-01') - pd.Timedelta(hours=off)
    out = simulate(spec, wx, start, start + pd.Timedelta(days=365), dt='1h')
    ts = out['ts']
    lst = ts.index.tz_localize(None) + pd.Timedelta(hours=off)
    ref = ref.set_index('time_lst').reindex(lst)
    annual = {}
    for g, (sk, rk) in ENDUSE.items():
        rv = float(sum(np.nansum(ref[c].values) for c in rk if c in ref))
        if sk is None:
            if g == 'electricity':
                sv = float((ts['meter_electricity_kW'] + ts['pv_kW']).sum())
            else:
                sv = float(sum(ts[c].sum() for c in ts.columns if c.startswith('meter_')) + ts['pv_kW'].sum())
        else:
            sv = float(sum(ts[k + '_kW'].sum() for k in sk))
        annual[g] = (sv, rv)
    sim_el = (ts['meter_electricity_kW'] + ts['pv_kW']).values
    ref_el = ref['electricity_total_kWh'].values
    def r(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 and np.std(a[m]) > 0 and np.std(b[m]) > 0 \
            else None
    site_s = sum(ts[c].values for c in ts.columns if c.startswith('meter_')) + ts['pv_kW'].values
    daily = lambda x: pd.Series(x, index=lst).resample('D').sum().values
    return dict(bldg_id=int(row['bldg_id']), type=str(row['in.comstock_building_type']),
                cz=str(row['in.as_simulated_ashrae_iecc_climate_zone_2006']), hvac=str(row['in.hvac_system_type']),
                system=spec['system'], area_m2=spec['floor_area_m2'], annual=annual,
                hourly_r=dict(electricity=r(sim_el, ref_el)),
                daily_r=dict(electricity=r(daily(sim_el), daily(ref_el)),
                             site=r(daily(site_s), daily(ref['site_energy_total_kWh'].values))),
                ledger_closure=out['ledger']['closure_rel'],
                weekday_profile=dict(sim=pd.Series(sim_el, index=lst)[lst.dayofweek < 5].groupby(
                    lst[lst.dayofweek < 5].hour).mean().tolist(),
                    ref=pd.Series(ref_el, index=lst)[lst.dayofweek < 5].groupby(lst[lst.dayofweek < 5].hour).mean().tolist()))


def _pick(seed, exclude=()):
    smp = pd.read_parquet(f'{BLDG_ROOT}/data/proc/stock/comstock_sample.parquet')
    smp['type'] = smp['stratum'].str.split('|').str[0]
    smp = smp[~smp.bldg_id.isin(list(exclude))]
    idx = smp.groupby('type').sample(n=3, random_state=seed).index
    return smp.loc[idx]


def main(seed=12):
    global OUT
    cal_ids = set(_pick(11).bldg_id)
    pick = _pick(11) if seed == 11 else _pick(seed, exclude=cal_ids)
    OUT = OUT if seed == 11 else OUT + f'_heldout{seed}'
    os.makedirs(OUT, exist_ok=True)
    meta = ds.dataset(f'{BLDG_ROOT}/data/proc/stock/comstock_meta.parquet').to_table(
        columns=COM_COLS, filter=ds.field('bldg_id').isin([int(x) for x in pick.bldg_id])).to_pandas()
    hourly = ds.dataset(f'{BLDG_ROOT}/data/proc/stock/comstock_hourly.parquet')
    for _, p in pick.iterrows():
        f = os.path.join(OUT, f'{p.bldg_id}_{p.state}.json')
        if os.path.exists(f):
            continue
        rows = meta[(meta.bldg_id == int(p.bldg_id)) & (meta['in.state'].astype(str) == str(p.state))]
        if len(rows) == 0:
            rows = meta[meta.bldg_id == int(p.bldg_id)]
        row = rows.iloc[0]
        ref = hourly.to_table(filter=ds.field('bldg_id') == int(p.bldg_id)).to_pandas()
        try:
            res = one(row, p, ref)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = dict(error=repr(e))
        if res is None:
            continue
        json.dump(res, open(f, 'w'))
        if 'annual' in res:
            print(res['type'], res['cz'], res['system'], {k: round(v[0] / max(v[1], 1), 2) for k, v in res['annual'].items()},
                  flush=True)


def summarize(outdir=None):
    outdir = outdir or OUT
    rows = [json.load(open(os.path.join(outdir, f))) for f in sorted(os.listdir(outdir))]
    rows = [r for r in rows if 'annual' in r]
    def st(key, min_ref=500.0):
        rat = np.array([r['annual'][key][0] / r['annual'][key][1] for r in rows if r['annual'][key][1] > min_ref])
        if not len(rat):
            return None
        return dict(n=len(rat), median_ratio=float(np.median(rat)), p25=float(np.percentile(rat, 25)),
                    p75=float(np.percentile(rat, 75)), within_25pct=float(np.mean(np.abs(rat - 1) <= 0.25)),
                    sum_ratio=float(sum(r['annual'][key][0] for r in rows if r['annual'][key][1] > min_ref) /
                                    sum(r['annual'][key][1] for r in rows if r['annual'][key][1] > min_ref)))
    by_type = {}
    for r in rows:
        by_type.setdefault(r['type'], []).append(r['annual']['site'][0] / max(r['annual']['site'][1], 1))
    rr = lambda sec, k: [r[sec][k] for r in rows if r[sec].get(k) is not None]
    s = dict(what='BuildingSim twins of ComStock 2025 R3 models (simulator vs simulator, NOT reality)',
             n_buildings=len(rows), annual_ratio_sim_over_comstock={k: st(k) for k in ENDUSE},
             hourly_correlation_electricity=dict(median=float(np.median(rr('hourly_r', 'electricity')))),
             daily_correlation=dict(electricity=float(np.median(rr('daily_r', 'electricity'))),
                                    site=float(np.median(rr('daily_r', 'site')))),
             site_ratio_by_type={k: dict(n=len(v), median=float(np.median(v))) for k, v in by_type.items()},
             max_ledger_closure=float(max(r['ledger_closure'] for r in rows)),
             notes=['ComStock envelope U-values are not in the metadata: BuildingSim uses its code-level table for the '
                    'energy code of construction / last roof replacement', 'refrigeration is booked with equipment',
                    'latitude from the state centroid'])
    return s


def report():
    cal = summarize(f'{BLDG_ROOT}/checks/out/comstock')
    held = summarize(f'{BLDG_ROOT}/checks/out/comstock_heldout12')
    write('comstock_twins.json', dict(
        what='BuildingSim twins of ComStock 2025 R3 models (simulator vs simulator, NOT reality)',
        heldout_seed12=held, calibration_seed11_in_sample=cal,
        corrections_made_on_seed11=['lighting-generation LPD multipliers x0.66', 'non-residential DHW volume x0.6',
                                    'non-residential exterior lighting x1.8',
                                    'night base fractions +0.05 lighting / +0.10 plug'],
        note='the seed-11 sample motivated the corrections (lighting, DHW, exterior lighting, night base; before them '
             'its site ratio was 0.96 median with lighting 1.52x and DHW 1.61x) and is shown re-simulated after them '
             '(in-sample); seed 12 is a disjoint draw and is the number to quote'))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
        report()
