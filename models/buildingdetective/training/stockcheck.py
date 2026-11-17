"""Simulator-vs-simulator check: the Detective reads NREL ResStock / ComStock EnergyPlus meters (AMY2018 county
weather). NOT reality. The truth for UA, tau, balance points comes from a BuildingSim twin built from the SAME stock
characteristics (buildingsim.stock mapping); heating / cooling class, floor area and vintage come from the stock
model's own characteristics. Used only as a report (never for choosing).

Sample (fixed seed 5, disjoint from nothing we train on: the bank never sees stock meters):
  ResStock: 12 dwellings per building type (5 types; vacant skipped) = 60
  ComStock: 6 models per building type (15 types) = 90
Features: hourly / monthly / annual exactly as for real buildings (no quirk emulation: stock meters are clean).
usage: python3 stockcheck.py [n_res_per_type=12] [n_com_per_type=6]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import common as C
import features as F

warnings.filterwarnings('ignore')
from _paths import BLDG  # noqa: E402

sys.path.insert(0, os.path.join(BLDG, 'checks'))
sys.path.insert(0, os.path.join(BLDG, 'buildingsim', 'tools'))
P = os.path.join(BLDG, 'data', 'proc', 'stock') + os.sep
OD = os.path.join(C.ROOT, 'feats', 'stock')


def res_heat(s):
    s = str(s)
    if 'ASHP' in s or 'MSHP' in s or 'GSHP' in s:
        return 'heat_pump'
    if s.startswith('Electricity'):
        return 'resistance'
    if s.startswith(('Natural Gas', 'Propane', 'Fuel Oil', 'Other Fuel')):
        return 'fuel'
    if s == 'None':
        return 'none'
    return None


def com_heat(fuel, hvac):
    fuel, hvac = str(fuel), str(hvac)
    if fuel == 'DistrictHeating' or 'district hot water' in hvac:
        return 'district'
    if 'HP' in hvac or 'heat pump' in hvac:
        return 'heat_pump' if 'boiler' not in hvac else 'fuel'
    if fuel == 'Electricity':
        return 'resistance'
    return 'fuel'


def com_cool(hvac):
    hvac = str(hvac)
    if 'district chilled' in hvac:
        return 'district'
    if 'chiller' in hvac:
        return 'chiller'
    return 'dx'


def vintage_mid(v):
    v = str(v)
    if v.startswith('<'):
        return 1930.0
    if v[:4].isdigit():
        return float(v[:4]) + 5
    return np.nan


def feats_for(meters, wdf, tz, lat):
    loc = wdf.index.tz_convert(tz)
    T, G = wdf['T_out'].to_numpy(float), wdf['GHI'].to_numpy(float)
    h = F.feat_hourly(meters, T, G, loc, lat)
    mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(meters, T, loc)
    m = F.feat_monthly(mm, Tm, Hm, Cm, nd, lat)
    d = np.asarray(loc.normalize().asi8)
    _, inv = np.unique(d, return_inverse=True)
    Td = np.bincount(inv, weights=T) / np.bincount(inv)
    kwh = {c: float(np.nansum(v)) for c, v in meters.items()}
    a = F.feat_annual(kwh, float(np.maximum(F.HB - Td, 0).sum()), float(np.maximum(Td - F.HB, 0).sum()))
    return h, m, a, kwh


def twin_labels(spec, wx):
    import buildingsim as bs
    from buildingsim.generator import labels
    f = bs.fill_spec(spec)
    off = wx[1]['std_offset']
    start = pd.Timestamp('2018-01-01') - pd.Timedelta(hours=off)
    out = bs.simulate(f, wx, start.tz_localize('UTC'), (start + pd.Timedelta(days=365)).tz_localize('UTC'), dt='1h',
                      record_trend=False)
    L = labels(f, out)
    y, cls = C.derived_labels(f, out, L, wx[1]['tz'])
    return y, cls, out


def run(kind, n_per):
    from buildingsim.stock import resstock_to_spec, comstock_to_spec, read_stock_weather_parquet
    from build_priors import RES_COLS, COM_COLS
    from check_comstock import STATE_LATLON
    smp = pd.read_parquet(P + f'{kind}_sample.parquet')
    smp['type'] = smp['stratum'].str.split('|').str[0]
    pick = smp.loc[smp.groupby('type').sample(n=n_per, random_state=5, replace=False).index]
    cols = RES_COLS if kind == 'resstock' else COM_COLS
    meta = ds.dataset(P + f'{kind}_meta.parquet').to_table(
        columns=cols, filter=ds.field('bldg_id').isin([int(x) for x in pick.bldg_id])).to_pandas()
    hourly = ds.dataset(P + f'{kind}_hourly.parquet')
    acc = dict(hourly=[], monthly=[], annual=[], Y=[], K=[])
    labs = []
    for _, p in pick.iterrows():
        rows = meta[meta.bldg_id == int(p.bldg_id)]
        if kind == 'comstock' and 'in.state' in rows:
            r2 = rows[rows['in.state'].astype(str) == str(p.state)]
            rows = r2 if len(r2) else rows
        if not len(rows):
            continue
        row = rows.iloc[0]
        if kind == 'resstock' and str(row['in.vacancy_status']) == 'Vacant':
            continue
        wf = P + f"weather/{p.county}.parquet"
        if not os.path.exists(wf):
            continue
        if kind == 'resstock':
            lat, lon = float(row['in.weather_file_latitude']), float(row['in.weather_file_longitude'])
        else:
            lat, lon = STATE_LATLON.get(str(p.state), (38.0, -95.0))
        wx = read_stock_weather_parquet(wf, lat, lon)
        try:
            spec = resstock_to_spec(row, whole_building=False) if kind == 'resstock' else comstock_to_spec(row)
            y, cls, out = twin_labels(spec, wx)
        except Exception as e:
            print('ERR twin', p.bldg_id, repr(e)[:200], flush=True)
            continue
        ref = hourly.to_table(filter=ds.field('bldg_id') == int(p.bldg_id)).to_pandas()
        off = wx[1]['std_offset']
        idx = out['ts'].index
        lst = idx.tz_localize(None) + pd.Timedelta(hours=off)
        ref = ref.drop_duplicates('time_lst').set_index('time_lst').reindex(lst)
        g = lambda c: ref[c].fillna(0).to_numpy(float) if c in ref else np.zeros(len(ref))
        met = dict(elec=g('electricity_total_kWh'),
                   fuel=g('natural_gas_total_kWh') + g('propane_total_kWh') + g('fuel_oil_total_kWh'))
        if kind == 'comstock':
            met['dheat'] = g('district_heating_total_kWh')
            met['dcool'] = g('district_cooling_total_kWh')
        met = {c: v for c, v in met.items() if v.sum() > 1.0}
        wdf = wx[0].reindex(idx).interpolate(limit=6).bfill().ffill()
        h, m, a, kwh = feats_for(met, wdf, wx[1]['tz'], lat)
        if kind == 'resstock':
            heat, cool = res_heat(row['in.hvac_heating_type_and_fuel']), None
            year = vintage_mid(row['in.vintage'])
            use = 'residential_single' if 'Single' in p.type or 'Mobile' in p.type else 'residential_multi'
        else:
            heat = com_heat(row['in.heating_fuel'], row['in.hvac_system_type'])
            cool = com_cool(row['in.hvac_system_type'])
            year = float(row['in.year_built']) if pd.notna(row['in.year_built']) else np.nan
            use = spec.get('primary_use')
        acc['hourly'].append(h); acc['monthly'].append(m); acc['annual'].append(a)
        acc['Y'].append([y[k] for k in C.REG])
        labs.append(dict(id=f'{kind}_{p.bldg_id}', type=p.type, use=use, heat=heat, cool=cool,
                         twin_heat=cls['heat'], twin_cool=cls['cool'], A=float(spec['floor_area_m2']), year=year,
                         kwh=kwh, system=spec.get('system'),
                         stock_heating=str(row.get('in.hvac_heating_type_and_fuel', row.get('in.heating_fuel'))),
                         stock_hvac=str(row.get('in.hvac_system_type', row.get('in.hvac_cooling_type')))))
        print(kind, len(labs), p.type, heat, cool, 'UA %.0f' % 10 ** y['logUA'], flush=True)
    os.makedirs(OD, exist_ok=True)
    np.savez(os.path.join(OD, f'{kind}.npz'), hourly=np.array(acc['hourly'], np.float32),
             monthly=np.array(acc['monthly'], np.float32), annual=np.array(acc['annual'], np.float32),
             Y=np.array(acc['Y'], np.float32))
    json.dump(labs, open(os.path.join(OD, f'{kind}.json'), 'w'), default=float)


if __name__ == '__main__':
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run('resstock', nr)
    run('comstock', nc)
