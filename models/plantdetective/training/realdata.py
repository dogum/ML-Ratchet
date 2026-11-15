"""Real CAMPD units: hourly records (truth side) and the context PlantDetective may see (EIA-860/923, prices).

The model never sees CAMPD heat input.  Context comes only from EIA-860 inventory (prime mover, CHP flag, nameplate,
minimum load, cold-start time), EIA-923 annual fuel and net generation of the linked generators (the same construction
as the baseline in grid/data/scripts/baselines.py), ERCOT day-ahead hub prices and Henry Hub daily gas.
"""
from common import *
import functools
import numpy as np, pandas as pd
import griddata as gd
from gridsim.fleet import henry_hub_daily

PUDL = f'{GRID}/data/raw/pudl'
COLD = {'10M': 0.17, '1H': 1.0, '12H': 12.0, 'OVER': 24.0}   # EIA-860 time from cold shutdown to full load


@functools.lru_cache(maxsize=1)
def units():
    u = gd.campd_units()
    # never carry the CAMPD-derived characteristic columns forward (exam truth)
    keep = ['unit_id', 'plant_id_epa', 'unit_id_epa', 'facility_name', 'unit_type', 'primary_fuel', 'plant_id_eia',
            'generator_ids', 'crosswalk', 'ba', 'split', 'first_year', 'last_year']
    return u[keep].copy()


def truth_tech(row):
    """Technology label from CAMPD unit type + fuel (metadata, not measurements)."""
    ut, fu = str(row['unit_type']), str(row['primary_fuel'])
    if ut == 'Combined cycle':
        return 'gas_cc'
    if ut == 'Combustion turbine':
        return 'gas_ct'
    if 'Coal' in fu:
        return 'coal_steam'
    if 'Gas' in fu or 'Natural' in fu:
        return 'gas_steam'
    return 'other'


@functools.lru_cache(maxsize=1)
def _gens():
    cols = ['plant_id_eia', 'generator_id', 'report_date', 'prime_mover_code', 'associated_combined_heat_power',
            'minimum_load_mw', 'time_cold_shutdown_full_load_code', 'capacity_mw', 'energy_source_code_1',
            'unit_id_pudl', 'generator_operating_date']
    g = pd.read_parquet(f'{PUDL}/out_eia__yearly_generators.parquet', columns=cols)
    g = g[g.plant_id_eia.isin(units().plant_id_eia.dropna().astype(int).unique())]
    g['generator_id'] = g.generator_id.astype(str)
    return g.sort_values('report_date').groupby(['plant_id_eia', 'generator_id']).last().reset_index()


@functools.lru_cache(maxsize=1)
def _gf():
    hr = pd.read_parquet(f'{PUDL}/out_eia923__yearly_generation_fuel_by_generator.parquet')
    hr['year'] = pd.to_datetime(hr.report_date).dt.year
    hr['generator_id'] = hr.generator_id.astype(str)
    return hr.groupby(['plant_id_eia', 'generator_id', 'year'])[['fuel_consumed_for_electricity_mmbtu',
                                                               'net_generation_mwh']].sum()


def context(unit_id):
    """EIA-860/923 context of one CAMPD unit (dict; None where unknown)."""
    u = units().set_index('unit_id').loc[unit_id]
    pid = int(u.plant_id_eia)
    gens = [g for g in str(u.generator_ids).split(';') if g and g != 'None']
    G = _gens()
    gg = G[(G.plant_id_eia == pid) & G.generator_id.isin(gens)]
    # how many CAMPD units share each generator (a CC steam part shared by several CTs)
    share = {}
    for g_ in gens:
        n = sum(g_ in str(x).split(';') for x in units()[units().plant_id_eia == pid].generator_ids)
        share[g_] = max(n, 1)
    pm = gg.prime_mover_code.astype(str).tolist()
    own = gg[~gg.prime_mover_code.isin(['CA'])]
    st = gg[gg.prime_mover_code.isin(['CA'])]
    np_own = float(own.capacity_mw.sum()) if len(own) else None
    np_st = float(sum(r.capacity_mw / share[r.generator_id] for r in st.itertuples())) if len(st) else 0.0
    ml = own.minimum_load_mw.dropna()
    cold = own.time_cold_shutdown_full_load_code.dropna().map(COLD).dropna()
    ctx = dict(pm_ct=float(any(p in ('CT', 'GT') for p in pm)), pm_st=float(any(p == 'ST' for p in pm)),
               pm_ca=float(len(st) > 0 or any(p in ('CA', 'CS') for p in pm)),
               fuel_coal=float(any(str(e) in ('SUB', 'BIT', 'LIG', 'RC', 'WC') for e in gg.energy_source_code_1)),
               chp=float(bool(gg.associated_combined_heat_power.fillna(False).any())) if len(gg) else None,
               np_own=np_own, np_st_share=np_st,
               minload=float(ml.sum()) if len(ml) == len(own) and len(ml) else None,
               cold_h=float(cold.max()) if len(cold) else None,
               online_year=float(pd.to_datetime(own.generator_operating_date).dt.year.min()) if len(own) else None)
    # EIA-923 annual heat rate of the linked generators (same construction as the baseline)
    gf = _gf()
    hr = {}
    for y in range(2018, 2026):
        k = [(pid, g_, y) for g_ in gens if (pid, g_, y) in gf.index]
        if not k:
            continue
        s = gf.loc[k].sum()
        if s.net_generation_mwh > 0 and s.fuel_consumed_for_electricity_mmbtu > 0:
            h = s.fuel_consumed_for_electricity_mmbtu / s.net_generation_mwh
            if 5 < h < 25:
                hr[y] = float(h)
    ctx['hr923'] = hr
    return ctx


def record(unit_id, start=None, end=None):
    """Hourly CAMPD record on a regular UTC grid: op (h), P (gross MW while operating), HI (MMBtu), meas (bool)."""
    h = gd.campd_unit(unit_id, start, end)
    if h.empty:
        return h
    idx = pd.date_range(h.index.min(), h.index.max(), freq='h')
    h = h[~h.index.duplicated()].reindex(idx)
    meas = h.hi_meas.astype(str).str.contains('Measured|Calculated', case=False, na=False) | h.hi_meas.isna()
    return pd.DataFrame(dict(op=h.op_time.fillna(0).astype(float).values,
                             P=h.gross_load_MW.fillna(0).astype(float).values,
                             HI=h.heat_input_MMBtu.astype(float).values,
                             meas=meas.values), index=idx)


@functools.lru_cache(maxsize=4)
def prices(start='2018-01-01', end='2026-01-01'):
    """Hourly ERCOT day-ahead hub-average price ($/MWh) and Henry Hub daily gas ($/MMBtu) on the hourly grid."""
    p = gd.prices('ERCOT', 'HB_HUBAVG', start, end, kind='da')['price']
    idx = pd.date_range(pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC'), freq='h', inclusive='left')
    p = p[~p.index.duplicated()].reindex(idx).interpolate(limit=6)
    hh = henry_hub_daily()
    g = hh.reindex(pd.date_range(idx[0].tz_localize(None).normalize(), idx[-1].tz_localize(None).normalize(),
                                 freq='D')).ffill().bfill()
    gas = g.reindex(idx.tz_localize(None).normalize()).values
    return pd.DataFrame(dict(price=p.values, gas=gas), index=idx)
