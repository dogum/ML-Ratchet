"""Frozen case lists: simulated validation (val), dev gate (dev) and the exam (exam_recs, exam_hue, exam_wpuq,
exam_bdg2). Every list is drawn once with a fixed seed. A case = dict(spec_nom, spec_true, wsrc, year, month,
tariff, pref, res, seed, id, group).

Exam twins: the controller knows the twin built from the unit's real metadata (spec_nom); the simulated house is a
perturbed copy (spec_true), so neither MPC nor ThermoPilot gets the house exactly.
"""
import functools, os
import numpy as np
import pandas as pd
import buildingdata as bd
import buildingsim as bs
from buildingsim import stock
from . import twins, env

DRAW_SEED = 20261117


def val_cases(n=60):
    out = []
    s = 900000
    while len(out) < n:
        c = twins.train_case(s)
        s += 1
        if c is None:
            continue
        c['id'] = f'val{c["seed"]}'; c['group'] = 'val'
        out.append(c)
    return out


def _recs_cases(split, per_slot, months, seed):
    x = bd.recs(split=split)
    x = x[x.system.isin(['split_ac_furnace', 'heat_pump_air', 'baseboard_resistance'])]
    r = np.random.default_rng(seed)
    out = []
    for st, (site, tariffs) in twins.STATE_SITE.items():
        pool = x[x.state == st]
        for tar in tariffs:
            order = r.permutation(len(pool))
            got = 0
            for i in order:
                if got >= per_slot:
                    break
                row = pool.iloc[int(i)]
                tw = twins.recs_twin(row, seed=int(r.integers(1 << 30)))
                if tw is None:
                    continue
                spec, pref = tw
                true = twins.perturb(spec, int(r.integers(1 << 30)))
                for mo in months:
                    out.append(dict(spec_nom=spec, spec_true=true, wsrc=f'site:{site}', year=2017, month=mo, tariff=tar,
                                    pref=pref, res=True, seed=int(r.integers(1 << 30)), id=f'{row["id"]}_{tar}_{mo}',
                                    unit=str(row['id']), group=f'recs_{split}', state=st))
                got += 1
    return out


def dev_cases():
    """Dev gate: 1 RECS dev respondent per (state, tariff) slot x Jan/Jul (2017 weather) + 4 BDG2 dev small
    buildings x Jan/Jul."""
    out = _recs_cases('dev', 1, [1, 7], DRAW_SEED + 1)
    out += _bdg2_cases('dev', 1, [1, 7], DRAW_SEED + 2)
    return out


def exam_recs():
    return _recs_cases('exam', 6, [1, 7], DRAW_SEED)


# ------------------------------------------------------------------------------------------------ BDG2
def bdg2_twin(bid):
    s = bd.bdg2_spec(bid)
    s = {k: v for k, v in s.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    meters = str(s.pop('meters', ''))
    s['system'] = 'packaged_rtu' if 'gas' in meters else 'heat_pump_air'
    s['hvac'] = {}
    if 'n_floors' not in s and s.get('floor_area_m2'):
        s['n_floors'] = max(1, int(round(np.sqrt(s['floor_area_m2']) / 30)))
    s['pv'] = None; s['battery'] = None
    return s


def _bdg2_cases(split, per_group, months, seed):
    m = bd.bdg2_meta(split=split)
    ms = m['meters'].astype(str)
    ok = (m.floor_area_m2 < 4000) & (m.floor_area_m2 > 150) & ms.str.contains('electricity') & \
         ~ms.str.contains('chilled|steam|hot') & m.site_id.isin(list(twins.SITE_COM_TARIFF)) & \
         m.primary_use.isin(['office', 'retail', 'education', 'assembly', 'other', 'lodging', 'food'])
    m = m[ok]
    groups = {'CA': ['Bear', 'Swan'], 'FL': ['Panther', 'Gator'], 'MN': ['Hog'], 'DC': ['Rat']}
    r = np.random.default_rng(seed)
    out = []
    for g, sites in groups.items():
        pool = m[m.site_id.isin(sites)]
        if len(pool) == 0:
            continue
        order = r.permutation(len(pool))[:per_group]
        for i in order:
            b = pool.iloc[int(i)]
            spec = bdg2_twin(b['id'])
            true = twins.perturb(spec, int(r.integers(1 << 30)))
            lo = float(r.uniform(20.5, 21.5))
            pref = dict(lo=lo, hi=lo + 2.5, w_sleep_h=0.0, w_away_h=4.0, w_sleep_c=0.0, w_away_c=4.0)
            tar = twins.SITE_COM_TARIFF[b['site_id']]
            for mo in months:
                out.append(dict(spec_nom=spec, spec_true=true, wsrc=f'site:{b["site_id"]}', year=2017, month=mo,
                                tariff=tar, pref=pref, res=False, seed=int(r.integers(1 << 30)), id=f'{b["id"]}_{mo}',
                                unit=b['id'], group=f'bdg2_{split}'))
    return out


def exam_bdg2():
    return _bdg2_cases('exam', 4, [1, 7], DRAW_SEED + 3)


# ------------------------------------------------------------------------------------------------ HUE, WPuQ
def _rs_match(bt, cz_list, heat, cool, seed):
    rs = twins._rs()
    m = rs['in.geometry_building_type_recs'].astype(str).isin(bt)
    m &= rs['in.ashrae_iecc_climate_zone_2004'].astype(str).isin(cz_list)
    m &= rs['in.hvac_heating_type_and_fuel'].astype(str).isin(heat)
    ct = rs['in.hvac_cooling_type'].astype(str)
    if cool is not None:
        m &= (ct != 'None') if cool else (ct == 'None')
    m &= rs['in.vacancy_status'].astype(str) != 'Vacant'
    cand = rs[m.values]
    r = np.random.default_rng(seed)
    return cand.iloc[int(r.integers(len(cand)))] if len(cand) else None


HUE_BT = {'bungalow': ['Single-Family Detached'], 'character': ['Single-Family Detached'],
          'modern': ['Single-Family Detached'], 'special': ['Single-Family Detached'],
          'duplex': ['Single-Family Attached'], 'laneway': ['Single-Family Detached'],
          'apartment': ['Multi-Family with 5+ Units']}
HUE_HEAT = {'heat_pump_air': ['Electricity ASHP'], 'split_ac_furnace': ['Natural Gas Fuel Furnace'],
            'baseboard_resistance': ['Electricity Baseboard']}


def exam_hue():
    h = pd.read_parquet(os.path.join(env.PROC, 'other', 'hue_houses.parquet'))
    h = h[h.system.isin(list(HUE_HEAT)) & h.HouseType.notna()]
    r = np.random.default_rng(DRAW_SEED + 4)
    out = []
    for _, b in h.iterrows():
        cool = bool(b['has_ac']) or b['system'] == 'heat_pump_air'
        row = _rs_match(HUE_BT.get(b['HouseType'], ['Single-Family Detached']), ['4C', '5B', '4A'],
                        HUE_HEAT[b['system']], cool, int(r.integers(1 << 30)))
        if row is None:
            continue
        spec = stock.resstock_to_spec(row, whole_building=False)
        if spec['system'] not in twins.SUPPORTED_SYSTEMS:
            continue
        spec['geometry']['orientation_deg'] = {'North': 0, 'NNW': 340, 'NW': 315, 'NE': 45, 'ENE': 67, 'East': 90,
                                               'SE': 135, 'South': 180, 'SW': 225, 'West': 270}.get(b['Facing'], 180) + 0.0
        spec['pv'] = None; spec['id'] = f'HUE_{b["house"]}'
        spec['provenance'] = dict(source='HUE house twin (type, facing, heating/cooling system; ResStock 4C match)')
        true = twins.perturb(spec, int(r.integers(1 << 30)))
        pref = dict(lo=20.5, hi=24.5, w_sleep_h=2.0, w_away_h=2.0, w_sleep_c=0.0, w_away_c=2.0)
        months = [1, 7] if cool else [1]
        for mo in months:
            out.append(dict(spec_nom=spec, spec_true=true, wsrc='hue', year=2017, month=mo, tariff='pge_e_tou_c',
                            pref=pref, res=True, seed=int(r.integers(1 << 30)), id=f'HUE_{b["house"]}_{mo}',
                            unit=f'HUE_{b["house"]}', group='hue'))
    return out


def wpuq_homes():
    w = pd.read_parquet(os.path.join(env.PROC, 'other', 'wpuq_60min.parquet'), columns=['path'])
    p = sorted(set(str(x) for x in w['path'].unique() if str(x).endswith('HEATPUMP')))
    return p


def wpuq_twin(path, seed):
    row = _rs_match(['Single-Family Detached'], ['5A', '4A', '5B'], ['Electricity ASHP'], None, seed)
    spec = stock.resstock_to_spec(row, whole_building=False)
    spec['hvac'] = dict(spec.get('hvac') or {}, cooling=False)
    spec['pv'] = None
    spec['id'] = 'WPuQ_' + path.split('/')[1]
    spec['provenance'] = dict(source='WPuQ heat-pump home twin (generic ResStock 5A ASHP house; no metadata)')
    return spec


def exam_wpuq():
    r = np.random.default_rng(DRAW_SEED + 5)
    out = []
    for p in wpuq_homes():
        spec = wpuq_twin(p, int(r.integers(1 << 30)))
        true = twins.perturb(spec, int(r.integers(1 << 30)))
        pref = dict(lo=20.5, hi=26.0, w_sleep_h=2.0, w_away_h=2.0, w_sleep_c=0.0, w_away_c=0.0)
        out.append(dict(spec_nom=spec, spec_true=true, wsrc='wpuq', year=2019, month=1, tariff='xcel_mn_a04_tod',
                        pref=pref, res=True, seed=int(r.integers(1 << 30)), id=spec['id'] + '_1', unit=spec['id'],
                        group='wpuq'))
    return out


SETS = dict(val=val_cases, dev=dev_cases, exam_recs=exam_recs, exam_bdg2=exam_bdg2, exam_hue=exam_hue,
            exam_wpuq=exam_wpuq)
