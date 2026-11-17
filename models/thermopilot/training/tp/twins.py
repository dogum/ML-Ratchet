"""Buildings for ThermoPilot: generated training buildings, RECS respondent twins, HUE / WPuQ / BDG2 twins, and the
'true vs nominal' perturbation (the controller knows the nominal spec; the simulated house is a perturbed copy).

Rules (PHASE2_BRIEF): training may use real metadata only from RECS prior + dev respondents and BDG2 dev buildings.
RECS exam respondents, HUE, WPuQ and BDG2 exam buildings are used only for the final exam.
"""
import copy, functools
import numpy as np
import pandas as pd
from . import env
import buildingsim as bs
import buildingdata as bd
from buildingsim import stock
from buildingsim.generator import _load

SUPPORTED_SYSTEMS = ('split_ac_furnace', 'heat_pump_air', 'baseboard_resistance', 'packaged_rtu', 'ptac')

# RECS housing type -> ResStock building type
BT = {'single_detached': 'Single-Family Detached', 'single_attached': 'Single-Family Attached',
      'apartment_2_4': 'Multi-Family with 2 - 4 Units', 'apartment_5plus': 'Multi-Family with 5+ Units',
      'mobile_home': 'Mobile Home'}
VINT = {'<1940': 1930, '1940s': 1945, '1950s': 1955, '1960s': 1965, '1970s': 1975, '1980s': 1985, '1990s': 1995,
        '2000s': 2005, '2010s': 2015}
# RECS states used in the exam -> (BDG2 site whose real weather is used, list of real tariffs of that state)
STATE_SITE = {'CA': ('Bear', ['pge_e_tou_c']), 'AZ': ('Fox', ['aps_r3', 'srp_e26']), 'TX': ('Bull', ['austin_res_tou', 'rtp_ercot']),
              'FL': ('Panther', ['duke_fl_rt']), 'UT': ('Bobcat', ['rmp_ut_sched1_tou']),
              'CO': ('Bobcat', ['xcel_co_re_tou']), 'MN': ('Hog', ['xcel_mn_a04_tod']),
              'NY': ('Cockatoo', ['nyseg_sc12_tou', 'coned_sc1_rate3']), 'NJ': ('Eagle', ['pseg_rs_tou_3p'])}
# commercial tariffs by BDG2 site
SITE_COM_TARIFF = {'Bear': 'pge_b19_sec', 'Swan': 'pge_b19_sec', 'Panther': 'ouc_gsd_tou', 'Gator': 'ouc_gsd_tou',
                   'Hog': 'xcel_mn_a15_tod', 'Rat': 'pepco_gtlv'}


@functools.lru_cache(maxsize=1)
def _rs():
    return _load('resstock_prior.parquet')


def _heat_ok(rs_heat, recs_row):
    sysm = recs_row['system']
    fuel = str(recs_row.get('heating_fuel'))
    if sysm == 'heat_pump_air':
        return rs_heat == 'Electricity ASHP'
    if sysm == 'baseboard_resistance':
        return rs_heat == 'Electricity Baseboard'
    if sysm == 'split_ac_furnace':
        if fuel == 'electricity':
            return rs_heat == 'Electricity Electric Furnace'
        if fuel == 'propane':
            return rs_heat == 'Propane Fuel Furnace'
        if fuel == 'fuel_oil':
            return rs_heat == 'Fuel Oil Fuel Furnace'
        return rs_heat == 'Natural Gas Fuel Furnace'
    return False


def recs_twin(row, seed=0):
    """Twin of a RECS respondent: the nearest ResStock prior dwelling with the same building type, climate zone,
    heating system/fuel and cooling presence (by floor area and vintage), with the respondent's own floor area,
    year built, stories, occupants and thermostat settings. Returns (spec, pref) or None."""
    rs = _rs()
    if row['system'] not in SUPPORTED_SYSTEMS or row['housing_type'] not in BT:
        return None
    m = rs['in.geometry_building_type_recs'].astype(str) == BT[row['housing_type']]
    cz = str(row['climate_zone'])
    mcz = rs['in.ashrae_iecc_climate_zone_2004'].astype(str) == cz
    if (m & mcz).sum() > 30:
        m &= mcz
    heat = rs['in.hvac_heating_type_and_fuel'].astype(str)
    m &= heat.map(lambda h: _heat_ok(h, row))
    has_cool = str(row['cooling_equipment']) not in ('none', 'nan', 'None', 'evaporative')
    ct = rs['in.hvac_cooling_type'].astype(str)
    m &= (ct != 'None') if has_cool else (ct == 'None') | (row['system'] == 'heat_pump_air')
    m &= rs['in.vacancy_status'].astype(str) != 'Vacant'
    cand = rs[m.values]
    if len(cand) == 0:
        return None
    area = float(row['heated_area_m2'] if row.get('heated_area_m2') and row['heated_area_m2'] > 10 else row['floor_area_m2'])
    yb = 0.5 * (float(row['year_built_lo'] or 1970) + float(row['year_built_hi'] or 1990))
    a = cand['in.sqft..ft2'].astype(float).values * stock.FT2
    v = cand['in.vintage'].astype(str).map(VINT).fillna(1980).values
    score = np.abs(np.log(np.maximum(a, 10) / area)) + np.abs(v - yb) / 30.0
    top = np.argsort(score)[:5]
    r = np.random.default_rng(seed)
    pick = cand.iloc[int(top[r.integers(len(top))])]
    spec = stock.resstock_to_spec(pick, whole_building=False)
    if spec['system'] not in SUPPORTED_SYSTEMS:
        return None
    spec['floor_area_m2'] = area
    spec['year_built'] = int(yb)
    if row.get('n_floors') and row['n_floors'] == row['n_floors']:
        spec['n_floors'] = int(min(max(row['n_floors'], 1), 3)) if not row['housing_type'].startswith('apartment') else 1
    ig = spec.setdefault('internal_gains', {})
    if row.get('n_occupants') and row['n_occupants'] == row['n_occupants']:
        ig['occupants'] = float(row['n_occupants'])
        ig['plug'] = None
    spec['hvac'] = dict(spec.get('hvac') or {}, cooling=bool(has_cool or row['system'] == 'heat_pump_air'))
    spec['id'] = str(row['id'])
    spec['provenance'] = dict(source='RECS 2020 respondent twin (ResStock prior nearest match)', recs_id=str(row['id']),
                              resstock_bldg_id=int(pick['bldg_id']))
    spec.pop('pv', None); spec['pv'] = None
    return spec, recs_pref(row)


def _f(x, d):
    try:
        x = float(x)
        return d if not np.isfinite(x) or x < 5 or x > 35 else x
    except (TypeError, ValueError):
        return d


def recs_pref(row):
    """Comfort band from the respondent's own thermostat settings (home / night / away, winter and summer)."""
    lo = _f(row.get('T_heat_home_C'), 20.5)
    hi = _f(row.get('T_cool_home_C'), lo + 3.0)
    lo = float(np.clip(lo, 17.0, 23.5)); hi = float(np.clip(max(hi, lo + 1.5), lo + 1.5, 28.0))
    wsh = float(np.clip(lo - _f(row.get('T_heat_night_C'), lo), 0, 6))
    wah = float(np.clip(lo - _f(row.get('T_heat_away_C'), lo), 0, 6))
    wsc = float(np.clip(_f(row.get('T_cool_night_C'), hi) - hi, 0, 6))
    wac = float(np.clip(_f(row.get('T_cool_away_C'), hi) - hi, 0, 6))
    return dict(lo=lo, hi=hi, w_sleep_h=wsh, w_away_h=wah, w_sleep_c=wsc, w_away_c=wac)


def perturb(spec, seed, scale=1.0):
    """The 'true' house that the controller does not know exactly: envelope U x LN(0, 0.12), air leakage
    x LN(0, 0.25), thermal mass class shifted with p 0.3, plug loads x LN(0, 0.2), occupants +-1, sizing redrawn."""
    r = np.random.default_rng(seed)
    s = copy.deepcopy(spec)
    for k in ('wall_U', 'roof_U', 'window_U'):
        if s.get(k) is not None:
            s[k] = float(s[k]) * float(np.exp(r.normal(0, 0.12 * scale)))
    for k in ('ach50', 'infiltration_ach'):
        if s.get(k) is not None:
            s[k] = float(s[k]) * float(np.exp(r.normal(0, 0.25 * scale)))
    tm = s.get('thermal_mass')
    if isinstance(tm, str) and r.random() < 0.3 * scale:
        order = ['light', 'medium', 'heavy']
        i = order.index(tm) if tm in order else 1
        s['thermal_mass'] = order[int(np.clip(i + r.choice([-1, 1]), 0, 2))]
    elif isinstance(tm, (int, float)):
        s['thermal_mass'] = float(tm) * float(np.exp(r.normal(0, 0.25 * scale)))
    ig = s.setdefault('internal_gains', {})
    if ig.get('plug'):
        ig['plug'] = float(ig['plug']) * float(np.exp(r.normal(0, 0.2 * scale)))
    h = s.setdefault('hvac', {})
    for k in ('sizing_factor_heat', 'sizing_factor_cool'):
        if h.get(k):
            h[k] = float(h[k]) * float(np.exp(r.normal(0, 0.08 * scale)))
    return s


# ------------------------------------------------------------------------------------------- training cases
TRAIN_SITES = ['Bear', 'Fox', 'Bull', 'Panther', 'Bobcat', 'Hog', 'Cockatoo', 'Eagle', 'Rat', 'Swan', 'Gator']
EPW_CZ = {'Fairbanks': '8', 'Phoenix': '2B', 'Los.Angeles': '3B', 'San.Francisco': '3C', 'Boulder': '5B',
          'Denver': '5B', 'Miami': '1A', 'Atlanta': '3A', 'Chicago': '5A', 'Baltimore': '4A', 'Duluth': '7',
          'Minneapolis': '6A', 'Helena': '6B', 'New.York': '4A', 'El.Paso': '3B', 'Houston': '2A', 'Seattle': '4C'}
MONTH_P = np.array([2, 1.5, 1, 0.6, 0.6, 1, 2, 2, 1, 0.6, 1, 1.6], float)


def train_pref(r, res):
    if res:
        lo = float(r.uniform(19.0, 22.5)); hi = lo + float(r.uniform(1.5, 4.5))
        return dict(lo=lo, hi=hi, w_sleep_h=float(r.choice([0, 1, 2, 3])), w_away_h=float(r.choice([0, 2, 3, 5])),
                    w_sleep_c=float(r.choice([0, 0, 1, 2])), w_away_c=float(r.choice([0, 2, 3, 4])))
    lo = float(r.uniform(20.0, 22.0)); hi = lo + float(r.uniform(1.5, 3.5))
    w = float(r.uniform(2, 6))
    return dict(lo=lo, hi=hi, w_sleep_h=0.0, w_away_h=w, w_sleep_c=0.0, w_away_c=w)


@functools.lru_cache(maxsize=1)
def _recs_train():
    x = bd.recs(split='prior')        # dev respondents are kept for the dev gate
    return x[x.system.isin(SUPPORTED_SYSTEMS)].reset_index(drop=True)


def train_case(seed):
    """One random training episode: (spec_nom, spec_true, weather source, year, month, tariff, pref, kind)."""
    r = np.random.default_rng(1000003 * seed + 17)
    month = int(r.choice(np.arange(1, 13), p=MONTH_P / MONTH_P.sum()))
    if r.random() < 0.6:
        site = TRAIN_SITES[r.integers(len(TRAIN_SITES))]
        wsrc, cz = f'site:{site}', env.SITE_CZ.get(site, '4A')
    else:
        city = list(EPW_CZ)[r.integers(len(EPW_CZ))]
        wsrc, cz = f'epw:{city}', EPW_CZ[city]
    year = 2016
    u = r.random()
    pref = None
    if u < 0.4:
        recs = _recs_train()
        same = recs[recs.climate_zone.astype(str).str[0] == cz[0]]
        pool = same if len(same) > 20 else recs
        for _ in range(20):
            row = pool.iloc[int(r.integers(len(pool)))]
            tw = recs_twin(row, seed=int(r.integers(1 << 30)))
            if tw is not None:
                spec, pref = tw
                break
        else:
            spec = None
        res = True
    if u >= 0.4 or pref is None:
        res = u < 0.78
        if res:
            cands = bs.generate_buildings(1, seed=int(r.integers(1 << 30)), primary_use='residential_single',
                                          climate_zone=cz if cz in ('1A', '2A', '2B', '3A', '3B', '3C', '4A', '4C', '5A', '5B', '6A', '6B', '7', '8') else None,
                                          system=['split_ac_furnace', 'heat_pump_air', 'baseboard_resistance'])
        else:
            cands = bs.generate_buildings(1, seed=int(r.integers(1 << 30)), primary_use=['office', 'retail', 'other', 'assembly', 'education'],
                                          system=['packaged_rtu', 'heat_pump_air', 'ptac', 'split_ac_furnace'])
            cands = [c for c in cands if c['floor_area_m2'] < 4000]
        if not cands:
            return None
        spec = cands[0]
        pref = train_pref(r, res)
        if r.random() < 0.3:
            pref = dict(pref, w_sleep_h=0.0, w_away_h=0.0, w_sleep_c=0.0, w_away_c=0.0)
    spec['pv'] = None
    true = perturb(spec, int(r.integers(1 << 30)))
    # tariff: real (same sector), synthetic, or real-time wholesale
    v = r.random()
    if res:
        tariff = (env.RES_TARIFFS[r.integers(len(env.RES_TARIFFS))] if v < 0.55 else
                  env.synth_tariff(int(r.integers(1 << 30))) if v < 0.88 else 'rtp_ercot')
    else:
        tariff = (env.COM_TARIFFS[r.integers(len(env.COM_TARIFFS))] if v < 0.6 else
                  env.synth_tariff(int(r.integers(1 << 30)), demand=True))
    return dict(spec_nom=spec, spec_true=true, wsrc=wsrc, year=year, month=month, tariff=tariff, pref=pref,
                res=res, seed=int(seed))
