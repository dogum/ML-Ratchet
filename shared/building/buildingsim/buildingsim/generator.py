"""Random but physically consistent buildings of every use type and climate.

Each building bootstraps one real combination of characteristics from the priors, then jitters continuous values
so no two generated buildings are identical:
- residential_single / residential_multi: NREL ResStock 2025 R1 baseline (120,000 dwelling models drawn with the
  stock weights from agent B's whole-stock metadata); multi-family units become whole buildings
  (unit area x units). Envelope R-values, windows, ACH50, setpoints and offsets, heating/cooling type and
  efficiency, water heater, lighting technology, plug-load level, PV come from the row.
- office, education, lodging, retail, healthcare, warehouse, food: NREL ComStock 2025 R3 baseline (149,272 models):
  type, size, stories, aspect ratio, rotation, vintage and energy codes (envelope level), wall construction, window
  type, WWR category, airtightness, HVAC system type, heating fuel, operating hours, thermostat setpoints and
  setbacks, lighting generation.
- assembly, lab, other: CBECS 2018 respondents of the 'prior' split (real survey answers: size, floors, vintage
  bin, WWR, window type, system, heating fuel, weekly hours), use-type defaults for the rest.
Physical consistency: envelope U-values follow the vintage/code and climate zone of the row; HVAC capacities are
autosized from each building's own design loads at simulation time (sizing factor jittered 1.1-1.5); fans and
pumps scale with capacity; the schedule drives gains, setpoints and ventilation together.

Jitter (multiplicative log-normal unless stated): U-values sd 10 %, infiltration 20 %, lighting and plug 15 %,
people density 20 %, DHW volume 25 %, WWR within its category, setpoints +-0.5 K, operating hours +-0.5 h,
heating/cooling sizing factors uniform 1.1-1.5 / 1.0-1.3.
"""
import copy
import os

import numpy as np
import pandas as pd

from .spec import USE_TYPES, climate_zone_number
from .stock import resstock_to_spec, comstock_to_spec, cbecs_to_spec, COM_USE
from .weather import EPW_CLIMATE_ZONE

PRI = os.path.join(os.environ.get('ML_RATCHET_BUILDING') or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'buildingsim', 'refdata', 'priors')   # data root: $ML_RATCHET_BUILDING or shared/building
_CACHE = {}


def _load(name):
    if name not in _CACHE:
        _CACHE[name] = pd.read_parquet(os.path.join(PRI, name))
    return _CACHE[name]


def _cz_ok(val, cz):
    if cz is None:
        return True
    v = str(val)
    return v == str(cz) or (len(str(cz)) == 1 and v.startswith(str(cz)))


def weather_hint(cz):
    """Name of a bundled TMYx city in (or nearest to) climate zone cz."""
    best, bd = None, 99
    zn = climate_zone_number(cz)
    for city, z in EPW_CLIMATE_ZONE.items():
        d = abs(climate_zone_number(z) - zn) + (0 if str(z)[-1:] == str(cz)[-1:] else 0.5)
        if d < bd:
            best, bd = city, d
    return best


def _jit(rng, x, sd):
    return float(x) * float(np.exp(rng.normal(0, sd))) if x is not None else None


def jitter(spec, rng):
    s = copy.deepcopy(spec)
    for k in ('wall_U', 'roof_U', 'window_U'):
        if s.get(k) is not None:
            s[k] = _jit(rng, s[k], 0.10)
    if s.get('ach50') is not None:
        s['ach50'] = _jit(rng, s['ach50'], 0.20)
    if s.get('infiltration_ach') is not None:
        s['infiltration_ach'] = _jit(rng, s['infiltration_ach'], 0.20)
    ig = s.setdefault('internal_gains', {})
    from .spec import USE_DEFAULTS
    U = USE_DEFAULTS.get(s['primary_use'], USE_DEFAULTS['other'])
    ig['lighting'] = _jit(rng, ig.get('lighting', U['lighting']), 0.15)
    ig['plug'] = _jit(rng, ig.get('plug', U['plug']), 0.15)
    if not s['primary_use'].startswith('residential'):
        ig['people_W_m2'] = _jit(rng, (U['people_sens'] + U['people_lat']) / U['people_m2'], 0.20)
    sp = s.setdefault('setpoints', {})
    for k in ('heat', 'cool'):
        if sp.get(k) is not None:
            sp[k] = float(sp[k] + rng.uniform(-0.5, 0.5))
    sch = s.get('occupancy_schedule')
    if isinstance(sch, dict) and sch.get('type') == 'commercial':
        for k in ('weekday', 'saturday', 'sunday'):
            if sch.get(k):
                o, c = sch[k]
                sh = rng.uniform(-0.5, 0.5)
                sch[k] = (float(o + sh), float(c + sh + rng.uniform(-0.5, 0.5)))
    h = s.setdefault('hvac', {})
    h['sizing_factor_heat'] = float(rng.uniform(1.1, 1.5))
    h['sizing_factor_cool'] = float(rng.uniform(1.0, 1.3))
    d = s.get('dhw')
    if isinstance(d, dict) and d.get('L_per_day') is not None:
        d['L_per_day'] = _jit(rng, d['L_per_day'], 0.25)
    s['dhw_scale'] = float(np.exp(rng.normal(0, 0.25)))
    return s


def generate_buildings(n, seed=0, primary_use=None, climate_zone=None, system=None, mix='balanced',
                       jitter_values=True):
    """n random building specs. primary_use / climate_zone / system: a value or list to restrict to.
    mix='balanced' draws use types uniformly; mix='stock' follows ResStock/ComStock/CBECS weights by count
    (dominated by houses)."""
    rng = np.random.default_rng(seed)
    uses = [primary_use] if isinstance(primary_use, str) else list(primary_use or USE_TYPES)
    czs = [climate_zone] if isinstance(climate_zone, str) else (list(climate_zone) if climate_zone else None)
    systems = [system] if isinstance(system, str) else (list(system) if system else None)
    rs, cs, cb = _load('resstock_prior.parquet'), _load('comstock_prior.parquet'), _load('cbecs_prior.parquet')
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        use = uses[rng.integers(len(uses))] if mix == 'balanced' else _stock_use(rng, uses)
        cz = czs[rng.integers(len(czs))] if czs else None
        if use.startswith('residential'):
            mf = use == 'residential_multi'
            bt = rs['in.geometry_building_type_recs'].astype(str)
            m = bt.str.startswith('Multi-Family') if mf else ~bt.str.startswith('Multi-Family')
            if cz:
                m &= rs['in.ashrae_iecc_climate_zone_2004'].astype(str).map(lambda v: _cz_ok(v, cz))
            m &= rs['in.vacancy_status'].astype(str) != 'Vacant'
            cand = np.flatnonzero(m.values)
            if len(cand) == 0:
                continue
            row = rs.iloc[cand[rng.integers(len(cand))]]
            spec = resstock_to_spec(row, whole_building=True)
        elif use in ('assembly', 'lab', 'other'):
            cand = np.flatnonzero((cb['primary_use'] == use).values)
            if len(cand) == 0:
                continue
            row = cb.iloc[cand[rng.integers(len(cand))]]
            spec = cbecs_to_spec(row, rng)
            spec['climate_zone'] = cz or _random_cz(rng)
        else:
            types = [t for t, u in COM_USE.items() if u == use]
            m = cs['in.comstock_building_type'].astype(str).isin(types)
            if cz:
                m &= cs['in.as_simulated_ashrae_iecc_climate_zone_2006'].astype(str).map(lambda v: _cz_ok(v, cz))
            cand = np.flatnonzero(m.values)
            if len(cand) == 0:
                continue
            row = cs.iloc[cand[rng.integers(len(cand))]]
            spec = comstock_to_spec(row, rng)
        if systems and spec['system'] not in systems:
            continue
        if jitter_values:
            spec = jitter(spec, rng)
        spec['id'] = f'gen{seed}_{len(out):05d}'
        spec['weather_hint'] = weather_hint(spec.get('climate_zone') or '4A')
        out.append(spec)
    return out


def _random_cz(rng):
    """Climate zone drawn from the ComStock stock (weighted), so CBECS-based types see a realistic climate mix."""
    cs = _load('comstock_prior.parquet')
    z = cs['in.as_simulated_ashrae_iecc_climate_zone_2006'].astype(str).values
    w = cs['weight'].values.astype(float)
    return str(z[rng.choice(len(z), p=w / w.sum())])


def _stock_use(rng, uses):
    w = {'residential_single': 0.55, 'residential_multi': 0.25}
    rest = [u for u in uses if u not in w]
    p = np.array([w.get(u, 0.2 / max(len(rest), 1)) for u in uses], float)
    return uses[rng.choice(len(uses), p=p / p.sum())]


def labels(spec, out):
    """Ground-truth labels for BuildingDetective from a spec and its simulation result.
    UA_network_W_K: what an air-temperature-controlled meter sees (5R1C: air exchange + the series of the interior
    coupling with windows + opaque; multi-node: sum of U A + air exchange). balance_point_heat_C: occupied heating
    setpoint minus mean internal + solar gains over UA_network (a degree-day style label)."""
    import numpy as np
    s = out['spec']
    S = out['summary']
    sim = out['sim']
    z = sim.zone
    H_ve = float(np.mean(np.asarray(sim.r_inf) * 1.2 * 1020.0)) + float(np.mean(sim.m_v)) * 1020.0
    if getattr(z, 'kind', '') == 'multinode':
        UA_net = z.UA + H_ve
    else:
        UA_net = H_ve + 1.0 / (1.0 / z.H_tr_is + 1.0 / (z.H_tr_w + z.H_op))
    gains = float(np.mean(sim.phi_int) + np.mean(sim.phi_sol))
    hv = s['hvac']
    return dict(UA_envelope_W_K=z.UA, UA_network_W_K=UA_net, H_air_exchange_W_K=H_ve, C_mass_J_K=z.C_m,
                tau_h=S['tau_h'], balance_point_heat_C=s['setpoints']['heat'] - gains / UA_net,
                mean_gains_W=gains, floor_area_m2=s['floor_area_m2'], primary_use=s['primary_use'],
                system=s['system'], heat_source=hv.get('heat_source'), fuel=hv.get('fuel'),
                cooling=hv.get('cooling'), year_built=s['year_built'], heat_sp=s['setpoints']['heat'],
                cool_sp=s['setpoints']['cool'], setback_heat=s['setpoints'].get('setback_heat'),
                window_wwr=s['window_wwr'], occupancy_schedule=s['occupancy_schedule'],
                climate_zone=s.get('climate_zone'), sizing=S['sizing'])
