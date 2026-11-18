"""Phase 2 wrappers, upstreamed from the four building models (HANDOFF section 4, P0.4). All opt-in.

Each function is a port of the model's own v1.0 training code (named in its docstring); nothing here is called by
`simulate()`. Importing this module changes nothing. The two flags that act inside the simulator are elsewhere:
`simulate(..., options={'holidays': 'auto'})` (LoadCast's country calendars) and the hvac field `district_eff_chw`
(BuildingDetective W2, a chilled-water distribution factor apart from heat's `district_eff`).
`checks/check_phase2.py` shows each port reproduces the model's wrapper (checks/out/phase2_equivalence.json).

LoadCast        spec_from_meters, CAMPUS_KNOBS, scale_meters, DEV_METER_SCALE
Detective       apply_intensity + W1_INTENSITY, make_campus + W2_CAMPUS, flat_base_loads + W3_BASE, base_share,
                clean_labels
FaultFinder     log_commands, damper_command / installed_damper, static_offsets, OperatorJitter
(ThermoPilot's RECS twins are stock.recs_to_spec / stock.recs_comfort_band; its tariff and RTP builders are in
 buildingdata.)
"""
import copy

import numpy as np
import pandas as pd

# ================================================================================================ LoadCast
# Source: models/loadcast/training/lcsim.py (make_spec, _sched, _clean_hvac, post) and calib_summary.json.

#: Modal knobs of LoadCast's 52 BDG2 dev fits (calib.py grid, best CV(RMSE) per building, local 2016):
#: schedule 24x7 in 22 of 52 (default 21, extended 9), outdoor air x2.5 in 37 of 52, VAV minimum 0.3 with supply-air
#: reset in 43 of 52 (0.8 without reset in 9). 'sys' stays 'auto': the meters decide the plant.
CAMPUS_KNOBS = dict(sys='auto', sched='24x7', oa=2.5, vav_min=0.3, sat_reset=True)

#: Median dev fit of y_real = a * y_sim + b * A per BDG2 meter (LoadCast calib_summary.json; b in W/m2 of floor).
#: n buildings: electricity 47, chilled water 20, steam 10, hot water 7, gas 2.
DEV_METER_SCALE = {'electricity': (0.42875098395029027, 6.2807771278765),
                   'chilledwater': (0.9160066568085539, 31.448604055625992),
                   'steam': (0.3608859677040277, 3.481664727914508),
                   'hotwater': (0.40380687911969143, 5.9568913872569),
                   'gas': (0.0, 61.89728601556404)}


def _sched(use, kind):
    from .schedules import DEFAULT_SCHEDULES
    s = copy.deepcopy(DEFAULT_SCHEDULES.get(use, DEFAULT_SCHEDULES['other']))
    if kind == 'default' or s.get('type') != 'commercial':
        return s          # 24x7 and lodging types already run their air handlers around the clock
    if kind == 'extended':
        o, c = s['weekday']
        s['weekday'] = (max(o - 1.5, 4.0), min(c + 4.0, 23.5))
        s['saturday'] = s.get('saturday') or (9.0, 17.0)
        s['sunday'] = s.get('sunday') or (11.0, 17.0)
        s['light_base'] = s.get('light_base', 0.15) + 0.1
        s['plug_base'] = s.get('plug_base', 0.4) + 0.1
        return s
    o, c = s['weekday'] if s.get('weekday') else (8, 18)
    return dict(type='24x7', day=(o, c), night_occ=0.1, night_light=min(s.get('light_base', 0.15) + 0.25, 0.8),
                night_plug=min(s.get('plug_base', 0.4) + 0.15, 0.9), weekend=0.6)


def _clean_hvac(h):
    h = dict(h or {})
    for k in ('heat_source', 'cooling_source', 'reheat', 'district_heat', 'fuel', 'wshp_tower', 'central_ac', 'room_ac',
              'heating_capacity_W', 'cooling_capacity_W', 'backup_capacity_W'):
        h.pop(k, None)
    return h


def spec_from_meters(base, meters, knobs=None):
    """LoadCast `lcsim.make_spec`: a BDG2-style building (use, area, year, climate zone, which meters exist) plus
    calibration knobs -> a spec. The meters a building has decide its plant: chilled water + steam/hot water ->
    district_steam_chw; steam or hot water only -> VAV with district heat; chilled water only -> district chilled water
    (+ gas DHW if a gas meter); gas only -> gas RTU or VAV; electricity only -> knob 'sys'.
    knobs (all optional; {} = Phase 1 generic twin): sys 'auto'|'district'|'electric'|'gas_rtu'|'gas_vav'|'keep';
    sched 'default'|'extended'|'24x7'; oa (outdoor-air multiplier on 62.1 rates; >= 2 turns the economizer off);
    vav_min; sat_reset (False keeps 12.8 C supply air all year); elec (lighting + plug multiplier).
    base: spec dict (bd.bdg2_spec or a generator spec). meters: BDG2 meter names that exist."""
    k = dict(knobs or {})
    s = {kk: v for kk, v in copy.deepcopy(base).items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    for drop in ('meters', 'defaults_source', 'src_fill', 'weather_hint', 'hvac_hint'):
        s.pop(drop, None)
    use = s.get('primary_use', 'other')
    ms = set(meters)
    chw, stm, hw, gas = 'chilledwater' in ms, 'steam' in ms, 'hotwater' in ms, 'gas' in ms
    h = s.get('hvac') or {}
    sysk = k.get('sys', 'auto')
    if chw and (stm or hw):
        s['system'] = 'district_steam_chw'; h = _clean_hvac(h)  # noqa: E702
        if hw and not stm: h['district_heat'] = 'hot_water'  # noqa: E701
        s['dhw'] = dict(fuel='district')
    elif stm or hw:
        s['system'] = 'vav_chiller_boiler'; h = _clean_hvac(h); h['heat_source'] = 'district'  # noqa: E702
        if hw: h['district_heat'] = 'hot_water'  # noqa: E701
        s['dhw'] = dict(fuel='district')
    elif chw:
        s['system'] = 'district_steam_chw'; h = _clean_hvac(h)  # noqa: E702
        if gas:            # district cooling + gas meter: keep district heat (unmetered), DHW on gas
            s['dhw'] = dict(fuel='gas')
    elif gas:
        if sysk in ('auto', 'gas_rtu') or sysk == 'keep' and s.get('system') not in ('packaged_rtu', 'vav_chiller_boiler'):
            s['system'] = 'packaged_rtu'; h = _clean_hvac(h); h['heat_source'] = 'gas'  # noqa: E702
        elif sysk == 'gas_vav':
            s['system'] = 'vav_chiller_boiler'; h = _clean_hvac(h)  # noqa: E702
        s['dhw'] = dict(fuel='gas')
    else:   # electricity only: HVAC unknown
        if sysk == 'district':
            s['system'] = 'district_steam_chw'; h = _clean_hvac(h)  # noqa: E702
        elif sysk == 'electric':
            s['system'] = 'packaged_rtu'; h = _clean_hvac(h); h['heat_source'] = 'hp'  # noqa: E702
            s['dhw'] = dict(fuel='electric')
        elif sysk == 'gas_rtu':
            s['system'] = 'packaged_rtu'; h = _clean_hvac(h); h['heat_source'] = 'gas'  # noqa: E702
        elif sysk == 'auto':
            s.pop('system', None); h = _clean_hvac(h)  # noqa: E702
        # 'keep': generator's own system
    s['hvac'] = h
    sc = k.get('sched', 'default')
    if sc != 'default':
        s['occupancy_schedule'] = _sched(use, sc)
    v = dict(s.get('ventilation') or {})
    if k.get('oa', 1.0) != 1.0:
        from .spec import USE_DEFAULTS
        U = USE_DEFAULTS.get(use, USE_DEFAULTS['other'])
        v['L_s_person'] = (v.get('L_s_person') or U['oa_p']) * k['oa']
        v['L_s_m2'] = (v.get('L_s_m2') or U['oa_a']) * k['oa']
        if k['oa'] >= 2.0: v['economizer'] = False  # noqa: E701
    s['ventilation'] = v
    if 'vav_min' in k: h['vav_min_frac'] = float(k['vav_min'])  # noqa: E701
    if k.get('sat_reset') is False: h['sat_reset'] = False  # noqa: E701
    if 'elec' in k:      # interior lighting + plug multiplier
        ig = dict(s.get('internal_gains') or {})
        from .spec import USE_DEFAULTS
        U = USE_DEFAULTS.get(use, USE_DEFAULTS['other'])
        ig['lighting'] = (ig.get('lighting') or U['lighting']) * k['elec']
        ig['plug'] = (ig.get('plug') or U['plug']) * k['elec']
        s['internal_gains'] = ig
    if not s.get('n_floors') and s.get('floor_area_m2'):
        s['n_floors'] = max(1, int(round(np.sqrt(s['floor_area_m2']) / 30)))
    return s


def scale_meters(meters_kW, knobs, area):
    """LoadCast `lcsim.post`: empirical meter calibration y = a_m * sim_m + b_m * A / 1000 (kW; b in W/m2 of floor),
    a = distribution/plant losses and scale, b = unmodelled base load. knobs: {'a_<meter>': a, 'b_<meter>': b} with
    BDG2 meter names (DEV_METER_SCALE has the dev medians); missing knobs leave a meter unchanged. Keys starting
    with '_' are dropped. Returns a new dict; the simulation output and its ledger are not touched."""
    out = {}
    for mt, x in meters_kW.items():
        if mt.startswith('_'): continue  # noqa: E701
        a = float(knobs.get(f'a_{mt}', 1.0)); b = float(knobs.get(f'b_{mt}', 0.0))  # noqa: E702
        out[mt] = a * x + b * area / 1000.0
    return out


# ================================================================================================ BuildingDetective
# Source: models/buildingdetective/training/wrappers.py, common.py (clean_labels, derived_labels), simone.py,
# and the shipped training/calibration.json.

#: W1: one factor on per-area intensities per use type, fitted (calib_intensity.py) so the median generated site EUI
#: matches the weighted median of CBECS 2018 / RECS 2020 prior + dev respondents (never exam).
W1_INTENSITY = {'office': 0.9905691832166055, 'warehouse': 0.7788347772805057, 'retail': 0.7292949713349939,
                'residential_multi': 0.615934632888116, 'other': 0.7429057869350414,
                'assembly': 0.6620784525257961, 'residential_single': 0.6453917238046647, 'food': 0.5443876007122788,
                'education': 0.549283049096764, 'healthcare': 0.5593522006737923, 'lab': 0.41983690470803603,
                'lodging': 1.549059117990416}

#: W2: district distribution factors (meter = delivered / eff) fitted as sim(eff=1)/real on the BDG2 dev twins, 2017
#: (calib_campus.py; 21 heat and 24 chilled-water building-meters); eff_sd = 0.8 x log-sd of all district ratios.
W2_CAMPUS = dict(eff_heat_median=0.18481490847795473, eff_chw_median=0.3293591393583672, eff_sd=0.8, p_campus=0.3)

#: W3: share of each district meter's annual energy moved to a flat base (mean, sd), and an added flat electricity
#: load (x the meter mean; drawn with probability p_elec_base), from the dev twin test (devtwin_v1 vs BDG2 dev, 2017).
W3_BASE = dict(phi_heat=(0.70, 0.15), phi_chw=(0.50, 0.15), elec_base=(0.50, 0.35), p_elec_base=0.85)


def apply_intensity(f, fac):
    """Detective W1 (`wrappers.apply_intensity`): scale the per-area intensities of a FILLED spec (fill_spec output;
    modified in place and returned) by fac: lighting, plug, process, cooking, people W/m2, exterior lighting, outdoor
    air per m2 and per person, DHW volume; homes also scale infiltration (ach50, infiltration_ach) by sqrt(fac).
    fac = W1_INTENSITY[primary_use] for the fitted factors."""
    ig = f['internal_gains']
    for k in ('lighting', 'plug', 'process', 'cooking', 'people_W_m2', 'exterior_lighting'):
        if ig.get(k) is not None:
            ig[k] = float(ig[k]) * fac
    v = f['ventilation']
    for k in ('L_s_m2', 'L_s_person'):
        if v.get(k) is not None:
            v[k] = float(v[k]) * fac
    d = f.get('dhw')
    if isinstance(d, dict) and d.get('L_per_day') is not None:
        d['L_per_day'] = float(d['L_per_day']) * fac
    if f['primary_use'].startswith('residential'):
        if f.get('ach50') is not None:
            f['ach50'] = float(f['ach50']) * np.sqrt(fac)
        if f.get('infiltration_ach') is not None:
            f['infiltration_ach'] = float(f['infiltration_ach']) * np.sqrt(fac)
    f['_w1_intensity'] = fac
    return f


def make_campus(f, rng, kind=None, campus=None):
    """Detective W2 (`wrappers.make_campus`): convert a FILLED commercial spec (in place) to campus district service.
    kind 'both' | 'heat' | 'cool' (None: drawn 0.55 / 0.25 / 0.20). One lognormal draw sets both distribution
    factors around the dev medians (heat 0.185, chilled water 0.329, log-sd 0.8), clipped to [0.05, 1]; VAV minimum
    U(0.3, 0.6); outdoor air per m2 x U(1, 2); economizer with p 0.5; around-the-clock air handlers with p 0.35.
    Upstream difference: the chilled-water factor goes to hvac['district_eff_chw'] (in the plant, so the ledgers
    close) instead of the model's '_eff_chw' + meter rescale (campus_meters); the meters are the same to round-off.
    Returns (f, kind)."""
    c = dict(W2_CAMPUS, **(campus or {}))
    kind = kind or rng.choice(['both', 'heat', 'cool'], p=[0.55, 0.25, 0.20])
    h = f['hvac']
    if kind == 'both':
        f['system'] = 'district_steam_chw'
        h['district_heat'] = 'hot_water' if rng.random() < 0.3 else 'steam'
    elif kind == 'heat':
        f['system'] = 'vav_chiller_boiler'
        h['heat_source'] = 'district'
        h.pop('cooling_source', None)
    else:
        f['system'] = 'district_steam_chw'          # chilled water metered; heat from the campus too but unmetered
    z = rng.normal(0, c['eff_sd'])
    h['district_eff'] = float(np.clip(c['eff_heat_median'] * np.exp(z), 0.05, 1.0))
    h['district_eff_chw'] = float(np.clip(c['eff_chw_median'] * np.exp(z), 0.05, 1.0))
    h['vav_min_frac'] = float(rng.uniform(0.3, 0.6))
    v = f['ventilation']
    v['L_s_m2'] = float(v.get('L_s_m2') or 0.3) * float(rng.uniform(1.0, 2.0))
    v['economizer'] = bool(rng.random() < 0.5)
    sch = f.get('occupancy_schedule')
    if isinstance(sch, dict) and sch.get('type') == 'commercial' and rng.random() < 0.35:
        sch['weekday'] = (0.0, 24.0)               # air handlers run around the clock (labs, residences, hospitals)
        sch['saturday'] = (0.0, 24.0)
        sch['sunday'] = (0.0, 24.0)
    f['_w2_campus'] = kind
    return f, kind


_HEAT = ('dheat', 'steam', 'hot_water', 'hotwater')
_COOL = ('dcool', 'chilled_water', 'chilledwater')
_ELEC = ('elec', 'electricity')


def flat_base_loads(meters, rng, commercial=True, params=None):
    """Detective W3 (`wrappers.w3_base_loads`): meters = {name: hourly kW} (BuildingSim names 'electricity',
    'chilled_water', 'steam', 'hot_water', BDG2 names, or the Detective's 'elec' / 'dheat' / 'dcool'). District heat
    and chilled water: x <- (1 - phi) x + phi mean(x), annual total unchanged (phi ~ N(0.70, 0.15) heat,
    N(0.50, 0.15) chilled water, clipped [0.05, 0.95]). Electricity: with p 0.85, + b mean(x), b ~ N(0.5, 0.35)
    clipped [0, 1.5] (x 0.3 for homes). Draws in dict order. Returns a new dict; other meters pass through."""
    P = dict(W3_BASE, **(params or {}))
    out = {}
    for ch, x in meters.items():
        x = np.asarray(x, float)
        mu = float(np.nanmean(x))
        if (ch in _HEAT or ch in _COOL) and mu > 0:
            m, s = P['phi_heat'] if ch in _HEAT else P['phi_chw']
            phi = float(np.clip(rng.normal(m, s), 0.05, 0.95))
            x = (1 - phi) * x + phi * mu
        elif ch in _ELEC and mu > 0 and rng.random() < P['p_elec_base']:
            m, s = P['elec_base']
            b = float(np.clip(rng.normal(m, s), 0.0, 1.5)) * (1.0 if commercial else 0.3)
            x = x + b * mu
        out[ch] = x
    return out


def base_share(out):
    """Detective label (common.derived_labels): 1 - (heating + cooling + fans + pumps + heat rejection) / site energy,
    from the summary end uses. Unclean: it explodes when site energy is ~0 (three training buildings had -8e12)."""
    eu = out['summary']['kWh_by_end_use']
    site = float(out['summary']['site_kWh'])
    heat_site = eu.get('heating_electricity', 0) + eu.get('heating_gas', 0) + eu.get('heating_district', 0)
    cool_site = eu.get('cooling_electricity', 0) + eu.get('cooling_district', 0)
    hvac_aux = eu.get('fans_electricity', 0) + eu.get('pumps_electricity', 0) + eu.get('heat_rejection_electricity', 0)
    return 1.0 - (heat_site + cool_site + hvac_aux) / max(site, 1e-9)


def clean_labels(y):
    """Detective label hygiene (`common.clean_labels`) on a dict of labels: base_share is undefined when the metered
    site energy is ~0 (NaN outside [-1, 1], else clipped to [-0.2, 1]); balance points (Tb_heat, Tb_cool,
    balance_point_heat_C) clipped to [-25, 30] degC. float32, as the model computes it. Returns a new dict."""
    y = dict(y)
    if 'base_share' in y:
        b = np.float32(y['base_share'])
        y['base_share'] = np.float32(np.nan) if np.abs(b) > 1 else np.clip(b, np.float32(-0.2), np.float32(1.0))
    for k in ('Tb_heat', 'Tb_cool', 'balance_point_heat_C'):
        if k in y:
            y[k] = np.clip(np.float32(y[k]), np.float32(-25), np.float32(30))
    return y


# ================================================================================================ FaultFinder
# Source: models/faultfinder/training/simbank.py (log_commands, Operator) and noise.py (Noise.damper_command,
# Noise.apply offsets). The model works on 15-min day arrays of fractions; these ports work on BuildingSim's trend
# frame (percent columns) and give the same numbers (checks/check_phase2.py).

_ACT = {'oa_damper_stuck': 'OA_damper_pct', 'cooling_valve_stuck': 'cool_cmd_pct', 'heating_valve_stuck': 'heat_cmd_pct',
        'oa_damper_leak': 'OA_damper_pct', 'cooling_valve_leak': 'cool_cmd_pct', 'heating_valve_leak': 'heat_cmd_pct'}


def log_commands(trend_healthy, trend_faulted, fault, rng, p_feedback=0.2):
    """FaultFinder `simbank.log_commands`: BMS trend logs record CONTROL SIGNALS, not actuator positions (twin test
    on dev: FLEXLAB heating valve stuck at 100 % logs a 0 % command; a 40 % leak logs 0 %). BuildingSim logs the
    actual position, so for stuck and leaking actuators the logged signal is rebuilt while the fan runs: stuck ->
    the healthy run's command (damper as is; valves wound up to 0 / 100 % when the healthy command is beyond the
    stuck position +-5 %); leak -> faulted signal minus the leak. With p_feedback (0.2) the actual position is kept
    (units with position feedback). trend_healthy / trend_faulted: out['trend'] of the same window without and with
    `fault` (a fault dict). Returns a new faulted trend frame."""
    t = fault['type']
    if rng.random() < p_feedback:
        return trend_faulted
    tf = trend_faulted.copy()
    col = _ACT.get(t)
    if col is None:
        return tf
    on = np.nan_to_num(tf['fan_status'].to_numpy(float)) > 0.5
    xf = tf[col].to_numpy(float) / 100.0
    if t.endswith('_stuck'):
        p = fault['position']
        h = trend_healthy[col].to_numpy(float) / 100.0
        if col == 'OA_damper_pct':
            cmd = h.copy()
        else:
            cmd = np.where(h > p + 0.05, 1.0, np.where(h < p - 0.05, 0.0, h))
        xf = np.where(on, cmd, xf)
    else:
        xf = np.where(on, np.clip(xf - fault['leak'], 0, 1), xf)
    tf[col] = xf * 100.0
    return tf


def damper_command(frac, rng):
    """FaultFinder `Noise.damper_command` (installed damper characteristic, dev calibration v4): BuildingSim logs the
    outdoor-air FRACTION as the damper signal; real dev units log a COMMAND whose OA fraction is a + (b - a) cmd^g
    (ERS / FLEXLAB at 100 % command mix only 0.81-0.90 outdoor air; Nesbitt at 15 % mixes 0.14-0.28). Inverse map
    with a ~ U(0, 0.15), b ~ U(0.75, 1), g ~ U(0.6, 1.6); 30 % of units keep the linear map. frac in 0..1."""
    if rng.random() < 0.3:
        return frac
    a, b, g = rng.uniform(0, 0.15), rng.uniform(0.75, 1.0), rng.uniform(0.6, 1.6)
    c = np.clip((frac - a) / (b - a), 0, 1) ** (1 / g)
    return np.where(np.isfinite(frac), c, np.nan).astype(np.float32)


def installed_damper(trend, rng):
    """damper_command applied to a trend frame's OA_damper_pct (one unit = one draw). Returns a new frame."""
    t = trend.copy()
    frac = t['OA_damper_pct'].to_numpy(np.float32) / np.float32(100)
    t['OA_damper_pct'] = np.asarray(damper_command(frac, rng), float) * 100.0
    return t


def static_offsets(trend, rng, level=1.0, large=True):
    """FaultFinder `Noise.apply` static sensor offsets (one constant per sensor per call; the model draws one set per
    unit-day). large=True is dev calibration v4 (big_offsets: mixed-air stratification, FLEXLAB MA 0.9 K below both
    OA and RA; zone vs return 1.5 K apart on the ORNL RTU; Korea supply-tracking tail): sd SAT 0.5 (p 0.2: 1.2),
    MAT 0.8, RAT 0.4, OAT 0.6, ZAT 0.9 K. large=False is v3: SAT 0.45 (p 0.2: 1.0), MAT 0.4, RAT 0.3, OAT 0.5,
    ZAT 0.3 K. All x level. Returns (new frame, {column: offset K})."""
    t = trend.copy()
    L = level
    if large:
        d = {'SAT': rng.normal(0, 0.5) * L if rng.random() < 0.8 else rng.normal(0, 1.2) * L,
             'MAT': rng.normal(0, 0.8) * L, 'RAT': rng.normal(0, 0.4) * L, 'OAT': rng.normal(0, 0.6) * L,
             'ZAT': rng.normal(0, 0.9) * L}
    else:
        d = {'SAT': rng.normal(0, 0.45) * L if rng.random() < 0.8 else rng.normal(0, 1.0) * L,
             'MAT': rng.normal(0, 0.4) * L, 'RAT': rng.normal(0, 0.3) * L, 'OAT': rng.normal(0, 0.5) * L,
             'ZAT': rng.normal(0, 0.3) * L}
    for c, v in d.items():
        t[c] = t[c].to_numpy(float) + v
    return t, d


class OperatorJitter:
    """FaultFinder `simbank.Operator` (real-operation irregularity, dev calibration from the Korean AHUs and the dev
    days): every local day the air handler starts and stops a little early or late (normal, sd in steps: start 2,
    stop 3 at 15 min) and with p 0.15 is switched off for a block of 1-3 h. shift=('start'|'stop', steps) from the
    timestamp shift_from (local, naive) moves start earlier or stop later (FaultFinder's schedule_shift fault).
    A controller: simulate(..., controller=OperatorJitter(seed, tz)); it forces hvac_on through the hook, so the zone
    physics responds."""

    def __init__(self, seed, tz, sd_start=2.0, sd_stop=3.0, p_block=0.15, shift=None, shift_from=None):
        self.seed, self.tz = seed, tz
        self.sd_start, self.sd_stop, self.p_block = sd_start, sd_stop, p_block
        self.shift, self.shift_from = shift, shift_from
        self.mod = None

    def _build(self, sim):
        on = np.asarray(sim.hvac_on, float).copy()
        t = pd.DatetimeIndex(sim.times).tz_convert(self.tz).tz_localize(None)
        day = (t.normalize() - t.normalize()[0]).days.to_numpy()
        rng = np.random.default_rng(self.seed)
        mod = on.copy()
        for d in np.unique(day):
            ii = np.where(day == d)[0]
            o = on[ii] > 0.5
            if not o.any() or o.all():
                continue
            a, b = ii[np.argmax(o)], ii[len(o) - 1 - np.argmax(o[::-1])]
            s1 = int(round(rng.normal(0, self.sd_start)))
            s2 = int(round(rng.normal(0, self.sd_stop)))
            if self.shift is not None and t[ii[0]] >= self.shift_from:
                if self.shift[0] == 'start':
                    s1 = -self.shift[1]
                else:
                    s2 = self.shift[1]
            a2, b2 = max(ii[0], a + s1), min(ii[-1], b + s2)
            mod[ii] = 0.0
            mod[a2:b2 + 1] = 1.0
            if rng.random() < self.p_block and b2 - a2 > 16:
                c = rng.integers(a2 + 4, b2 - 8)
                mod[c:c + rng.integers(4, 12)] = 0.0
        self.mod = mod.tolist()

    def __call__(self, obs):
        if self.mod is None:
            self._build(obs['sim'])
        return {'hvac_on': self.mod[obs['k']] > 0.5}
