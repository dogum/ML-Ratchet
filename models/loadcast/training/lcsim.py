"""LoadCast wrappers around BuildingSim (no edits to the shared package).

1. Country holidays: BuildingSim's calendar knows US holidays only. `use_country(tz)` swaps its holiday function
   for LoadCast's own (US, Canada/Ontario, UK, Ireland) before a run (a monkeypatch inside our own process).
2. `make_spec(base, meters, knobs)`: a BDG2-style building (use, area, year, climate zone, which meters exist) plus
   calibration knobs -> a BuildingSim spec. Knobs (all optional; defaults = Phase 1 generic twin):
     sys     'auto' | 'district' | 'electric' | 'gas_rtu' | 'gas_vav' | 'keep'   (HVAC when meters do not decide it)
     sched   'default' | 'extended' | '24x7'      (operating hours / 24-7 air handlers)
     oa      outdoor-air multiplier on ASHRAE 62.1 rates (campus labs and old AHUs run high OA)
     vav_min VAV box minimum flow fraction (reheat; simultaneous heating and cooling)
     sat_reset  False keeps supply air at 12.8 C all year (more reheat)
3. `post(meters_kW, knobs, area)`: empirical meter calibration y = a_m * sim_m + b_m * A (W/m2), fitted on dev
   2016 per building (calib.py): a = distribution/plant losses and scale, b = unmodelled base loads (process steam,
   humidification, DHW loops, servers).
"""
import copy
import numpy as np, pandas as pd
import lc_common as C
import buildingsim as bs
from buildingsim import weather as _bw

_ORIG_HOL = _bw._holidays
_COUNTRY = ['US']


def _holidays_country(d):
    if _COUNTRY[0] == 'US':
        return _ORIG_HOL(d)
    return C.is_holiday(pd.DatetimeIndex(d), _COUNTRY[0])


_bw._holidays = _holidays_country


def use_country(tz):
    _COUNTRY[0] = C.country_of_tz(tz)


def _sched(use, kind):
    from buildingsim.schedules import DEFAULT_SCHEDULES
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


def make_spec(base, meters, knobs=None):
    """base: spec dict (BDG2 spec or generator spec). meters: iterable of BDG2 meter names that exist."""
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
        s['system'] = 'district_steam_chw'; h = _clean_hvac(h)
        if hw and not stm: h['district_heat'] = 'hot_water'
        s['dhw'] = dict(fuel='district')
    elif stm or hw:
        s['system'] = 'vav_chiller_boiler'; h = _clean_hvac(h); h['heat_source'] = 'district'
        if hw: h['district_heat'] = 'hot_water'
        s['dhw'] = dict(fuel='district')
    elif chw:
        s['system'] = 'district_steam_chw'; h = _clean_hvac(h)
        if gas:            # district cooling + gas meter: heat from an on-site gas boiler would need a new system;
            s['dhw'] = dict(fuel='gas')   # keep district heat (unmetered) and put DHW on gas
    elif gas:
        if sysk in ('auto', 'gas_rtu') or sysk == 'keep' and s.get('system') not in ('packaged_rtu', 'vav_chiller_boiler'):
            s['system'] = 'packaged_rtu'; h = _clean_hvac(h); h['heat_source'] = 'gas'
        elif sysk == 'gas_vav':
            s['system'] = 'vav_chiller_boiler'; h = _clean_hvac(h)
        s['dhw'] = dict(fuel='gas')
    else:   # electricity only: HVAC unknown
        if sysk == 'district':
            s['system'] = 'district_steam_chw'; h = _clean_hvac(h)
        elif sysk == 'electric':
            s['system'] = 'packaged_rtu'; h = _clean_hvac(h); h['heat_source'] = 'hp'
            s['dhw'] = dict(fuel='electric')
        elif sysk == 'gas_rtu':
            s['system'] = 'packaged_rtu'; h = _clean_hvac(h); h['heat_source'] = 'gas'
        elif sysk == 'auto':
            s.pop('system', None); h = _clean_hvac(h)
        # 'keep': generator's own system
    s['hvac'] = h
    sc = k.get('sched', 'default')
    if sc != 'default':
        s['occupancy_schedule'] = _sched(use, sc)
    v = dict(s.get('ventilation') or {})
    if k.get('oa', 1.0) != 1.0:
        from buildingsim.spec import USE_DEFAULTS
        U = USE_DEFAULTS.get(use, USE_DEFAULTS['other'])
        v['L_s_person'] = (v.get('L_s_person') or U['oa_p']) * k['oa']
        v['L_s_m2'] = (v.get('L_s_m2') or U['oa_a']) * k['oa']
        if k['oa'] >= 2.0: v['economizer'] = False
    s['ventilation'] = v
    if 'vav_min' in k: h['vav_min_frac'] = float(k['vav_min'])
    if k.get('sat_reset') is False: h['sat_reset'] = False
    if 'elec' in k:      # interior lighting + plug multiplier
        ig = dict(s.get('internal_gains') or {})
        from buildingsim.spec import USE_DEFAULTS
        U = USE_DEFAULTS.get(use, USE_DEFAULTS['other'])
        ig['lighting'] = (ig.get('lighting') or U['lighting']) * k['elec']
        ig['plug'] = (ig.get('plug') or U['plug']) * k['elec']
        s['internal_gains'] = ig
    if not s.get('n_floors') and s.get('floor_area_m2'):
        s['n_floors'] = max(1, int(round(np.sqrt(s['floor_area_m2']) / 30)))
    return s


def simulate_site(spec, site, start=None, end=None):
    """Run a spec through a BDG2 site's real weather; meters (kW) on the site's UTC frame (NaN outside the run)."""
    w, m = C.site_weather(site)
    use_country(m['tz'])
    idx = w.index
    s0 = idx[0] if start is None else start
    e0 = idx[-1] + pd.Timedelta(hours=1) if end is None else end
    out = bs.simulate(spec, (w, m), s0, e0, dt='1h', record_trend=False)
    ts = out['ts']
    res = {}
    for mt, col in C.SIM_COL.items():
        x = ts[col].reindex(idx).values if col in ts else np.zeros(len(idx))
        res[mt] = np.asarray(x, float)
    res['_sim_end_uses'] = None
    return res, out


def post(meters_kW, knobs, area):
    out = {}
    for mt, x in meters_kW.items():
        if mt.startswith('_'): continue
        a = float(knobs.get(f'a_{mt}', 1.0)); b = float(knobs.get(f'b_{mt}', 0.0))
        out[mt] = a * x + b * area / 1000.0
    return out
