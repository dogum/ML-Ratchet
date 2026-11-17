"""Building spec (the shared contract): defaults, filling and validation.

`fill_spec(spec, climate=None)` returns a complete copy; every field that was missing is listed in
spec['src_fill'] with the source of its default. Unknown contract fields may be None.

Default sources (approximate, rounded):
- internal loads and outdoor air by use: DOE commercial prototype buildings (PNNL, ASHRAE 90.1-2010 versions) and
  ASHRAE 62.1-2010 Table 6-1 (people + area rates); people heat gains ASHRAE Fundamentals 2017 ch. 18 Table 1;
  residential loads from the Building America House Simulation Protocols (Wilson et al. 2014).
- envelope by vintage and IECC climate zone: approximate code levels (pre-1980 typical existing stock, 1980-2003
  ASHRAE 90.1-1989 / MEC era, 2004+ ASHRAE 90.1-2007 non-residential and IECC 2009 residential prescriptive tables).
- equipment efficiency by vintage: US federal minimum efficiency standards (DOE 10 CFR 430/431) by year.
- thermal mass classes: ISO 13790:2008 Table 12.
"""
import copy
import math

USE_TYPES = ('office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse',
             'residential_single', 'residential_multi', 'food', 'lab', 'other')

# people_m2: floor area per person at peak; people_W: total heat per person (sensible, latent) W
USE_DEFAULTS = {
    'office':      dict(people_m2=18.6, people_sens=75, people_lat=55, lighting=9.7, plug=10.8, oa_p=2.5, oa_a=0.30,
                        heat=21.1, cool=23.9, sb_heat=15.6, sb_cool=26.7, ftf=3.96, ceiling=2.74, aspect=1.5,
                        wwr=0.33, dhw_L_m2d=0.35, ext_light=1.0, mass='medium', system='packaged_rtu', cooking=0.0),
    'education':   dict(people_m2=8.0, people_sens=70, people_lat=45, lighting=10.0, plug=8.0, oa_p=5.0, oa_a=0.60,
                        heat=21.1, cool=23.9, sb_heat=15.6, sb_cool=26.7, ftf=4.0, ceiling=2.74, aspect=2.0,
                        wwr=0.35, dhw_L_m2d=0.6, ext_light=1.0, mass='medium', system='packaged_rtu', cooking=2.0),
    'lodging':     dict(people_m2=25.0, people_sens=70, people_lat=45, lighting=8.0, plug=6.0, oa_p=2.5, oa_a=0.30,
                        heat=21.1, cool=23.9, sb_heat=None, sb_cool=None, ftf=3.05, ceiling=2.6, aspect=3.0,
                        wwr=0.30, dhw_L_m2d=5.0, ext_light=1.0, mass='medium', system='ptac', cooking=2.0),
    'retail':      dict(people_m2=10.0, people_sens=75, people_lat=55, lighting=15.0, plug=4.0, oa_p=3.8, oa_a=0.60,
                        heat=21.1, cool=23.9, sb_heat=15.6, sb_cool=29.4, ftf=6.1, ceiling=5.0, aspect=1.3,
                        wwr=0.10, dhw_L_m2d=0.2, ext_light=2.0, mass='medium', system='packaged_rtu', cooking=0.0),
    'healthcare':  dict(people_m2=10.0, people_sens=75, people_lat=55, lighting=12.0, plug=20.0, oa_p=2.5, oa_a=1.50,
                        heat=21.1, cool=23.9, sb_heat=None, sb_cool=None, ftf=4.27, ceiling=2.74, aspect=1.5,
                        wwr=0.30, dhw_L_m2d=3.5, ext_light=1.0, mass='heavy', system='vav_chiller_boiler', cooking=3.0),
    'assembly':    dict(people_m2=8.0, people_sens=70, people_lat=35, lighting=10.0, plug=5.0, oa_p=2.5, oa_a=0.30,
                        heat=21.1, cool=23.9, sb_heat=15.6, sb_cool=29.4, ftf=5.0, ceiling=4.0, aspect=1.3,
                        wwr=0.20, dhw_L_m2d=0.5, ext_light=1.0, mass='heavy', system='packaged_rtu', cooking=1.0),
    'warehouse':   dict(people_m2=100.0, people_sens=100, people_lat=105, lighting=7.0, plug=2.0, oa_p=0.0,
                        oa_a=0.30, heat=15.6, cool=26.7, sb_heat=10.0, sb_cool=32.2, ftf=8.5, ceiling=8.0, aspect=2.0,
                        wwr=0.02, dhw_L_m2d=0.1, ext_light=0.8, mass='light', system='packaged_rtu', cooking=0.0),
    'residential_single': dict(people_m2=70.0, people_sens=70, people_lat=45, lighting=2.0, plug=6.0, oa_p=3.5,
                               oa_a=0.15, heat=21.1, cool=24.4, sb_heat=None, sb_cool=None, ftf=2.74, ceiling=2.44,
                               aspect=1.3, wwr=0.15, dhw_L_m2d=None, ext_light=0.1, mass='light',
                               system='split_ac_furnace', cooking=0.0),
    'residential_multi': dict(people_m2=40.0, people_sens=70, people_lat=45, lighting=2.5, plug=7.0, oa_p=3.5,
                              oa_a=0.15, heat=21.1, cool=24.4, sb_heat=None, sb_cool=None, ftf=3.0, ceiling=2.6,
                              aspect=2.0, wwr=0.20, dhw_L_m2d=None, ext_light=0.3, mass='medium', system='ptac',
                              cooking=0.0),
    'food':        dict(people_m2=5.0, people_sens=80, people_lat=80, lighting=10.7, plug=40.0, oa_p=3.8, oa_a=2.0,
                        heat=21.1, cool=23.9, sb_heat=15.6, sb_cool=29.4, ftf=3.05, ceiling=2.74, aspect=1.0,
                        wwr=0.25, dhw_L_m2d=8.0, ext_light=2.0, mass='medium', system='packaged_rtu',
                        cooking=120.0),
    'lab':         dict(people_m2=15.0, people_sens=75, people_lat=55, lighting=12.0, plug=20.0, oa_p=0.0, oa_a=2.5,
                        heat=21.1, cool=23.9, sb_heat=None, sb_cool=None, ftf=4.27, ceiling=2.9, aspect=1.5,
                        wwr=0.30, dhw_L_m2d=1.0, ext_light=1.0, mass='heavy', system='vav_chiller_boiler',
                        cooking=0.0),
    'other':       dict(people_m2=20.0, people_sens=75, people_lat=55, lighting=9.0, plug=6.0, oa_p=2.5, oa_a=0.30,
                        heat=21.1, cool=23.9, sb_heat=15.6, sb_cool=26.7, ftf=4.0, ceiling=3.0, aspect=1.5,
                        wwr=0.20, dhw_L_m2d=0.3, ext_light=1.0, mass='medium', system='packaged_rtu', cooking=0.0),
}

# ISO 13790 Table 12: (A_m / A_f, C_m / A_f in J/m2K)
MASS_CLASSES = {'very_light': (2.5, 80e3), 'light': (2.5, 110e3), 'medium': (2.5, 165e3), 'heavy': (3.0, 260e3),
                'very_heavy': (3.5, 370e3)}

SYSTEMS = {
    'ptac': 'packaged terminal AC per zone, electric resistance heat (heat_source="hp" gives a PTHP), constant fan',
    'packaged_rtu': 'packaged rooftop unit: DX cooling, gas furnace (or electric/HP), constant-volume fan, economizer',
    'split_ac_furnace': 'residential central AC + gas furnace, cycling blower, no outdoor-air intake',
    'heat_pump_air': 'air-source heat pump (single/two/variable speed), defrost, electric backup strips',
    'heat_pump_ground': 'ground-source (water-to-air) heat pump with a lumped ground-loop temperature',
    'vav_chiller_boiler': 'VAV AHU (economizer, preheat), reheat boxes, chiller (water/air-cooled, EIR curves), '
                          'cooling tower, gas boiler, pumps',
    'fan_coil': '4-pipe fan coils + dedicated outdoor-air unit, chiller and boiler plant',
    'radiant': 'radiant floor (heat into the mass node) from a boiler or heat pump, optional chiller, DOAS',
    'district_steam_chw': 'VAV AHU with district steam heating and district chilled water (meters steam, chilled_water)',
    'baseboard_resistance': 'electric baseboard heat, optional window AC',
    'hydronic_baseboard': 'baseboards / radiators from a boiler (or district), optional room or central AC '
                          '(extension; ResStock boilers and shared heating)',
    'wall_furnace': 'gas (or electric) wall/floor furnace without ducts, optional room AC (extension; ResStock)',
    'none': 'no HVAC (free-floating)',
    'ideal': 'ideal loads: unlimited heating/cooling exactly to setpoint, energy = load (for checks and sizing)',
}


_CAL = None


def _calib():
    global _CAL
    if _CAL is None:
        import json
        import os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration.json')
        _CAL = json.load(open(f)) if os.path.exists(f) else {}
    return _CAL


def list_systems():
    return dict(SYSTEMS)


def climate_zone_number(cz):
    if cz is None:
        return 4
    s = str(cz).strip().upper()
    d = ''.join(ch for ch in s if ch.isdigit())
    return int(d) if d else 4


def default_envelope(year, cz, use):
    """Approximate code-level envelope (W/m2K, -, ACH) by vintage and IECC climate zone number."""
    z = climate_zone_number(cz)
    res = use.startswith('residential')
    year = 1985 if year is None else year
    if year < 1980:
        e = dict(wall_U=0.9 if res else 1.3, roof_U=0.45 if res else 0.7, window_U=4.8 if res else 5.0,
                 SHGC=0.65 if res else 0.6, infiltration_ach=0.9 if res else 0.6)
    elif year < 2004:
        e = dict(wall_U=0.55 if res else 0.7, roof_U=0.25 if res else 0.4, window_U=3.2 if res else 3.4,
                 SHGC=0.55, infiltration_ach=0.5 if res else 0.4)
    else:
        if res:
            wall = {1: 0.48, 2: 0.48, 3: 0.48, 4: 0.48, 5: 0.33, 6: 0.33, 7: 0.32, 8: 0.32}[min(max(z, 1), 8)]
            roof = {1: 0.19, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.12, 6: 0.12, 7: 0.12, 8: 0.12}[min(max(z, 1), 8)]
            win = {1: 4.0, 2: 3.4, 3: 2.8, 4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0}[min(max(z, 1), 8)]
            shgc = 0.30 if z <= 3 else 0.40
        else:
            wall = {1: 0.70, 2: 0.48, 3: 0.48, 4: 0.36, 5: 0.36, 6: 0.31, 7: 0.31, 8: 0.31}[min(max(z, 1), 8)]
            roof = 0.27 if z < 8 else 0.21
            win = {1: 3.4, 2: 3.4, 3: 3.2, 4: 2.6, 5: 2.6, 6: 2.4, 7: 2.0, 8: 2.0}[min(max(z, 1), 8)]
            shgc = 0.25 if z <= 3 else (0.40 if z <= 6 else 0.45)
        e = dict(wall_U=wall, roof_U=roof, window_U=win, SHGC=shgc, infiltration_ach=0.35 if res else 0.3)
    return e


def default_efficiencies(year, system):
    """Rated efficiencies by vintage (federal minimums of the time, rounded)."""
    year = 1995 if year is None else year
    if year < 1992:
        seer, hspf, afue, eer_c, boiler, cop_ch = 9.0, 6.8, 0.70, 8.5, 0.75, 4.2
    elif year < 2006:
        seer, hspf, afue, eer_c, boiler, cop_ch = 10.0, 6.8, 0.78, 9.5, 0.78, 4.9
    elif year < 2015:
        seer, hspf, afue, eer_c, boiler, cop_ch = 13.0, 7.7, 0.80, 11.0, 0.80, 5.5
    else:
        seer, hspf, afue, eer_c, boiler, cop_ch = 14.0, 8.2, 0.80, 11.2, 0.82, 6.1
    return dict(SEER=seer, HSPF=hspf, AFUE=afue, EER=eer_c, boiler_eff=boiler, chiller_COP=cop_ch)


def seer_to_eer(seer, speed='single'):
    """EER (Btu/Wh) from SEER: single speed EER ~ 0.875 SEER (DOE rule of thumb); variable speed EER/SEER is lower."""
    return {'single': 0.875, 'two': 0.80, 'variable': 0.62}.get(speed, 0.875) * seer


def hspf_to_cop47(hspf, speed='single'):
    """Rated heating COP at 8.3 C from HSPF (rule of thumb fitted to AHRI-listed equipment: ~3.3 at HSPF 7.7,
    ~3.6 at 9 for single/two stage; inverter units carry a lower COP47 for the same HSPF, so we add nothing)."""
    return 0.38 * hspf + 0.40


def fill_spec(spec, climate=None):
    """Return a completed deep copy of `spec`. `climate`: optional dict from weather.prepare()['climate'] used for
    sizing defaults."""
    s = copy.deepcopy(spec)
    src = s.setdefault('src_fill', {})

    def dflt(key, val, why, d=None):
        d = s if d is None else d
        if d.get(key) is None:
            d[key] = val
            src[key] = why

    use = s.get('primary_use') or 'office'
    if use not in USE_DEFAULTS:
        use = 'other'
    s['primary_use'] = use
    U = USE_DEFAULTS[use]
    dflt('id', 'bldg', 'default')
    dflt('floor_area_m2', 180.0 if use == 'residential_single' else (2000.0 if use != 'residential_multi' else
                                                                     2500.0), 'typical size')
    res = use.startswith('residential')
    dflt('n_floors', 2 if use == 'residential_single' else max(1, int(round(math.sqrt(s['floor_area_m2']) / 30))),
         'typical')
    dflt('year_built', 1985, 'stock median-ish')
    dflt('climate_zone', None, 'unknown')
    env = default_envelope(s['year_built'], s.get('climate_zone'), use)
    for k, v in env.items():
        dflt(k, v, f'code-level default for vintage {s["year_built"]} and zone {s.get("climate_zone")}')
    dflt('window_wwr', U['wwr'], 'use-type default')
    dflt('thermal_mass', U['mass'], 'use-type default (ISO 13790 class)')
    g = s.setdefault('geometry', {})
    dflt('aspect_ratio', U['aspect'], 'use default', g)
    dflt('floor_to_floor_m', U['ftf'], 'DOE prototype', g)
    dflt('ceiling_m', U['ceiling'], 'DOE prototype', g)
    dflt('orientation_deg', 0.0, 'long facade faces south', g)
    dflt('foundation', 'slab', 'default', g)
    dflt('slab_F', 1.26 if s['year_built'] < 2004 else 0.9, 'ASHRAE 90.1 F-factor (uninsulated / insulated slab)',
         g)
    dflt('window_overhang', 0.0, 'none (projection / window height)', g)
    dflt('shading_factor', 0.9 if not res else 0.95, 'surroundings, frames, screens', g)
    if 'interior_shading' not in g:
        g['interior_shading'] = (0.7, 0.85) if res else None
        src['interior_shading'] = 'ResStock/OS-HPXML default summer 0.7 / winter 0.85 (residential)'
    dflt('roof_absorptance', 0.7, 'medium roof', g)
    dflt('wall_absorptance', 0.6, 'medium wall', g)
    dflt('attic', res, 'residential: vented attic above the ceiling', g)
    ig = s.setdefault('internal_gains', {})
    dflt('people', U['people_sens'] + U['people_lat'] if U['people_m2'] else 0.0, 'ASHRAE Fund. ch.18', ig)
    ig['people_W_m2'] = ig.get('people_W_m2') or (ig['people'] / U['people_m2'])
    dflt('people_latent_frac', U['people_lat'] / (U['people_sens'] + U['people_lat']), 'ASHRAE Fund. ch.18', ig)
    if res:
        from .stock import residential_plug
        n_occ = ig.get('occupants') or max(1.0, s['floor_area_m2'] / U['people_m2'])
        n_u = max(1, int(round(s['floor_area_m2'] / 85.0))) if use == 'residential_multi' else 1
        dflt('lighting', 0.0, 'residential: lighting is inside the ResStock-fitted plug model', ig)
        dflt('plug', residential_plug(s['floor_area_m2'] / n_u, n_occ / n_u), 'ResStock non-HVAC electricity fit', ig)
    dflt('lighting', U['lighting'], 'DOE prototype / 90.1-2010 LPD', ig)
    dflt('plug', U['plug'], 'DOE prototype', ig)
    dflt('cooking', U['cooking'], 'food service: gas cooking W/m2 at peak', ig)
    dflt('cooking_fuel', 'gas', 'default', ig)
    dflt('cooking_to_space', 0.3, 'hood removes ~70 % (ASHRAE Fund. ch.18)', ig)
    dflt('exterior_lighting', U['ext_light'] * (1.0 if res else 1.8), 'W per m2 floor, dusk to dawn (x1.8 non-'
         'residential after the ComStock calibration sample)', ig)
    dflt('process', 0.0, 'constant process/server load W/m2', ig)
    if res:
        dflt('occupants', max(1.0, s['floor_area_m2'] / U['people_m2'] * 1.0), 'BA HSP style', ig)
    sched = s.get('occupancy_schedule')
    from .schedules import DEFAULT_SCHEDULES
    if sched is None or isinstance(sched, str):
        name = sched if isinstance(sched, str) and sched in DEFAULT_SCHEDULES else use
        s['occupancy_schedule'] = copy.deepcopy(DEFAULT_SCHEDULES[name])
        src['occupancy_schedule'] = f'default schedule {name}'
    sp = s.setdefault('setpoints', {})
    dflt('heat', U['heat'], 'use default', sp)
    dflt('cool', U['cool'], 'use default', sp)
    # an explicit None means "no setback"; a missing key takes the use-type default
    if 'setback_heat' not in sp:
        sp['setback_heat'] = U['sb_heat']
        src['setback_heat'] = 'use default (None = no setback)'
    if 'setback_cool' not in sp:
        sp['setback_cool'] = U['sb_cool']
        src['setback_cool'] = 'use default (None = no setup)'
    dflt('recovery_h', 0.0, 'setpoint ramp before occupancy (h); commercial early start is in the schedule', sp)
    if res:
        dflt('schedule', 'night_setback' if sp.get('setback_heat') is not None else 'constant', 'default', sp)
    v = s.setdefault('ventilation', {})
    dflt('L_s_person', U['oa_p'], 'ASHRAE 62.1 Table 6-1', v)
    dflt('L_s_m2', U['oa_a'], 'ASHRAE 62.1 Table 6-1 (residential: 62.2-like continuous rate)', v)
    dflt('economizer', (not res) and climate_zone_number(s.get('climate_zone')) not in (1,), 'ASHRAE 90.1', v)
    dflt('economizer_type', 'drybulb', 'differential dry-bulb', v)
    dflt('econ_high_limit', 21.0, 'ASHRAE 90.1 fixed dry-bulb high limit (zones 4-8 use 21-24 C)', v)
    dflt('erv', False, 'default', v)
    dflt('erv_eff', 0.7, 'sensible effectiveness when ERV present', v)
    dflt('dcv', False, 'demand-controlled ventilation', v)
    dflt('mech_vent', (s['year_built'] >= 2010) if res else True, 'residential: exhaust fan in newer homes', v)
    dflt('natural_ventilation', bool(res), 'residential: windows opened in the cooling season (as ResStock)', v)
    dflt('nv_open_frac', 0.1, 'effective open area / window area for natural ventilation', v)
    dflt('system', U['system'], 'use default')
    h = s.setdefault('hvac', {})
    eff = default_efficiencies(s['year_built'], s['system'])
    dflt('SEER', eff['SEER'], 'federal minimum of the vintage', h)
    dflt('HSPF', eff['HSPF'], 'federal minimum of the vintage', h)
    dflt('EER', eff['EER'], 'federal minimum (commercial packaged) of the vintage', h)
    dflt('AFUE', eff['AFUE'], 'federal minimum of the vintage', h)
    dflt('boiler_eff', eff['boiler_eff'], 'federal minimum of the vintage', h)
    dflt('boiler_type', 'noncondensing', 'default', h)
    dflt('chiller_COP', eff['chiller_COP'], 'ASHRAE 90.1 path of the vintage', h)
    dflt('chiller_condenser', 'water' if s['floor_area_m2'] > 15000 else 'air', 'size rule (DOE prototypes)', h)
    dflt('speed', 'single', 'single-speed compressor', h)
    dflt('heat_source', None, 'per system type', h)
    dflt('cooling', s['system'] not in ('baseboard_resistance', 'hydronic_baseboard', 'wall_furnace', 'none')
         and not (use == 'warehouse'),
         'warehouses: heating only by default', h)
    dflt('heating_capacity_W', None, 'autosized', h)
    dflt('cooling_capacity_W', None, 'autosized', h)
    dflt('sizing_factor_heat', 1.25, 'ACCA Manual S / common practice', h)
    dflt('sizing_factor_cool', 1.15, 'ACCA Manual S / common practice', h)
    dflt('fan_W_per_m3s', None, 'per system type', h)
    dflt('backup_capacity_W', None, 'autosized (heat pumps: full design load)', h)
    dflt('hp_min_T', -17.8 if h['speed'] != 'variable' else -25.0, 'compressor lockout', h)
    dflt('static_reset', True, 'VAV duct static pressure reset (90.1 App G fan curve)', h)
    dflt('sat_cool', 12.8, 'VAV supply-air temperature setpoint', h)
    dflt('vav_min_frac', 0.3, 'VAV box minimum flow fraction', h)
    dflt('chw_supply_C', 6.67, 'AHRI 550/590 rating point', h)
    dflt('district_eff', 1.0, 'meter reads delivered energy', h)
    dhw = s.get('dhw')
    if dhw is None or isinstance(dhw, str):
        fuel = dhw if isinstance(dhw, str) else ('gas' if s['system'] in ('split_ac_furnace', 'packaged_rtu',
                                                                        'vav_chiller_boiler', 'fan_coil', 'radiant',
                                                                        'hydronic_baseboard', 'wall_furnace')
                                                 else 'electric')
        if s['system'] == 'district_steam_chw' and not isinstance(dhw, str):
            fuel = 'district'
        s['dhw'] = dict(fuel=fuel)
        src['dhw'] = 'fuel follows the heating system'
    d = s['dhw']
    dflt('fuel', 'electric', 'default', d)
    dflt('type', 'storage', 'storage tank', d)
    if res:
        dflt('L_per_day', 50.0 * ig['occupants'], 'about 50 L of tank hot water per person per day', d)
    else:
        # x0.6 after the ComStock calibration sample (water heating was 1.6x ComStock with CBECS-derived volumes)
        dflt('L_per_day', 0.6 * (U['dhw_L_m2d'] or 0.3) * s['floor_area_m2'], 'CBECS water-heating EUI by use x 0.6', d)
    if s.get('dhw_scale'):
        d['L_per_day'] = d['L_per_day'] * float(s['dhw_scale'])
        s['dhw_scale'] = None
    dflt('T_set', 51.7 if res else 60.0, '125 F residential, 140 F commercial', d)
    dflt('efficiency', {'gas': 0.78 if s['year_built'] < 2015 else 0.82, 'electric': 0.98, 'heat_pump': 3.2,
                        'district': 1.0}.get(d['fuel'], 0.9), 'recovery efficiency / rated COP', d)
    dflt('UA_W_K', (2.5 if d['fuel'] == 'gas' else 1.5) * (1 if res else max(1.0, d['L_per_day'] / 400.0)),
         'tank standby UA', d)
    dflt('in_conditioned_space', True, 'default', d)
    pv = s.get('pv')
    if pv:
        pv.setdefault('tilt', 25.0 if res else 10.0)
        pv.setdefault('azimuth', 180.0)
        pv.setdefault('losses', 0.14)
        pv.setdefault('gamma', -0.0037)
        pv.setdefault('dc_ac', 1.2)
    bat = s.get('battery')
    if bat:
        bat.setdefault('kW', bat.get('kWh', 10.0) / 2.0)
        bat.setdefault('eff_charge', 0.95)
        bat.setdefault('eff_discharge', 0.95)
        bat.setdefault('soc_min', 0.1)
        bat.setdefault('soc0', 0.5)
        bat.setdefault('mode', 'self_consumption')
    m = s.setdefault('meters', {})
    dflt('gas_fuels', ['gas'], 'default', m)
    if s.get('infiltration_factor') is None:
        s['infiltration_factor'] = _calib().get('infiltration_factor_residential', 1.0) if res else 1.0
        src['infiltration_factor'] = 'data/calibration.json (residential: fitted to ResStock twins)'
    s.setdefault('zones', 'single')
    s.setdefault('zone_model', '5R1C')
    s.setdefault('moisture_capacitance', 5.0)
    return s
