"""Translate NREL ResStock / ComStock characteristics and CBECS respondents into the building-spec contract.

Used by the generator (bootstrap of real-stock characteristic combinations) and by the simulator-vs-simulator
check (twins of individual ResStock / ComStock models). Every mapping is an approximation of the source model;
the assumptions are written next to each table.

Units: ResStock uses IP strings ("R-13", "68F", "15 ACH50"); converted here to SI.
"""
import math
import re

import numpy as np
import pandas as pd

from .spec import default_envelope, default_efficiencies

FT2 = 0.09290304
RSI_PER_RIP = 0.1761102

# ---------------------------------------------------------------- ResStock option tables (approximate, SI)
# whole-assembly U (W/m2K) incl. films: framing factor 0.25 for studs; masonry walls with their mass
def _wall_U(s):
    s = str(s)
    m = re.search(r'R-(\d+)', s)
    R_ins = float(m.group(1)) * RSI_PER_RIP if m else 0.0
    if s.startswith('Wood Stud'):
        R_base = 0.62                  # films + gypsum + sheathing + siding + uninsulated cavity air
        if R_ins == 0:
            return 1.6
        R_cav = R_ins + 0.17
        U_cav = 1 / (R_base - 0.17 + R_cav)
        U_stud = 1 / (R_base - 0.17 + 0.089 * 3.5 / 0.0254 * 0.0254 / 0.12 * 0.0)  # placeholder overwritten
        U_stud = 1 / (R_base + 0.68)   # 3.5 in softwood stud ~ R-4.4 (RSI 0.78)
        return 0.75 * U_cav + 0.25 * U_stud
    if s.startswith('Brick'):
        return 1.0 / (0.55 + R_ins * 0.85) if R_ins else 1.75
    if s.startswith('CMU'):
        return 1.0 / (0.55 + R_ins * 0.85) if R_ins else 2.2
    return 1.0 / (0.5 + R_ins) if R_ins else 1.6


def _ceiling_U(s):
    s = str(s)
    m = re.search(r'R-(\d+)', s)
    if m:
        return 1.0 / (0.3 + float(m.group(1)) * RSI_PER_RIP * 0.93)
    if 'Uninsulated' in s:
        return 2.5
    return None


WINDOWS = {   # (U W/m2K, SHGC) whole window, approximate ResStock/NFRC values
    'Single, Clear, Metal': (6.4, 0.76), 'Single, Clear, Non-metal': (4.9, 0.64),
    'Double, Clear, Metal, Air': (4.4, 0.67), 'Double, Clear, Non-metal, Air': (2.8, 0.56),
    'Double, Low-E, Non-metal, Air, M-Gain': (2.0, 0.44), 'Triple, Low-E, Non-metal, Air, L-Gain': (1.6, 0.26),
    'Single, Clear, Metal, Exterior Clear Storm': (3.4, 0.67), 'Single, Clear, Non-metal, Exterior Clear Storm':
        (2.8, 0.56), 'Double, Clear, Metal, Air, Exterior Clear Storm': (2.6, 0.55),
    'Double, Clear, Non-metal, Air, Exterior Clear Storm': (2.0, 0.49)}

ORIENT = {'North': 180.0, 'Northeast': 225.0, 'East': 270.0, 'Southeast': 315.0, 'South': 0.0, 'Southwest': 45.0,
          'West': 90.0, 'Northwest': 135.0}
VINTAGE_YEAR = {'<1940': 1930, '1940s': 1945, '1950s': 1955, '1960s': 1965, '1970s': 1975, '1980s': 1985,
                '1990s': 1995, '2000s': 2005, '2010s': 2013}
FOUNDATION = {'Slab': 'slab', 'Vented Crawlspace': 'crawlspace', 'Unvented Crawlspace': 'crawlspace_unvented',
              'Unheated Basement': 'unconditioned_basement', 'Heated Basement': 'basement', 'Ambient': 'ambient'}


def _rval(s):
    """R-value (IP) in an option string such as 'Ceiling R-19' or 'Wall R-10, Exterior'; 0 if none."""
    m = re.search(r'R-?(\d+)', str(s))
    return float(m.group(1)) if m else 0.0


def _f2c(s, default=None):
    m = re.search(r'(-?\d+)F', str(s))
    return (float(m.group(1)) - 32) * 5 / 9 if m else default


def _num(s, default=None):
    m = re.search(r'(-?\d+\.?\d*)', str(s))
    return float(m.group(1)) if m else default


def residential_plug(area_m2, occupants, mult=1.0):
    """Peak plug W/m2 for the ResStock-derived shape: annual non-HVAC electricity
    E = 2967 + 4.78 A + 500 N kWh (least squares on 147 occupied ResStock 2025 R1 dwellings, R2 0.20;
    includes lighting and appliances), spread with schedules.RES_PLUG_SHAPE."""
    from .schedules import RES_PLUG_PEAK
    E = (2967.0 + 4.78 * area_m2 + 500.0 * occupants) * mult
    return E * 1000.0 / 8760.0 / max(area_m2, 10.0) * RES_PLUG_PEAK * 1.083


def resstock_to_spec(r, whole_building=False):
    """One ResStock dwelling unit (row of metadata) -> spec. whole_building=True turns a multi-family unit into
    its whole building (unit area x number of units, all exterior surfaces exposed)."""
    bt = str(r['in.geometry_building_type_recs'])
    mf = bt.startswith('Multi-Family')
    sfa = bt == 'Single-Family Attached'
    area = float(r['in.sqft..ft2']) * FT2
    stories = int(_num(r['in.geometry_stories'], 1))
    n_units = int(_num(r['in.geometry_building_number_units_mf'], 1) or 1) if mf else 1
    year = VINTAGE_YEAR.get(str(r['in.vintage']), 1980)
    cz = str(r['in.ashrae_iecc_climate_zone_2004'])
    g = dict(orientation_deg=ORIENT.get(str(r['in.orientation']), 0.0), aspect_ratio=1.5 if not mf else 2.0,
             floor_to_floor_m=2.74, ceiling_m=2.44)
    fnd = FOUNDATION.get(str(r['in.geometry_foundation_type']), 'slab')
    attic = str(r['in.geometry_attic_type'])
    g['attic'] = attic in ('Vented Attic', 'Unvented Attic')
    floor_ins = str(r['in.insulation_floor'])
    if fnd in ('crawlspace', 'crawlspace_unvented', 'unconditioned_basement', 'ambient'):
        R = _rval(floor_ins)
        g['floor_U'] = 1.0 / (0.6 + R * RSI_PER_RIP * 0.9)
    slab = str(r['in.insulation_slab'])
    g['slab_F'] = 1.26 if ('Uninsulated' in slab or slab == 'None') else (0.95 if 'R5' in slab else 0.85)
    fw = str(r['in.insulation_foundation_wall'])
    if fnd == 'basement':
        Rb = _rval(fw)
        g['basement_U'] = 1.0 / (0.7 + Rb * RSI_PER_RIP)
        g['below_grade_floors'] = 1           # ResStock conditioned area includes the heated basement
    exp_wall = 1.0
    if mf and not whole_building:
        lvl = str(r.get('in.geometry_building_level_mf', 'Middle'))
        hz = str(r.get('in.geometry_building_horizontal_location_mf', 'Middle'))
        exp_wall = 0.6 if hz in ('Left', 'Right') else 0.4
        if lvl != 'Bottom':
            fnd = 'adiabatic'
            g['floor_U'] = 0.0
        g['roof_exposed'] = lvl == 'Top'
        stories_unit = 1
    elif sfa:
        exp_wall = 0.7
        stories_unit = min(stories, 3)
    else:
        stories_unit = min(stories, 3) if not mf else stories
    g['foundation'] = fnd
    g['wall_exposed_frac'] = exp_wall
    wall_U = _wall_U(r['in.insulation_wall'])
    cu = _ceiling_U(r['in.insulation_ceiling'])
    roof_s = str(r['in.insulation_roof'])
    roof_U = cu if cu is not None else (_ceiling_U(roof_s) or 2.0)
    if cu is None and 'Uninsulated' in roof_s:
        roof_U = 2.5
    win = WINDOWS.get(str(r['in.windows']), (3.0, 0.6))
    wa = re.findall(r'(\d+)', str(r['in.window_areas']))
    wwr = (float(wa[0]) / 100.0) if wa else 0.15
    ach50 = _num(r['in.infiltration'], 10.0)
    if 'in.air_leakage_to_outside_ach50' in r and pd.notna(r.get('in.air_leakage_to_outside_ach50')):
        ach50 = float(r['in.air_leakage_to_outside_ach50'])     # attached units: leakage to outside only
    occ = _num(r['in.occupants'], 2.0) or 0.0
    if str(r['in.occupants']).startswith('10'):
        occ = 10.0
    vac = str(r.get('in.vacancy_status', 'Occupied')) == 'Vacant'
    # setpoints
    hsp = _f2c(r['in.heating_setpoint'], 20.0)
    csp = _f2c(r['in.cooling_setpoint'], 24.0)
    hmag = (_num(r['in.heating_setpoint_offset_magnitude'], 0) or 0) * 5 / 9
    cmag = (_num(r['in.cooling_setpoint_offset_magnitude'], 0) or 0) * 5 / 9
    hper = str(r['in.heating_setpoint_offset_period'])
    sched = 'constant'
    if hmag > 0 and hper != 'None':
        sched = 'night_setback' if hper.startswith('Night') else ('away_setback' if hper.startswith('Day') and
                                                                  'Night' not in hper else 'setback')
    sp = dict(heat=hsp, cool=max(csp, hsp + 1.0), setback_heat=hsp - hmag if hmag > 0 else None,
              setback_cool=(csp + cmag) if cmag > 0 else None, schedule=sched)
    # HVAC
    htf = str(r['in.hvac_heating_type_and_fuel'])
    heff = str(r['in.hvac_heating_efficiency'])
    ctype = str(r['in.hvac_cooling_type'])
    ceff = str(r['in.hvac_cooling_efficiency'])
    shared = str(r['in.hvac_has_shared_system'])
    h = {}
    fuel = 'gas'
    if 'Fuel Oil' in htf:
        fuel = 'fuel_oil'
    elif 'Propane' in htf:
        fuel = 'propane'
    cooling = ctype not in ('None',)
    seer = _num(ceff.split('SEER')[1], None) if 'SEER' in ceff else None
    eer_room = _num(ceff.split('EER')[1], None) if ceff.startswith('Room AC') else None
    if 'ASHP' in heff or 'MSHP' in heff:
        system = 'heat_pump_air'
        seer = _num(heff.split('SEER')[1], 13.0)
        h['HSPF'] = _num(heff.split('SEER')[1].split(',')[1], 7.7)
        h['SEER'] = seer
        h['speed'] = 'variable' if ('MSHP' in heff or seer >= 18) else ('two' if seer >= 16 else 'single')
        if 'MSHP' in heff:
            h['fan_W_per_cfm'] = 0.07
        cooling = True
    elif 'Shared Heating' in heff or shared in ('Heating Only', 'Heating and Cooling'):
        system = 'fan_coil' if shared == 'Heating and Cooling' else 'hydronic_baseboard'
        h['heat_source'] = 'elec' if 'Electricity' in htf else 'boiler'
        h['boiler_eff'] = 0.80
        if system == 'fan_coil':
            h['chiller_condenser'] = 'air'
    elif 'Fuel Furnace' in heff:
        system = 'split_ac_furnace'
        h['AFUE'] = _num(heff.split(',')[1], 80.0) / 100.0
    elif 'Electric Furnace' in heff or 'Electric Wall Furnace' in heff:
        system = 'split_ac_furnace' if 'Wall' not in heff else 'wall_furnace'
        h['heat_source'] = 'elec'
    elif 'Wall/Floor Furnace' in heff:
        system = 'wall_furnace'
        h['AFUE'] = _num(heff.split(',')[1], 68.0) / 100.0
    elif 'Boiler' in heff:
        system = 'hydronic_baseboard'
        h['heat_source'] = 'boiler'
        h['boiler_eff'] = _num(heff.split(',')[1], 80.0) / 100.0
    elif 'Baseboard' in heff:
        system = 'baseboard_resistance'
    elif htf == 'None' or heff == 'None':
        system = 'split_ac_furnace' if ctype == 'Central AC' else ('baseboard_resistance' if cooling else 'none')
        h['heating_capacity_W'] = 1.0 if system != 'none' else None
    else:
        system = 'split_ac_furnace'
    h['cooling'] = bool(cooling)
    if seer and 'SEER' not in h:
        h['SEER'] = seer
    if eer_room:
        h['SEER'] = eer_room / 0.875
        h['room_ac'] = True
    if ctype == 'Room AC' and system == 'split_ac_furnace':
        h['room_ac'] = True
    if ctype == 'Central AC' and system in ('baseboard_resistance', 'hydronic_baseboard', 'wall_furnace'):
        h['central_ac'] = True
    h['fuel'] = fuel if 'Electricity' not in htf else 'electricity'
    # water heating
    whe = str(r['in.water_heater_efficiency'])
    whf = str(r['in.water_heater_fuel'])
    if 'Heat Pump' in whe:
        dhw = dict(fuel='heat_pump', efficiency=_num(whe.split(',')[-1], 3.45) * 0.9)
    elif whf == 'Electricity':
        dhw = dict(fuel='electric', efficiency=0.99 if 'Tankless' in whe else 0.92 if 'Standard' in whe else 0.95)
    elif whf in ('Natural Gas', 'Propane', 'Fuel Oil'):
        dhw = dict(fuel='gas', efficiency=0.82 if 'Tankless' in whe else (0.67 if 'Premium' in whe else 0.59))
        if 'Tankless' in whe:
            dhw['UA_W_K'] = 0.0
    else:
        dhw = dict(fuel='electric')
    # lighting / plug: BA HSP-like peak W/m2 scaled by technology and plug-load multiplier
    lt = str(r['in.lighting'])
    lmult = {'100% LED': 0.45, '100% CFL': 0.6, '100% Incandescent': 1.6}.get(lt, 1.0)
    pmult = (_num(r['in.plug_loads'], 100.0) or 100.0) / 100.0
    pmult *= {'Low': 0.85, 'Medium': 1.0, 'High': 1.15}.get(str(r['in.usage_level']), 1.0)
    if vac:
        occ, pmult, lmult = 0.0, 0.3, 0.1
    unit_area = area
    if mf and whole_building:
        area = unit_area * n_units
        g['wall_exposed_frac'] = 1.0
        g['foundation'] = FOUNDATION.get(str(r['in.geometry_foundation_type']), 'slab')
        g['roof_exposed'] = True
        stories_unit = stories
        occ = occ * n_units
    # non-HVAC electricity (plug + appliances + lighting): regression on ResStock dwellings (see residential_plug)
    gas_cook = str(r['in.cooking_range']) in ('Gas', 'Propane')
    gas_dry = str(r['in.clothes_dryer']) in ('Gas', 'Propane')
    use = 'residential_multi' if mf else 'residential_single'
    n_u = n_units if (mf and whole_building) else 1
    occ_u = occ / n_u if n_u > 1 else occ
    plug_peak = residential_plug(area / n_u, occ_u, pmult * (0.92 + 0.08 * lmult)) * (1.0 if not vac else 0.3)
    gas_kwh = (600.0 if gas_cook else 0.0) + (700.0 if gas_dry else 0.0)
    ig = dict(lighting=0.0, plug=plug_peak, occupants=max(occ, 0.0), cooking=0.0,
              cooking_kWh_yr=gas_kwh * n_u, cooking_fuel='gas', cooking_to_space=0.3)
    if occ == 0:
        ig['people'] = 0.0
    pv = None
    pvs = str(r['in.pv_system_size'])
    if pvs not in ('None', 'nan'):
        pv = dict(kW=_num(pvs, 5.0), tilt=25.0, azimuth=180.0)
    lat = float(r['in.weather_file_latitude']) if pd.notna(r.get('in.weather_file_latitude')) else None
    lon = float(r['in.weather_file_longitude']) if pd.notna(r.get('in.weather_file_longitude')) else None
    spec = dict(id=f"resstock_{int(r['bldg_id'])}", primary_use=use, floor_area_m2=area,
                n_floors=max(1, stories_unit), year_built=year, climate_zone=cz, lat=lat, lon=lon,
                wall_U=wall_U, roof_U=roof_U, window_U=win[0], SHGC=win[1], window_wwr=wwr,
                ach50=ach50, thermal_mass='light' if str(r['in.geometry_wall_type']) == 'Wood Frame' else 'medium',
                geometry=g, internal_gains=ig, setpoints=sp,
                ventilation=dict(L_s_person=0.0, L_s_m2=0.0, mech_vent=False, economizer=False),
                system=system, hvac=h, dhw=dhw, pv=pv,
                provenance=dict(source='ResStock 2025 R1 (modelled stock)', bldg_id=int(r['bldg_id']),
                                county=str(r['in.county']), state=str(r['in.state']), unit_area_m2=unit_area,
                                n_units=n_units, whole_building=bool(whole_building)))
    if r.get('in.vacancy_status') == 'Vacant':
        spec['setpoints'] = dict(heat=12.8, cool=32.0, setback_heat=None, setback_cool=None, schedule='constant')
    return spec


# ---------------------------------------------------------------- ComStock
COM_USE = {'SmallOffice': 'office', 'MediumOffice': 'office', 'LargeOffice': 'office', 'Warehouse': 'warehouse',
           'RetailStandalone': 'retail', 'RetailStripmall': 'retail', 'Grocery': 'retail',
           'PrimarySchool': 'education', 'SecondarySchool': 'education', 'FullServiceRestaurant': 'food',
           'QuickServiceRestaurant': 'food', 'LargeHotel': 'lodging', 'SmallHotel': 'lodging',
           'Hospital': 'healthcare', 'Outpatient': 'healthcare'}
COM_FTF = {'SmallOffice': 3.05, 'MediumOffice': 3.96, 'LargeOffice': 3.96, 'Warehouse': 8.53,
           'RetailStandalone': 6.1, 'RetailStripmall': 5.18, 'Grocery': 6.1, 'PrimarySchool': 4.0,
           'SecondarySchool': 4.0, 'FullServiceRestaurant': 3.05, 'QuickServiceRestaurant': 3.05,
           'LargeHotel': 3.05, 'SmallHotel': 3.05, 'Hospital': 4.27, 'Outpatient': 3.05}
COM_WIN = {'Single - No LowE - Clear - Aluminum': (6.4, 0.78), 'Single - No LowE - Clear - Wood': (5.0, 0.70),
           'Single - No LowE - Tinted/Reflective - Aluminum': (6.4, 0.60),
           'Single - No LowE - Tinted/Reflective - Wood': (5.0, 0.55),
           'Double - No LowE - Clear - Aluminum': (3.7, 0.68), 'Double - No LowE - Tinted/Reflective - Aluminum':
               (3.7, 0.50), 'Double - LowE - Clear - Aluminum': (2.9, 0.42),
           'Double - LowE - Tinted/Reflective - Aluminum': (2.9, 0.32),
           'Double - LowE - Tinted/Reflective - Thermally Broken Aluminum': (2.5, 0.30),
           'Double - LowE - Clear - Thermally Broken Aluminum': (2.5, 0.40),
           'Triple - LowE - Clear - Thermally Broken Aluminum': (1.8, 0.35),
           'Triple - LowE - Tinted/Reflective - Thermally Broken Aluminum': (1.8, 0.25)}
WWR_CAT = {'_1pct': (0.0, 0.01), '2_10pct': (0.02, 0.10), '11_25pct': (0.11, 0.25), '26_50pct': (0.26, 0.50),
           '51_75pct': (0.51, 0.75), '76_100pct': (0.76, 0.90)}
# lighting-generation multipliers on the use-type LPD; scaled by 0.66 after the ComStock calibration sample
# (seed 11: BuildingSim interior lighting was 1.52x ComStock with the unscaled table)
LIGHT_GEN = {'gen1_t12_incandescent': 0.89, 'gen2_t8_halogen': 0.73, 'gen3_t5_cfl': 0.59, 'gen4_led': 0.43,
             'gen5_led': 0.33}


def _code_year(code):
    s = str(code)
    if 'Pre-1980' in s or 'Pre-1975' in s:
        return 1970
    if '1980-2004' in s or 'DEER 1985' in s or 'DEER 1996' in s or 'DEER 2003' in s:
        return 1992
    if '2004' in s or '2007' in s:
        return 2006
    if '2010' in s or '2011' in s:
        return 2011
    if '2013' in s or '2014' in s or '2015' in s or '2017' in s:
        return 2015
    return 1990


COM_HVAC = {   # ComStock hvac_system_type -> (system, hvac overrides)
    'PSZ-AC with gas coil': ('packaged_rtu', dict(heat_source='gas')),
    'PSZ-AC with electric coil': ('packaged_rtu', dict(heat_source='elec')),
    'PSZ-AC with gas boiler': ('packaged_rtu', dict(heat_source='boiler')),
    'PSZ-AC with district hot water': ('packaged_rtu', dict(heat_source='district')),
    'PSZ-HP': ('heat_pump_air', {}),
    'Residential AC with residential forced air furnace': ('split_ac_furnace', {}),
    'PVAV with gas boiler reheat': ('vav_chiller_boiler', dict(cooling_source='dx')),
    'PVAV with PFP boxes': ('vav_chiller_boiler', dict(cooling_source='dx', heat_source='elec', reheat='elec')),
    'PVAV with gas heat with electric reheat': ('vav_chiller_boiler', dict(cooling_source='dx', reheat='elec')),
    'PTAC with electric coil': ('ptac', dict(heat_source='elec')),
    'PTAC with gas boiler': ('ptac', dict(heat_source='boiler')),
    'PTAC with gas coil': ('ptac', dict(heat_source='gas')),
    'PTHP': ('ptac', dict(heat_source='hp')),
    'VAV chiller with gas boiler reheat': ('vav_chiller_boiler', dict(chiller_condenser='water')),
    'VAV air-cooled chiller with gas boiler reheat': ('vav_chiller_boiler', dict(chiller_condenser='air')),
    'VAV chiller with PFP boxes': ('vav_chiller_boiler', dict(chiller_condenser='water', heat_source='elec',
                                                              reheat='elec')),
    'VAV air-cooled chiller with PFP boxes': ('vav_chiller_boiler', dict(chiller_condenser='air', heat_source='elec',
                                                                         reheat='elec')),
    'VAV chiller with district hot water reheat': ('vav_chiller_boiler', dict(heat_source='district')),
    'VAV district chilled water with district hot water reheat': ('district_steam_chw',
                                                                  dict(district_heat='hot_water')),
    'DOAS with fan coil air-cooled chiller with boiler': ('fan_coil', dict(chiller_condenser='air')),
    'DOAS with fan coil chiller with boiler': ('fan_coil', dict(chiller_condenser='water')),
    'DOAS with fan coil chiller with district hot water': ('fan_coil', dict(heat_source='district')),
    'DOAS with fan coil chiller with baseboard electric': ('fan_coil', dict(heat_source='elec')),
    'DOAS with water source heat pumps cooling tower with boiler': ('heat_pump_ground', dict(wshp_tower=True)),
    'DOAS with water source heat pumps with ground source heat pump': ('heat_pump_ground', {}),
}


def comstock_to_spec(r, rng=None):
    """One ComStock model -> spec. rng jitters values drawn inside a category (WWR); None = midpoints."""
    bt = str(r['in.comstock_building_type'])
    use = COM_USE.get(bt, 'other')
    area = float(r['in.sqft..ft2']) * FT2
    nf = int(_num(r['in.number_stories'], 1) or 1)
    year = int(r['in.year_built']) if pd.notna(r['in.year_built']) else 1985
    cz = str(r['in.as_simulated_ashrae_iecc_climate_zone_2006'])
    y_env = _code_year(r['in.energy_code_followed_during_original_building_construction'])
    y_roof = _code_year(r['in.energy_code_followed_during_last_roof_replacement'])
    y_hvac = _code_year(r['in.energy_code_followed_during_last_hvac_replacement'])
    env = default_envelope(y_env, cz, use)
    roof = default_envelope(max(y_roof, y_env), cz, use)['roof_U']
    wct = str(r['in.wall_construction_type'])
    wall_U = env['wall_U'] * {'Mass': 1.15, 'SteelFramed': 1.0, 'WoodFramed': 0.85, 'Metal Building': 1.2}.get(wct, 1)
    mass = {'Mass': 'heavy', 'SteelFramed': 'medium', 'WoodFramed': 'light', 'Metal Building': 'light'}.get(wct,
                                                                                                         'medium')
    win = COM_WIN.get(str(r['in.window_type']), (3.5, 0.55))
    lo, hi = WWR_CAT.get(str(r['in.window_to_wall_ratio_category']), (0.1, 0.3))
    wwr = 0.5 * (lo + hi) if rng is None else float(rng.uniform(lo, hi))
    ftf = COM_FTF.get(bt, 4.0)
    aspect = float(r['in.aspect_ratio']) if pd.notna(r['in.aspect_ratio']) else 1.5
    geo_A = area / nf
    L = math.sqrt(geo_A * aspect)
    W = geo_A / L
    A_env = 2 * (L + W) * nf * ftf + geo_A
    V = area * (ftf - 0.6)
    q75 = float(r['in.airtightness..m3_per_m2_h']) if pd.notna(r['in.airtightness..m3_per_m2_h']) else 20.0
    ach50 = q75 * (50 / 75) ** 0.65 * A_env / V
    sys_name, hv = COM_HVAC.get(str(r['in.hvac_system_type']), ('packaged_rtu', {}))
    eff = default_efficiencies(y_hvac, sys_name)
    h = dict(hv)
    h.update(EER=eff['EER'], SEER=eff['SEER'], HSPF=eff['HSPF'], AFUE=max(eff['AFUE'], 0.78),
             boiler_eff=eff['boiler_eff'], chiller_COP=eff['chiller_COP'])
    hf = str(r['in.heating_fuel'])
    h['fuel'] = {'NaturalGas': 'gas', 'Electricity': 'electricity', 'Propane': 'propane', 'FuelOil': 'fuel_oil',
                 'DistrictHeating': 'district'}.get(hf, 'gas')
    # schedule
    def hrs(o, d):
        try:
            o, d = float(o), float(d)
        except (TypeError, ValueError):
            return None
        return None if d <= 0.1 else (o, min(o + d, o + 23.9))
    from .schedules import DEFAULT_SCHEDULES
    import copy
    sch = copy.deepcopy(DEFAULT_SCHEDULES[use])
    if bt == 'Outpatient':     # clinics keep office-like hours, not the 24/7 hospital schedule
        sch = copy.deepcopy(DEFAULT_SCHEDULES['office'])
        sch.update(weekday=(7, 19), saturday=(8, 14), plug_base=0.3)
    if sch.get('type') == 'commercial':
        wd = hrs(r['in.weekday_opening_time..hr'], r['in.weekday_operating_hours..hr'])
        we = hrs(r['in.weekend_opening_time..hr'], r['in.weekend_operating_hours..hr'])
        if wd:
            sch['weekday'] = wd
        sch['saturday'] = we
        sch['sunday'] = we if bt not in ('SmallOffice', 'MediumOffice', 'LargeOffice', 'PrimarySchool',
                                         'SecondarySchool', 'Warehouse') else None
    hsp_f = _num(r['in.tstat_htg_sp_f..f'], 999)
    csp_f = _num(r['in.tstat_clg_sp_f..f'], 999)
    sp = {}
    if hsp_f and hsp_f < 900:
        sp['heat'] = (hsp_f - 32) * 5 / 9
        d = _num(r['in.tstat_htg_delta_f..delta_f'], 0) or 0
        sp['setback_heat'] = sp['heat'] - d * 5 / 9 if d > 0 else None
    if csp_f and csp_f < 900:
        sp['cool'] = (csp_f - 32) * 5 / 9
        d = _num(r['in.tstat_clg_delta_f..delta_f'], 0) or 0
        sp['setback_cool'] = sp['cool'] + d * 5 / 9 if d > 0 else None
    from .spec import USE_DEFAULTS
    U = USE_DEFAULTS[use]
    lg = LIGHT_GEN.get(str(r['in.interior_lighting_generation']), 1.0)
    ig = dict(lighting=U['lighting'] * lg)
    if bt == 'Grocery':
        ig['plug'] = 8.0
        ig['process'] = 25.0             # refrigerated cases (remote condensers: heat mostly outside)
    if bt == 'QuickServiceRestaurant':
        ig['cooking'] = 200.0
    if bt == 'Outpatient':
        ig['plug'] = 12.0
    if bt in ('SmallOffice',):
        ig['plug'] = 6.5
    swh = str(r['in.service_water_heating_fuel'])
    dhw = dict(fuel={'NaturalGas': 'gas', 'Electricity': 'electric', 'Propane': 'gas', 'FuelOil': 'gas',
                     'DistrictHeating': 'district'}.get(swh, 'gas'))
    spec = dict(id=f"comstock_{int(r['bldg_id'])}", primary_use=use, floor_area_m2=area, n_floors=nf, year_built=year,
                climate_zone=cz, wall_U=wall_U, roof_U=roof, window_U=win[0], SHGC=win[1], window_wwr=wwr,
                ach50=ach50, thermal_mass=mass,
                geometry=dict(aspect_ratio=max(aspect, 1.0), floor_to_floor_m=ftf, ceiling_m=max(ftf - 0.9, 2.6),
                              orientation_deg=float(r['in.rotation..degrees']) if pd.notna(r['in.rotation..degrees'])
                              else 0.0),
                internal_gains=ig, occupancy_schedule=sch, setpoints=sp, system=sys_name, hvac=h, dhw=dhw,
                ventilation=dict(L_s_m2=0.6 if bt == 'Outpatient' else None, economizer=sys_name in ('packaged_rtu', 'vav_chiller_boiler', 'district_steam_chw')
                                 and y_hvac >= 1992 and area > 500),
                provenance=dict(source='ComStock 2025 R3 (modelled stock)', bldg_id=int(r['bldg_id']),
                                county=str(r['in.as_simulated_nhgis_county_gisjoin']), state=str(r['in.state']),
                                building_type=bt, hvac_system_type=str(r['in.hvac_system_type'])))
    if bt == 'Warehouse':
        spec['hvac']['cooling'] = str(r['in.hvac_system_type']).startswith('PSZ') and area < 20000 and \
            str(r.get('in.building_subtype', '')) != 'warehouse_bulk'
    return spec


# ---------------------------------------------------------------- CBECS (assembly, lab, other, and a check set)
CB_SYS = {'vav_chiller_boiler': 'vav_chiller_boiler', 'heat_pump_air': 'heat_pump_air', 'packaged_rtu': 'packaged_rtu',
          'split_ac_furnace': 'split_ac_furnace', 'ptac': 'ptac', 'baseboard_resistance': 'baseboard_resistance',
          'fan_coil': 'fan_coil', 'district_steam_chw': 'district_steam_chw', 'radiant': 'radiant', 'none': 'none'}


def cbecs_to_spec(r, rng=None):
    rng = rng or np.random.default_rng(0)
    use = str(r['primary_use']) if str(r['primary_use']) in ('office', 'education', 'lodging', 'retail',
                                                               'healthcare', 'assembly', 'warehouse', 'food', 'lab',
                                                               'other') else 'other'
    lo, hi = r['year_built_lo'], r['year_built_hi']
    year = int(rng.integers(int(max(lo, 1900)), int(hi) + 1)) if pd.notna(lo) and pd.notna(hi) else 1985
    sysn = CB_SYS.get(str(r['system']), 'packaged_rtu')
    hf = str(r['heating_fuel'])
    h = {}
    if hf == 'electricity' and sysn in ('packaged_rtu', 'vav_chiller_boiler', 'ptac'):
        h['heat_source'] = 'elec'
    if hf == 'district' and sysn not in ('district_steam_chw',):
        h['heat_source'] = 'district'
    hpw = float(r['hours_per_week']) if pd.notna(r['hours_per_week']) else 50.0
    from .schedules import DEFAULT_SCHEDULES
    import copy
    sch = copy.deepcopy(DEFAULT_SCHEDULES[use])
    if sch.get('type') == 'commercial':
        if int(r.get('open24', 0) or 0) == 1 or hpw >= 160:
            sch = dict(type='24x7', day=(7, 19), night_occ=0.4, night_light=0.5, night_plug=0.7, weekend=0.8)
        else:
            days = 5 if hpw <= 60 else (6 if hpw <= 84 else 7)
            d = min(hpw / days, 23.5)
            o = 8.0 if d < 12 else max(5.0, 14.0 - d / 2)
            sch['weekday'] = (o, o + d)
            sch['saturday'] = (o, o + d) if days >= 6 else None
            sch['sunday'] = (o, o + d) if days >= 7 else None
    wwr = r['window_wwr_mid'] if pd.notna(r.get('window_wwr_mid')) else 0.2
    spec = dict(id=str(r['id']), primary_use=use, floor_area_m2=float(r['floor_area_m2']),
                n_floors=int(r['n_floors']) if pd.notna(r['n_floors']) else 1, year_built=year,
                window_wwr=float(wwr), system=sysn, hvac=h, occupancy_schedule=sch,
                provenance=dict(source='CBECS 2018 prior split (real survey respondent)', id=str(r['id'])))
    if str(r.get('window_type')) == 'single':
        spec['window_U'], spec['SHGC'] = 5.8, 0.7
    if str(r.get('wall_material')) in ('masonry', 'concrete'):
        spec['thermal_mass'] = 'heavy'
    return spec


# ---------------------------------------------------------------- stock weather (AMY 2018 county files on OEDI)
def read_stock_weather(path, lat, lon, std_offset_h, elev=0.0, tz=None):
    """ResStock/ComStock county weather CSV (hour-ending, local standard time) -> contract frame + meta."""
    from . import psychro as ps
    d = pd.read_csv(path)
    t = pd.to_datetime(d['date_time']) - pd.Timedelta(hours=1)          # interval start
    idx = pd.DatetimeIndex(t - pd.Timedelta(hours=std_offset_h)).tz_localize('UTC')
    T = d.iloc[:, 1].values.astype(float)
    RH = d.iloc[:, 2].values.astype(float)

    def pair(x):
        return 0.5 * (x + np.r_[x[0], x[:-1]])
    Ti, RHi = pair(T), np.clip(pair(RH), 1, 100)
    P = float(ps.pressure_from_elevation(elev))
    W = ps.w_from_rh(Ti, RHi, P)
    df = pd.DataFrame(dict(T_out=Ti, RH=RHi, dewpoint=ps.dewpoint_from_w(W, P), wind_speed=pair(d.iloc[:, 3].values),
                           wind_dir=d.iloc[:, 4].values, GHI=d.iloc[:, 5].values, DNI=d.iloc[:, 6].values,
                           DHI=d.iloc[:, 7].values, pressure=P), index=idx)
    tzname = tz or {-5: 'America/New_York', -6: 'America/Chicago', -7: 'America/Denver', -8: 'America/Los_Angeles',
                    -9: 'America/Anchorage', -10: 'Pacific/Honolulu'}.get(int(std_offset_h), 'UTC')
    return df, dict(lat=lat, lon=lon, elev=elev, tz=tzname, source='OEDI AMY2018 ' + str(path).split('/')[-1])


def read_stock_weather_parquet(path, lat, lon, elev=0.0):
    """Agent B's county weather parquet (time_lst hour-beginning, values as published hour-ending) -> contract."""
    from . import psychro as ps
    w = pd.read_parquet(path)
    off = float(w['std_utc_offset_h'].iloc[0])
    idx = pd.DatetimeIndex(w.index - pd.Timedelta(hours=off)).tz_localize('UTC')

    def pair(x):
        x = np.asarray(x, float)
        return 0.5 * (x + np.r_[x[0], x[:-1]])
    T = pair(w['T_out'])
    RH = np.clip(pair(w['RH']), 1, 100)
    P = float(ps.pressure_from_elevation(elev))
    W = ps.w_from_rh(T, RH, P)
    df = pd.DataFrame(dict(T_out=T, RH=RH, dewpoint=ps.dewpoint_from_w(W, P), wind_speed=pair(w['wind_speed']),
                           wind_dir=w['wind_dir'].values, GHI=w['GHI'].values, DNI=w['DNI'].values,
                           DHI=w['DHI'].values, pressure=P), index=idx)
    tz = {-5: 'America/New_York', -6: 'America/Chicago', -7: 'America/Denver', -8: 'America/Los_Angeles',
          -9: 'America/Anchorage', -10: 'Pacific/Honolulu'}.get(int(off), 'UTC')
    return df, dict(lat=lat, lon=lon, elev=elev, tz=tz, std_offset=off, source='NREL AMY2018 county ' +
                    str(path).split('/')[-1])
