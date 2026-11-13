"""Vehicle spec (shared contract) -> defaults -> compiled Vehicle object.

A spec is a plain dict / JSON. Missing fields are filled from class- and architecture-dependent defaults so a minimal
spec such as {"class": "car", "architecture": "bev"} runs. `load_vehicle` accepts a dict, a path to a .json, or the id of
a vehicle in roadsim/data/vehicles/*.json.
"""
import os, json, copy, math
import numpy as np
from .constants import LBF, MPH, LB
from .components import Engine, Motor, Battery, TorqueConverter, Cabin

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CLASSES = ['car', 'suv', 'pickup', 'van', 'bus', 'truck_md', 'truck_hd']
ARCHS = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev']

# Representative class defaults. Ranges used by the generator live in generator.py.
CLASS_DEFAULTS = {
    'car':      dict(mass_kg=1650, CdA_m2=0.62, Crr=0.0085, Crr_speed_coef=0.004, wheel_radius_m=0.32, driveline_eff=0.975, rot_frac=0.025, power_kW=140, aux_W=350, UA=120, Ccab=9e4, solar=0.45, yaw_k=0.0007),
    'suv':      dict(mass_kg=2000, CdA_m2=0.85, Crr=0.0090, Crr_speed_coef=0.004, wheel_radius_m=0.355, driveline_eff=0.97, rot_frac=0.025, power_kW=170, aux_W=400, UA=150, Ccab=1.2e5, solar=0.55, yaw_k=0.0008),
    'pickup':   dict(mass_kg=2450, CdA_m2=1.10, Crr=0.0095, Crr_speed_coef=0.004, wheel_radius_m=0.39, driveline_eff=0.965, rot_frac=0.03, power_kW=260, aux_W=450, UA=160, Ccab=1.3e5, solar=0.6, yaw_k=0.0010),
    'van':      dict(mass_kg=2600, CdA_m2=1.25, Crr=0.0090, Crr_speed_coef=0.004, wheel_radius_m=0.37, driveline_eff=0.965, rot_frac=0.03, power_kW=200, aux_W=500, UA=220, Ccab=2.0e5, solar=0.8, yaw_k=0.0010),
    'bus':      dict(mass_kg=15000, CdA_m2=6.0, Crr=0.0070, Crr_speed_coef=0.002, wheel_radius_m=0.50, driveline_eff=0.96, rot_frac=0.05, power_kW=220, aux_W=3500, UA=900, Ccab=1.5e6, solar=4.0, yaw_k=0.0015),
    'truck_md': dict(mass_kg=10000, CdA_m2=5.0, Crr=0.0072, Crr_speed_coef=0.002, wheel_radius_m=0.48, driveline_eff=0.96, rot_frac=0.05, power_kW=200, aux_W=1500, UA=160, Ccab=1.5e5, solar=0.7, yaw_k=0.0018),
    'truck_hd': dict(mass_kg=30000, CdA_m2=5.9, Crr=0.0062, Crr_speed_coef=0.0015, wheel_radius_m=0.50, driveline_eff=0.965, rot_frac=0.04, power_kW=340, aux_W=2500, UA=180, Ccab=2.0e5, solar=0.8, yaw_k=0.0025),
}

GEAR_SETS = {
    ('AT', 6): [4.15, 2.37, 1.56, 1.16, 0.86, 0.69],
    ('AT', 8): [4.71, 3.14, 2.11, 1.67, 1.29, 1.00, 0.84, 0.67],
    ('AT', 9): [4.71, 2.84, 1.91, 1.38, 1.00, 0.81, 0.70, 0.58, 0.48],
    ('AT', 10): [4.70, 2.99, 2.15, 1.80, 1.52, 1.28, 1.00, 0.85, 0.69, 0.64],
    ('DCT', 6): [3.92, 2.43, 1.45, 1.03, 0.77, 0.63],
    ('DCT', 7): [3.64, 2.26, 1.48, 1.03, 0.85, 0.67, 0.56],
    ('DCT', 8): [4.66, 2.95, 1.94, 1.39, 1.09, 0.87, 0.69, 0.56],
    ('AMT', 12): [12.8, 9.98, 7.73, 6.01, 4.67, 3.63, 2.82, 2.19, 1.70, 1.32, 1.00, 0.78],
    ('AMT', 10): [12.8, 9.25, 6.76, 4.90, 3.58, 2.61, 1.89, 1.38, 1.00, 0.73],
    ('AT_HD', 6): [3.51, 1.91, 1.43, 1.00, 0.75, 0.64],     # Allison B400R / 3000-series type
    ('AT_HD', 5): [3.49, 1.86, 1.41, 1.00, 0.75],
}

ENGINE_DEFAULTS = {
    'gasoline': 'gas_na_toyota_a25a',
    'gasoline_turbo': 'gas_turbo_ford_27',
    'gasoline_small_turbo': 'gas_turbo_honda_15',
    'gasoline_atkinson': 'gas_atk_toyota_tnga_hev_paper',
    'gasoline_v8': 'gas_v8_ford_73',
    'diesel': 'diesel_gm_30',
    'diesel_hd': 'willans_hd_diesel',
}


def deep_update(d, u):
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def abc_to_si(A, B, C):
    """EPA target coefficients (lbf, lbf/mph, lbf/mph^2) -> SI (N, N/(m/s), N/(m/s)^2)."""
    return A * LBF, B * LBF / MPH, C * LBF / MPH ** 2


def default_spec(cls='car', arch='conv'):
    c = CLASS_DEFAULTS[cls]
    hd = cls in ('bus', 'truck_md', 'truck_hd')
    P = c['power_kW']
    s = dict(id='default_%s_%s' % (cls, arch), **{'class': cls}, architecture=arch,
             mass_kg=c['mass_kg'], payload_kg=0.0, CdA_m2=c['CdA_m2'], Crr=c['Crr'], Crr_speed_coef=c['Crr_speed_coef'],
             yaw_drag_coef=c['yaw_k'], rot_mass_frac=c['rot_frac'], wheel_radius_m=c['wheel_radius_m'],
             final_drive=None, driveline_eff=c['driveline_eff'], tire={'pressure_ratio': 1.0, 'K_t': 0.010 if hd else 0.008},
             engine=None, transmission=None, motors=[], battery=None,
             aux={'base_W': c['aux_W'], 'HVAC': {'UA_W_K': c['UA'], 'C_J_K': c['Ccab'], 'solar_area_m2': c['solar'],
                                                  'heat_max_W': 6000 if not hd else (30000 if cls == 'bus' else 8000),
                                                  'cool_max_W': 5000 if not hd else (35000 if cls == 'bus' else 7000),
                                                  'blower_W': 150 if not hd else (1500 if cls == 'bus' else 300),
                                                  'occupant_W': 100 if cls != 'bus' else 2000}},
             regen={'blend': 'series', 'max_decel_g': 0.30 if not hd else 0.15, 'min_speed_mps': 2.0, 'max_share': 1.0},
             control={'strategy': None, 'params': {}})
    fuel = 'diesel' if hd else 'gasoline'
    if arch == 'conv':
        if hd:
            s['engine'] = dict(fuel='diesel', displacement_L=15.0 if cls == 'truck_hd' else (8.9 if cls == 'bus' else 6.7),
                               max_power_kW=P, bsfc_map='willans_hd_diesel', idle_rpm=600)
            s['transmission'] = dict(type='AMT', gears=GEAR_SETS[('AMT', 12)]) if cls == 'truck_hd' else dict(type='AT', gears=GEAR_SETS[('AT_HD', 6)])
        else:
            s['engine'] = dict(fuel='gasoline', displacement_L=round(P / 60.0, 1), max_power_kW=P, bsfc_map=ENGINE_DEFAULTS['gasoline'])
            s['transmission'] = dict(type='AT', gears=GEAR_SETS[('AT', 8)])
    elif arch in ('hev_parallel_p2',):
        Pe = P * 0.8
        s['engine'] = dict(fuel=fuel, displacement_L=round(Pe / 55.0, 1) if not hd else 6.7, max_power_kW=Pe,
                           bsfc_map='willans_hd_diesel' if hd else 'gas_atk_toyota_tnga_hev_paper')
        s['transmission'] = dict(type='DCT' if not hd else 'AT', gears=GEAR_SETS[('DCT', 6)] if not hd else GEAR_SETS[('AT_HD', 6)])
        s['motors'] = [dict(role='P2', max_power_kW=max(30.0, 0.25 * P), eff_map='ipm_generic_150', max_rpm=6500)]
        s['battery'] = dict(chemistry='NMC', capacity_kWh=1.5 if not hd else 12.0, nominal_V=270 if not hd else 600, soc_window=[0.3, 0.8])
    elif arch in ('hev_series', 'phev_series'):
        Pe = P * (0.7 if arch == 'hev_series' else 0.5)
        s['engine'] = dict(fuel=fuel, displacement_L=round(Pe / 55.0, 1) if not hd else 6.7, max_power_kW=Pe,
                           bsfc_map='willans_hd_diesel' if hd else 'gas_atk_toyota_tnga_hev_paper')
        s['transmission'] = dict(type='single')
        s['motors'] = [dict(role='traction', max_power_kW=P, eff_map='ipm_generic_150'),
                       dict(role='generator', max_power_kW=Pe * 1.05, eff_map='ipm_generic_150')]
        cap = (1.5 if not hd else 30.0) if arch == 'hev_series' else (16.0 if not hd else 150.0)
        s['battery'] = dict(chemistry='NMC', capacity_kWh=cap, nominal_V=350 if not hd else 650,
                            soc_window=[0.3, 0.8] if arch == 'hev_series' else [0.15, 0.95])
    elif arch in ('hev_powersplit', 'phev_powersplit'):
        Pe = P * 0.75
        s['engine'] = dict(fuel='gasoline', displacement_L=round(Pe / 55.0, 1), max_power_kW=Pe, bsfc_map='gas_atk_toyota_tnga_hev_paper')
        s['transmission'] = dict(type='eCVT', ring_teeth=78, sun_teeth=30, mg2_ratio=2.636)
        s['motors'] = [dict(role='MG1', max_power_kW=0.6 * Pe, max_torque_Nm=0.6 * Pe * 4.0, eff_map='ipm_generic_150', max_rpm=13500),
                       dict(role='MG2', max_power_kW=0.75 * P, eff_map='ipm_generic_150', max_rpm=13500)]
        cap = 1.3 if arch == 'hev_powersplit' else 13.0
        s['battery'] = dict(chemistry='NiMH' if arch == 'hev_powersplit' else 'NMC', capacity_kWh=cap, nominal_V=245 if arch == 'hev_powersplit' else 350,
                            soc_window=[0.4, 0.8] if arch == 'hev_powersplit' else [0.15, 0.95])
    elif arch == 'bev':
        s['transmission'] = dict(type='single')
        s['motors'] = [dict(role='traction', max_power_kW=P, eff_map='ipm_generic_150')]
        cap = {'car': 70, 'suv': 85, 'pickup': 130, 'van': 90, 'bus': 450, 'truck_md': 200, 'truck_hd': 700}[cls]
        s['battery'] = dict(chemistry='NMC' if not hd else 'LFP', capacity_kWh=cap, nominal_V=380 if not hd else 650,
                            soc_window=[0.02, 0.98], thermal={'heater_W': 3000 if not hd else 8000, 'cooler_W': 2000 if not hd else 6000})
    if arch != 'conv':
        s['aux']['HVAC']['source'] = ('heat_pump' if arch == 'bev' and not hd else
                                      'ptc' if arch in ('bev', 'phev_series', 'phev_powersplit') else 'engine')
    return s


def fill_spec(spec):
    """Return a complete spec: defaults for the class/architecture, overlaid with the given fields."""
    spec = copy.deepcopy(spec)
    cls = spec.get('class', 'car'); arch = spec.get('architecture', 'conv')
    assert cls in CLASSES, cls
    assert arch in ARCHS, arch
    full = default_spec(cls, arch)
    # sub-dicts given explicitly replace the defaults except for missing keys
    for k, v in spec.items():
        if k in ('engine', 'transmission', 'battery') and isinstance(v, dict) and isinstance(full.get(k), dict):
            full[k] = deep_update(full[k], v)
        elif k == 'motors' and v:
            dm = {m['role']: m for m in full.get('motors', [])}
            full['motors'] = [deep_update(copy.deepcopy(dm.get(m.get('role'), {})), m) for m in v]
        elif isinstance(v, dict) and isinstance(full.get(k), dict):
            full[k] = deep_update(full[k], v)
        else:
            full[k] = copy.deepcopy(v)
    rl = full.get('road_load_ABC')
    if rl and 'A_N' not in rl and 'A_lbf' in rl:
        A, B, C = abc_to_si(rl['A_lbf'], rl['B_lbf_mph'], rl['C_lbf_mph2'])
        rl.update(A_N=A, B_N_per_mps=B, C_N_per_mps2=C)
    tr = full.get('transmission') or {}
    if tr.get('type') in ('AT', 'DCT', 'AMT') and not tr.get('gears'):
        n = tr.get('n_gears', 8)
        key = (tr['type'], n)
        tr['gears'] = GEAR_SETS.get(key) or list(np.round(np.geomspace(4.6, 0.65, n), 3))
    if tr.get('type') == 'CVT':
        tr.setdefault('ratio_range', [2.6, 0.40])
    if full.get('final_drive') is None:
        full['final_drive'] = _default_final_drive(full)
    return full


def _default_final_drive(s):
    """final drive such that the engine/motor runs at a sensible speed at the class's typical cruising speed."""
    r = s['wheel_radius_m']
    tr = s.get('transmission') or {}
    t = tr.get('type')
    hd = s['class'] in ('bus', 'truck_md', 'truck_hd')
    if t in ('AT', 'DCT', 'AMT'):
        top = tr['gears'][-1]
        v = (65 if s['class'] == 'truck_hd' else 55 if hd else 62) * MPH
        rpm = 1150 if s['class'] == 'truck_hd' else (1700 if hd else 1650 if len(tr['gears']) >= 8 else 2000)
        return round(rpm * math.pi / 30 * r / (v * top), 3)
    if t == 'CVT':
        return round(1550 * math.pi / 30 * r / (62 * MPH * tr['ratio_range'][1]), 3)
    if t == 'eCVT':
        return 3.27 if not hd else 4.0
    # single speed: motor max speed at vehicle top speed
    m = [mm for mm in s['motors'] if mm.get('role') in ('traction', 'MG2', 'P2')]
    vtop = {'car': 190, 'suv': 180, 'pickup': 170, 'van': 150, 'bus': 105, 'truck_md': 120, 'truck_hd': 120}[s['class']] / 3.6
    if m and m[0].get('max_rpm'):
        wmax = m[0]['max_rpm'] * math.pi / 30
    else:
        from .maps import load_maps
        _, meta = load_maps()
        wmax = meta['motors'][(m[0].get('eff_map') if m else None) or 'ipm_generic_150']['w_max']
    return round(wmax * r / vtop, 3)


def load_vehicle(v):
    """dict | path to json | id in data/vehicles -> complete spec dict."""
    if isinstance(v, str):
        p = v if os.path.exists(v) else os.path.join(DATA, 'vehicles', v + '.json')
        v = json.load(open(p))
    return fill_spec(v)


class Vehicle:
    """Compiled vehicle: components built once, reused across trips."""

    def __init__(self, spec):
        s = fill_spec(spec)
        self.spec = s
        self.id = s.get('id')
        self.cls = s['class']; self.arch = s['architecture']
        self.hd = self.cls in ('bus', 'truck_md', 'truck_hd')
        self.mass = float(s['mass_kg']) + float(s.get('payload_kg', 0.0) or 0.0)
        self.m_eff = self.mass * (1.0 + s.get('rot_mass_frac', 0.025))
        self.r = s['wheel_radius_m']; self.FD = s['final_drive']; self.eta_dl = s['driveline_eff']
        rl = s.get('road_load_ABC')
        # road load: if EPA ABC is given and use_abc, A/B/C define rolling + aero directly
        self.CdA = s['CdA_m2']; self.Crr = s['Crr']; self.Crr_v = s['Crr_speed_coef']
        self.yaw_k = s.get('yaw_drag_coef', 0.0008)
        self.parasitic_N = s.get('parasitic_N', 0.0)
        t = s.get('tire', {}) or {}
        self.tire_K_t = t.get('K_t', 0.008); self.tire_p_ratio = t.get('pressure_ratio', 1.0)
        self.tire_warm_amp = t.get('warmup_amp', 0.15); self.tire_warm_tau = t.get('warmup_tau_s', 1200.0)
        self.tire_temp_effects = t.get('temp_effects', True)
        self.engine = Engine(s['engine']) if s.get('engine') else None
        if self.engine is not None and self.arch != 'conv' and 'cold_fric_Nm_per_L' not in (s['engine'].get('warmup_model') or {}):
            from . import calib as _cal
            self.engine.c_fric *= _cal.get('cold_fric_hybrid_factor')
        self.fuel = (s['engine'] or {}).get('fuel', 'none') if s.get('engine') else 'none'
        from . import calib
        self.motors = {}
        for m in s.get('motors', []):
            if self.arch == 'bev' and 'loss_scale' not in m:
                m = dict(m, loss_scale=calib.get('bev_motor_loss_scale') * calib.get('motor_loss_scale'))
            self.motors[m['role']] = Motor(m)
        self.battery = Battery(s['battery']) if s.get('battery') else None
        self.trans = s.get('transmission') or {'type': 'single'}
        self.ttype = self.trans.get('type', 'single')
        self.gears = list(self.trans.get('gears', [1.0]))
        self.tc = None
        if self.engine is not None and self.ttype in ('AT', 'CVT') and self.arch in ('conv', 'hev_parallel_p2'):
            stall = 2300.0 if self.engine.fuel == 'gasoline' else 1700.0
            if self.hd: stall = 1600.0
            self.tc = TorqueConverter(self.engine.T_peak, stall)
        a = s.get('aux', {}) or {}
        self.aux_base = a.get('base_W', 400.0)
        hv = dict(a.get('HVAC', {}) or {})
        src = hv.get('source', 'engine' if self.engine is not None else 'ptc')
        self.cabin = Cabin(hv, src)
        self.regen = s.get('regen', {}) or {}
        self.control = s.get('control', {}) or {}

    def road_load_coeffs(self, rho=1.2):
        """equivalent A [N], B [N/(m/s)], C [N/(m/s)^2] on flat road, 20 C, no wind."""
        A = self.mass * 9.80665 * self.Crr + self.parasitic_N
        B = self.mass * 9.80665 * self.Crr * self.Crr_v
        C = 0.5 * rho * self.CdA
        return A, B, C


def spec_from_abc(spec, A_lbf, B_lbf_mph, C_lbf_mph2, rho_ref=1.1767, mass_kg=None):
    """Set CdA, Crr and Crr_speed_coef so the spec reproduces EPA target road-load coefficients at the reference air
    density (SAE J2263/J1263 standard conditions 20 C, 98.21 kPa -> 1.167 kg/m3; EPA uses ~1.1767 at 20 C 1 atm)."""
    A, B, C = abc_to_si(A_lbf, B_lbf_mph, C_lbf_mph2)
    m = mass_kg or spec['mass_kg']
    s = copy.deepcopy(spec)
    s['CdA_m2'] = max(C / (0.5 * rho_ref), 0.2)
    s['Crr'] = max(A / (m * 9.80665), 0.003)
    s['Crr_speed_coef'] = B / (m * 9.80665 * s['Crr'])
    s['road_load_ABC'] = dict(A_lbf=A_lbf, B_lbf_mph=B_lbf_mph, C_lbf_mph2=C_lbf_mph2, A_N=A, B_N_per_mps=B, C_N_per_mps2=C)
    return s
