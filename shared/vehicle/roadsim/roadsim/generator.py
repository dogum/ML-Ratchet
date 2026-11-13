"""Random but physically consistent vehicle library.

Light duty (car, suv, pickup, van): each vehicle starts from a REAL EPA test-car record of the same class and fuel group
(test weight, effective CdA and Crr from the target road-load coefficients, rated power, transmission), drawn at random
and jittered a little, so the joint distribution (heavy vehicles have big frontal areas and more power, ...) is that of
the real fleet. Battery energy for EVs is drawn from fueleconomy.gov (range x consumption). Hybrid hardware is sized
from the rated power with ratios seen in production hybrids (README).
Heavy duty (bus, truck_md, truck_hd): uniform/lognormal draws inside ranges from 40 CFR 1037 (standard payloads, GEM
inputs) and published specifications (README); no public per-vehicle road-load data was available.

    from roadsim.generator import generate_vehicles
    specs = generate_vehicles(1000, seed=1)              # mixed classes and architectures
    specs = generate_vehicles(50, seed=2, cls='bus', arch='bev')
"""
import os, json, math, hashlib
import numpy as np
from .vehicle import GEAR_SETS, fill_spec
from .strategies import STRATEGIES

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_CAL = None

CLASS_W = {'car': 0.28, 'suv': 0.30, 'pickup': 0.14, 'van': 0.07, 'bus': 0.07, 'truck_md': 0.07, 'truck_hd': 0.07}
ARCH_W = {
    'car': {'conv': 0.42, 'hev_powersplit': 0.12, 'hev_parallel_p2': 0.07, 'hev_series': 0.05, 'phev_powersplit': 0.06, 'phev_series': 0.04, 'bev': 0.24},
    'suv': {'conv': 0.42, 'hev_powersplit': 0.12, 'hev_parallel_p2': 0.08, 'hev_series': 0.05, 'phev_powersplit': 0.06, 'phev_series': 0.03, 'bev': 0.24},
    'pickup': {'conv': 0.55, 'hev_powersplit': 0.05, 'hev_parallel_p2': 0.12, 'phev_series': 0.05, 'bev': 0.23},
    'van': {'conv': 0.50, 'hev_powersplit': 0.18, 'phev_powersplit': 0.10, 'bev': 0.22},
    'bus': {'conv': 0.40, 'hev_series': 0.22, 'hev_parallel_p2': 0.10, 'bev': 0.28},
    'truck_md': {'conv': 0.55, 'hev_parallel_p2': 0.15, 'phev_series': 0.05, 'bev': 0.25},
    'truck_hd': {'conv': 0.70, 'hev_parallel_p2': 0.10, 'bev': 0.20},
}
# strategy mix for generated vehicles (dp/mpc excluded: they need the trip in advance / are slow)
STRAT_W = {
    'hev_series': {'aecms': 0.4, 'series_power_follower': 0.3, 'series_thermostat': 0.15, 'ecms': 0.15},
    'hev_powersplit': {'aecms': 0.5, 'powersplit_rule': 0.35, 'ecms': 0.15},
    'hev_parallel_p2': {'aecms': 0.5, 'parallel_rule': 0.35, 'ecms': 0.15},
    'phev_series': {'phev_cdcs': 0.6, 'phev_blended': 0.4},
    'phev_powersplit': {'phev_cdcs': 0.6, 'phev_blended': 0.4},
    'bev': {'bev_regen_series': 0.6, 'bev_regen_parallel': 0.2, 'bev_one_pedal': 0.2},
    'conv': {'conv_rule': 0.7, 'conv_opt': 0.15, 'conv_stopstart': 0.15},
}


def _cal():
    global _CAL
    if _CAL is None:
        _CAL = json.load(open(os.path.join(DATA, 'generator_calib.json')))
    return _CAL


def _pick(rng, w):
    k = list(w); p = np.array([w[x] for x in k], float); p /= p.sum()
    return k[int(rng.choice(len(k), p=p))]


def _ln(rng, s):
    return float(np.exp(rng.normal(0.0, s)))


def _engine_map(fuel, P_kW, disp, hybrid, rng):
    if fuel == 'diesel':
        return 'diesel_gm_30' if rng.random() < 0.6 else 'diesel_bmw_n57'
    if hybrid:
        return 'gas_atk_toyota_tnga_hev_paper' if rng.random() < 0.8 else 'gas_atk_toyota_tnga_hev'
    if P_kW / max(disp, 0.5) > 80:
        return 'gas_turbo_honda_15' if disp < 2.2 else 'gas_turbo_ford_27'
    if disp >= 4.5:
        return 'gas_v8_ford_73'
    if disp >= 3.0:
        return 'gas_v6_gm_43'
    return 'gas_na_toyota_a25a' if rng.random() < 0.7 else 'gas_na_mazda_20'


def _ld(rng, cls, arch):
    cal = _cal()
    grp = 'bev' if arch == 'bev' else 'gasoline'
    if arch == 'conv' and cls in ('pickup', 'suv') and rng.random() < 0.06:
        grp = 'diesel'
    key = f'{cls}|{grp}'
    if key not in cal['testcar'] or len(cal['testcar'][key]['mass_kg']) < 20:
        key = f'suv|{grp}' if f'suv|{grp}' in cal['testcar'] else f'suv|gasoline'
    T = cal['testcar'][key]
    i = int(rng.integers(len(T['mass_kg'])))
    row = {k: T[k][i] for k in T}
    mass = row['mass_kg'] * _ln(rng, 0.03)
    if arch in ('phev_powersplit', 'phev_series'):
        mass *= 1.08                                    # pack mass relative to the gasoline donor record
    P = row['P_kW'] if row['P_kW'] == row['P_kW'] else 150.0
    P = float(np.clip(P * _ln(rng, 0.06), 60, 600))
    disp = row['disp_L'] if row['disp_L'] and row['disp_L'] < 9 else round(P / 65, 1)
    s = {'class': cls, 'architecture': arch, 'mass_kg': round(mass, 1),
         'payload_kg': round(float(rng.uniform(0, 250 if cls in ('car', 'suv') else 450)), 1),
         'CdA_m2': round(row['CdA'] * _ln(rng, 0.04), 4), 'Crr': round(row['Crr'] * _ln(rng, 0.06), 5),
         'Crr_speed_coef': round(float(np.clip(row['Crr_v'], -0.01, 0.02)), 5),
         'wheel_radius_m': round(float({'car': 0.32, 'suv': 0.36, 'pickup': 0.39, 'van': 0.36}[cls] * _ln(rng, 0.03)), 3),
         'generator': {'donor': key, 'donor_row': i, 'donor_year': row.get('year')}}
    hyb = arch not in ('conv', 'bev')
    if arch == 'conv':
        fuel = 'diesel' if grp == 'diesel' else 'gasoline'
        s['engine'] = dict(fuel=fuel, displacement_L=disp, max_power_kW=round(P, 1), bsfc_map=_engine_map(fuel, P, disp, False, rng),
                           eff_scale=round(_ln(rng, 0.03), 3), idle_stop=bool(rng.random() < 0.3))
        tr = row['trans']; n = row['n_gears']
        if tr == 'CVT':
            s['transmission'] = dict(type='CVT', ratio_range=[round(float(rng.uniform(2.3, 2.8)), 3), round(float(rng.uniform(0.38, 0.45)), 3)])
        else:
            if (tr, n) not in GEAR_SETS: n = 8 if tr == 'AT' else 7
            if (tr, n) not in GEAR_SETS: tr = 'AT'
            s['transmission'] = dict(type=tr, n_gears=n, gears=GEAR_SETS[(tr, n)])
    elif arch == 'bev':
        ev = cal['ev_battery'].get(cls) or cal['ev_battery']['suv']
        cap = float(np.clip(ev['batt_kWh'][int(rng.integers(len(ev['batt_kWh'])))] * _ln(rng, 0.05), 25, 220))
        V = 800.0 if rng.random() < 0.25 else 380.0
        s['motors'] = [dict(role='traction', max_power_kW=round(P, 1), eff_map='bolt_edu_150' if rng.random() < 0.3 else 'ipm_generic_150',
                            max_rpm=round(float(rng.uniform(12000, 18000))))]
        s['battery'] = dict(chemistry='LFP' if rng.random() < 0.3 else 'NMC', capacity_kWh=round(cap, 1), nominal_V=V, soc_window=[0.02, 0.98],
                            thermal={'heater_W': 3000 if rng.random() < 0.7 else 0, 'cooler_W': 2000})
        s['aux'] = {'HVAC': {'source': 'heat_pump' if rng.random() < 0.5 else 'ptc'}}
    else:
        Pe = P * float(rng.uniform(0.65, 0.85))
        s['engine'] = dict(fuel='gasoline', displacement_L=round(min(disp, Pe / 50), 1), max_power_kW=round(Pe, 1),
                           bsfc_map=_engine_map('gasoline', Pe, disp, arch != 'hev_parallel_p2' or rng.random() < 0.5, rng),
                           eff_scale=round(_ln(rng, 0.03), 3))
        plug = arch.startswith('phev')
        if arch in ('hev_powersplit', 'phev_powersplit'):
            s['transmission'] = dict(type='eCVT', ring_teeth=78, sun_teeth=30, mg2_ratio=round(float(rng.uniform(2.6, 3.4)), 3))
            s['final_drive'] = round(float(rng.uniform(2.9, 3.7)), 3)
            s['motors'] = [dict(role='MG1', max_power_kW=round(Pe * float(rng.uniform(0.45, 0.65)), 1), eff_map='ipm_generic_150', max_rpm=13500),
                           dict(role='MG2', max_power_kW=round(P * float(rng.uniform(0.55, 0.8)), 1), eff_map='ipm_generic_150', max_rpm=17000)]
        elif arch in ('hev_series', 'phev_series'):
            s['transmission'] = dict(type='single')
            s['motors'] = [dict(role='traction', max_power_kW=round(P * float(rng.uniform(0.85, 1.0)), 1), eff_map='ipm_generic_150', max_rpm=14000),
                           dict(role='generator', max_power_kW=round(Pe * 1.05, 1), eff_map='ipm_generic_150', max_rpm=14000)]
        else:
            n = int(rng.choice([6, 8])) if cls in ('car', 'suv') else 10
            tr = 'DCT' if (cls in ('car', 'suv') and n == 6) else 'AT'
            if (tr, n) not in GEAR_SETS: tr, n = 'AT', 8
            s['transmission'] = dict(type=tr, n_gears=n, gears=GEAR_SETS[(tr, n)])
            s['motors'] = [dict(role='P2', max_power_kW=round(float(rng.uniform(30, 60)), 1), eff_map='ipm_generic_150', max_rpm=7000)]
        if plug:
            cap = float(rng.uniform(8, 25)); V = 350.0
            s['battery'] = dict(chemistry='NMC', capacity_kWh=round(cap, 1), nominal_V=V, soc_window=[0.15, 0.95])
        else:
            cap = float(rng.uniform(0.8, 1.9)); V = float(rng.choice([220.0, 245.0, 260.0, 280.0]))
            s['battery'] = dict(chemistry='NiMH' if rng.random() < 0.3 else 'NMC', capacity_kWh=round(cap, 2), nominal_V=V,
                                soc_window=[0.35, 0.8] if arch == 'hev_powersplit' else [0.3, 0.8])
    s.setdefault('aux', {})['base_W'] = round(float(rng.uniform(250, 600)), 0)
    s['aux'].setdefault('HVAC', {}).update(UA_W_K=round({'car': 120, 'suv': 150, 'pickup': 160, 'van': 220}[cls] * _ln(rng, 0.15), 1),
                                           T_heat_set=round(float(rng.uniform(19, 23)), 1), T_cool_set=round(float(rng.uniform(20, 24)), 1))
    return s


def _hd(rng, cls, arch):
    """heavy duty: ranges from 40 CFR 1037 (payloads, GEM inputs) and published specs (README)."""
    if cls == 'bus':
        curb = rng.uniform(12000, 14500); pay = rng.uniform(0, 6000); CdA = rng.uniform(5.5, 7.0); Crr = rng.uniform(0.0060, 0.0085)
        disp = float(rng.choice([6.7, 8.9])); P = rng.uniform(190, 260); r = 0.50
        gears = GEAR_SETS[('AT_HD', 6)]; ttype = 'AT'; base = rng.uniform(3000, 7000); bat = rng.uniform(300, 660)
        if arch == 'bev': curb += 1500
    elif cls == 'truck_md':
        curb = rng.uniform(6000, 9000); pay = rng.uniform(0, 7000); CdA = rng.uniform(4.5, 6.5); Crr = rng.uniform(0.0065, 0.0090)
        disp = 6.7; P = rng.uniform(190, 260); r = 0.47; gears = [3.10, 1.81, 1.41, 1.00, 0.71, 0.61]; ttype = 'AT'
        base = rng.uniform(1000, 2500); bat = rng.uniform(150, 300)
        if arch == 'bev': curb += 1200
    else:
        curb = rng.uniform(14000, 16500); pay = rng.uniform(0, 20000); CdA = rng.uniform(4.8, 6.8); Crr = rng.uniform(0.0050, 0.0075)
        disp = float(rng.choice([11.0, 13.0, 15.0])); P = rng.uniform(300, 450); r = 0.505
        gears = GEAR_SETS[('AMT', 12)] if rng.random() < 0.7 else GEAR_SETS[('AMT', 10)]; ttype = 'AMT'
        base = rng.uniform(1500, 3500); bat = rng.uniform(500, 1000)
        if arch == 'bev': curb += 3500
    s = {'class': cls, 'architecture': arch, 'mass_kg': round(float(curb), 1), 'payload_kg': round(float(pay), 1),
         'CdA_m2': round(float(CdA), 3), 'Crr': round(float(Crr), 5), 'Crr_speed_coef': 0.0015, 'wheel_radius_m': r,
         'aux': {'base_W': round(float(base))}, 'generator': {'donor': 'hd_ranges'}}
    if arch == 'conv' or arch.startswith('hev') or arch.startswith('phev'):
        Pe = P if arch == 'conv' else P * float(rng.uniform(0.75, 0.95))
        s['engine'] = dict(fuel='diesel', displacement_L=disp, max_power_kW=round(float(Pe), 1), bsfc_map='willans_hd_diesel',
                           eff_scale=round(_ln(rng, 0.025), 3), idle_rpm=600 if cls == 'truck_hd' else 650)
    if arch == 'conv' or arch == 'hev_parallel_p2':
        s['transmission'] = dict(type=ttype, gears=gears)
    if arch == 'hev_parallel_p2':
        s['motors'] = [dict(role='P2', max_power_kW=round(float(rng.uniform(80, 200)), 1), eff_map='ipm_generic_150', max_rpm=3500)]
        s['battery'] = dict(chemistry='NMC', capacity_kWh=round(float(rng.uniform(10, 40)), 1), nominal_V=650, soc_window=[0.3, 0.8],
                            max_power_kW=200)
    elif arch in ('hev_series', 'phev_series'):
        s['transmission'] = dict(type='single')
        s['motors'] = [dict(role='traction', max_power_kW=round(float(P), 1), eff_map='ipm_generic_150', max_rpm=6000),
                       dict(role='generator', max_power_kW=round(float(P) * 0.85, 1), eff_map='ipm_generic_150', max_rpm=3000)]
        cap = rng.uniform(15, 45) if arch == 'hev_series' else rng.uniform(100, 250)
        s['battery'] = dict(chemistry='NMC', capacity_kWh=round(float(cap), 1), nominal_V=650,
                            soc_window=[0.3, 0.8] if arch == 'hev_series' else [0.15, 0.95], max_power_kW=250)
    elif arch == 'bev':
        s['transmission'] = dict(type='single')
        s['motors'] = [dict(role='traction', max_power_kW=round(float(P) * (1.3 if cls == 'truck_hd' else 1.0), 1), eff_map='ipm_generic_150', max_rpm=8000)]
        s['battery'] = dict(chemistry='LFP' if rng.random() < 0.6 else 'NMC', capacity_kWh=round(float(bat), 0), nominal_V=650 if cls != 'truck_hd' else 800,
                            soc_window=[0.05, 0.98], thermal={'heater_W': 8000, 'cooler_W': 6000})
        s['aux']['HVAC'] = {'source': 'ptc' if rng.random() < 0.6 else 'heat_pump'}
    return s


def generate_vehicle(rng, cls=None, arch=None, strategy=True):
    cls = cls or _pick(rng, CLASS_W)
    arch = arch or _pick(rng, ARCH_W[cls])
    s = _ld(rng, cls, arch) if cls in ('car', 'suv', 'pickup', 'van') else _hd(rng, cls, arch)
    if strategy:
        st = _pick(rng, {k: v for k, v in STRAT_W[arch].items() if k in STRATEGIES[arch]})
        s['control'] = {'strategy': st, 'params': {}}
    s['regen'] = {'blend': 'series' if rng.random() < 0.8 else 'parallel',
                  'max_decel_g': round(float(rng.uniform(0.18, 0.35)) if cls in ('car', 'suv', 'pickup', 'van') else float(rng.uniform(0.10, 0.18)), 3),
                  'min_speed_mps': round(float(rng.uniform(1.0, 3.0)), 2)}
    h = hashlib.md5(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()[:10]
    s['id'] = f'gen_{cls}_{arch}_{h}'
    fill_spec(s)            # validates
    return s


def generate_vehicles(n, seed=0, cls=None, arch=None, strategy=True):
    rng = np.random.default_rng(seed)
    return [generate_vehicle(rng, cls, arch, strategy) for _ in range(n)]
