"""EnergyNet shared helpers: paths, spec completion for partial (VED/EPA-like) specs, warm starts, energy targets.

Target energy (MJ), same definition as RoadData's baselines:
  conv, hev_*     : fuel chemical energy = fuel g x 0.042 MJ/g
  bev             : battery net discharge (DC) kWh x 3.6
  phev_*          : fuel g x 0.042 + battery net discharge kWh x 3.6
"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, copy, math
ROOT = _P.VEHICLE
EN = f'{ROOT}/energynet'
for p in (f'{ROOT}/roadsim', ROOT, f'{EN}/src'):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np

LHV = 0.042            # MJ/g (E10 gasoline, as RoadData baselines)
ARCHS = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev']
CLASSES = ['car', 'suv', 'pickup', 'van', 'bus', 'truck_md', 'truck_hd']
HD = ('bus', 'truck_md', 'truck_hd')


def arch_group(arch):
    if arch == 'conv': return 'ICE'
    if arch == 'bev': return 'EV'
    if arch.startswith('phev'): return 'PHEV'
    return 'HEV'


def warm_temps(T_amb=20.0):
    return {'engine': 85.0, 'driveline': 70.0, 'tire_warm': 1.0, 'cabin': 21.0, 'battery': max(20.0, float(T_amb))}


def sim_target(out, arch):
    """Energy target (MJ) from a roadsim output, per second (array) -> same definition as the real label."""
    fr = np.nan_to_num(out['fuel_rate'].astype(float))
    pb = np.nan_to_num(out['batt_current'].astype(float) * out['batt_voltage'].astype(float))   # W, + discharge
    g = arch_group(arch)
    if g in ('ICE', 'HEV'):
        return fr * LHV
    if g == 'EV':
        return pb / 1e6
    return fr * LHV + pb / 1e6


def real_target(arr, ved_type):
    """Per-second energy (MJ) of a real VED trip, valid seconds only (RoadData baseline definition)."""
    ok = arr['valid'] == 1
    if ved_type in ('ICE', 'HEV'):
        y = np.nan_to_num(arr['fuel_rate'].astype(float)) * LHV
    else:
        pe = np.nan_to_num(arr['batt_current'].astype(float) * arr['batt_voltage'].astype(float)) / 1e6
        y = pe if ved_type == 'EV' else np.nan_to_num(arr['fuel_rate'].astype(float)) * LHV + pe
    return np.where(ok, y, 0.0)


def unit_of(arch):
    g = arch_group(arch)
    return {'ICE': 'g', 'HEV': 'g', 'EV': 'kWh', 'PHEV': 'MJ'}[g]


def mj_to_unit(E, arch):
    u = unit_of(arch)
    return E / LHV if u == 'g' else (E / 3.6 if u == 'kWh' else E)


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if isinstance(d, (np.floating,)): return float(d)
    if isinstance(d, (np.integer,)): return int(d)
    return d


# Known production hardware for the VED plug-ins / EV (public spec sheets); used ONLY to build twins for dev/exam
# vehicles whose architecture RoadData already identified (Volt, Ford Energi / Prius Prime, 2013 Leaf).
PLUGIN_HW = {
    ('phev_series', 1.4): dict(battery=dict(capacity_kWh=16.5, nominal_V=360, soc_window=[0.2, 0.85]), P=111),
    ('phev_series', 1.5): dict(battery=dict(capacity_kWh=18.4, nominal_V=360, soc_window=[0.15, 0.9]), P=111),
    ('phev_powersplit', 2.0): dict(battery=dict(capacity_kWh=7.6, nominal_V=300, soc_window=[0.15, 0.95]), P=145),
    ('phev_powersplit', 1.8): dict(battery=dict(capacity_kWh=8.8, nominal_V=350, soc_window=[0.15, 0.95]), P=90),
}


def complete_spec(partial):
    """Partial contract spec (unknowns None) -> spec that roadsim can run. Only physically generic fills:
    rated power from displacement (55 kW/L NA, 75 kW/L boosted, 30 kW/L HD diesel), class defaults for CdA/Crr,
    road-load ABC turned into CdA/Crr when given."""
    s = _clean(copy.deepcopy(partial))
    for k in ('meta', 'labels_fueleconomy', 'split', 'source', 'mass_note', 'results', 'arch_source', 'drive', 'n_v_ratio', 'model', 'make',
              'model_year'):
        s.pop(k, None)
    arch, cls = s['architecture'], s['class']
    eng = s.get('engine') or {}
    if arch == 'bev':
        s.pop('engine', None)
    if eng and arch != 'bev':
        d = eng.get('displacement_L')
        if not eng.get('max_power_kW') and d:
            kwl = 30 if eng.get('fuel') == 'diesel' and cls in HD else (75 if eng.get('boosted') else (45 if arch != 'conv' else 55))
            eng['max_power_kW'] = float(d) * kwl
        for k in ('cylinders', 'boosted'):
            eng.pop(k, None)
        if eng.get('fuel') not in ('gasoline', 'diesel'): eng['fuel'] = 'gasoline'
        s['engine'] = eng
    tr = s.get('transmission') or {}
    tr = {k: v for k, v in tr.items() if k in ('type', 'gears', 'n_gears')}
    if tr.get('type') == 'manual': tr['type'] = 'AMT'
    if isinstance(tr.get('gears'), (int, float)): tr['n_gears'] = int(tr.pop('gears'))
    if arch in ('hev_powersplit', 'phev_powersplit'): tr = {'type': 'eCVT'}
    if arch in ('bev', 'phev_series', 'hev_series'): tr = {'type': 'single'}
    if tr.get('type') in (None,):
        tr = {}
    if tr: s['transmission'] = tr
    else: s.pop('transmission', None)
    rl = s.pop('road_load_ABC', None)
    if rl and rl.get('A_N') and s.get('mass_kg'):
        # EPA target coefficients -> CdA, Crr (C = 0.5 rho CdA at rho_ref 1.1767; A = m g Crr; B -> speed coefficient)
        m = float(s['mass_kg']); A, B, C = rl['A_N'], rl['B_N_per_mps'], rl['C_N_per_mps2']
        s['CdA_m2'] = float(max(0.3, 2 * C / 1.1767)); Crr = float(max(0.004, A / (m * 9.81)))
        s['Crr'] = Crr; s['Crr_speed_coef'] = float(np.clip(B / max(A, 1.0), -0.01, 0.03))
    if s.get('battery') is None or not s['battery'].get('nominal_V'):
        d = (eng or {}).get('displacement_L')
        hw = PLUGIN_HW.get((arch, d))
        if hw:
            s['battery'] = hw['battery']
            if arch.startswith('phev'):
                s.setdefault('motors', None)
        elif arch == 'bev' and s.get('battery'):
            cap = s['battery'].get('capacity_kWh', 24.0)
            s['battery'] = {'capacity_kWh': cap, 'nominal_V': 360, 'chemistry': 'NMC', 'soc_window': [0.02, 0.98]}
        else:
            s.pop('battery', None)
    if s.get('battery'):
        s['battery'] = {k: v for k, v in s['battery'].items() if k in ('capacity_kWh', 'nominal_V', 'chemistry', 'soc_window', 'max_power_kW', 'thermal')}
        if s['battery'].get('chemistry') == 'Li-ion': s['battery']['chemistry'] = 'NMC'
    for k in ('motors', 'aux', 'regen', 'control'):
        if not s.get(k): s.pop(k, None)
    if arch == 'bev' and 'motors' not in s:
        s['motors'] = [{'role': 'traction', 'max_power_kW': 80.0 if cls == 'car' else 150.0, 'eff_map': 'ipm_generic_150'}]
    return s
