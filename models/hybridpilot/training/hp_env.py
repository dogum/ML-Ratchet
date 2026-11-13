"""HybridPilot environment: vehicles, routes and initial conditions for training, the frozen set, dev and exam.

Everything here is deterministic given a seed. Real VED routes come only from `train_noise` (training / frozen),
`dev` (dev gate, calibration) and `exam` (final exam, never used for choosing anything).
"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, math, hashlib, copy
sys.path.insert(0, _P.VEHICLE + '/roadsim' + ''); sys.path.insert(0, _P.VEHICLE)
import numpy as np

ROOT = _P.WORK
from roadsim import load_vehicle, make_trip
from roadsim.generator import generate_vehicles
from roadsim.sim import get_vehicle, normalize_trip
from roadsim.trips import cycle

HYBRID_ARCHS = ['hev_powersplit', 'hev_parallel_p2', 'hev_series', 'phev_powersplit', 'phev_series']
# (class, architecture) mix for generated hybrids: light duty 75 %, heavy duty 25 %
COMBOS = [('car', 'hev_powersplit', 0.13), ('suv', 'hev_powersplit', 0.08), ('van', 'hev_powersplit', 0.03),
          ('car', 'hev_parallel_p2', 0.07), ('suv', 'hev_parallel_p2', 0.06), ('pickup', 'hev_parallel_p2', 0.05),
          ('car', 'hev_series', 0.07), ('suv', 'hev_series', 0.04),
          ('car', 'phev_powersplit', 0.06), ('suv', 'phev_powersplit', 0.05), ('van', 'phev_powersplit', 0.02),
          ('car', 'phev_series', 0.05), ('pickup', 'phev_series', 0.04),
          ('bus', 'hev_series', 0.08), ('bus', 'hev_parallel_p2', 0.04),
          ('truck_md', 'hev_parallel_p2', 0.05), ('truck_md', 'phev_series', 0.03), ('truck_hd', 'hev_parallel_p2', 0.05)]
WARM = {'engine': 85.0, 'driveline': 70.0, 'tire_warm': 1.0, 'cabin': 21.0}

# seeds: one per role, never shared
SEED = dict(train=101, frozen=202, dev=303, exam=404, lab=505)


def is_phev(arch):
    return arch.startswith('phev')


def is_hd(cls):
    return cls in ('bus', 'truck_md', 'truck_hd')


def gen_fleet(n, role):
    """n generated hybrids for a role ('train' | 'frozen' | 'dev' | 'exam'), drawn from COMBOS."""
    rng = np.random.default_rng(SEED[role])
    p = np.array([c[2] for c in COMBOS]); p /= p.sum()
    out = []
    for i in range(n):
        cls, arch, _ = COMBOS[int(rng.choice(len(COMBOS), p=p))]
        s = generate_vehicles(1, seed=int(rng.integers(1 << 30)), cls=cls, arch=arch)[0]
        s['id'] = f'{role}_{i:04d}_{cls}_{arch}'
        out.append(s)
    return out


def fleet_balanced(role, per_combo):
    """per_combo vehicles of every (class, arch) combo (for exam / dev strata)."""
    rng = np.random.default_rng(SEED[role] + 7)
    out = []
    for cls, arch, _ in COMBOS:
        for j in range(per_combo):
            s = generate_vehicles(1, seed=int(rng.integers(1 << 30)), cls=cls, arch=arch)[0]
            s['id'] = f'{role}_{cls}_{arch}_{j}'
            out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ VED twins
# Engine power by displacement for the hybrids VED holds (likely models from RoadData's architecture inference).
_HEV_KW = {1.5: 55.0, 1.8: 73.0, 2.0: 105.0, 2.3: 99.0, 2.4: 110.0, 2.5: 118.0, 3.3: 155.0, 3.5: 183.0, 1.6: 77.0, 1.3: 73.0,
           1.0: 50.0, 1.4: 63.0}


def twin_spec(vehicle_id):
    """A RoadSim spec for a VED hybrid: VED class, mass, architecture, displacement; hardware sized like the likely
    production model; road load from class defaults. Unknown fields stay at RoadSim defaults."""
    import realdata as rd
    s0 = rd.load_vehicle(vehicle_id)
    arch = s0['architecture']; cls = s0['class'] if s0['class'] in ('car', 'suv') else 'suv'
    disp = float(s0['engine']['displacement_L']) if s0.get('engine') else 1.8
    s = {'class': cls, 'architecture': arch, 'mass_kg': float(s0['mass_kg']), 'payload_kg': 0.0, 'id': f'twin_{vehicle_id}'}
    Pe = _HEV_KW.get(round(disp, 1), 60.0 * disp)
    eng = {'fuel': 'gasoline', 'displacement_L': disp, 'max_power_kW': Pe, 'bsfc_map': 'gas_atk_toyota_tnga_hev_paper'}
    if arch in ('hev_powersplit', 'phev_powersplit'):
        P2 = {1.8: 53.0, 2.5: 105.0, 2.0: 88.0, 3.5: 123.0, 3.3: 123.0, 1.5: 45.0, 2.3: 70.0, 2.4: 105.0}.get(round(disp, 1), 0.9 * Pe)
        s['transmission'] = {'type': 'eCVT', 'ring_teeth': 78, 'sun_teeth': 30, 'mg2_ratio': 2.636}
        s['motors'] = [{'role': 'MG1', 'max_power_kW': round(0.55 * Pe, 1), 'eff_map': 'ipm_generic_150', 'max_rpm': 13500},
                       {'role': 'MG2', 'max_power_kW': P2, 'eff_map': 'ipm_generic_150', 'max_rpm': 17000}]
        if arch == 'phev_powersplit':
            cap = 7.6 if disp == 2.0 else 8.8     # Ford Energi 7.6 kWh; Prius Prime 8.8 kWh (Plug-in 4.4)
            s['battery'] = {'chemistry': 'NMC', 'capacity_kWh': cap, 'nominal_V': 350.0, 'soc_window': [0.15, 0.95]}
        else:
            s['battery'] = {'chemistry': 'NiMH', 'capacity_kWh': 1.3, 'nominal_V': 245.0 if disp < 2.2 else 245.0, 'soc_window': [0.4, 0.8]}
    elif arch == 'phev_series':      # Volt: 1.4 L (gen 1, 16 kWh) or 1.5 L (gen 2, 18.4 kWh), 111 kW traction
        eng['max_power_kW'] = 63.0 if disp < 1.45 else 75.0
        s['transmission'] = {'type': 'single'}
        s['motors'] = [{'role': 'traction', 'max_power_kW': 111.0, 'eff_map': 'ipm_generic_150', 'max_rpm': 12000},
                       {'role': 'generator', 'max_power_kW': 55.0 if disp < 1.45 else 87.0, 'eff_map': 'ipm_generic_150', 'max_rpm': 12000}]
        s['battery'] = {'chemistry': 'NMC', 'capacity_kWh': 16.0 if disp < 1.45 else 18.4, 'nominal_V': 360.0, 'soc_window': [0.15, 0.95]}
    elif arch == 'hev_parallel_p2':
        s['transmission'] = {'type': 'DCT', 'n_gears': 6}
        s['motors'] = [{'role': 'P2', 'max_power_kW': 32.0, 'eff_map': 'ipm_generic_150', 'max_rpm': 7000}]
        s['battery'] = {'chemistry': 'NMC', 'capacity_kWh': 1.56, 'nominal_V': 240.0, 'soc_window': [0.3, 0.8]}
    s['engine'] = eng
    s['meta'] = {'twin_of': vehicle_id, 'likely_models': s0.get('meta', {}).get('likely_models'), 'split': s0.get('split')}
    return s


# ------------------------------------------------------------------------------------------------ routes
def _smooth_hills(rng, dist_m, amp_m, lam_m):
    """elevation profile: sum of a few random sinusoids over distance; returns grade(dist)."""
    z = np.zeros_like(dist_m)
    for _ in range(4):
        lam = lam_m * float(np.exp(rng.normal(0, 0.5)))
        z += amp_m * float(rng.uniform(0.2, 1.0)) * np.sin(2 * np.pi * dist_m / lam + rng.uniform(0, 2 * np.pi))
    g = np.gradient(z, dist_m) if len(dist_m) > 2 else np.zeros_like(dist_m)
    return np.clip(np.nan_to_num(g), -0.08, 0.08)


def synth_route(rng, kind='mixed', hd=False, max_s=None):
    """synthetic speed trace (1 Hz): stops, accelerations, cruising with fluctuation; grade from random hills.
    kind: urban | suburban | highway | mixed | bus | hilly"""
    if max_s is None:
        max_s = int(rng.uniform(600, 2400))
    a_up = (0.5, 1.2) if hd else (0.8, 2.4)
    a_dn = (0.6, 1.5) if hd else (0.8, 2.8)
    v = [0.0]
    seg_kinds = {'urban': ['urban'], 'suburban': ['suburban', 'urban'], 'highway': ['highway', 'suburban'],
                 'mixed': ['urban', 'suburban', 'highway'], 'bus': ['bus'], 'hilly': ['suburban', 'highway']}[kind]
    while len(v) < max_s:
        sk = seg_kinds[int(rng.integers(len(seg_kinds)))]
        if sk == 'urban':
            vc, L, dwell = rng.uniform(7, 16), rng.uniform(120, 700), rng.uniform(0, 45)
        elif sk == 'suburban':
            vc, L, dwell = rng.uniform(14, 24), rng.uniform(400, 2500), rng.uniform(0, 40)
        elif sk == 'bus':
            vc, L, dwell = rng.uniform(8, 15), rng.uniform(150, 600), rng.uniform(10, 45)
        else:
            vc, L, dwell = rng.uniform(24, 33 if not hd else 28), rng.uniform(2000, 15000), rng.uniform(0, 20)
        if rng.random() < 0.3 and sk != 'bus':
            dwell = 0.0
        au, ad = rng.uniform(*a_up), rng.uniform(*a_dn)
        x = 0.0; cur = v[-1]
        # accelerate / cruise with an Ornstein-Uhlenbeck fluctuation
        dev = 0.0
        while x < L and len(v) < max_s:
            dev += -0.05 * dev + rng.normal(0, 0.25 if sk != 'highway' else 0.15)
            tgt = max(vc + dev, 2.0)
            remaining = L - x
            vstop = math.sqrt(max(2 * ad * remaining, 0.0)) if dwell > 0 else 1e9
            tgt = min(tgt, vstop)
            if cur < tgt:
                cur = min(cur + au * (1 - 0.6 * cur / max(vc, 1)), tgt)
            else:
                cur = max(cur - ad, tgt)
            v.append(cur); x += cur
            if dwell > 0 and remaining < 3 and cur < 1.5:
                break
        if dwell > 0:
            while v[-1] > 0:
                v.append(max(v[-1] - ad, 0.0))
            v.extend([0.0] * int(dwell))
    v = np.array(v[:max_s], float)
    v[-1] = 0.0
    if len(v) > 5:
        k = int(np.argmax(v[::-1] > 0)) if np.any(v > 0) else 0
    d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))])
    amp = {'hilly': rng.uniform(10, 40)}.get(kind, rng.uniform(0, 15))
    grade = _smooth_hills(rng, d + 1e-3 * np.arange(len(d)), amp, rng.uniform(400, 2500))
    return v, grade


def draw_weather(rng):
    """ambient temperature from a VED-like distribution (Ann Arbor 5/50/95 %: -9 / 8 / 28 C), wind 0-8 m/s."""
    T = float(np.clip(rng.normal(9.0, 11.0), -20, 36))
    return dict(T_amb=T, wind_speed=float(rng.uniform(0, 8)), wind_dir=float(rng.uniform(0, 360)), heading=float(rng.uniform(0, 360)))


def trip_from_trace(v, grade, w, name, kind):
    t = make_trip(v, T_amb=w['T_amb'], grade=grade, wind_speed=w['wind_speed'], wind_dir=w['wind_dir'], heading=w['heading'])
    t['_name'] = name; t['_kind'] = kind
    return t


CYCLES_LD = ['udds', 'hwfet', 'us06', 'sc03', 'nycc', 'la92']
CYCLES_HD = ['hd_transient', 'cbd14', 'gem_55', 'gem_65', 'nycc', 'udds']


def ved_routes(split, arch=None, vehicles=None, limit=None, seed=0, label_ok=None, route_ok=True, min_s=120, max_s=2700):
    """VED trips as route inputs (inputs only; labels are never read here)."""
    import realdata as rd
    out = []
    kw = dict(arch=arch, vehicles=vehicles, limit=limit, seed=seed)
    if label_ok is not None:
        kw['label_ok'] = label_ok
    else:
        kw['route_ok'] = route_ok
    for meta, arr in rd.load_trips(split, **kw):
        tr = {k: np.asarray(arr[k], float) for k in ['t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface']}
        if len(tr['v']) < min_s:
            continue
        if len(tr['v']) > max_s:
            tr = {k: x[:max_s] for k, x in tr.items()}
        tr['surface'] = np.nan_to_num(tr['surface']).astype(int)
        tr['_name'] = meta['trip_id']; tr['_kind'] = 'ved_' + split; tr['_vehicle'] = meta['vehicle_id']
        out.append(tr)
    return out


def train_noise_partition():
    """train_noise VED vehicles split by a fixed seed: 12 for training routes, 5 for the frozen set."""
    import realdata as rd
    d = rd.load_trips('train_noise', route_ok=True, meta_only=True)
    vids = sorted(d.vehicle_id.unique())
    rng = np.random.default_rng(777)
    fro = sorted(rng.choice(vids, 5, replace=False).tolist())
    return [v for v in vids if v not in fro], fro


def init_conditions(rng, spec, trip, warm_p=0.6):
    """random start state: SOC (HEV around the window middle; PHEV from full to CS level), warm fragment or cold soak."""
    v = get_vehicle(spec); b = v.battery
    lo, hi = b.soc_lo, b.soc_hi; mid = 0.5 * (lo + hi)
    if is_phev(v.arch):
        u = rng.random()
        soc0 = 0.9 if u < 0.5 else (float(rng.uniform(lo + 0.05, 0.9)) if u < 0.85 else lo + 0.05)
    else:
        soc0 = float(np.clip(mid + rng.uniform(-0.1, 0.1), lo + 0.03, hi - 0.03))
    warm = rng.random() < warm_p or trip.get('_kind', '').startswith('ved') and rng.random() < 0.8
    temps = dict(WARM) if warm else None
    return soc0, temps


def train_cases(n_cases, seed, role='train', fleet=None, ved=None, ved_frac=0.45, cyc_frac=0.2):
    """(spec, trip, soc0, temps, nav_known, preview_s) tuples. Routes: VED train_noise (warm fragments), EPA/HD cycles
    with random weather, synthetic routes (incl. bus routes and hilly roads)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_cases):
        spec = fleet[int(rng.integers(len(fleet)))]
        hd = is_hd(spec['class'])
        u = rng.random()
        if u < ved_frac and ved and not (spec['class'] == 'truck_hd'):
            tr = copy.deepcopy(ved[int(rng.integers(len(ved)))])
        elif u < ved_frac + cyc_frac:
            name = (CYCLES_HD if hd else CYCLES_LD)[int(rng.integers(6))]
            tr = make_trip(name, **draw_weather(rng)); tr['_name'] = name; tr['_kind'] = 'cycle'
        else:
            kinds = ['bus', 'bus', 'urban', 'mixed'] if spec['class'] == 'bus' else (
                ['highway', 'hilly', 'mixed', 'suburban'] if spec['class'] == 'truck_hd' else ['urban', 'suburban', 'highway', 'mixed', 'hilly'])
            kind = kinds[int(rng.integers(len(kinds)))]
            vtr, g = synth_route(rng, kind, hd=hd)
            tr = trip_from_trace(vtr, g, draw_weather(rng), f'synth_{kind}_{seed}_{i}', 'synth_' + kind)
        soc0, temps = init_conditions(rng, spec, tr)
        nav = bool(rng.random() < 0.5)
        prev = int(rng.choice([0, 5, 10, 20, 30, 60]))
        out.append(dict(spec=spec, trip=tr, soc0=soc0, temps=temps, nav=nav, preview=prev, case_id=f'{role}{seed}_{i}'))
    return out
