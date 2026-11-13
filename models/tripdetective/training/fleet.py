"""Vehicles for the bank (RoadSim generator, light duty) and twin specs for real dev cars."""
from common import *
from roadsim.generator import generate_vehicle
from roadsim.vehicle import fill_spec
import copy

CLS_W = {'car': 0.42, 'suv': 0.35, 'pickup': 0.13, 'van': 0.10}
ARCH_W = {'conv': 0.34, 'hev_powersplit': 0.17, 'hev_parallel_p2': 0.10, 'hev_series': 0.07,
          'phev_powersplit': 0.10, 'phev_series': 0.08, 'bev': 0.14}
ALLOWED = {'pickup': {'conv', 'hev_powersplit', 'hev_parallel_p2', 'phev_series', 'bev'},
           'van': {'conv', 'hev_powersplit', 'phev_powersplit', 'bev'}}


def _pick(rng, w):
    k = list(w); p = np.array([w[x] for x in k]); p = p / p.sum()
    return k[int(rng.choice(len(k), p=p))]


def draw_vehicle(rng, cls_w=None, quick=False, prior='v1'):
    """prior v3: rejection sampling to the VED-era mass prior (priors.accept_v3). quick=True returns only the total mass."""
    cw = cls_w or CLS_W
    while True:
        arch = _pick(rng, ARCH_W)
        cls = _pick(rng, {c: w for c, w in cw.items() if c not in ALLOWED or arch in ALLOWED[c]})
        s = generate_vehicle(rng, cls, arch)
        m = s['mass_kg'] + s['payload_kg']
        if quick: return m
        if prior != 'v3': return s
        import priors
        if priors.accept_v3(m, rng): return s


def truth(spec):
    """vehicle-level targets from a spec"""
    m = float(spec['mass_kg']) + float(spec.get('payload_kg') or 0)
    eng = spec.get('engine') or {}
    crr = float(spec['Crr']); cda = float(spec['CdA_m2']); cv = float(spec.get('Crr_speed_coef') or 0)
    return dict(arch=spec['architecture'], cls=spec['class'], strategy=spec['control']['strategy'], mass=m,
                disp=float(eng['displacement_L']) if eng.get('displacement_L') else np.nan,
                P_kW=float(eng.get('max_power_kW') or np.nan), CdA=cda, Crr=crr, Crr_v=cv,
                A_N=m * G * crr, B_N=m * G * crr * cv, C_N=0.5 * 1.2 * cda)


# ---------------- twins of real dev cars ----------------
_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        rng = np.random.default_rng(777)
        _LIB = []
        for arch in ARCHS:
            for cls in CLS_W:
                if cls in ALLOWED and arch not in ALLOWED[cls]: continue
                for _ in range(60):
                    _LIB.append(generate_vehicle(rng, cls, arch))
    return _LIB


PLUG_OVERRIDE = {'VED_537': dict(capacity_kWh=18.4), 'VED_11': dict(capacity_kWh=7.6), 'VED_455': dict(capacity_kWh=24.0)}


def twin_spec(vid, row=None, strategy=None):
    """A full RoadSim spec for a real VED car: the nearest generated vehicle of the same architecture and class
    (by mass and displacement), set to the car's published mass (curb + 136 kg) and displacement."""
    import realdata as rd
    if row is None: row = rd.ved_vehicles().loc[vid]
    arch = row.architecture; cls = row.cls if row.cls in CLS_W else 'car'
    if cls in ALLOWED and arch not in ALLOWED[cls]: cls = 'suv'
    m = (row.gen_weight_lb * LB + CURB_OFFSET) if np.isfinite(row.gen_weight_lb) else 1600.0
    d = row.displacement_L if np.isfinite(row.displacement_L) else None
    cands = [s for s in _library() if s['architecture'] == arch and s['class'] == cls]
    def dist(s):
        e = (np.log((s['mass_kg'] + s['payload_kg']) / m)) ** 2
        if d and s.get('engine'): e += (np.log(s['engine']['displacement_L'] / d)) ** 2
        return e
    s = copy.deepcopy(min(cands, key=dist))
    s['payload_kg'] = 0.0; s['mass_kg'] = round(float(m), 1)
    if d and s.get('engine'):
        r = d / s['engine']['displacement_L']
        s['engine']['displacement_L'] = d
        s['engine']['max_power_kW'] = round(float(s['engine']['max_power_kW'] * r), 1)
    if vid in PLUG_OVERRIDE: s['battery'].update(PLUG_OVERRIDE[vid])
    if strategy: s['control'] = {'strategy': strategy, 'params': {}}
    s['id'] = f'twin_{vid}'
    fill_spec(s)
    return s
