"""Export the garage: one page where a visitor picks a vehicle and a real route and sees all three vehicle models on the
same trip (site/src/pages/garage.astro). RoadSim is Python, so the simulator runs here, over a fixed grid; the three
networks run live in the browser on what this script stores.

    .venv/Scripts/python.exe shared/vehicle/garage/export_garage.py [--workers 14] [--quick]
    -> site/public/garage/  routes.{json,bin}, v_<vehicle>.{json,bin}, tdbus.{json,bin}, reference.json

Grid (nothing here is chosen by a result):
  vehicles  six RoadSim specs: RoadSim's calibrated twins (checks/check_ved_twin.py, label-matched) of VED exam cars
            470 (Corolla-type petrol sedan) and 10 (2013 Leaf), HybridPilot's copies of the same builder's twins of VED
            exam cars 428 (Prius-type power-split) and 492 (Volt-type plug-in), a P2 hybrid from RoadSim's generator
            (seed 2026, closest of 300 draws to 1,500 kg / 1.6 L), and the ZTBus trolleybus twin of checks/check_ztbus.py.
  routes    the EnergyNet and HybridPilot example trips that last at least 10 minutes (four real VED exam trips, Ann
            Arbor), and one ZTBus exam leg drawn with seed 2026 among the 20-30 minute legs with good GPS. The cars
            drive all five; the bus drives the Zurich leg only. The exams are already taken; these trips choose nothing.
  controls  air temperature (route mean -10, 20 or 35 C; EnergyNet's own perturb(dT): every second shifted, air density
            at constant pressure) x extra load (cars +0 / +400 kg, bus +0 / +60 passengers at 70 kg): 6 variants.
            The same changed input goes to RoadSim, EnergyNet (told the new mass) and TripDetective. A third load level
            would have taken the garage past its ~8 MB budget: HybridPilot's inputs at every decision second dominate.

Per cell (vehicle x route x temperature x load), RoadSim physics v3.1 (v2 plus the hybrid fixes), pinned:
  * the run itself (RoadSim's default controller for the architecture: A-ECMS for the power-split, charge-depleting
    then sustaining for the plug-in; warm start, as the VED fragments are): ledger, summary, display traces (wheel, fuel and battery power, HVAC, SOC, engine on);
  * what a VED-style OBD logger records of it (roadsim.obd.emulate, fixed channel set per powertrain, no dropouts),
    quantised exactly as TripDetective's instrument stores logs: TripDetective reads it back in the browser;
  * hybrids: HybridPilot, A-ECMS, the rule and the DP optimum on the same trip, graded as HybridPilot's exam grades
    (hp_score: % of DP at matched final charge), and HybridPilot's own network inputs at every decision second, so
    the browser re-runs the network over the whole trip.
Reference outputs for the browser parity test (site/tests/garage.test.ts) use the weights exactly as the browser loads
them (the instruments' float16 packs) and the stored inputs exactly as the browser decodes them.

Inference code is untouched: this script imports each model's own modules (and the instrument exporters' perturb() and
HybridPilot capture) and may use mlratchet.webexport, which inference never does.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(VEH))
MODELS = os.path.join(REPO, 'models')
# first match wins: HybridPilot's pipeline.py, not TripDetective's step list of the same name
sys.path[:0] = [p for p in (os.path.join(MODELS, 'hybridpilot'), os.path.join(MODELS, 'hybridpilot', 'training'), os.path.join(VEH, 'roadsim'),
                            VEH, REPO, os.path.join(MODELS, 'energynet'), os.path.join(MODELS, 'tripdetective')) if p not in sys.path]
os.environ.setdefault('OMP_NUM_THREADS', '1')

import realdata as rd  # noqa: E402
from roadsim import simulate, obd  # noqa: E402
from roadsim.sim import get_vehicle  # noqa: E402
from roadsim.generator import generate_vehicle  # noqa: E402
from roadsim.strategies import DEFAULT  # noqa: E402
import pipeline as PL  # noqa: E402  (HybridPilot: RULE, WARM, PHYSICS)
import hybridpilot as HP  # noqa: E402
import hp_score as S  # noqa: E402
import tripdetective as TD  # noqa: E402
from energynet import EnergyNet  # noqa: E402
from mlratchet.webexport import write_json, write_pack  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENX = _load('garage_en_export', os.path.join(MODELS, 'energynet', 'web', 'export_instrument.py'))      # perturb()
HPX = _load('garage_hp_export', os.path.join(MODELS, 'hybridpilot', 'web', 'export_instrument.py'))    # capture_strategy()
ZTB = _load('garage_check_ztbus', os.path.join(VEH, 'checks', 'check_ztbus.py'))                      # bus_spec()
CVT = _load('garage_check_ved_twin', os.path.join(VEH, 'checks', 'check_ved_twin.py'))                # build_twin()

PHYSICS = 'roadsim-physics-v3.1'   # HybridPilot v2.1's pin; non-hybrid runs are bit-identical to v2
assert PL.PHYSICS == PHYSICS
OUT = os.path.join(REPO, 'site', 'public', 'garage')
INST = os.path.join(REPO, 'site', 'public', 'instruments')
FIELDS = ENX.FIELDS                                   # EnergyNet's trip fields (v, grade, T_amb, wind..., surface)
TEMPS = [-10.0, 20.0, 35.0]                           # route-mean air temperature, C
LOADS = {'car': [0.0, 400.0], 'bus': [0.0, 60 * 70.0]}
VED_ROUTES = ['VED_428_465', 'VED_458_1353', 'VED_470_1423', 'VED_492_2826']   # example trips >= 10 min (sorted)
LHV_KJ_G = 42.0                                       # EnergyNet / RoadData target definition, 0.042 MJ per g
# TripDetective's integer scales (models/tripdetective/web/export_instrument.py Q / LABQ); aux in W; bus power in 100 W
Q = dict(fuel=1000, rpm=1000, batt=10, soc=1000, aux=1000, pbus=0.01)
HYBRID = ('hev_powersplit', 'hev_parallel_p2', 'phev_series', 'phev_powersplit')
XCOL = list(range(12, 37)) + [65]                     # HybridPilot features that change second by second


# ------------------------------------------------------------------------------------------------ packs
def read_pack(prefix):
    """an mlratchet pack exactly as site/src/lib/nn/pack.ts decodes it (values as float64)."""
    man = json.load(open(prefix + '.json', encoding='utf-8'))
    buf = open(prefix + '.bin', 'rb').read()
    out = {}
    for k, d in man['tensors'].items():
        dt = np.dtype({'f32': '<f4', 'f16': '<f2', 'i32': '<i4', 'u8': 'u1'}[d['dtype']])
        out[k] = np.frombuffer(buf, dt, count=d['bytes'] // dt.itemsize, offset=d['offset']).reshape(d['shape']).astype(np.float64)
    return out, man['meta']


def as_f32(x):
    return np.asarray(x, np.float32).astype(np.float64)


def bits(mask):
    return np.packbits(np.asarray(mask, bool).astype(np.uint8), bitorder='little')


def int16_bytes(q):
    q = np.asarray(np.round(q), np.int64)
    assert np.abs(q).max(initial=0) < 32768, np.abs(q).max()
    return np.frombuffer(q.astype('<i2').tobytes(), np.uint8)


# ------------------------------------------------------------------------------------------------ routes
def local_xy(lat, lon):
    lat0, lon0 = float(lat[0]), float(lon[0])
    x = (np.asarray(lon, float) - lon0) * np.cos(np.deg2rad(lat0)) * 111320.0
    y = (np.asarray(lat, float) - lat0) * 110540.0
    return x, y


def load_routes():
    R = []
    for tid in VED_ROUTES:
        meta, a = rd.load_trip(tid)
        R.append(dict(id=tid, kind='ved', meta=meta, a=a))
    d = rd.ztbus_full('exam', label_ok=True)
    d = d[d.q_route_ok & (d.gnss_frac >= 0.95) & (d.duration_s >= 1200) & (d.duration_s <= 1800)].sort_values('trip_id')
    row = d.iloc[int(np.random.default_rng(2026).integers(len(d)))]
    meta, a = rd.load_trip(row.trip_id)
    R.append(dict(id=row.trip_id, kind='ztbus', meta=dict(meta, passengers_mean=float(row.passengers_mean), route=str(row.route),
                                                          season=str(row.season), elec_kWh=float(row.elec_kWh)), a=a))
    out = []
    for r in R:
        a = r['a']; n = len(a['v'])
        t = {}
        for k in FIELDS:
            x = np.asarray(a[k], float) if k in a else np.zeros(n)
            if k == 'T_amb':
                x = np.where(np.isfinite(x), x, np.nanmean(x))
            elif k == 'rho_air':
                x = np.where(np.isfinite(x) & (x > 0.5), x, 1.2)
            else:
                x = np.nan_to_num(x)
            t[k] = x
        if r['kind'] == 'ztbus':                       # as checks/check_ztbus.py: the bus record has no wind
            t['wind_speed'] = np.zeros(n); t['wind_dir'] = np.zeros(n)
        t['surface'] = np.clip(np.round(t['surface']), 0, 4)
        # every model sees exactly the stored float32 values
        t = {k: (as_f32(v) if k != 'surface' else v.astype(np.float64)) for k, v in t.items()}
        elev = np.nan_to_num(np.asarray(a['elev'], float), nan=float(np.nanmean(a['elev'])))
        x, y = local_xy(a['lat'], a['lon'])
        ok = np.isfinite(x) & np.isfinite(y)
        x = np.interp(np.arange(n), np.where(ok)[0], x[ok]); y = np.interp(np.arange(n), np.where(ok)[0], y[ok])
        v = t['v']; dist = float(v.sum())
        Tm = float(t['T_amb'].mean())
        m = r['meta']
        info = dict(id=r['id'], kind=r['kind'], n=n, km=dist / 1000, minutes=n / 60, T_mean=Tm,
                    vmax_kmh=float(v.max() * 3.6), v_mean_kmh=float(dist / n * 3.6),
                    fast_share=float((v > 70 / 3.6).mean()), stops=int(np.sum((v[1:] < 0.5) & (v[:-1] >= 0.5))),
                    climb_m=float(np.clip(np.diff(elev), 0, None).sum()), elev_range_m=float(elev.max() - elev.min()),
                    date=str(m.get('date', ''))[:10], dT=[T - Tm for T in TEMPS])
        if r['kind'] == 'ztbus':
            info.update(passengers=m['passengers_mean'], line=m['route'], season=m['season'], real_kWh=m['elec_kWh'],
                        place='Zurich, VBZ trolleybus line ' + m['route'])
        else:
            info.update(place='Ann Arbor, Michigan', vehicle=m.get('vehicle_id'))
        out.append(dict(info=info, trip=t, elev=elev, x=x, y=y))
    return out


# ------------------------------------------------------------------------------------------------ vehicles
def vehicles(routes):
    ex = {e['trip_id']: e for e in json.load(open(os.path.join(MODELS, 'energynet', 'examples', 'examples.json')))}
    twin = lambda t: json.load(open(os.path.join(MODELS, 'hybridpilot', 'examples', f'{t}_twin.json')))  # noqa: E731
    V = []
    def ved_twin(vid):
        # RoadSim's own calibrated twin (checks/check_ved_twin.py, the builder HybridPilot's twins come from): the VED
        # record completed generically, road-load scale and engine level matched to the unadjusted fueleconomy.gov values
        _, spec, info = CVT.build_twin((vid, PHYSICS, True))
        spec['id'] = f'twin_{vid}'
        return spec, info

    # 1. petrol sedan: twin of VED exam car 470
    conv, info = ved_twin('VED_470')
    en = dict(ex['VED_470_1423']['vehicle'])
    V.append(dict(id='sedan', name='Petrol sedan', short='Petrol', type=f"Conventional, {conv['engine']['displacement_L']:.1f} L",
                  likely=info['labels']['example_models'],
                  source="RoadSim's calibrated twin of VED exam car 470 (checks/check_ved_twin.py: road load and engine level matched to its fueleconomy.gov label)",
                  spec=conv, en=en, soc0=None))
    # 2. Prius-type power-split hybrid: HybridPilot's calibrated twin of VED exam car 428
    ps = twin('VED_428_465')
    V.append(dict(id='prius', name='Power-split hybrid', short='Hybrid', type='Toyota-type power-split, 1.8 L', likely=ps['meta']['likely_models'],
                  source='HybridPilot\'s calibrated twin of VED exam car 428 (road load and engine level matched to its fueleconomy.gov label)',
                  spec=ps, en={'class': 'car', 'architecture': 'hev_powersplit', 'mass_kg': ps['mass_kg'] + ps['payload_kg'], 'mass_kind': 'reported',
                               'fuel': 'gasoline', 'thermal': 'warm', 'displacement_L': 1.8, 'transmission': 'eCVT',
                               'label_city': 48.0, 'label_hwy': 43.0, 'model_year': 2011.0}, soc0=0.6))
    # 3. P2 parallel hybrid from the generator (v2 donors: hybrid bodies from real hybrid records)
    rng = np.random.default_rng(2026); best = None
    for _ in range(300):
        s = generate_vehicle(rng, 'car', 'hev_parallel_p2', donor_model='v2')
        e = abs(s['mass_kg'] + s['payload_kg'] - 1500) / 1500 + abs(s['engine']['displacement_L'] - 1.6) / 1.6
        if best is None or e < best[0]:
            best = (e, s)
    p2 = best[1]; p2.pop('control', None)
    V.append(dict(id='p2', name='P2 parallel hybrid', short='P2 hybrid', type=f"Parallel P2, {p2['engine']['displacement_L']:.1f} L, {p2['transmission']['n_gears']}-speed DCT",
                  likely='Ioniq / Niro-class layout (generated, not a real car)',
                  source=f"RoadSim generator (v2 donors), seed 2026: the closest of 300 draws to 1,500 kg and 1.6 L (donor year {p2['generator']['donor_year']})",
                  spec=p2, en={'class': 'car', 'architecture': 'hev_parallel_p2', 'mass_kg': p2['mass_kg'] + p2['payload_kg'], 'mass_kind': 'reported',
                               'fuel': 'gasoline', 'thermal': 'warm', 'displacement_L': p2['engine']['displacement_L'],
                               'max_power_kW': p2['engine']['max_power_kW'], 'transmission': 'DCT', 'n_gears': p2['transmission']['n_gears'],
                               'battery_kWh': p2['battery']['capacity_kWh'], 'model_year': float(p2['generator']['donor_year'])}, soc0=0.6))
    # 4. Volt-type plug-in: HybridPilot's calibrated twin of VED exam car 492
    ph = twin('VED_492_2826')
    lo, hi = ph['battery']['soc_window']
    V.append(dict(id='volt', name='Plug-in hybrid', short='Plug-in', type='Volt-type series plug-in, 16.5 kWh', likely=ph['meta']['likely_models'],
                  source='HybridPilot\'s calibrated twin of VED exam car 492', spec=ph,
                  en={'class': 'car', 'architecture': 'phev_series', 'mass_kg': ph['mass_kg'] + ph['payload_kg'], 'mass_kind': 'reported',
                      'fuel': 'gasoline', 'thermal': 'warm', 'displacement_L': 1.4, 'transmission': 'single', 'battery_kWh': 16.5,
                      'label_city': 35.0, 'label_hwy': 40.0, 'label_city_kWh_per_100mi': 33.0, 'label_hwy_kWh_per_100mi': 36.0, 'model_year': 2013.0},
                  soc0=0.30, usable=(0.30 - lo) / (hi - lo)))
    # 5. Leaf: VED exam car 10, filled as EnergyNet's complete_spec does for a VED BEV record
    leaf, _ = ved_twin('VED_10')
    lo, hi = leaf['battery'].get('soc_window') or [0.02, 0.98]
    en = dict(ex['VED_10_2909']['vehicle'])
    V.append(dict(id='leaf', name='Electric car', short='Electric', type='2013 Leaf-type, 24 kWh', likely='2013 Nissan Leaf',
                  source="RoadSim's calibrated twin of VED exam car 10 (checks/check_ved_twin.py: road-load scale matched to its label)",
                  spec=leaf, en=en, soc0=0.8, usable=(0.8 - lo) / (hi - lo)))
    # 6. ZTBus trolleybus twin (checks/check_ztbus.py): 19 t curb + 70 kg per passenger of the leg's mean
    bus_route = [r for r in routes if r['info']['kind'] == 'ztbus'][0]['info']
    bus = ZTB.bus_spec(bus_route['passengers'])
    V.append(dict(id='bus', name='Trolleybus', short='Bus', type='18.7 m articulated battery-trolleybus', likely='VBZ Zurich trolleybuses 183 and 208 (the ZTBus record)',
                  source='RoadSim\'s ZTBus twin (checks/check_ztbus.py): 19,000 kg curb (a convention) + 70 kg per passenger', spec=bus,
                  en={'class': 'bus', 'architecture': 'bev', 'transmission': 'single', 'thermal': 'warm'}, soc0=0.6))
    for v in V:
        v['en'] = {k: x for k, x in v['en'].items() if x is not None}
        v['en']['obd_maf'] = False           # the truth is RoadSim's own fuel, not an OBD estimate of it
        v['en']['thermal'] = 'warm'
        if v.get('usable') is not None:
            v['en']['soc_start'] = float(v['usable'])
        else:
            v['en'].pop('soc_start', None)
        v['arch'] = v['spec']['architecture']
        v['cls'] = v['spec']['class']
        v['routes'] = [i for i, r in enumerate(routes) if (r['info']['kind'] == 'ztbus') or v['cls'] != 'bus']
    return V


def logger(arch):
    """fixed VED-style channel set (TripDetective's VED_MIX, the most common choice), no dropouts."""
    c = dict(cad_speed=1.0, cad_fast=2.0, cad_slow=5.0, cut_k=0.7, maf_bias=0.0, idle_scale=1.0, p_dropout=0.0)
    if arch == 'bev':
        c.update(fuel=None, rpm=False, batt=True, aux=True)
    elif arch.startswith('phev'):
        c.update(fuel='pid', rpm=True, batt=True, aux=True)        # Volt-type: the OBD fuel-rate PID
    else:
        c.update(fuel='maf', rpm=True, batt=False, aux=False)
    fl = dict(has_fuel=c['fuel'] is not None, fuel_est=c['fuel'] == 'maf', has_rpm=c['rpm'], has_batt=c['batt'], has_aux=c['aux'])
    return c, fl


# ------------------------------------------------------------------------------------------------ one cell
def perturbed(route_trip, dT):
    t, _ = ENX.perturb(route_trip, {}, {'dT': dT} if dT else {})
    return t


def run_hybrid(spec, trip, soc0, temps):
    veh = get_vehicle(spec)
    phev = veh.arch.startswith('phev')
    pts, outs, stage = S.dp_curve(spec, trip, soc0, temps, S.dp_targets(veh, soc0, phev), veh=veh)
    nat = soc0 if not (phev and soc0 > veh.battery.soc_lo + 0.08) else veh.battery.soc_lo + 0.05
    jn = int(np.argmin([abs(p['target'] - nat) for p in pts]))
    runs = {'dp': (outs[pts[jn]['target']][0], dict(pts[jn]))}
    cap = HPX.capture_strategy()
    o = simulate(spec, trip, strategy=cap, init_soc=soc0, init_temps=temps, record=True, options=PHYSICS)
    runs['hybridpilot'] = (o, S.metrics(o, veh.arch, soc0))
    for nm in ('aecms', 'rule'):
        o = simulate(spec, trip, strategy=PL.RULE[veh.arch] if nm == 'rule' else nm, init_soc=soc0, init_temps=temps, record=True, options=PHYSICS)
        runs[nm] = (o, S.metrics(o, veh.arch, soc0))
    case = dict(spec=spec, trip=trip, soc0=soc0, temps=temps)
    for nm, (o, m) in runs.items():
        if nm != 'dp':
            S.add_matched(case, veh, stage, pts, m['soc_end'])
    for nm, (o, m) in runs.items():
        S.grade(m, pts, S.LHV.get(veh.fuel, 42.9e3) / 1e3, soc0)
    return runs, cap, veh


def run_cell(job):
    V, R, ri, ti, li = job['V'], job['R'], job['ri'], job['ti'], job['li']
    t0 = time.time()
    dT = R['info']['dT'][ti]
    trip = perturbed(R['trip'], dT)
    trip_sim = dict(trip, t=np.arange(len(trip['v']), dtype=float), elev=R['elev'])
    trip_sim['surface'] = trip['surface'].astype(int)
    spec = copy.deepcopy(V['spec'])
    load = LOADS[V['cls']][li]
    spec['payload_kg'] = float(spec.get('payload_kg') or 0.0) + load
    spec['id'] = f"{spec.get('id', V['id'])}_garage_l{int(load)}"
    arch = V['arch']
    hp = None
    if V['cls'] == 'bus':
        temps = dict(cabin=21.0, battery=float(trip['T_amb'].mean()))
    else:
        temps = dict(PL.WARM)
    out = simulate(spec, trip_sim, strategy=DEFAULT[arch], init_soc=V['soc0'], init_temps=temps, record=True, options=PHYSICS)
    if arch in HYBRID:
        runs, cap, veh = run_hybrid(spec, trip_sim, V['soc0'], temps)
        hp = dict(runs=runs, cap=cap.cap, veh=veh, ref=cap.ref)
    n = len(trip['v'])
    L = out['ledger']; Sm = out['summary']
    g = lambda k: np.nan_to_num(np.asarray(out[k], float))[:n]  # noqa: E731
    pb = g('batt_current') * g('batt_voltage')
    fr = g('fuel_rate')
    eng = bool(spec.get('engine')) and arch != 'bev'
    E_true = (fr.sum() * LHV_KJ_G / 1e3 if arch == 'conv' or arch.startswith('hev') else 0.0) \
        + (pb.sum() / 1e6 if arch == 'bev' else 0.0) + ((fr.sum() * LHV_KJ_G / 1e3 + pb.sum() / 1e6) if arch.startswith('phev') else 0.0)
    km = Sm['distance_m'] / 1000
    hvac = float(g('hvac_power').sum() / 1e3)
    res = dict(ri=ri, ti=ti, li=li, wall=0.0, n=n, strategy=DEFAULT[arch], E_MJ=float(E_true), fuel_g=float(fr.sum()), batt_kWh=float(pb.sum() / 3.6e6),
               km=float(km), soc_end=float(out['soc'][-1]) if out.get('soc') is not None else None,
               ledger={k: float(L[k]) for k in ('fuel', 'batt_chem_out', 'aero', 'rolling', 'grade', 'brake', 'eng_loss', 'mot_loss',
                                                 'trans_loss', 'batt_loss', 'aux', 'kinetic_change', 'residual', 'closure_rel')},
               hvac_kJ=hvac, trace_rms=float(Sm['trace_rms_mps']), T_cabin_end=float(out['T_cabin'][-1]),
               starts=int(np.sum((g('eng_on')[1:] > 0.5) & (g('eng_on')[:-1] <= 0.5))) if eng else 0,
               on_share=float((g('eng_on') > 0.5).mean()) if eng else 0.0)
    T = {}
    # display traces
    T['pw'] = g('P_wheel') / 1e3
    T['hv'] = g('hvac_power') / 1e3
    if eng:
        T['pf'] = fr * LHV_KJ_G
        T['on'] = bits(g('eng_on') > 0.5)
    if arch != 'conv':
        T['pb'] = pb / 1e3
        T['soc'] = np.round(np.clip(g('soc'), 0, 1) * 250).astype(np.uint8)
    # the OBD log TripDetective reads
    rng = np.random.default_rng([job['vi'], ri, ti, li])
    if V['cls'] == 'bus':
        T['q_pbus'] = int16_bytes(pb * Q['pbus'])
    else:
        cfg, fl = logger(arch)
        lg = obd.emulate(out, cfg, rng, spec)
        a = {k: trip[k] for k in ('grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air')} | lg
        a['v'] = trip['v']; a['valid'] = np.ones(n)
        raw = TD.raw_from_log(a, fl)
        for k, row in (('fuel', 5), ('rpm', 6), ('batt', 7), ('soc', 8), ('aux', 9)):
            on = dict(fuel=fl['has_fuel'], rpm=fl['has_rpm'], batt=fl['has_batt'], soc=fl['has_batt'], aux=fl['has_aux'])[k]
            if on:
                T[f'q_{k}'] = int16_bytes(raw[row].astype(np.float64) * Q[k])
        res['flags'] = [float(fl[k]) for k in TD.FLAGS]
    # what TripDetective is graded against: the run's ledger per km (TripDetective's ledger groups)
    res['td_ledger'] = [L['aero'] / km, L['rolling'] / km, L['grade'] / km, L['brake'] / km,
                        (L['eng_loss'] + L['mot_loss'] + L['trans_loss'] + L['batt_loss']) / km, L['aux'] / km]
    res['mass_true'] = float(spec['mass_kg'] + spec['payload_kg'])
    if hp is not None:
        runs, cap, veh = hp['runs'], hp['cap'], hp['veh']
        res['hp'] = {nm: dict(fuel_g=float(o['summary']['fuel_g']), E_kWh=float(m['E_kJ']) / 3600, soc_end=float(o['soc'][-1]),
                              pct_dp=float(m['pct_dp']), extrap=bool(m['extrap']), starts_km=float(m['starts_per_km']),
                              dist_km=float(m['dist_km']))
                     for nm, (o, m) in runs.items()}
        for nm, tag in (('hybridpilot', 'hp'), ('aecms', 'ae'), ('dp', 'dp')):
            o = runs[nm][0]
            T[f'on_{tag}'] = bits(np.asarray(o['eng_on'], float)[:n] > 0.5)
            T[f'soc_{tag}'] = np.round(np.clip(np.asarray(o['soc'], float)[:n], 0, 1) * 250).astype(np.uint8)
        # a second at the powertrain's limit is iterated, so select() can run more than once: keep the last call
        last = {c['k']: c for c in cap}
        cap = [last[q] for q in sorted(last)]
        X = np.stack([c['x'] for c in cap])
        k = np.array([c['k'] for c in cap], int)
        assert k.max() < n
        assert np.all(X[:, 37:65] == 0), 'no preview, no navigation'
        assert np.all(X[:, :12] == X[:1, :12]), 'vehicle features constant'
        T['hp_mask'] = bits(np.isin(np.arange(n), k))
        T['hp_x'] = X[:, XCOL]
        T['hp_soc'] = np.array([c['st']['soc'] for c in cap])
        T['hp_I'] = np.array([c['I0'] for c in cap])
        res['hp_const'] = [float(q) for q in X[0, :12]]
        res['hp_srec'] = [float(c['s']) for c in cap]           # for the check in main(), not exported
        res['hp_meta'] = dict(ref=float(hp['ref']), phev=bool(veh.arch.startswith('phev')), soc_lo=float(veh.battery.soc_lo),
                              soc_hi=float(veh.battery.soc_hi), prior_key=hp_prior_key(veh))
    res['wall'] = time.time() - t0
    return res, T


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--quick', action='store_true', help='one route, one variant per vehicle (smoke test; writes to work/)')
    a = ap.parse_args()
    t00 = time.time()
    out_dir = OUT if not a.quick else os.path.join(REPO, 'work', 'garage_quick')
    os.makedirs(out_dir, exist_ok=True)
    routes = load_routes()
    V = vehicles(routes)
    jobs = []
    for vi, v in enumerate(V):
        vlite = {k: x for k, x in v.items()}
        for ri in v['routes']:
            if a.quick and ri not in (0, 4):
                continue
            for ti in range(len(TEMPS)):
                for li in range(len(LOADS[v['cls']])):
                    if a.quick and not (ti == 1 and li == 0):
                        continue
                    jobs.append(dict(V=vlite, R=routes[ri], vi=vi, ri=ri, ti=ti, li=li))
    jobs.sort(key=lambda j: -(len(j['R']['trip']['v']) * (8 if j['V']['arch'] in HYBRID else 1)))   # long jobs first
    print(f'{len(jobs)} cells on {a.workers} workers', flush=True)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            res = p.map(run_cell, jobs, chunksize=1)
    else:
        res = [run_cell(j) for j in jobs]
    print(f'simulated in {time.time() - t00:.0f} s', flush=True)
    cells = {}
    for j, (r, T) in zip(jobs, res):
        cells[(j['vi'], j['ri'], j['ti'], j['li'])] = (r, T)

    # ---- weights exactly as the browser loads them
    EW, EM = read_pack(os.path.join(INST, 'energynet', 'weights'))
    en = EnergyNet()
    for jx, m in enumerate(en.net.m):
        m.mu, m.sd = EW[f'm{jx}_x_mu'], EW[f'm{jx}_x_sd']
        m.L = [(EW[f'm{jx}_body_{i}_weight'], EW[f'm{jx}_body_{i}_bias']) for i in (0, 2, 4, 6)]
        m.H = (EW[f'm{jx}_head_weight'], EW[f'm{jx}_head_bias'])
        m.mask = list(EM['masked'][jx]) if EM.get('masked') else m.mask
    TW, _ = read_pack(os.path.join(INST, 'tripdetective', 'weights'))
    td = TD.TripDetective()
    for k in TW:
        td.W[k] = TW[k].astype(np.float32)
    bus = TD.TripDetectiveBus()
    bfl = {k: x for k, x in bus.td.W.items() if k not in ('meta', 'n_blocks', 'down') and not k.endswith('_s')}
    nb = int(bus.td.W['n_blocks'])
    bmeta = dict(bus.td.meta, n_blocks=nb, strides=[int(bus.td.W[f'c{i}_s']) for i in range(nb)], params=int(sum(x.size for x in bfl.values())),
                 scale=bus.SCALE)
    write_pack(os.path.join(out_dir, 'tdbus'), bfl, dtype='f16', meta=bmeta)
    BW, _ = read_pack(os.path.join(out_dir, 'tdbus'))
    for k in BW:
        bus.td.W[k] = BW[k].astype(np.float32)
    HW, _ = read_pack(os.path.join(INST, 'hybridpilot', 'weights'))
    hpnet = HP.MLP(); hpnet.p = HW

    # ---- routes pack
    rt, rmeta = {}, []
    for i, r in enumerate(routes):
        for k in FIELDS:
            rt[f'r{i}_{k}'] = r['trip'][k]
        rt[f'r{i}_elev'] = r['elev']
        rt[f'r{i}_xy'] = np.stack([r['x'], r['y']], 1)
        rmeta.append(r['info'])
    keep = {f'r{i}_surface': 'u8' for i in range(len(routes))} | {f'r{i}_elev': 'f16' for i in range(len(routes))}
    from roadsim.vehicle import fill_spec
    full = {v['id']: fill_spec(v['spec']) for v in V}
    vmeta = [dict(id=v['id'], name=v['name'], short=v['short'], type=v['type'], likely=v['likely'], source=v['source'], arch=v['arch'],
                  cls=v['cls'], en=v['en'], routes=v['routes'], soc0=v['soc0'],
                  truth=dict(mass=float(v['spec']['mass_kg'] + v['spec'].get('payload_kg', 0.0)), CdA=float(full[v['id']]['CdA_m2']),
                             Crr=float(full[v['id']]['Crr']), disp=(v['spec'].get('engine') or {}).get('displacement_L') if v['arch'] != 'bev' else None,
                             battery_kWh=(full[v['id']].get('battery') or {}).get('capacity_kWh'),
                             P_kW=(v['spec'].get('engine') or {}).get('max_power_kW'), arch=v['arch'],
                             curb=float(v['spec']['mass_kg']) if v['cls'] == 'car' else float(ZTB.CURB_KG)),
                  logger=logger(v['arch'])[1] if v['cls'] != 'bus' else dict(power=True))
             for v in V]
    meta = dict(physics=PHYSICS, temps=TEMPS, loads=LOADS, routes=rmeta, vehicles=vmeta, q=Q, xcol=XCOL, lhv_kJ_g=LHV_KJ_G,
                td_ledger=TD_LEDGER, policy=dict(HP.DEFAULTS), built='14 Nov 2026',
                note='RoadSim physics v3.1 runs precomputed on this grid (Python); EnergyNet, TripDetective and HybridPilot run live.')
    write_pack(os.path.join(out_dir, 'routes'), rt, dtype='f32', keep=keep, meta=meta)
    RW, _ = read_pack(os.path.join(out_dir, 'routes'))
    rtrip = [{k: RW[f'r{i}_{k}'] for k in FIELDS} for i in range(len(routes))]
    for i in range(len(routes)):
        assert all(np.array_equal(rtrip[i][k], routes[i]['trip'][k]) for k in FIELDS), 'route round trip'

    cases = []
    stats = dict(en=0, td=0, hp=0, hp_s_max_abs=0.0, hp_s_med_abs=[])
    # ---- vehicle packs
    for vi, v in enumerate(V):
        tens, cmeta = {}, {}
        for (vj, ri, ti, li), (r, T) in sorted(cells.items()):
            if vj != vi:
                continue
            tag = f'c{ri}_{ti}_{li}'
            for k, x in T.items():
                tens[f'{tag}_{k}'] = x
            r = dict(r)
            srec = r.pop('hp_srec', None)
            cmeta[tag] = r
            if srec is not None:
                cmeta[tag]['_srec'] = srec
        keepv = {}
        for k in tens:
            if k.endswith(('_hp_soc', '_hp_I')):
                keepv[k] = 'f32'                                 # SOC feedback: float16 would move s by ~1e-3
            elif k.endswith(('_on', '_on_hp', '_on_ae', '_on_dp', '_hp_mask', '_soc', '_soc_hp', '_soc_ae', '_soc_dp')) or '_q_' in k:
                keepv[k] = 'u8'
        srecs = {t: cm.pop('_srec') for t, cm in cmeta.items() if '_srec' in cm}
        write_pack(os.path.join(out_dir, f"v_{v['id']}"), tens, dtype='f16', keep=keepv, meta=dict(id=v['id'], cells=cmeta))
        P, _ = read_pack(os.path.join(out_dir, f"v_{v['id']}"))

        # ---- reference outputs, from the stored inputs as the browser decodes them
        for (vj, ri, ti, li), (r, _) in sorted(cells.items()):
            if vj != vi:
                continue
            tag = f'c{ri}_{ti}_{li}'
            dT = routes[ri]['info']['dT'][ti]
            trip = perturbed(rtrip[ri], dT)
            veh = en_vehicle(v, li)
            o = en_outputs(en, trip, veh)
            cases.append(dict(kind='en', v=v['id'], r=ri, t=ti, l=li, outputs=o)); stats['en'] += 1
            if (ti, li) in ((1, 0), (0, 1), (2, 1)):
                raw, fl = td_raw(v, P, tag, trip, rmeta[ri]['n'])
                net = bus.td if v['cls'] == 'bus' else td
                cases.append(dict(kind='td', v=v['id'], r=ri, t=ti, l=li, pool=[ri], outputs=td_outputs(net, [raw], [fl]).tolist())); stats['td'] += 1
            if v['arch'] in HYBRID and (ti, li) in ((1, 0), (0, 1)):
                X = hp_inputs(P, tag, r['hp_const'])
                y = np.array([hpnet(x) for x in X])
                # the stored inputs reproduce the s RoadSim used (float16 storage, so not exactly)
                srec = np.asarray(srecs[tag])
                sl = np.array([equiv(hpnet, x, s, I, r['hp_meta']) for x, s, I in zip(X, P[f'{tag}_hp_soc'], P[f'{tag}_hp_I'])])
                stats['hp_s_max_abs'] = max(stats['hp_s_max_abs'], float(np.max(np.abs(sl - srec))))
                stats['hp_s_med_abs'].append(float(np.median(np.abs(sl - srec))))
                idx = np.unique(np.linspace(0, len(X) - 1, 6).astype(int))
                cases.append(dict(kind='hp', v=v['id'], r=ri, t=ti, l=li, steps=idx.tolist(),
                                  outputs=[float(q) for i in idx for q in (y[i, 0], y[i, 1])] + [float(np.mean(y[:, 0])), float(np.mean(y[:, 1]))]))
                stats['hp'] += 1
        # pooled reading: every route of this car at the middle temperature, no load
        if v['cls'] != 'bus':
            raws, fls = [], []
            for ri in v['routes']:
                if (vi, ri, 1, 0) not in cells:
                    continue
                trip = perturbed(rtrip[ri], routes[ri]['info']['dT'][1])
                raw, fl = td_raw(v, P, f'c{ri}_1_0', trip, rmeta[ri]['n'])
                raws.append(raw); fls.append(fl)
            if len(raws) > 1:
                cases.append(dict(kind='td', v=v['id'], r=v['routes'][0], t=1, l=0, pool=[ri for ri in v['routes'] if (vi, ri, 1, 0) in cells],
                                  outputs=td_outputs(td, raws, fls).tolist())); stats['td'] += 1
        print(f"  {v['id']:6s} {sum(os.path.getsize(os.path.join(out_dir, f'v_{v['id']}.' + e)) for e in ('json', 'bin')) / 1e6:.2f} MB", flush=True)
    write_json(os.path.join(out_dir, 'reference.json'), {'cases': cases, 'tolerance': 2e-3})
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"reference cases: EnergyNet {stats['en']}, TripDetective {stats['td']}, HybridPilot {stats['hp']}; "
          f"HybridPilot s (equivalence factor, typically 1-4) from the stored inputs and float16 weights vs RoadSim's run: "
          f"max |difference| {stats['hp_s_max_abs']:.1e}, median {np.median(stats['hp_s_med_abs']):.1e}")
    print(f'wrote {out_dir}: {size / 1e6:.2f} MB, {time.time() - t00:.0f} s')
    enc = {(c['v'], c['r'], c['t'], c['l']): c['outputs'] for c in cases if c['kind'] == 'en'}
    for (vi, ri, ti, li), (r, _) in sorted(cells.items()):
        if ti == 1 and li == 0:
            e = enc[(V[vi]['id'], ri, ti, li)]
            print(f"  {V[vi]['id']:6s} route {ri}: EnergyNet {e[0]:.2f} MJ [{e[4] / e[3] * e[0]:.2f}, {e[5] / e[3] * e[0]:.2f}] vs RoadSim {r['E_MJ']:.2f} MJ")
            hp = r.get('hp')
            print(f"  {V[vi]['id']:6s} route {ri}: E {r['E_MJ']:.2f} MJ, closure {r['ledger']['closure_rel']:.1e}, trace rms {r['trace_rms']:.2f} m/s"
                  + (f", % of DP: HP {hp['hybridpilot']['pct_dp'] * 100:.2f}, A-ECMS {hp['aecms']['pct_dp'] * 100:.2f}" if hp else ''))


TD_LEDGER = ['aero', 'rolling', 'grade', 'brake', 'powertrain', 'aux']


def en_vehicle(v, li):
    """EnergyNet's vehicle record for a load level: the record's mass is scaled by EnergyNet's own mass control."""
    veh = dict(v['en'])
    load = LOADS[v['cls']][li]
    if load and veh.get('mass_kg'):
        _, veh = ENX.perturb({k: np.zeros(1) for k in FIELDS}, veh, {'mass': (veh['mass_kg'] + load) / veh['mass_kg']})
    return veh


def en_outputs(en, trip, veh):
    r = en.predict(trip, veh)
    return [float(q) for q in [r['energy_MJ'], r['sigma_log'], r['physics_estimate_MJ'], r['value'], r['lo68'], r['hi68']]
            + list(r['components_MJ'].values())]


def td_raw(v, P, tag, trip, n):
    """TripDetective's raw log of one cell, decoded from the stored integers exactly as the browser does it."""
    raw = np.zeros((len(TD.RAW), n), np.float32)
    raw[0] = trip['v']; raw[1] = trip['grade'] * 100; raw[2] = trip['T_amb']
    raw[3] = trip['wind_speed'] * np.cos((trip['wind_dir'] - trip['heading']) * (np.pi / 180))
    raw[4] = trip['rho_air']; raw[10] = 1.0
    dec = lambda k: np.frombuffer(P[f'{tag}_q_{k}'].astype(np.uint8).tobytes(), '<i2').astype(np.float64)  # noqa: E731
    if v['cls'] == 'bus':
        pw = dec('pbus') / Q['pbus']
        raw[3] = 0.0
        raw[7] = (pw / 1000.0 / 12.0 * 1000.0) / 1000.0
        raw[8] = 0.5
        return raw, np.array([0, 0, 0, 1, 0], np.float32)
    fl = logger(v['arch'])[1]
    for k, row in (('fuel', 5), ('rpm', 6), ('batt', 7), ('soc', 8), ('aux', 9)):
        if f'{tag}_q_{k}' in P:
            raw[row] = dec(k) / Q[k]
    return raw, np.array([float(fl[k]) for k in TD.FLAGS], np.float32)


def td_outputs(net, raws, fls):
    """raw network outputs over every window of every trip (as the instrument exporter's run_net)."""
    I = []
    for raw, fl in zip(raws, fls):
        for s, e in TD.windows(raw.shape[1]):
            I.append(TD.net_input(raw[:, s:e], fl, feat=net.feat))
    h = net.heads([net.encode(*i) for i in I], [i[2] for i in I])
    if hasattr(net, 'apply_offstop'):   # TripDetective v2.1: engine-stop constraint
        h = net.apply_offstop(h, I)
    out = [h['arch'], h['strat'], h['mu'], h['sig']]
    for mu, sg in h['led']:
        out += [mu, sg]
    return np.concatenate([np.ravel(o) for o in out]).astype(float)


def hp_inputs(P, tag, const):
    Xv = P[f'{tag}_hp_x']
    X = np.zeros((len(Xv), HP.N_FEAT))
    X[:, :12] = np.asarray(const, np.float32).astype(np.float64)
    X[:, XCOL] = Xv
    return X


def hp_prior_key(veh):
    """HybridPilot's class key, as make_roadsim_strategy sets it: 'mild' | 'bus_<arch>' | arch."""
    mild = ((getattr(veh, 'control', None) or {}).get('params') or {}).get('engine_on_moving')
    return 'mild' if mild else ('bus_' + veh.arch if veh.cls == 'bus' else veh.arch)


def equiv(net, x, soc, I0, m):
    """the release's own equivalence factor (HybridPilotCore.equivalence), from the integral before this second."""
    core = HP.HybridPilotCore(net=net)
    core.I = I0
    s, _ = core.equivalence(x, soc, m['ref'], m['phev'], window=(m['soc_lo'], m['soc_hi']), prior_key=m.get('prior_key'))
    return s


if __name__ == '__main__':
    main()
