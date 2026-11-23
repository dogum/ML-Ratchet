"""Simulated trips from RoadSim (LawFinder, law 5: trip energy per km).

One sample = one light-duty vehicle from RoadSim's generator (car, SUV, pickup or van; petrol, P2 / power-split /
series hybrid, or battery-electric) driving one training-allowed route:
  - 60 %: a RoadData train_noise VED trip (real Ann Arbor route, lidar grade and airport weather; vehicles with no
    energy signal, so no label exists), driven as recorded or (40 % of those) with the temperature shifted by N(0, 8) C;
  - 40 %: a dynamometer schedule (UDDS, HWFET, US06, SC03, NYCC, LA92, the 20 F cold FTP) on a dyno spec as RoadSim's
    EPA checks run it (no tyre-temperature effects, HVAC only for SC03 and the cold FTP), at the test temperature.
The simulator drives the TRUE weather (ambient + N(0, 1.5) C, wind x U(0.4, 0.9) on road routes); the recorded inputs
are the OBSERVED ones, as EnergyNet's bank does. Starts: VED routes that begin stopped start cold 40 % of the time;
UDDS dyno draws 43 % (the FTP's cold share); otherwise warm.
Vehicle spread (as EnergyNet's fleet.py broadens the generator): CdA x exp(N(0, 0.10)), Crr x exp(N(0, 0.12)), engine
efficiency scale exp(N(-0.08, 0.10)) (a simple stand-in for EnergyNet's model-year prior), BEV motor loss scale
0.47 x U(0.7, 1.5). The same vehicles and routes are drawn for both physics versions (the seed does not depend on it).
    p1 = RoadSim physics v1      p2 = physics v2 (dev-calibrated on-road loads and engine maps; the default)
Recorded: E (MJ/km, fuel at LHV or DC battery) and the law's inputs (law_road.trip_inputs / vehicle_inputs).
usage: python3 road_bank.py <ver> <train|val> <n> [seed] [worker] [n_workers]   (resumable shards of 100)
"""
import os, sys, math, time, copy
import numpy as np, pandas as pd
from _paths import ROOT
import law_road as L

SHARD = 100
TR_KEYS = ('t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface')
DYNO = {'UDDS': 24.0, 'HWFET': 24.0, 'US06': 24.0, 'SC03': 35.0, 'NYCC': 24.0, 'LA92': 24.0, 'FTP_cold_20F': -6.7}
ARCH_P = {'conv': 0.50, 'hev_powersplit': 0.15, 'hev_parallel_p2': 0.08, 'hev_series': 0.05, 'bev': 0.22}
CLS_P = {'car': 0.50, 'suv': 0.30, 'pickup': 0.12, 'van': 0.08}
_ROUTES = None; _CYC = None


def rho_air(T, elev):
    p = 101325.0 * (1 - 2.25577e-5 * np.asarray(elev, float)) ** 5.25588
    return p / (287.05 * (np.asarray(T, float) + 273.15))


def routes():
    global _ROUTES
    if _ROUTES is None:
        import realdata as rd
        _ROUTES = [{k: np.asarray(arr[k]) for k in TR_KEYS} for _, arr in rd.load_trips('train_noise', route_ok=True)]
        print('train_noise routes', len(_ROUTES), flush=True)
    return _ROUTES


def cycles():
    global _CYC
    if _CYC is None:
        import realdata as rd
        allc = rd.cycles(); _CYC = {}
        for n, T in DYNO.items():
            v = np.asarray(allc[n][1]['v'], np.float32); N = len(v)
            _CYC[n] = dict(t=np.arange(N, dtype=np.float32), v=v, grade=np.zeros(N, np.float32), elev=np.zeros(N, np.float32),
                           heading=np.zeros(N, np.float32), T_amb=np.full(N, T, np.float32), wind_speed=np.zeros(N, np.float32),
                           wind_dir=np.zeros(N, np.float32), rho_air=rho_air(np.full(N, T), 0.0).astype(np.float32),
                           surface=np.zeros(N, np.int8))
    return _CYC


def warm_temps(T_amb):
    return {'engine': 85.0, 'oil': 90.0, 'driveline': 70.0, 'axle': 60.0, 'tire_warm': 1.0, 'cabin': 21.0,
            'battery': max(20.0, float(T_amb))}


def dyno_spec(spec, hvac):
    s = copy.deepcopy(spec)
    s['tire'] = dict(s.get('tire') or {}, K_t=0.0, temp_effects=False)
    s.setdefault('aux', {}).setdefault('HVAC', {})['enabled'] = bool(hvac)
    return s


def make_vehicle(rng):
    from roadsim.generator import generate_vehicle
    from roadsim.vehicle import fill_spec
    cls = str(rng.choice(list(CLS_P), p=list(CLS_P.values())))
    arch = str(rng.choice(list(ARCH_P), p=list(ARCH_P.values())))
    s = copy.deepcopy(generate_vehicle(rng, cls=cls, arch=arch, donor_model='v2'))
    s['CdA_m2'] = round(s['CdA_m2'] * float(np.exp(rng.normal(0, 0.10))), 4)
    s['Crr'] = round(s['Crr'] * float(np.exp(rng.normal(0, 0.12))), 5)
    if s.get('engine'):
        s['engine']['eff_scale'] = round(float(np.exp(rng.normal(-0.08, 0.10))), 3)
    if arch == 'bev' and s.get('motors'):
        for m in s['motors']:
            m['loss_scale'] = round(0.47 * float(rng.uniform(0.7, 1.5)), 3)
    full = fill_spec(s)
    return s, full


def draw_route(rng):
    if rng.random() < 0.6:
        R = routes(); tr = dict(R[int(rng.integers(len(R)))])
        if rng.random() < 0.4:
            dT = float(rng.normal(0, 8))
            tr['T_amb'] = (tr['T_amb'] + dT).astype(np.float32)
            tr['rho_air'] = rho_air(tr['T_amb'], tr['elev']).astype(np.float32)
        return tr, 'ved', False, None
    C = cycles(); n = list(DYNO)[int(rng.integers(len(DYNO)))]
    return dict(C[n]), 'dyno', True, n


def one(rng, ver):
    from roadsim import simulate
    spec, full = make_vehicle(rng)
    arch = spec['architecture']
    obs, src, dyno, name = draw_route(rng)
    tru = {k: np.asarray(v).copy() for k, v in obs.items()}
    if not dyno:
        tru['T_amb'] = (obs['T_amb'] + rng.normal(0, 1.5)).astype(np.float32)
        tru['wind_speed'] = (obs['wind_speed'] * rng.uniform(0.4, 0.9)).astype(np.float32)
        tru['rho_air'] = (obs['rho_air'] * (273.15 + obs['T_amb']) / (273.15 + tru['T_amb'])).astype(np.float32)
    stopped = obs['v'][0] <= 1.0
    cold = (dyno and name == 'UDDS' and rng.random() < 0.43) or (not dyno and stopped and rng.random() < 0.4) \
        or name == 'FTP_cold_20F'
    b = full.get('battery') or {}
    lo, hi = (b.get('soc_window') or [0.2, 0.9])
    soc = None if arch == 'conv' else (float(rng.uniform(lo + 0.05, hi - 0.05)) if arch != 'bev' else float(rng.uniform(0.3, 0.95)))
    s = dyno_spec(spec, name in ('SC03', 'FTP_cold_20F')) if dyno else spec
    seed = int(rng.integers(1 << 30))
    out = simulate(s, tru, options=L.PHYSICS[ver], init_soc=soc, init_temps=None if cold else warm_temps(float(tru['T_amb'][0])),
                   record=True, rng=seed)
    grp = L.group_of(arch)
    if grp == 'EV':
        E = float(np.nansum(np.asarray(out['batt_current'], float) * np.asarray(out['batt_voltage'], float)) / 1e6)
    else:
        E = float(np.nansum(np.asarray(out['fuel_rate'], float)) * L.LHV)
    ti = L.trip_inputs(obs['v'], obs['grade'], obs['T_amb'])
    m = float(full['mass_kg']) + float(full.get('payload_kg') or 0)
    vi = L.vehicle_inputs(m, full['CdA_m2'], full['Crr'], arch)
    return dict(E=E / (ti['dist_m'] / 1e3), E_total_MJ=E, src=src, cycle=name or '', cold=float(cold), arch=arch,
                cls=spec['class'], **ti, **vi, trace_rms=float(out['summary'].get('trace_rms_mps', np.nan)))


def build(ver, split, n, seed=1, worker=0, n_workers=1):
    out = f'{ROOT}/banks/road_{ver}'
    os.makedirs(out, exist_ok=True)
    nsh = int(math.ceil(n / SHARD))
    for k in range(worker, nsh, n_workers):
        fn = f'{out}/{split}_{k:03d}.parquet'
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rows = []
        for j in range(SHARD):
            # the seed does not depend on the physics version: p1 and p2 see the same vehicles and routes
            rng = np.random.default_rng([seed, 0 if split == 'train' else 1, k, j])
            try:
                rows.append(one(rng, ver))
            except Exception as e:
                print('skip', repr(e)[:150], flush=True)
        pd.DataFrame(rows).to_parquet(fn + '.tmp'); os.replace(fn + '.tmp', fn)
        print(f'{ver} {split} shard {k+1}/{nsh} {len(rows)} rows {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    a = sys.argv
    build(a[1], a[2], int(a[3]), int(a[4]) if len(a) > 4 else 1, int(a[5]) if len(a) > 5 else 0, int(a[6]) if len(a) > 6 else 1)
