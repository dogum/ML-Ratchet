"""Check: simulation speed, single thread (OMP_NUM_THREADS=1), full output recording on.

Times simulate() on (a) UDDS and (b) a 93-minute mixed trip (UDDS+HWFET+US06+LA92 twice, with grade, wind and
temperature variation) for one vehicle of each architecture (car class), the Class 8 diesel and the BEV bus, and the
throughput over 200 generated vehicles on random cycles. Reports x real time = simulated seconds / wall seconds.
Also times the offline DP solve on UDDS.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
sys.path.insert(0, HERE)
import _opts
_opts.init()          # --options NAME|JSON (physics options for every simulate() call; default v1)
from roadsim import simulate, make_trip, load_vehicle, cycle
from roadsim.sim import get_vehicle
from roadsim.generator import generate_vehicles


def long_trip():
    v = np.concatenate([cycle(c) for c in ('udds', 'hwfet', 'us06', 'la92')] * 2)
    N = len(v); t = np.arange(N)
    return make_trip(v, T_amb=5 + 10 * np.sin(t / 3000), grade=0.03 * np.sin(t / 200), wind_speed=4 + 2 * np.sin(t / 700),
                     wind_dir=(t * 0.1) % 360, heading=(t * 0.05) % 360, surface=(t // 900) % 3)


def timeit(spec, trip, strat=None, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        o = simulate(spec, trip, strategy=strat, init_temps={'engine': 60.0}, options=_opts.get())
        best = min(best, time.perf_counter() - t)
    return len(trip['v']) / best, o['ledger']['closure_rel']


def _cpu():
    import platform
    try:
        return open('/proc/cpuinfo').read().split('model name')[1].split(chr(10))[0].strip(': ')
    except OSError:
        return platform.processor() or platform.machine()


def main():
    res = {}
    L = long_trip(); U = make_trip('udds', T_amb=20)
    cases = [('car conv (AT8)', {'class': 'car', 'architecture': 'conv'}, None),
             ('car conv (CVT)', {'class': 'car', 'architecture': 'conv', 'transmission': {'type': 'CVT'}}, None),
             ('car bev', {'class': 'car', 'architecture': 'bev'}, None),
             ('car hev_powersplit aecms', {'class': 'car', 'architecture': 'hev_powersplit'}, 'aecms'),
             ('car hev_powersplit rule', {'class': 'car', 'architecture': 'hev_powersplit'}, 'powersplit_rule'),
             ('car hev_parallel_p2 aecms', {'class': 'car', 'architecture': 'hev_parallel_p2'}, 'aecms'),
             ('car hev_series aecms', {'class': 'car', 'architecture': 'hev_series'}, 'aecms'),
             ('car phev_powersplit cdcs', {'class': 'car', 'architecture': 'phev_powersplit'}, 'phev_cdcs'),
             ('class8 diesel', 'ref_class8_sleeper_diesel', None),
             ('bus bev', 'ref_bus40_bev', None)]
    for name, spec, strat in cases:
        spec = load_vehicle(spec)
        get_vehicle(spec)   # compile once (cached), as in batch use
        xu, cu = timeit(spec, U, strat)
        xl, cl = timeit(spec, L, strat, reps=1)
        res[name] = dict(x_realtime_udds=round(xu), x_realtime_93min_trip=round(xl), closure=max(cu, cl))
        print(name, res[name], flush=True)
    # mixed generated fleet
    V = generate_vehicles(200, seed=5)
    rng = np.random.default_rng(0)
    cyc = ['udds', 'hwfet', 'us06', 'la92', 'nycc', 'hd_transient', 'cbd14']
    secs = 0; t0 = time.perf_counter()
    for v in V:
        c = cyc[int(rng.integers(len(cyc)))]
        o = simulate(v, make_trip(c, T_amb=float(rng.uniform(-10, 35))), record=True, options=_opts.get())
        secs += len(o['v'])
    wall = time.perf_counter() - t0
    res['generated fleet (200 vehicles, random cycles, incl. compile)'] = dict(sim_seconds=secs, wall_s=round(wall, 2), x_realtime=round(secs / wall))
    # DP
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})
    t = time.perf_counter(); simulate(spec, U, strategy='dp', options=_opts.get()); res['DP on UDDS (solve + forward run), s'] = round(time.perf_counter() - t, 2)
    t = time.perf_counter(); simulate(spec, U, strategy='mpc', options=_opts.get()); res['MPC 15 s horizon on UDDS, s'] = round(time.perf_counter() - t, 2)
    out = dict(check='speed benchmark (single thread)', method=__doc__.strip(), target='>= 200x real time for a 1 Hz car trip', results=res,
               cpu=_cpu())
    json.dump(out, open(os.path.join(HERE, _opts.out('speed.json')), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
