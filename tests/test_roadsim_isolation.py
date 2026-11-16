"""RoadSim runs are independent of what the same process simulated before (15 Nov 2026).

HybridPilot's v2.2 exam notes a "reproducibility gap" on a generated SUV plug-in on VED_381_1188 (31.8 g / end SOC 0.455
in one run, 39.7 g / 0.835 in another) and suspected RoadSim state surviving between runs (the vehicle cache keeps one
compiled Vehicle per spec). It was not RoadSim: the exam set holds two cases with that case_id (start SOC 0.49 and 0.90)
and the scorer keeps whichever finishes last. All 432 exam cases run in one process, forwards and backwards, give
bit-identical results, equal to the stored 3-worker results.

This test keeps it that way: trip B on its own in a fresh process must equal trip B after other trips A on the same and
other vehicles in this process (other strategies, start SOC, temperatures, physics presets, battery aging, sub-steps),
bit for bit, on physics v2, v3 and v3.1 (< 20 s)."""
import hashlib, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROADSIM = os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim')
sys.path.insert(0, ROADSIM)

PHYS = ['roadsim-physics-v2', 'roadsim-physics-v3', 'roadsim-physics-v3.1']
COMBOS = [('suv', 'phev_powersplit'), ('car', 'hev_parallel_p2'), ('car', 'hev_series'), ('car', 'conv')]


def _specs(phys):
    from roadsim.generator import generate_vehicles
    return [generate_vehicles(1, seed=7, cls=c, arch=a, options=phys)[0] for c, a in COMBOS]


def _digest(out):
    """every output array, the ledger and the summary (without wall-clock timing), hashed."""
    m = hashlib.sha256()
    for k in sorted(out):
        v = out[k]
        if isinstance(v, np.ndarray):
            m.update(k.encode()); m.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for q in sorted(v):
                if q not in ('wall_s', 'x_realtime'):
                    m.update(f'{k}.{q}={v[q]!r}'.encode())
        else:
            m.update(f'{k}={v!r}'.encode())
    return m.hexdigest()


def _trip_b(spec, phys):
    """the reported case in miniature: a plug-in starting at SOC 0.49, cold, on a city cycle."""
    from roadsim import simulate, make_trip
    tr = make_trip('udds', T_amb=-7.0, wind_speed=4.0, wind_dir=30.0)
    return _digest(simulate(spec, tr, init_soc=0.49, init_temps={'engine': -7.0, 'cabin': -7.0, 'battery': -7.0},
                            record=True, options=phys))


def _trips_a(spec, phys):
    """different start SOC, warm start, other strategies, physics v1, an aged pack with sub-steps."""
    from roadsim import simulate, make_trip, list_strategies
    tr = make_trip('nycc', T_amb=35.0, solar=800.0, grade=0.02)
    hot = {'engine': 90.0, 'cabin': 30.0, 'battery': 40.0}
    strats = [s for s in list_strategies(spec['architecture']) if s not in ('dp', 'mpc')]
    simulate(spec, tr, strategy=strats[-1], init_soc=0.9, init_temps=hot, record=False, options=phys)
    simulate(spec, tr, strategy=strats[0], init_soc=0.3, init_temps=hot, options='roadsim-physics-v1')
    simulate(spec, tr, init_soc=0.9, options={'preset': phys, 'substeps': 2, 'batt_age': {'years': 8.0, 'efc': 900.0}})


def _fresh(phys):
    """trip B on each vehicle in a new process (started for all presets at once)."""
    code = ('import sys, json; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); '
            'import test_roadsim_isolation as T; '
            'print(json.dumps([T._trip_b(s, sys.argv[3]) for s in T._specs(sys.argv[3])]))')
    return subprocess.Popen([sys.executable, '-c', code, ROADSIM, HERE, phys], stdout=subprocess.PIPE, text=True,
                            env=dict(os.environ, OMP_NUM_THREADS='1'))


def test_trip_b_alone_equals_trip_b_after_trip_a():
    procs = {phys: _fresh(phys) for phys in PHYS}
    for phys in PHYS:
        specs = _specs(phys)
        n = len(specs)
        got = {}
        for i in reversed(range(n)):                    # other order than the fresh process
            _trips_a(specs[i], phys)                    # same vehicle (shares the cached Vehicle)
            _trips_a(specs[(i + 1) % n], phys)          # and another vehicle
            got[i] = _trip_b(specs[i], phys)
        out, _ = procs[phys].communicate(timeout=120)
        assert procs[phys].returncode == 0
        fresh = json.loads(out.strip().splitlines()[-1])
        for i in range(n):
            assert got[i] == fresh[i], f'{phys} {COMBOS[i]}: trip B differs after other trips in the same process'
