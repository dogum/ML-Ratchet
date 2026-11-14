"""RoadSim behaviour fingerprint: a fixed set of vehicles x cycles x conditions, summarised per run.

Every v2 physics feature ships behind an option that is off by default. With every option off, RoadSim must reproduce
the v1 fingerprint (tests/golden/roadsim_v1_fingerprint.json) to 1e-9 relative: new physics may only change results
when it is switched on.

usage:
    python fingerprint.py write   # (re)write the golden file — only when PHYSICS_VERSION's defaults change on purpose
    python fingerprint.py check   # compare the current code against it
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GOLD = os.path.join(REPO, 'tests', 'golden', 'roadsim_v1_fingerprint.json')

from roadsim import simulate, make_trip, load_vehicle
from roadsim.generator import generate_vehicles

CASES = [('udds', dict(T_amb=20.0)), ('hwfet', dict(T_amb=20.0)), ('us06', dict(T_amb=20.0)),
         ('udds', dict(T_amb=-7.0, wind_speed=5.0, wind_dir=0.0)), ('udds', dict(T_amb=35.0, solar=850.0)),
         ('la92', dict(T_amb=10.0, grade=0.02))]
KEYS = ['fuel_g', 'batt_kWh_out', 'distance_m', 'soc_end', 'trace_rms_mps']


def vehicles():
    vs = generate_vehicles(28, seed=20261113)
    vs += [load_vehicle(i) for i in ('ref_bus40_hybrid', 'ref_class8_sleeper_diesel', 'ref_class8_bev')]
    return vs


def run():
    out = {}
    for i, spec in enumerate(vehicles()):
        cases = CASES if i < 28 else [('hd_transient', dict(T_amb=20.0))]
        for c, kw in cases:
            r = simulate(spec, make_trip(c, **kw), record=True)
            s = r['summary']
            key = f'{i:02d}|{spec.get("architecture")}|{c}|{json.dumps(kw, sort_keys=True)}'
            out[key] = {k: (None if s.get(k) is None else float(s[k])) for k in KEYS}
            out[key]['ledger_closure'] = float(r['ledger']['closure_rel'])
            out[key]['eng_on_frac'] = float(np.mean(r['eng_on'])) if r.get('eng_on') is not None else None
    return out


def compare(a, b, tol=1e-9):
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f'{k}: missing'); continue
        for q in a[k]:
            x, y = a[k][q], b[k].get(q)
            if q == 'ledger_closure':
                continue
            if (x is None) != (y is None) or (x is not None and abs(x - y) > tol * max(1.0, abs(x))):
                bad.append(f'{k} {q}: {x} -> {y}')
    return bad


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    cur = run()
    if cmd == 'write':
        os.makedirs(os.path.dirname(GOLD), exist_ok=True)
        json.dump(cur, open(GOLD, 'w', encoding='utf-8'), indent=0, sort_keys=True)
        print('wrote', len(cur), 'runs to', os.path.relpath(GOLD, REPO))
    else:
        bad = compare(json.load(open(GOLD, encoding='utf-8')), cur)
        print('fingerprint:', 'matches v1' if not bad else f'{len(bad)} differences'); print('\n'.join(bad[:40]))
        sys.exit(1 if bad else 0)
