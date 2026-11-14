"""RoadSim behaviour fingerprint: a fixed set of vehicles x cycles x conditions, summarised per run.

Every v2 physics feature ships behind an option that is off in physics v1. Run by name (options='roadsim-physics-v1',
since 13 Nov 2026 simulate()'s default is physics v2), RoadSim must reproduce the v1 fingerprint
(tests/golden/roadsim_v1_fingerprint.json) to 1e-9 relative: new physics may only change results when it is switched on.

Since 14 Nov 2026 physics v2 has its own golden file (tests/golden/roadsim_v2_fingerprint.json, options
'roadsim-physics-v2'): later presets (v2.1) may change hybrid and plug-in runs only, so every non-hybrid run of a later
preset must match the v2 golden to 1e-9 (tests/test_roadsim_v2_fingerprint.py). Physics v2.1 has its own golden too
(tests/golden/roadsim_v2_1_fingerprint.json, written 14 Nov 2026 by the v3 pass from the code before any v3 change), and
so has physics v3 (tests/golden/roadsim_v3_fingerprint.json, written 14 Nov 2026 by the v3.1 pass before any v3.1 change)
and physics v3.1 (tests/golden/roadsim_v3_1_fingerprint.json, written at the v3.1 freeze).

usage:
    python fingerprint.py write [--options roadsim-physics-v2]   # (re)write a golden file — only on purpose
    python fingerprint.py check [--options roadsim-physics-v2]   # compare the current code against it
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GOLD = os.path.join(REPO, 'tests', 'golden', 'roadsim_v1_fingerprint.json')
GOLDS = {'roadsim-physics-v1': GOLD, 'roadsim-physics-v2': os.path.join(REPO, 'tests', 'golden', 'roadsim_v2_fingerprint.json'),
         'roadsim-physics-v2.1': os.path.join(REPO, 'tests', 'golden', 'roadsim_v2_1_fingerprint.json'),
         'roadsim-physics-v3': os.path.join(REPO, 'tests', 'golden', 'roadsim_v3_fingerprint.json'),
         'roadsim-physics-v3.1': os.path.join(REPO, 'tests', 'golden', 'roadsim_v3_1_fingerprint.json')}
HYBRID_ARCHS = ('hev_powersplit', 'hev_parallel_p2', 'hev_series', 'phev_powersplit', 'phev_series')

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


def run(options='roadsim-physics-v1'):
    out = {}
    for i, spec in enumerate(vehicles()):
        cases = CASES if i < 28 else [('hd_transient', dict(T_amb=20.0))]
        for c, kw in cases:
            r = simulate(spec, make_trip(c, **kw), record=True, options=options)
            s = r['summary']
            key = f'{i:02d}|{spec.get("architecture")}|{c}|{json.dumps(kw, sort_keys=True)}'
            out[key] = {k: (None if s.get(k) is None else float(s[k])) for k in KEYS}
            out[key]['ledger_closure'] = float(r['ledger']['closure_rel'])
            out[key]['eng_on_frac'] = float(np.mean(r['eng_on'])) if r.get('eng_on') is not None else None
    return out


def is_hybrid_key(k):
    return k.split('|')[1] in HYBRID_ARCHS


def compare(a, b, tol=1e-9, keys=None):
    bad = []
    for k in sorted(set(a) | set(b)):
        if keys is not None and not keys(k):
            continue
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
    opt = sys.argv[sys.argv.index('--options') + 1] if '--options' in sys.argv else 'roadsim-physics-v1'
    gold = GOLDS[opt]
    cur = run(opt)
    if cmd == 'write':
        os.makedirs(os.path.dirname(gold), exist_ok=True)
        json.dump(cur, open(gold, 'w', encoding='utf-8', newline='\n'), indent=0, sort_keys=True)
        print('wrote', len(cur), 'runs to', os.path.relpath(gold, REPO))
    else:
        bad = compare(json.load(open(gold, encoding='utf-8')), cur)
        print('fingerprint:', f'matches {opt}' if not bad else f'{len(bad)} differences'); print('\n'.join(bad[:40]))
        sys.exit(1 if bad else 0)
