"""Re-run GridSim's checks and compare every number with the published v1 results.

usage: python shared/grid/checks/run_all.py [--quick] [--only name,...] [--ref PYTHON] [--compare-only]

Each check writes its JSON next to this file (as in v1). Before running, the committed JSONs are copied to
$ML_RATCHET_GRID/checks/out/published/; afterwards every numeric leaf is compared with them (relative tolerance 1e-6;
timings are reported, not compared) and the summary goes to checks/out/compare.json.

  --quick       skip the ERCOT years (MILP: 13-21 min per year and configuration) and the calibration
  --ref PYTHON  interpreter with pandapower, PYPOWER and ANDES (the reference libraries pin pandas 2, so they live in
                their own environment; see gridsim/README.md). Without it the two reference checks are skipped.
Needs the data (`mlratchet data grid`). Single thread.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('ML_RATCHET_GRID') or HERE.parent)
OUT = ROOT / 'checks' / 'out'
PUB = OUT / 'published'

# (name, script + args, json files written, needs reference venv, long)
CHECKS = [
    ('powerflow', ['check_powerflow.py'], ['powerflow.json'], True, False),
    ('network_dispatch', ['check_network_dispatch.py'], ['network_dispatch.json'], False, False),
    ('frequency_analytic', ['check_frequency_analytic.py'], ['frequency_analytic.json'], False, False),
    ('frequency_events', ['check_frequency_events.py'], ['frequency_events.json'], False, False),
    ('frequency_dev_events', ['check_frequency_dev_events.py'], ['frequency_dev_events.json'], False, False),
    ('andes', ['check_andes.py'], ['frequency_andes.json'], True, False),
    ('renewables', ['check_renewables.py', '2019', '2022'], ['renewables.json'], False, False),
    ('demand', ['check_demand.py'], ['demand.json'], False, False),
    ('generator', ['check_generator.py'], ['generator.json'], False, False),
    ('rt_scarcity', ['check_rt_scarcity.py'], ['rt_scarcity.json'], False, False),   # physics 1.2 process (any version)
    ('calibrate_ercot', ['calibrate_ercot.py'], [], False, True),
    ('ercot_2019', ['check_ercot_dispatch.py', '2019', 'calibrated', 'merit', 'economic'], ['ercot_dispatch_2019.json'], False, True),
    ('ercot_2022', ['check_ercot_dispatch.py', '2022', 'calibrated', 'merit', 'economic'], ['ercot_dispatch_2022.json'], False, True),
    ('summarize_ercot', ['summarize_ercot.py', '2019', '2022'], [], False, True),
    ('inertia_2019', ['check_ercot_inertia.py', '2019', 'calibrated'], ['ercot_inertia.json'], False, True),
    ('inertia_2022', ['check_ercot_inertia.py', '2022', 'calibrated'], ['ercot_inertia.json'], False, True),
    ('speed', ['check_speed.py'], ['speed.json'], False, False),
]
TIMING_KEYS = ('speed', 'seconds', 'wall', 'elapsed', 'runtime', 'minutes', 'per_state', 'time_s', 'solve_s', 'build_s')
TIMING_PREFIX = ('t_gridsim', 't_pypower', 't_pandapower', 't_andes', 't_solve', 't_total', 't_build')
NOISE = 1e-9   # both values below this: solver round-off (e.g. 2e-14 vs 1e-14 degrees), counted as equal


def leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f'{path}.{k}' if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f'{path}[{i}]')
    elif isinstance(x, bool) or x is None or isinstance(x, str):
        yield path, x
    elif isinstance(x, (int, float)):
        yield path, float(x)


def compare(old, new, rtol=1e-6, timing_file=False):
    a, b = dict(leaves(old)), dict(leaves(new))
    res = {'numbers': 0, 'equal': 0, 'timings': 0, 'differ': [], 'missing': [], 'new': sorted(set(b) - set(a))[:20]}
    for k, v in a.items():
        if k not in b:
            res['missing'].append(k)
            continue
        w = b[k]
        if not isinstance(v, float):
            if v != w:
                res['differ'].append([k, v, w])
            continue
        res['numbers'] += 1
        last = k.split('.')[-1].split('[')[0].lower()
        if timing_file or any(x in k.lower() for x in TIMING_KEYS) or last.startswith(TIMING_PREFIX):
            res['timings'] += 1
            continue
        if not isinstance(w, float):
            res['differ'].append([k, v, w])
        elif (math.isnan(v) and math.isnan(w)) or abs(v - w) <= rtol * max(abs(v), abs(w), 1e-12)                 or max(abs(v), abs(w)) < NOISE:
            res['equal'] += 1
        else:
            res['differ'].append([k, v, w])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--ref', help='python with pandapower / PYPOWER / ANDES')
    ap.add_argument('--compare-only', action='store_true')
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if not PUB.exists():
        PUB.mkdir(parents=True)
        for p in HERE.glob('*.json'):
            shutil.copy2(p, PUB / p.name)
    only = set(a.only.split(',')) if a.only else None
    env = dict(os.environ, OMP_NUM_THREADS='1', ML_RATCHET_GRID=str(ROOT), PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    ran = {}
    t_start = time.time()
    for name, cmd, outs, ref, long in CHECKS:
        if a.compare_only or (only and name not in only) or (a.quick and long and not only):
            continue
        py = a.ref if ref else sys.executable
        if ref and not a.ref:
            print(f'-- {name}: skipped (needs --ref)')
            continue
        t = time.time()
        r = subprocess.run([py, *cmd], cwd=HERE, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        ran[name] = {'rc': r.returncode, 'seconds': round(time.time() - t, 1)}
        print(f'-- {name}: rc {r.returncode}, {time.time() - t:.0f} s')
        if r.returncode:
            print(r.stderr[-2000:])
    summary = {}
    prev = json.loads((OUT / 'compare.json').read_text()).get('rerun', {}) if (OUT / 'compare.json').exists() else {}
    for p in sorted(PUB.glob('*.json')):
        cur = HERE / p.name
        if not cur.exists():
            continue
        c = compare(json.loads(p.read_text()), json.loads(cur.read_text()), timing_file=p.name == 'speed.json')
        rerun = cur.stat().st_mtime >= t_start or prev.get(p.name, False) or cur.read_bytes() != p.read_bytes()
        c['rerun'] = bool(rerun)
        summary[p.name] = c
        flag = ('same' if not c['differ'] and not c['missing'] else 'DIFFERS') if rerun else 'not re-run'
        print(f'{p.name:52s} {c["equal"]:5d}/{c["numbers"] - c["timings"]:<5d} numbers equal  {c["timings"]:3d} timings  {flag}')
        for d in c['differ'][:6]:
            print('    ', d)
    (OUT / 'compare.json').write_text(json.dumps({'ran': ran, 'rerun': {k: v['rerun'] for k, v in summary.items()},
                                                  'compare': summary}, indent=1, default=str))
    return 0 if all(r['rc'] == 0 for r in ran.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
