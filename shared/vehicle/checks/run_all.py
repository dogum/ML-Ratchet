"""Re-run the full verification chain (any OS). A physics version change means: recalibrate on dev, then every check.

usage: python run_all.py [--options roadsim-physics-v2] [--recalibrate-v1] [--only check_ledger,check_speed]
Logs go to roadsim/logs/ (<name>__<preset>.log for presets other than v1); results are the JSON files in this folder
(<name>__<preset>.json for presets other than v1); then the scoreboard is rewritten (scoreboard_<preset>.json).
epa_vehicles runs first; the checks after it run in parallel (one thread each).

--options: RoadSim physics options for every check (a preset name such as roadsim-physics-v2, or a JSON dict);
           default physics v1. Passed on through the environment (checks/_opts.py).
--recalibrate-v1: also re-fit the v1 constants (calibrate_dev.py -> roadsim/data/calibration.json). Off by default
           since 14 Nov 2026: physics v1 is frozen (tests/test_roadsim_fingerprint.py) and v2 has its own fitted numbers
           (roadsim/data/calibration_v2.json, fitted by the v2 calibration pass, not by this script).
"""
import os, sys, subprocess, time, argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), 'roadsim', 'logs')
FIRST = ['epa_vehicles']
CHECKS = ['check_cycles', 'check_coastdown', 'check_fastsim', 'check_hd', 'check_strategies', 'check_ledger',
          'check_generator', 'check_d3', 'check_ved_twin', 'check_epa_pool']
LAST = ['check_speed']        # timing: run alone


def run(name):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8')
    t = time.time()
    args = list(EXTRA.get(name, []))
    if OPT:
        env['ROADSIM_OPTIONS'] = OPT
        if name == 'check_d3':
            args += ['--options', OPT]
    tag = '' if not OPT or OPT == 'roadsim-physics-v1' else '__' + (OPT if not OPT.startswith('{') else 'custom')
    with open(os.path.join(LOGS, f'{name}{tag}.log'), 'w', encoding='utf-8') as log:
        r = subprocess.run([sys.executable, f'{name}.py'] + args, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'{name:<18} {"ok" if r.returncode == 0 else "FAILED (see roadsim/logs/" + name + ".log)"}'
          f'  {time.time() - t:6.0f} s', flush=True)
    return r.returncode


OPT = None
EXTRA = {'check_d3': ['--split', 'dev']}     # exam/held-out results are computed once, for the final report


def main():
    global OPT
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-calibrate', action='store_true', help='(kept for compatibility; v1 is no longer re-fitted by default)')
    ap.add_argument('--recalibrate-v1', action='store_true')
    ap.add_argument('--options', default=None)
    ap.add_argument('--only', default='')
    ap.add_argument('--all-splits', action='store_true', help='check_d3 on dev and exam (final report only)')
    a = ap.parse_args()
    OPT = a.options
    if a.all_splits:
        EXTRA['check_d3'] = ['--split', 'all']
    os.makedirs(LOGS, exist_ok=True)
    if a.only:
        names = a.only.split(',')
        with ThreadPoolExecutor(len(names)) as ex:
            bad = sum(ex.map(run, names))
        sys.exit(1 if bad else 0)
    for n in FIRST + (['calibrate_dev'] if a.recalibrate_v1 and not OPT else []):
        if run(n):
            sys.exit(1)
    with ThreadPoolExecutor(len(CHECKS)) as ex:
        bad = sum(ex.map(run, CHECKS))
    bad += sum(run(n) for n in LAST)
    run('scoreboard')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
