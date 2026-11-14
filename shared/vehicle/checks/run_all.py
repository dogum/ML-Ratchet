"""Re-run the full verification chain (any OS). A physics version change means: recalibrate on dev, then every check.

usage: python run_all.py [--skip-calibrate] [--only check_ledger,check_speed]
Logs go to roadsim/logs/; results are the JSON files in this folder; then the scoreboard is rewritten.
The two calibration steps run first and in order; the checks after them run in parallel (one thread each).
"""
import os, sys, subprocess, time, argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(os.path.dirname(HERE), 'roadsim', 'logs')
FIRST = ['epa_vehicles', 'calibrate_dev']
CHECKS = ['check_cycles', 'check_coastdown', 'check_fastsim', 'check_hd', 'check_strategies', 'check_ledger',
          'check_generator']
LAST = ['check_speed']        # timing: run alone


def run(name):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8')
    t = time.time()
    with open(os.path.join(LOGS, f'{name}.log'), 'w', encoding='utf-8') as log:
        r = subprocess.run([sys.executable, f'{name}.py'], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f'{name:<18} {"ok" if r.returncode == 0 else "FAILED (see roadsim/logs/" + name + ".log)"}'
          f'  {time.time() - t:6.0f} s', flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-calibrate', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.only:
        names = a.only.split(',')
        with ThreadPoolExecutor(len(names)) as ex:
            bad = sum(ex.map(run, names))
        sys.exit(1 if bad else 0)
    if not a.skip_calibrate:
        for n in FIRST:
            if run(n):
                sys.exit(1)
    with ThreadPoolExecutor(len(CHECKS)) as ex:
        bad = sum(ex.map(run, CHECKS))
    bad += sum(run(n) for n in LAST)
    run('scoreboard')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
