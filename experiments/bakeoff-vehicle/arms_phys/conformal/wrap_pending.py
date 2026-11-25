"""Wrap every registered arm that has no (current) Mondrian conformal wrapper yet. For agent C or the coordinator.

usage:
    source /home/claude/bakeoff/vehicle/arms_phys/env.sh
    python3 /home/claude/bakeoff/vehicle/arms_phys/conformal/wrap_pending.py             # list what would be wrapped
    python3 /home/claude/bakeoff/vehicle/arms_phys/conformal/wrap_pending.py --run       # fit + register each
    python3 .../wrap_pending.py --run --only tab_gbm,seq_tcn                             # a subset
    python3 .../wrap_pending.py --watch 600                                              # --run every 600 s until stopped

Skips: wrapper arms (family 'conformal' or id ending in _conformal), the dumb reference 'climatology', and arms whose
current wrapper was fitted on the SAME base hash. An arm re-registered with a new hash (register.py --replace) gets its
wrapper refit and re-registered with --replace. One arm at a time (nice 5, one thread; each wrapper costs one dev pass
of its base arm per track + seconds of fitting). Refuses once the exam has started. A wrapper that fails registration is
moved to conformal/failed/ (so it cannot block the exam) and reported; its base stays unwrapped.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.normpath(os.path.join(HERE, '..', '..', 'harness'))
sys.path.insert(0, HARNESS)
import register as REG  # noqa: E402

SKIP = {'climatology'}


def pending():
    regs = REG.registered()
    out = []
    for aid, r in sorted(regs.items()):
        fam = (r.get('meta') or {}).get('family')
        if aid in SKIP or aid.endswith('_conformal') or fam == 'conformal':
            continue
        cal = os.path.join(HERE, f'{aid}_conformal', 'calib.json')
        wrapped = json.load(open(cal)).get('base_hash') if os.path.exists(cal) else None
        reg_w = regs.get(f'{aid}_conformal')
        if reg_w and wrapped == r['hash']:
            continue
        out.append((aid, r['arm_dir'], 'refit (base re-registered)' if reg_w else 'new'))
    return out


def run_once(only=None):
    if REG.exam_started():
        print('the exam has started or finished: no new wrappers'); return 2
    todo = pending()
    if only:
        todo = [t for t in todo if t[0] in only]
    for aid, d, why in todo:
        print(f'{aid:28s} {why:28s} {d}', flush=True)
    rc = 0
    for aid, d, why in todo:
        print(f'== wrapping {aid} ({why}) at {time.strftime("%H:%M:%S")}', flush=True)
        r = subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HERE, 'fit_wrapper.py'), d, '--register'], cwd='/')
        print(f'== {aid}_conformal exit {r.returncode}', flush=True)
        rc = rc or r.returncode
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--watch', type=float, default=0, help='seconds between passes (implies --run)')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    if a.watch:
        while not REG.exam_started():
            run_once(only)
            time.sleep(a.watch)
        return
    if not a.run:
        for aid, d, why in [t for t in pending() if not only or t[0] in only]:
            print(f'{aid:28s} {why:28s} {d}')
        return
    sys.exit(run_once(only))


if __name__ == '__main__':
    main()
