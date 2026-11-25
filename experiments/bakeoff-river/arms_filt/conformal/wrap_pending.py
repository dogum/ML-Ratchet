"""Wrap every registered base arm that has no conformal wrapper yet (for agent C or the coordinator).

usage:
    source /home/claude/bakeoff/river/harness/env.sh
    python3 /home/claude/bakeoff/river/arms_filt/conformal/wrap_pending.py            # list what would be wrapped
    python3 /home/claude/bakeoff/river/arms_filt/conformal/wrap_pending.py --run      # fit + register each
    python3 .../wrap_pending.py --run --only seq_tcn,tab_gbm                          # a subset

Skips: wrapper arms (family 'conformal' or id ending in _conformal), the dumb reference 'climatology', arms already
wrapped with the SAME base hash. An arm re-registered with a new hash (register.py --replace) gets its wrapper refit
and re-registered with --replace. Runs one arm at a time (nice 5, one thread). Refuses once the exam has started.
Each wrapper costs one dev pass of its base arm (3 protocol/lead jobs for gauged arms) + seconds of fitting; the
wrapper itself is not re-scored with evaluate.py (--no-eval; its in-sample dev numbers are in work/fit_report.json and
the card), register.py smoke-tests it.
"""
import argparse
import json
import os
import subprocess
import sys

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
        w = regs.get(f'{aid}_conformal')
        cal = os.path.join(HERE, f'{aid}_conformal', 'calib.json')
        wrapped_hash = json.load(open(cal)).get('base_hash') if os.path.exists(cal) else None
        if w and wrapped_hash == r['hash']:
            continue
        out.append((aid, r['arm_dir'], 'refit (base re-registered)' if w else 'new'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    if REG.exam_started():
        sys.exit('the exam has started or finished: no new wrappers')
    todo = pending()
    if a.only:
        keep = set(a.only.split(','))
        todo = [t for t in todo if t[0] in keep]
    for aid, d, why in todo:
        print(f'{aid:28s} {why:28s} {d}')
    if not a.run:
        return
    for aid, d, why in todo:
        wdir = os.path.join(HERE, f'{aid}_conformal')
        replace = os.path.exists(os.path.join(REG.C.REGISTRY, f'{aid}_conformal.json'))
        print(f'== wrapping {aid} ({why})', flush=True)
        r = subprocess.run(['nice', '-n', '5', sys.executable, os.path.join(HERE, 'fit_wrapper.py'), d, '--no-eval'])
        if r.returncode:
            print(f'!! fit failed for {aid}; continuing', flush=True)
            continue
        cmd = [sys.executable, os.path.join(HARNESS, 'register.py'), wdir] + (['--replace'] if replace else [])
        r = subprocess.run(['nice', '-n', '5'] + cmd)
        print(f'== {aid}_conformal registration exit {r.returncode}', flush=True)


if __name__ == '__main__':
    main()
