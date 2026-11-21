"""The SnowSense ratchet log (ratchet.json).

python3 ratchet.py add <rung> <run> <phys> "<change>"   score run on every frozen val set + dev, decide, log
python3 ratchet.py rescore                                 re-score every rung on every frozen val set present
Rule: a rung is kept if it beats the incumbent by > 0.01 median per-series NSE on the NEWEST frozen simulated set
(val_<phys>) AND passes the dev gate: (1 - NSE_dev_new) <= 1.05 (1 - NSE_dev_incumbent) on the dev stations'
median NSE (WY2011-2025). Every decision is re-taken in order when a new frozen set appears.
"""
import glob
import json
import sys
from common import *
import evaluate as E

RJ = f'{ROOT}/ratchet.json'
MARGIN = 0.01


def load():
    return json.load(open(RJ)) if os.path.exists(RJ) else {'rule': __doc__.strip().split('\n')[3:6], 'rungs': []}


def vals():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{ROOT}/banks/val_p*.npz'))


def sim_scores(r):
    for v in vals():
        if v not in r['sim']:
            r['sim'][v] = E.sim_score(r['run'], v)['nse_median']


def decide(R):
    """Re-take every decision in order on the newest frozen set."""
    v = vals()[-1]
    inc = None
    for r in R['rungs']:
        if r.get('manual'):
            continue
        s, d = r['sim'][v], r['dev']['nse']
        if inc is None:
            r['kept'], r['reason'] = True, 'first rung'
            inc = r
            continue
        si, di = inc['sim'][v], inc['dev']['nse']
        gate = (1 - d) <= 1.05 * (1 - di)
        if s > si + MARGIN and gate:
            r['kept'], r['reason'] = True, f'{v}: {s:.3f} > {si:.3f} + {MARGIN}; dev {d:.3f} vs {di:.3f} passes'
            inc = r
        else:
            why = [] if s > si + MARGIN else [f'{v}: {s:.3f} not > {si:.3f} + {MARGIN}']
            if not gate:
                why.append(f'dev gate failed: {d:.3f} vs {di:.3f}')
            r['kept'], r['reason'] = False, '; '.join(why)
    R['incumbent'] = inc['rung'] if inc else None
    R['decided_on'] = v


if __name__ == '__main__':
    R = load()
    if sys.argv[1] == 'add':
        rung, run, phys, change = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        R['rungs'] = [r for r in R['rungs'] if r['rung'] != rung]
        dev = E.real_score(run, 'dev')
        R['rungs'].append({'rung': rung, 'run': run, 'phys': phys, 'change': change, 'sim': {}, 'dev': dev})
    for r in R['rungs']:
        sim_scores(r)
    decide(R)
    json.dump(R, open(RJ, 'w'), indent=1)
    for r in R['rungs']:
        print(r['rung'], r['run'], {k: round(v, 3) for k, v in r['sim'].items()}, 'dev', round(r['dev']['nse'], 3),
              r['kept'], r['reason'])
