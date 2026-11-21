"""Ratchet bookkeeping (ratchet.json).

Rule: a rung is kept only if its skill on the frozen simulated set of the CURRENT physics version beats the incumbent
(re-scored on the same set) by more than MARGIN = 0.01 AND its real dev composite is not lower than the incumbent's
by more than GATE_TOL = 0.02 (5 % of a composite near 0.4).
When the physics changes, a new frozen set is built and every earlier rung is re-scored on it (rescore).

python3 ratchet.py rung <name> <model.npz> <valphys> "<description>"
python3 ratchet.py rescore <valphys>
"""
import common  # noqa: F401
import json
import os
import sys
import time

from common import D
from net import Detective, load_bank
import evaluate as E

RJ = f'{D}/ratchet.json'
MARGIN, GATE_TOL = 0.01, 0.02


def load():
    if os.path.exists(RJ):
        return json.load(open(RJ))
    return {'rule': f'keep if frozen-sim skill > incumbent + {MARGIN} (both scored on the frozen set of the current '
                    f'physics) AND real dev composite >= incumbent - {GATE_TOL}',
            'skill': 'mean over 11 regression targets of max(R2, 0) and (2 AUC - 1) for regulated; frozen simulated '
                     'set = bank shards 900-903 (1,000 basins) of the physics version',
            'dev_composite': 'mean Spearman rho of the 6 out-of-period change targets on the 52 dev basins '
                             '(1999-2008 -> 1980-89, real Q truth) and (2 AUC - 1) for regulated (5 of 52)',
            'incumbent': None, 'rungs': [], 'rescores': []}


_VAL = {}


def val_set(phys):
    if phys not in _VAL:
        X, lab, _ = load_bank(phys, range(900, 904))
        _VAL[phys] = (X, lab)
    return _VAL[phys]


def score(path, phys):
    if not os.path.isabs(path):      # ratchet.json keeps paths relative to the workspace D (made relative on import)
        path = f'{D}/{path}'
    m = Detective(path)
    X, lab = val_set(phys)
    s = E.sim_scores(m, X, lab)
    d, _ = E.dev_gate(path)
    return s, d


def rung(name, path, phys, desc, force=False):
    R = load()
    s, d = score(path, phys)
    inc = R['incumbent']
    if inc is None:
        keep, why = True, 'first rung'
    else:
        si, di = score(inc['path'], phys)  # relative to D
        dv = s['_skill'] - si['_skill']
        okg = d['_composite'] >= di['_composite'] - GATE_TOL
        keep = (dv > MARGIN and okg) or force
        why = (f"sim skill {s['_skill']:.4f} vs incumbent {inc['name']} {si['_skill']:.4f} on val-{phys} "
               f"(delta {dv:+.4f}, need > {MARGIN}); dev composite {d['_composite']:.4f} vs {di['_composite']:.4f} "
               f"({'ok' if okg else 'FAIL'})" + (' [forced]' if force else ''))
    r = {'name': name, 'path': path, 'physics': phys, 'description': desc, 'sim_skill': s['_skill'],
         'sim': {k: v for k, v in s.items() if k != '_skill'}, 'dev_composite': d['_composite'],
         'dev': {k: v for k, v in d.items() if k != '_composite'}, 'kept': bool(keep), 'reason': why,
         'time': time.strftime('%Y-%m-%d %H:%M')}
    R['rungs'].append(r)
    if keep:
        R['incumbent'] = {'name': name, 'path': path, 'physics': phys}
    json.dump(R, open(RJ, 'w'), indent=1)
    print(name, 'KEPT' if keep else 'rejected', why)


def rescore(phys):
    R = load()
    rows = {}
    for r in R['rungs']:
        s, d = score(r['path'], phys)
        rows[r['name']] = {'sim_skill': s['_skill'], 'dev_composite': d['_composite']}
    R['rescores'].append({'val': phys, 'time': time.strftime('%Y-%m-%d %H:%M'), 'scores': rows})
    json.dump(R, open(RJ, 'w'), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'rung':
        rung(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], force='--force' in sys.argv)
    else:
        rescore(sys.argv[2])
