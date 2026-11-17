"""Ratchet bookkeeping: score a candidate rung on the frozen simulated validation set and the real dev gate.

A rung is kept only if its frozen-val skill beats the incumbent of the same mode by > MARGIN (0.01) AND the real dev
composite (evaluate.dev_gate, with the incumbents for the other modes) does not drop by more than GATE_TOL (0.02).
Every rung, kept or not, is appended to ratchet.json with the reason.
usage: python3 ratchet.py <rung_name> <mode> <run_dir> "<description>" [--physics v1] [--force-keep]
"""
import json
import os
import shutil
import sys
import time

import common as C
import evaluate as E
from detective import export_torch

RJ = os.path.join(C.ROOT, 'ratchet.json')
INC = os.path.join(C.ROOT, 'models', 'incumbent')
MARGIN, GATE_TOL = 0.01, 0.02


def load():
    if os.path.exists(RJ):
        return json.load(open(RJ))
    return dict(rule=f'keep if frozen-val skill > incumbent + {MARGIN} AND real dev composite >= incumbent - {GATE_TOL}',
                skill='mean over 12 regression targets (R2 clipped at 0) and 3 classifications (balanced accuracy '
                      'rescaled so chance = 0) on the frozen simulated validation set of the physics version',
                incumbent={}, rungs=[])


def main(name, mode, run, desc, physics='v1', force=False, val_override=None):
    R = load()
    res = json.load(open(os.path.join(C.ROOT, 'models', run, 'result.json')))
    skill = val_override if val_override is not None else res['val_skill']['_overall']
    os.makedirs(INC, exist_ok=True)
    cand = os.path.join(C.ROOT, 'models', 'cand')
    if os.path.exists(cand):
        shutil.rmtree(cand)
    shutil.copytree(INC, cand)
    export_torch(os.path.join(C.ROOT, 'models', run, 'final.pt'), os.path.join(cand, f'{mode}.npz'))
    inc = R['incumbent'].get(mode)
    gate_c, det_c = E.dev_gate(E.load_nets(cand))
    gate_i = R.get('gate_incumbent')
    if inc is None:
        keep, why = True, 'first rung for this mode'
    elif force:
        keep, why = True, 'forced (physics version change; re-baselined)'
    else:
        dv = skill - inc['val_skill']
        okg = gate_i is None or gate_c >= gate_i - GATE_TOL
        keep = dv > MARGIN and okg
        why = (f'val skill {skill:.4f} vs incumbent {inc["val_skill"]:.4f} (delta {dv:+.4f}, need > {MARGIN}); '
               f'dev gate {gate_c:.4f} vs {gate_i if gate_i is None else round(gate_i, 4)} '
               f'({"ok" if okg else "FAIL"})')
    rung = dict(name=name, mode=mode, run=run, description=desc, physics=physics, val_skill=skill,
                val_per_task=res['val_skill'], cfg=res['cfg'], dev_gate=gate_c, dev=E.strip(det_c), kept=bool(keep),
                reason=why, time=time.strftime('%Y-%m-%d %H:%M'))
    R['rungs'].append(rung)
    if keep:
        shutil.copy(os.path.join(cand, f'{mode}.npz'), os.path.join(INC, f'{mode}.npz'))
        R['incumbent'][mode] = dict(rung=name, run=run, val_skill=skill)
        R['gate_incumbent'] = gate_c
    json.dump(R, open(RJ, 'w'), indent=1, default=float)
    print(name, mode, 'KEPT' if keep else 'rejected', '|', why, '| gate %.4f' % gate_c)
    return keep


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    phys = 'v1'
    if '--physics' in sys.argv:
        phys = sys.argv[sys.argv.index('--physics') + 1]
        a = [x for x in a if x != phys]
    main(a[0], a[1], a[2], a[3], physics=phys, force='--force-keep' in sys.argv)


def rebaseline(physics, feat_tag, note):
    """Physics change: re-score every incumbent on the NEW frozen validation set (same shard ids, new physics) and
    make those the numbers to beat. Logged as a rung entry of kind 'rebaseline'."""
    import numpy as np
    import train as T
    from detective import Net
    R = load()
    for mode, inc in R['incumbent'].items():
        net = Net(os.path.join(INC, f'{mode}.npz'))
        Xv, Yv, Kv, _ = T.load_feats(feat_tag, mode, True)
        mu, sg, lg = net.raw(Xv)
        S = T.skills(mu, sg, lg, Yv, Kv)
        R['rungs'].append(dict(name=f'rebaseline-{physics}-{mode}', mode=mode, kind='rebaseline', physics=physics,
                               incumbent=inc['rung'], old_val_skill=inc['val_skill'], val_skill=S['_overall'],
                               val_per_task=S, kept=True, reason=note, time=time.strftime('%Y-%m-%d %H:%M')))
        inc['val_skill'] = S['_overall']
        inc['physics'] = physics
        print(mode, inc['rung'], 'old %.4f -> new frozen val %.4f' % (R['rungs'][-1]['old_val_skill'], S['_overall']))
    R['physics'] = physics
    R['frozen_val'] = f'bank v1 shards 900-904 featurized as {feat_tag}'
    json.dump(R, open(RJ, 'w'), indent=1, default=float)
