"""The v1.1 ladder (15 Nov 2026): rungs after the release, on the frozen set of bank v2 rebuilt with GridSim physics 1.1
(set v2p11: the HANDOFF P0.5 rebuild, training/outage_check.py), with the release's rule unchanged.

    python ladder_v1_1.py score <model> [<model> ...]   -> WORK/out/scores_<model>.json (val on v2p11 + dev gate)
    python ladder_v1_1.py decide                          -> training/ratchet_v1_1.json

Rule (as released, ratchet.py): keep a rung if its frozen-val skill beats the incumbent's by more than 0.01 AND its dev
skill (40 dev units, 2019 + 2021, mean of month / quarter / year windows) falls by no more than 0.005.  The incumbent is
the release (R4, the 3-network ensemble), re-scored on v2p11.
"""
from common import *
import json
import numpy as np
from ratchet import scores, MARGIN, DEV_TOL
from pdcore import REG

SET = 'v2p11'
RUNGS = [
    ('R4', 'R4_ens', 'v1.0 as released (incumbent): 3-network ensemble trained on bank v2 (GridSim physics 1.0)'),
    ('R5', 'R5_pl', 'R4 recipe (3 networks, seeds 0-2, 40 epochs) retrained on bank v2 rebuilt with GridSim physics '
                    '1.1 (set v2p11, training split)'),
    ('R6', 'R6_cs', 'R5 + count skips: starts per year, minimum up time, median run and minimum down time are '
                    'predicted as corrections to the statistic counted in the window itself (where the window has '
                    'one); 3 networks, seeds 0-2'),
]


def score_file(model):
    return f'{ROOT}/out/scores_{model}.json'


def main():
    if sys.argv[1] == 'score':
        os.makedirs(f'{ROOT}/out', exist_ok=True)
        for m in sys.argv[2:]:
            s = scores(m, SET)
            # per-target skills on val and dev (year windows) for the notebook
            from evaluate import val_score, dev_table, dev_score
            from model import load_any
            W = load_any(m)
            s['val_by_target'] = dict(zip(REG, np.round(val_score(W, SET)[2], 4).tolist()))
            rows = dev_table(W, kinds=('month', 'quarter', 'year'))
            for k in ('month', 'quarter', 'year'):
                s[f'dev_{k}_by_target'] = dict(zip(REG, np.round(dev_score(rows, k)[2], 4).tolist()))
            json.dump(s, open(score_file(m), 'w'), indent=1, default=float)
            print(m, json.dumps({k: s[k] for k in ('val', 'dev', 'dev_by_kind')}, default=float), flush=True)
        return
    out = dict(rule=__doc__, frozen=SET, margin=MARGIN, dev_tol=DEV_TOL, rungs=[])
    inc = None
    for rid, m, what in RUNGS:
        if not os.path.exists(score_file(m)):
            out['rungs'].append(dict(rung=rid, model=m, what=what, status='not scored'))
            continue
        s = json.load(open(score_file(m)))
        e = dict(rung=rid, model=m, what=what, **s)
        if inc is None:
            e.update(kept=True, reason='incumbent (v1.0 release), re-scored on v2p11')
        else:
            dv = s['val'] - inc['val']; dd = s['dev'] - inc['dev']
            ok = dv > MARGIN and dd >= -DEV_TOL
            e.update(kept=bool(ok), incumbent=inc['rung'], reason=(
                f"val {s['val']:.3f} vs {inc['rung']} {inc['val']:.3f} (d={dv:+.3f}, margin {MARGIN}); dev "
                f"{s['dev']:.3f} vs {inc['dev']:.3f} (d={dd:+.3f}, tol {DEV_TOL}) -> {'kept' if ok else 'rejected'}"))
        out['rungs'].append(e)
        if e['kept']:
            inc = e
    out['released'] = inc['rung'] if inc else None
    json.dump(out, open(os.path.join(SRC, 'ratchet_v1_1.json'), 'w'), indent=1, default=float)
    for e in out['rungs']:
        print(e['rung'], round(e.get('val', np.nan), 4), round(e.get('dev', np.nan), 4), e.get('kept'), e.get('reason', e.get('status')))
    print('released:', out['released'])


if __name__ == '__main__':
    main()
