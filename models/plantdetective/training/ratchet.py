"""Ratchet bookkeeping.  A rung is kept only if it beats the incumbent on the frozen simulated validation set of the
current bank version by more than MARGIN (skill score in [0,1]) AND passes the dev gate (real dev units, dev years:
its dev score must not fall below the incumbent's by more than DEV_TOL).  When the simulator (bank version) changes,
the incumbent is re-scored on the new frozen set first.  Every rung is logged with the reason.

usage: python ratchet.py score <model> <set_version>            (print scores)
       python ratchet.py rung <name> <model> <set_version> <incumbent model or -> "<what changed>"
"""
from common import *
import json, time
import numpy as np
from evaluate import val_score, dev_table, dev_score
from model import load_any as load

MARGIN = 0.01
DEV_TOL = 0.005
LOG = f'{ROOT}/ratchet.json'


def scores(model, set_version):
    W = load(model)
    v, acc, sk = val_score(W, set_version)
    rows = dev_table(W, kinds=('month', 'quarter', 'year'))
    d = {k: dev_score(rows, k)[0] for k in ('month', 'quarter', 'year')}
    return dict(val=v, val_acc=acc, dev=float(np.mean(list(d.values()))), dev_by_kind=d,
                dev_acc=dev_score(rows, 'year')[1])


def rung(name, model, set_version, incumbent, what):
    log = json.load(open(LOG)) if os.path.exists(LOG) else dict(margin=MARGIN, dev_tol=DEV_TOL, rungs=[])
    s = scores(model, set_version)
    entry = dict(rung=name, model=model, set=set_version, what=what, time=time.strftime('%Y-%m-%d %H:%M'), **s)
    if incumbent in (None, '-'):
        entry.update(kept=True, reason='first rung (incumbent by definition)')
    else:
        si = scores(incumbent, set_version)
        entry['incumbent'] = incumbent; entry['incumbent_val_on_this_set'] = si['val']; entry['incumbent_dev'] = si['dev']
        dv = s['val'] - si['val']; dd = s['dev'] - si['dev']
        ok = dv > MARGIN and dd >= -DEV_TOL
        entry['kept'] = bool(ok)
        entry['reason'] = (f'val {s["val"]:.3f} vs {si["val"]:.3f} (d={dv:+.3f}, margin {MARGIN}); '
                           f'dev {s["dev"]:.3f} vs {si["dev"]:.3f} (d={dd:+.3f}, tol {DEV_TOL}) -> '
                           + ('kept' if ok else 'rejected'))
    log['rungs'].append(entry)
    json.dump(log, open(LOG, 'w'), indent=1)
    print(json.dumps(entry, indent=1))
    return entry


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        print(json.dumps(scores(sys.argv[2], sys.argv[3]), indent=1))
    else:
        rung(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
