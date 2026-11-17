"""Ratchet bookkeeping. `python3 ratchet.py` scores every trained rung in configs/ladder.json on every frozen
simulated validation set and on the dev runs, then re-takes the keep/reject decisions in order on the NEWEST
frozen set (physics/noise version). Rule: keep if skill > incumbent skill + 0.01 AND the dev gate passes
(dev AUC >= max(0.60, incumbent dev AUC - 0.05)). Writes ratchet.json."""
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
from common import ROOT  # noqa: E402
from model import FFNet  # noqa: E402
import evaluate as ev  # noqa: E402

MARGIN = 0.01


def load_model(name):
    sd = torch.load(f'{ROOT}/models/{name}.pt')
    m = FFNet(w=sd['c1.0.weight'].shape[0])
    m.load_state_dict(sd)
    m.eval()
    return m


def score(name, versions):
    fn = f'{ROOT}/models/{name}.scores.json'
    sc = json.load(open(fn)) if os.path.exists(fn) else {}
    m = None
    for v in versions + ['dev']:
        if v in sc:
            continue
        m = m or load_model(name)
        sc[v] = ev.dev_eval(m) if v == 'dev' else ev.skill(m, ev.load_val(v))
        json.dump(sc, open(fn, 'w'), indent=1)
    return sc


def main():
    lad = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'ladder.json')))
    versions = lad['val_versions']
    newest = versions[-1]
    rows, inc = [], None
    for r in lad['rungs']:
        if not os.path.exists(f"{ROOT}/models/{r['name']}.pt"):
            rows.append(dict(**r, decision='not trained'))
            continue
        sc = score(r['name'], versions)
        s, d = sc[newest]['skill'], sc['dev']['dev_auc']
        if inc is None:
            dec, why = 'kept', 'first rung (incumbent by definition)'
        else:
            si, di = inc['scores'][newest]['skill'], inc['scores']['dev']['dev_auc']
            gate = d >= max(0.60, di - 0.05)
            if s > si + MARGIN and gate:
                dec, why = 'kept', f'skill {s:.3f} > {si:.3f} + {MARGIN} and dev AUC {d:.2f} passes gate (>= {max(0.6, di - 0.05):.2f})'
            elif s > si + MARGIN:
                dec, why = 'rejected', f'skill {s:.3f} beats {si:.3f} but dev gate fails (dev AUC {d:.2f} < {max(0.6, di - 0.05):.2f})'
            else:
                dec, why = 'rejected', f'skill {s:.3f} not > {si:.3f} + {MARGIN} (dev AUC {d:.2f})'
        row = dict(**r, decision=dec, reason=why,
                   scores={k: ({kk: round(vv, 4) for kk, vv in v.items() if kk != 'dev_p'} if isinstance(v, dict) else v)
                           for k, v in sc.items()})
        rows.append(row)
        if dec == 'kept':
            inc = dict(row, scores=sc)
    out = dict(rule=f'keep if skill(newest frozen sim val) > incumbent + {MARGIN} and dev AUC >= max(0.60, incumbent - 0.05); '
                    'skill = 0.5 detection AUC + 0.5 diagnosis accuracy (visible faults)',
               val_versions=versions, decided_on=newest, incumbent=inc['name'] if inc else None, rungs=rows)
    json.dump(out, open(f'{ROOT}/ratchet.json', 'w'), indent=1)
    for r in rows:
        print(r['name'], r['decision'], '|', r.get('reason', ''))
        if 'scores' in r:
            print('   ', {k: (v.get('skill'), v.get('auc'), v.get('diag_acc')) if 'skill' in v else
                         (v.get('dev_auc'), v.get('dev_fpr_at_0p5'), v.get('dev_tpr_at_0p5')) for k, v in r['scores'].items()})


if __name__ == '__main__':
    main()
