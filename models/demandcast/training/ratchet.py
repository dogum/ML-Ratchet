"""DemandCast ratchet.

Objective: on the FROZEN simulated validation bank (val_<physics version>), mean over demand, wind and solar of
skill = 1 - error / error of the physics prior alone (demand MAPE, wind/solar nMAE).
A rung is kept only if (a) the objective beats the incumbent by > 0.01, and (b) on the dev gate (30 fixed-seed real days
per BA from 2019/2021, never in the exam) the mean of its three errors relative to the incumbent's is <= 1.02.
The exam is never computed here.

python3 ratchet.py <ckpt> "<note>" [--val val_v1]
python3 ratchet.py --rescore --val val_v2     (physics changed: re-score every rung on the new frozen set, re-decide)
"""
import sys, os, json, numpy as np
from common import *
import evaluate as E
from train import load_banks

LOG = f'{ROOT}/ratchet.json'


def gate_rows(B, seed=7, per_ba=30):
    rng = np.random.default_rng(seed)
    rows = np.zeros(len(B['meta_ba']), bool)
    for ba in BAS:
        ix = np.where(B['meta_ba'] == ba)[0]
        rows[rng.choice(ix, size=min(per_ba, len(ix)), replace=False)] = True
    return rows


def dev_errors(nets, Bd, rows):
    pred = E.predict(nets, Bd)
    return {k: E.errors(Bd, pred, k, rows=rows) for k in ('d', 'w', 's')}


def decide(rec, inc):
    if inc is None:
        rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01
    rel = float(np.mean([rec['dev'][k]['err'] / max(inc['dev'][k]['err'], 1e-9) for k in ('d', 'w', 's')]))
    d_ok = rel <= 1.02
    rec['dev_rel'] = rel; rec['vs'] = inc['ckpt']; rec['accepted'] = bool(s_ok and d_ok)
    rec['why'] = (f"objective {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)')
                  + f"; dev errors {100 * (rel - 1):+.1f}% vs incumbent" + ('' if d_ok else ' (gate: <= +2%)'))


def score(ckpt, val):
    nets = E.load(ckpt)
    Bv = load_banks([val]); v = E.sim_score(Bv, E.predict(nets, Bv))
    Bd = dict(np.load(f'{DATA}/banks/real_dev.npz')); rows = gate_rows(Bd)
    return v, dev_errors(nets, Bd, rows)


def main(ckpt, note, val):
    v, dv = score(ckpt, val)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted'] and r['val_set'] == val), None)
    rec = dict(ckpt=ckpt, note=note, val_set=val, val=v, objective=v['objective'], dev=dv)
    decide(rec, inc); log.append(rec); jdump(log, LOG)
    print(f"{ckpt}: objective {v['objective']:.3f} | sim skill d {v['d']['skill']:.3f} w {v['w']['skill']:.3f} s {v['s']['skill']:.3f}"
          f" | sim err d {v['d']['err']:.2f}% (prior {v['d']['prior_err']:.2f}) w {v['w']['err']:.1f}% s {v['s']['err']:.1f}%\n"
          f"  dev gate: d {dv['d']['err']:.2f}% w {dv['w']['err']:.1f}% s {dv['s']['err']:.1f}% | cov68 d {dv['d']['cov68']:.2f} w {dv['w']['cov68']:.2f} s {dv['s']['cov68']:.2f}\n"
          f"  {'ACCEPTED' if rec['accepted'] else 'rejected'} - {rec['why']}")


def rescore(val):
    log = json.load(open(LOG)); inc = None
    for rec in log:
        v, dv = score(rec['ckpt'], val)
        rec.update(val_set=val, val=v, objective=v['objective'], dev=dv)
        decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['ckpt'], round(rec['objective'], 3), rec['accepted'], rec['why'])
    jdump(log, LOG)


if __name__ == '__main__':
    val = sys.argv[sys.argv.index('--val') + 1] if '--val' in sys.argv else 'val_v1'
    if sys.argv[1] == '--rescore': rescore(val)
    else: main(sys.argv[1], sys.argv[2], val)
