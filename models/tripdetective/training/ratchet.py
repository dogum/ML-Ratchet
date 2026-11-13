"""TripDetective ratchet.
Objective: mean skill on the frozen simulated validation set (average of single-trip and whole-car-pooled scoring):
  architecture (accuracy over the majority class), log mass, log displacement, log CdA, log Crr, engine on/off behaviour
  (EV share, on-threshold power, max EV speed), strategy (accuracy over majority), energy ledger (6 items, kJ/km).
  Skill = 1 - MAE / MAE of guessing the median, floored at 0.
Kept only if the objective rises by more than 0.01 AND the dev gate passes on the 16 real dev cars: the mean of the
dev errors (mass and displacement, per car pooled and per trip), each relative to the incumbent's, is no more than
5 % worse, and the per-trip VED-type accuracy drops by no more than 0.03. The exam is never computed here.
usage: python3 ratchet.py <ckpt> <val_tag> "<note>"   |   python3 ratchet.py --rescore <val_tag>"""
from common import *
import evaluate as E, simbank

LOG = f'{ROOT}/ratchet.json'
DKEYS = ('mass_mae_car', 'disp_mae_car', 'mass_mae_trip', 'disp_mae_trip')


def decide(rec, inc):
    if inc is None:
        rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01
    rel = float(np.mean([rec['dev'][k] / max(inc['dev'][k], 1e-9) for k in DKEYS]))
    a_ok = rec['dev']['vedtype_acc_trip'] >= inc['dev']['vedtype_acc_trip'] - 0.03
    d_ok = rel <= 1.05 and a_ok
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['ckpt']
    rec['why'] = (f"sim objective {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') +
                  f"; dev errors {100 * (rel - 1):+.0f}% vs incumbent" + ('' if rel <= 1.05 else ' (gate fails)') +
                  ('' if a_ok else '; dev VED-type accuracy dropped'))


def score(ckpt, val_tag):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    v = E.sim_scores(net, simbank.load_bank(val_tag))
    dv, _ = E.real_scores(net, 'dev')
    return v, dv


def main(ckpt, val_tag, note):
    v, dv = score(ckpt, val_tag)
    log = jload(LOG, [])
    inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=ckpt, note=note, val_tag=val_tag, val=v, objective=v['objective'], dev=dv)
    decide(rec, inc); log.append(rec); jsave(log, LOG)
    print(f"{ckpt}: objective {v['objective']:.3f}  single {json.dumps(v['single']['skill'])}\n  pooled {json.dumps(v['pooled']['skill'])}")
    print('  dev', {k: round(dv[k], 3) for k in list(DKEYS) + ['vedtype_acc_trip', 'vedtype_acc_car', 'mass_bias_car', 'mass_within_bin_car']})
    print(' ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])


def rescore(val_tag):
    log = jload(LOG, []); inc = None
    for rec in log:
        v, dv = score(rec['ckpt'], val_tag)
        rec['val'] = v; rec['objective'] = v['objective']; rec['dev'] = dv; rec['val_tag'] = val_tag
        decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['ckpt'], round(rec['objective'], 3), rec['accepted'], rec['why'])
    jsave(log, LOG)


if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore(sys.argv[2])
    else: main(sys.argv[1], sys.argv[2], sys.argv[3])
