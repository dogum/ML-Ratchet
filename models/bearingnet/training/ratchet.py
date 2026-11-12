"""BearingNet ratchet. Objective on the frozen simulated set (current physics version) = 0.8 * balanced accuracy (4 classes)
+ 0.1 * speed skill + 0.1 * size skill (skill = 1 - MAE / MAE of guessing the median, floored at 0).
A rung is kept only if (a) the objective beats the incumbent by more than 0.01 AND (b) dev gate: on the real dev readings
(CWRU 1 normal + 9 fault files, all sensors; Paderborn K006, KA01, KI03) balanced accuracy is no more than 0.03 below the
incumbent's. The exam is computed and stored for the record only; it never enters a decision.
Usage: ratchet.py CKPT "note"   |   ratchet.py --rescore VALNAME   (new physics version: re-score every rung, re-decide in order)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN
LOG = _P.WORK + '/ratchet.json'; CK = _P.WORK + '/ckpt/'

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01; d = rec['dev']['all']['bal_acc'] - inc['dev']['all']['bal_acc']; d_ok = d >= -0.03
    rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['ckpt']
    rec['why'] = f"sim objective {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') + f"; dev bal. acc {rec['dev']['all']['bal_acc']:.3f} vs {inc['dev']['all']['bal_acc']:.3f}" + ('' if d_ok else ' (dev gate failed)')

def score(ckpt, valname):
    net = BN.load_net(CK + ckpt); v = BN.val(net, valname); dv = BN.dev_score(net)
    ex = None
    if os.path.exists(BN.RD.D + 'real_exam.npz'):
        R = BN.readings(net, 'exam'); ex = dict(cwru_near=BN.summary(R, lambda r: r['kind'] == 'main' and r['src'] == 'CWRU' and r['loc'] == 'near'),
                                                pb=BN.summary(R, lambda r: r['src'] == 'PB' and r['cls'] in BN.CLS))
    return dict(val_name=valname, val=v, objective=v['objective'], dev=dv, exam=ex)

def main(ckpt, note, valname):
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=ckpt, note=note, **score(ckpt, valname)); decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1, default=float)
    v = rec['val']; print(f"{ckpt} [{valname}] objective {v['objective']:.3f} bal_acc {v['bal_acc']:.3f} speed {v['sp_skill']:.3f} size {v['sz_skill']:.3f} by snr {v['fault_acc_by_snr']}")
    print('  conf', v['conf']); print('  dev', json.dumps(rec['dev'])); print(' ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore(valname):
    log = json.load(open(LOG)); inc = None
    for rec in log:
        rec.update(score(rec['ckpt'], valname)); decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['ckpt'], round(rec['objective'], 3), rec['accepted'], rec['why'])
    json.dump(log, open(LOG, 'w'), indent=1, default=float)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore(sys.argv[2])
    else: main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else BN.VAL)
