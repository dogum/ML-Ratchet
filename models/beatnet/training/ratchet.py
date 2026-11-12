"""BeatNet ratchet. Objective: macro-F1 over N/S/V/F on the frozen simulated set (bnet.VAL). A rung is kept only if
(1) the objective rises by more than 0.01 over the incumbent, AND
(2) dev gate: on the 6 real dev records (DS1, fixed seed) the mean F1 of S and V does not fall by more than 0.02.
DS2, the rest of DS1 and SVDB are scored and logged but never used in the decision."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN
LOG = _P.WORK + '/ratchet.json'

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01; d_ok = rec['dev']['sv_f1'] >= inc['dev']['sv_f1'] - 0.02
    rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['ckpt']
    rec['why'] = (f"sim macro-F1 {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') +
                  f"; dev S/V F1 {rec['dev']['sv_f1']:.3f} vs {inc['dev']['sv_f1']:.3f}" + ('' if d_ok else ' (dev gate failed)'))

def evaluate(ckpt):
    net = BN.load_net(ckpt); v = BN.val(net); dv = BN.real(net, 'dev')[0]; ex = BN.real(net, 'ds2')[0]
    return dict(ckpt=os.path.basename(ckpt), val=v, objective=v['objective'], dev=dv, exam=ex, valset=os.path.basename(BN.VAL))

def fmt(s): return ' '.join(f"{c} Se {s['se'][i]:.2f} +P {s['ppv'][i]:.2f} |" for i, c in enumerate('NSVF'))

def main(ckpt, note):
    rec = evaluate(ckpt); rec['note'] = note; log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted']), None); decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    if 'af' in rec['dev']: print('  AF (sim / dev / exam):', *[f"Se {d['af']['se']:.2f} +P {d['af']['ppv']:.2f} AUC {d['af']['auc']:.3f}" for d in (rec['val'], rec['dev'], rec['exam'])])
    print(f"{rec['ckpt']}: objective {rec['objective']:.3f} (sim QRS MAE {rec['val']['qrs_mae_ms']:.1f} ms)\n  sim  {fmt(rec['val'])}\n  dev  {fmt(rec['dev'])}\n  exam {fmt(rec['exam'])}\n ",
          'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore():
    log = json.load(open(LOG)); inc = None
    for rec in log:
        r = evaluate(_P.WORK + '/ckpt/' + rec['ckpt']); rec.update({k: r[k] for k in ('val', 'objective', 'dev', 'exam', 'valset')}); decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['ckpt'], round(rec['objective'], 3), rec['accepted'], rec['why'])
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore()
    else: main(sys.argv[1], sys.argv[2])
