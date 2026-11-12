"""QuakeNet ratchet. Objective: mean skill (1 - MAE / MAE of guessing the median, floored at 0; back-azimuth: 1 - mean
angular error / 90 degrees) over magnitude, distance, log depth and back-azimuth on a frozen set of 5,000 simulated records
(seed 99). Kept only if the objective rises by more than 0.01 AND on the dev records (40 real events, never in the exam)
the mean of its four errors, each relative to the incumbent's, is no more than 5% worse. The exam is reported only."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); import qnet as QN
LOG = _P.WORK + '/ratchet.json'; KEYS = ('mw_mae', 'dist_mae', 'depth_med_km', 'baz_med_deg')

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01; rel = float(np.mean([rec['dev'][k] / max(inc['dev'][k], 1e-9) for k in KEYS])); d_ok = rel <= 1.05
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['ckpt']
    rec['why'] = f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') + ('' if d_ok else f"; dev errors {100 * (rel - 1):+.0f}% vs the incumbent")

def main(ckpt, note):
    net = QN.load_net(ckpt); v = QN.val(net); dv = QN.real(net, 'dev')[0]; ex = QN.real(net, 'exam')[0]
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=os.path.basename(ckpt), note=note, val=v, objective=v['objective'], dev=dv, exam=ex); decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    f = lambda s: ' '.join(f"{k} {s[k]:.2f}" for k in KEYS)
    print(f"{rec['ckpt']}: objective {v['objective']:.3f} skills {json.dumps({k: round(x, 3) for k, x in v['skill'].items()})}\n  sim  {f(v)}\n  dev  {f(dv)}\n  exam {f(ex)} | mw bias {ex['mw_bias']:+.2f} coverage mw {ex['mw_in_1sigma']:.2f} dist {ex['dist_in_1sigma']:.2f} depth {ex['logz_in_1sigma']:.2f}\n  ", 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore():
    log = json.load(open(LOG)); inc = None
    for rec in log:
        net = QN.load_net(_P.WORK + '/ckpt/' + rec['ckpt']); rec['val'] = QN.val(net); rec['objective'] = rec['val']['objective']; rec['dev'] = QN.real(net, 'dev')[0]; rec['exam'] = QN.real(net, 'exam')[0]
        decide(rec, inc); inc = rec if rec['accepted'] else inc
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore()
    else: main(sys.argv[1], sys.argv[2])
