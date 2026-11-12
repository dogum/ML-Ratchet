"""SwellCast ratchet. Objective: mean skill over log distance, t0, direction, log storm Hs and detection on the frozen simulated set
(visible events only, ridge contrast >= 0.3 dex, the same rule as the real exam). Skill = 1 - MAE / MAE of guessing the
median (direction: 1 - mean angular error / 90 deg), floored at 0.
A rung is kept only if the objective rises by more than 0.01 AND, on the real dev events (confirmed events of 18 dev storms, never in the exam),
the mean of its three errors (log D MAE, t0 MAE, mean direction error), each relative to the incumbent's, is no more than
5% worse. The exam is computed and stored but never used for a decision."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import evaluate as EVL
LOG = os.path.join(EVL.ROOT, 'ratchet.json'); KEYS = ('logD_mae', 't0_mae_h', 'dir_mean_deg')

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01; rel = float(np.mean([rec['dev'][k] / max(inc['dev'][k], 1e-9) for k in KEYS])); d_ok = rel <= 1.05
    rec['dev_rel'] = rel; rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['ckpt']
    rec['why'] = f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') + (f"; dev errors {100 * (rel - 1):+.0f}% vs incumbent" + ('' if d_ok else ' (gate fails)'))

def score(ck):
    net = EVL.load_net(ck); v = EVL.val(net); thr = EVL.det_threshold(net); sf = dict(det_thr=thr)
    dv = EVL.real(net, 'dev', sigf=sf); ex = EVL.real(net, 'exam', sigf=sf); exa = dict(visible=EVL.real(net, 'exam', True, sigf=sf), all=EVL.real(net, 'exam', None, sigf=sf), quiet=EVL.quiet(net, thr))
    return net, v, dv, ex, exa

def main(ck, note):
    net, v, dv, ex, exa = score(ck)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=os.path.basename(ck), note=note, val_set=EVL.VAL, val=v, objective=v['objective'], dev=dv, exam=ex, exam_all=exa); decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    f = lambda s: f"logD {s['logD_mae']:.3f} (D med {s['D_med_pct']:.0f}%) t0 {s['t0_mae_h']:.1f} h dir {s.get('dir_mean_deg', np.nan):.0f} deg det {s.get('detect_rate', s.get('det_auc', np.nan)):.2f}"
    print(f"{rec['ckpt']}: objective {v['objective']:.3f} skills {json.dumps({k: round(x, 3) for k, x in v['skill'].items()})}\n  sim  {f(v)}\n  dev  {f(dv)} n={dv['n']}\n  exam {f(ex)} n={ex['n']}  quiet-2025 false alarms {exa['quiet']['false_alarm']:.2f}\n ", 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore():
    log = json.load(open(LOG)); inc = None
    for rec in log:
        net, v, dv, ex, exa = score(rec['ckpt']); rec.update(val_set=EVL.VAL, val=v, objective=v['objective'], dev=dv, exam=ex, exam_all=exa)
        decide(rec, inc); inc = rec if rec['accepted'] else inc; print(rec['ckpt'], round(v['objective'], 3), rec['accepted'], rec['why'])
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore()
    else: main(sys.argv[1], sys.argv[2])
