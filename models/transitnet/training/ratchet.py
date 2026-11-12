"""TransitNet ratchet. Objective on a frozen set of 5,000 simulated readings (seed 99): 0.5 x (2 AUC - 1) for telling planets
from everything else, plus 0.5 x the mean skill (1 - MAE / MAE of guessing the median, floored at 0) on the four planet
parameters for planets with signal-to-noise of 7 or more. A candidate is kept only if the objective rises by more than 0.01
AND on the 20 + 20 real dev stars (never in the exam) its planet-vs-false-positive log-loss is no more than 0.05 worse and
its median radius-ratio error no more than 1 point worse. The exam is reported, never used to choose."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time
sys.path.insert(0, _P.SRC); import tnet as TN
LOG = _P.WORK + '/ratchet.json'

def decide(rec, inc):
    if inc is None: rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['objective'] > inc['objective'] + 0.01; d, di = rec['dev'], inc['dev']
    l_ok = d['logloss'] <= di['logloss'] + 0.05; k_ok = d['k_frac_med'] <= di['k_frac_med'] + 0.01
    rec['accepted'] = bool(s_ok and l_ok and k_ok); rec['vs'] = inc['ckpt']
    rec['why'] = (f"score {rec['objective']:.3f} vs {inc['objective']:.3f}" + ('' if s_ok else ' (needs +0.01)') + ('' if l_ok else f"; dev log-loss {d['logloss']:.3f} vs {di['logloss']:.3f}")
                  + ('' if k_ok else f"; dev radius-ratio error {100 * d['k_frac_med']:.1f}% vs {100 * di['k_frac_med']:.1f}%"))

def main(ckpt, note):
    t = time.time(); net = TN.load_net(ckpt); v = TN.val(net); dv = TN.dev(net); ex = TN.exam(net)[0]
    log = json.load(open(LOG)) if os.path.exists(LOG) else []; inc = next((r for r in reversed(log) if r['accepted']), None)
    rec = dict(ckpt=os.path.basename(ckpt), note=note, val=v, objective=v['objective'], dev=dv, exam=ex, eval_s=round(time.time() - t)); decide(rec, inc); log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    print(f"{rec['ckpt']}: objective {v['objective']:.3f} (AUC {v['auc']:.3f}; skills " + ', '.join(f"{p} {v[p]['skill']:.2f}" for p in ('log_k', 'b', 'log_aR', 'log_T14')) + ')')
    print('  dev', {k: round(x, 3) if isinstance(x, float) else x for k, x in dv.items()})
    print('  exam AUC %.3f planets called planet %.2f FPs called planet %.2f | k err median %.3f (naive %.3f) b err %.3f aR err %.3f' % (ex['auc'], ex['planets_called_planet'], ex['fp_called_planet'],
          ex['params_all']['k_frac']['median'], ex['params_all']['k_frac_naive']['median'], ex['params_all']['b_abs']['median'], ex['params_all']['aR_frac']['median']))
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])

def rescore():
    log = json.load(open(LOG)); inc = None
    for rec in log:
        net = TN.load_net(_P.WORK + '/ckpt/' + rec['ckpt']); rec['val'] = TN.val(net); rec['objective'] = rec['val']['objective']; rec['dev'] = TN.dev(net); rec['exam'] = TN.exam(net)[0]
        decide(rec, inc); inc = rec if rec['accepted'] else inc; print('rescored', rec['ckpt'], round(rec['objective'], 3), rec['accepted'])
    json.dump(log, open(LOG, 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == '--rescore': rescore()
    else: main(sys.argv[1], sys.argv[2])
