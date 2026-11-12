"""Evaluate a candidate and decide whether it beats the incumbent. Only improvements are kept (the ratchet)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, os, time
import numpy as np
sys.path.insert(0, _P.SRC)
from evaluate import load, evaluate
from realeval import exam
LOG = _P.WORK + '/ratchet.json'
RULE = 4   # rule 4 (rung 7+): like rule 3, but only LOUD (SNR>=12) one-detector chirps count, since quiet ones are physically ambiguous.
# rule 1: never lose Gaussian sensitivity. rule 2 (rung 3+): one objective. rule 3 (rung 6+): objective may not drop by >0.01 and single-detector false alarms must halve

def objective(r):
    """Gaussian-noise sensitivity + glitchy-noise sensitivity - 2 x median chirp-mass error"""
    return r['val']['sens'] + r['sens_glitch'] - 2 * r['val']['mc_relerr_med']

def rejudge(ckpt, note):
    """Re-apply the decision to an already-evaluated candidate under the current rule."""
    log = json.load(open(LOG)); old = next(x for x in log if x['ckpt'] == ckpt)
    rec = dict(old); rec['note'] = note; rec['objective'] = objective(rec)
    inc = next((x for x in reversed(log) if x.get('accepted')), None)
    rec['rule'] = 2; rec['accepted'] = rec['objective'] > objective(inc) + 0.01
    rec['why'] = f"objective {rec['objective']:.3f} vs {objective(inc):.3f}"; rec['vs'] = inc['ckpt']; rec['rejudged'] = True
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    print(rec['ckpt'], 'ACCEPTED' if rec['accepted'] else 'rejected', rec['why'])

def main(ckpt, note, tta=False):
    net = load(ckpt)
    t = time.time()
    r = evaluate(net, tta=tta); rg = evaluate(net, glitch=True, n_per=400, tta=tta)
    ex = exam(net, tta=tta)
    from realeval import hard_background
    hb = hard_background(net)
    rec = {'ckpt': os.path.basename(ckpt) + (' +TTA' if tta else ''), 'note': note, 'tta': tta, 'val': r, 'sens_glitch': rg['sens'], 'eff_glitch': rg['eff'],
           'real': {k: {kk: v[kk] for kk in ('score', 'mc', 'mc_lo', 'mc_hi', 'mc_ref', 'q', 'n_bg_louder', 't_pred')} for k, v in ex['events'].items()},
           'bg': {k: ex['background'][k] for k in ('n_windows', 'livetime_s', 'max', 'q999')}, 'eval_s': round(time.time() - t),
           'hard_bg': {'n': int(len(hb)), 'max': float(hb.max()), 'n_above_7': int(np.sum(hb > 7.0))}}
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    rec['objective'] = objective(rec)
    inc = next((x for x in reversed(log) if x.get('accepted')), None)
    if RULE >= 3 and inc is not None and 'single_fa' in inc['val']:
        rec['rule'] = RULE
        sf = (lambda r: np.mean([r['val']['single_fa'][k] for k in r['val']['single_fa'] if int(k) >= 12])) if RULE == 4 else (lambda r: np.mean(list(r['val']['single_fa'].values())))
        rec['accepted'] = bool(rec['objective'] > objective(inc) - 0.01 and sf(rec) < 0.5 * sf(inc))
        rec['why'] = f"objective {rec['objective']:.3f} vs {objective(inc):.3f}; {'loud ' if RULE == 4 else ''}single-detector false alarms {sf(rec):.3f} vs {sf(inc):.3f}"
        rec['vs'] = inc['ckpt']
    elif RULE >= 2 and inc is not None:
        rec['rule'] = 2
        rec['accepted'] = rec['objective'] > objective(inc) + 0.01
        rec['why'] = f"objective {rec['objective']:.3f} vs {objective(inc):.3f}" + ('' if rec['accepted'] else ' (no improvement)')
        rec['vs'] = inc['ckpt']
    elif inc is None:
        rec['accepted'] = True; rec['why'] = 'first rung'
    else:
        a, b = rec['val'], inc['val']
        better_sens = a['sens'] >= b['sens'] + 0.005 and a['mc_relerr_med'] <= b['mc_relerr_med'] * 1.05
        better_mc = a['mc_relerr_med'] <= b['mc_relerr_med'] * 0.95 and a['sens'] >= b['sens'] - 0.003
        better_glitch = rec['sens_glitch'] >= inc['sens_glitch'] + 0.02 and a['sens'] >= b['sens'] - 0.003
        rec['accepted'] = bool(better_sens or better_mc or better_glitch)
        rec['why'] = ('sensitivity ' if better_sens else '') + ('chirp-mass ' if better_mc else '') + ('glitch-robustness' if better_glitch else '') or 'no improvement'
        rec['vs'] = inc['ckpt']
    log.append(rec); json.dump(log, open(LOG, 'w'), indent=1)
    v = rec['val']
    print(f"{rec['ckpt']}: sens {v['sens']:.3f}  eff8 {v['eff'][8]:.2f} eff10 {v['eff'][10]:.2f}  mc_err {100*v['mc_relerr_med']:.1f}%  cov68 {v['mc_cov68']:.2f}  q_err {v['q_abserr_med']:.2f}  t_err {v['t_abserr_ms']:.0f}ms  sens_glitch {rec['sens_glitch']:.3f}")
    for k, e in rec['real'].items():
        print(f"   {k:10s} score {e['score']:6.2f}  Mc {e['mc']:5.1f} [{e['mc_lo']:.1f},{e['mc_hi']:.1f}] ref {e['mc_ref']:5.1f}  bg louder {e['n_bg_louder']}")
    print(f"   real background: {rec['bg']['n_windows']} windows, max {rec['bg']['max']:.2f}, q999 {rec['bg']['q999']:.2f}, thr@FAP1e-3 {v['thr_fap1e-3']:.2f}")
    print(f"   single-detector chirps above threshold: {v.get('single_fa')}   hard real background (events kept in): max {rec['hard_bg']['max']:.2f}, windows > 7: {rec['hard_bg']['n_above_7']} of {rec['hard_bg']['n']}")
    print('   ACCEPTED' if rec['accepted'] else '   rejected', '-', rec['why'])

if __name__ == '__main__':
    if sys.argv[1] == '--rejudge': rejudge(sys.argv[2], sys.argv[3])
    else: main(sys.argv[1], sys.argv[2], tta='--tta' in sys.argv)
