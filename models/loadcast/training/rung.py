"""Score a rung (a cold + warm network pair) and log it in ratchet.json.

    python3 rung.py <run> <frozen_bank> <frozen_noise> "<note>"      # score + decide against the incumbent
    python3 rung.py --rescore <frozen_bank> <frozen_noise>            # new frozen set: re-score every rung in order

Decision rule: keep the rung only if the frozen-set objective rises by more than 0.01 over the incumbent AND the dev gate
passes: dev score (mean of cold and warm median CV(RMSE) over the 84 dev building-meters, 2016 history -> 2017) no more
than 2 % (relative) worse than the incumbent's. Exam is never computed here.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E, lcnet as N

LOG = f'{C.ROOT}/ratchet.json'
_CACHE = {}


def frozen(bank, noise):
    key = (bank, noise)
    if key not in _CACHE:
        U = D.load_bank(bank, noise, seed=12345)
        for u in U: D.prepare(u)
        _CACHE[key] = (U, E.ref_cvs(U))
    return _CACHE[key]


def dev_units():
    if 'dev' not in _CACHE:
        U = D.real_units('dev')
        for u in U: D.prepare(u, ('full', 'w8', 'w2'))
        _CACHE['dev'] = U
    return _CACHE['dev']


def score(run, bank, noise):
    nets = {m: N.load(f'{C.ROOT}/ckpt/{run}_{m}.pt')[0] for m in ('cold', 'warm')}
    U, ref = frozen(bank, noise)
    df = E.eval_units(U, nets)
    sk = E.skill(df, ref)
    dv = E.eval_units(dev_units(), nets, modes=(('cold', 'full'), ('warm', 'full'), ('warm', 'w8'), ('warm', 'w2')))
    dsum = E.summarize(dv, keys=('cold', 'warm', 'warm_w8', 'warm_w2'))
    return dict(frozen=dict(bank=bank, noise=noise, **sk), dev=dict(score=E.dev_score(dv), by_meter=dsum)), dv


def decide(rec, inc):
    if inc is None:
        rec['accepted'] = True; rec['why'] = 'first rung'; return
    s_ok = rec['frozen']['objective'] > inc['frozen']['objective'] + 0.01
    rel = rec['dev']['score'] / inc['dev']['score']
    d_ok = rel <= 1.02
    rec['accepted'] = bool(s_ok and d_ok); rec['vs'] = inc['run']; rec['dev_rel'] = rel
    rec['why'] = (f"frozen objective {rec['frozen']['objective']:.3f} vs {inc['frozen']['objective']:.3f}" +
                  ('' if s_ok else ' (needs +0.01)') + f"; dev score {rec['dev']['score']:.1f} vs {inc['dev']['score']:.1f}" +
                  ('' if d_ok else ' (dev gate failed)'))


def main(run, bank, noise, note):
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    inc = next((r for r in reversed(log) if r['accepted']), None)
    t0 = time.time()
    res, dv = score(run, bank, noise)
    rec = dict(run=run, note=note, time=time.strftime('%Y-%m-%d %H:%M'), **res)
    decide(rec, inc)
    log.append(rec); C.jdump(log, LOG)
    dv.to_csv(f'{C.ROOT}/data/dev_eval_{run}.csv', index=False)
    f = rec['frozen']; d = rec['dev']['by_meter']
    print(f"{run}: frozen[{bank}/{noise}] objective {f['objective']:.3f} (cold skill {f['cold_skill']:.3f}, warm skill "
          f"{f['warm_skill']:.3f}; cold CV {f['cold_cv_med']:.1f} vs dumb {f['dumb_cv_med']:.1f}; warm CV {f['warm_cv_med']:.1f} "
          f"vs TOWT {f['towt_cv_med']:.1f})  dev score {rec['dev']['score']:.1f}  [{time.time() - t0:.0f} s]")
    for m in C.METERS:
        if m in d:
            print(f"   dev {m:12s} n {d[m]['n']:3d}  cold CV {d[m]['cold']['cv_med']:6.1f} NMBE {d[m]['cold']['nmbe_med']:6.1f} cov {d[m]['cold']['cov_mean']:.2f}"
                  f" | warm CV {d[m]['warm']['cv_med']:6.1f} NMBE {d[m]['warm']['nmbe_med']:6.1f} cov {d[m]['warm']['cov_mean']:.2f}"
                  f" | warm8w CV {d[m]['warm_w8']['cv_med']:6.1f} | warm2w CV {d[m]['warm_w2']['cv_med']:6.1f}")
    print('  ', 'ACCEPTED' if rec['accepted'] else 'rejected', '-', rec['why'])


def rescore(bank, noise):
    log = json.load(open(LOG)); inc = None
    for rec in log:
        res, _ = score(rec['run'], bank, noise)
        rec.setdefault('frozen_history', []).append(rec['frozen']); rec.update(res)
        decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['run'], round(rec['frozen']['objective'], 3), round(rec['dev']['score'], 1), rec['accepted'], rec['why'])
    C.jdump(log, LOG)


def redecide():
    """Recompute objectives from the stored skills (objective definition change) and re-take every decision in order."""
    log = json.load(open(LOG)); inc = None
    for rec in log:
        f = rec['frozen']; f['objective'] = 0.5 * (f['cold_skill'] + f['warm_skill'])
        decide(rec, inc); inc = rec if rec['accepted'] else inc
        print(rec['run'], round(f['objective'], 3), rec['accepted'], rec['why'])
    C.jdump(log, LOG)


if __name__ == '__main__':
    if sys.argv[1] == '--redecide': redecide()
    elif sys.argv[1] == '--rescore': rescore(sys.argv[2], sys.argv[3])
    else: main(*sys.argv[1:5])
