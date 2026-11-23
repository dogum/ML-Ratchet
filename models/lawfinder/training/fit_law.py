"""Search + ratchet driver for one law.  usage: python3 fit_law.py <law> [seconds]

For each physics version in order: symbolic regression on its train bank -> Pareto front -> rungs; every rung so far
is re-scored on this version's frozen set (val) and the dev set, and every decision is re-taken in order.
Writes results/<law>_search_<ver>.json (front) and results/<law>_ratchet.json (all rungs, all decisions, released).
Resumable: a finished version's front is reused.
"""
import sys, os, json, time, importlib
import numpy as np
import lawsearch as ls
import lawratchet as lr

ROOT = lr.ROOT


def main(law, seconds=None):
    L = importlib.import_module(f'law_{law}')
    dev = L.load_dev()
    Xd, yd = L.X_of(dev), L.target(dev)
    rungs = []
    history = []
    for vi, ver in enumerate(L.VERSIONS):
        if hasattr(L, 'bank_ready') and not L.bank_ready(ver):
            print(f'bank {ver} not complete: stopping after the versions before it'); break
        tr = L.load_bank(ver, 'train'); va = L.load_bank(ver, 'val')
        Xt, yt = L.X_of(tr), L.target(tr); Xv, yv = L.X_of(va), L.target(va)
        wt = L.weights(tr) if hasattr(L, 'weights') else None
        fn = f'{ROOT}/results/{law}_search_{ver}.json'
        if os.path.exists(fn):
            front = json.load(open(fn))['front']
        else:
            log = open(f'{ROOT}/logs/{law}_search_{ver}.log', 'a')
            t0 = time.time()
            front = lr.search(L.VARS, Xt, yt, wt, seconds=seconds or L.SEARCH_SECONDS, seed=vi,
                              seeds=L.seeds() if hasattr(L, 'seeds') else (), log=log,
                              **getattr(L, 'SEARCH_KW', {}))
            json.dump({'law': law, 'version': ver, 'n_train': int(len(yt)), 'seconds': time.time() - t0,
                       'front': front}, open(fn, 'w'), indent=1)
        # constant rung (r0) for the first version
        if vi == 0:
            c = float(np.average(yt, weights=wt)) if wt is not None else float(np.mean(yt))
            r0 = lr.make_law(('c', 0), [c], L.VARS)
            rungs.append({'id': 'r0', 'version': ver, 'change': 'A constant (the simulator mean)', **r0})
        for j, f in enumerate(front):
            if f['complexity'] == 1:
                continue
            rungs.append({'id': f'{ver}-c{f["complexity"]}', 'version': ver,
                          'change': f'Physics {ver}, complexity {f["complexity"]}', **f})
        # re-score every rung on this version's frozen set and on dev; re-take every decision in order
        for r in rungs:
            sim, devv = lr.score(r, Xv, yv, Xd, yd, dev_err=L.dev_error)
            r.setdefault('scores', {})[ver] = {'sim': sim, 'dev': devv}
            r['sim'], r['dev'] = sim, devv
            r['dev_r2'] = lr.r2(yd, lr.law_eval(r, Xd))
        dec, inc = lr.walk(rungs)
        history.append({'frozen_set': f'val_{ver}', 'decisions': dec, 'released': inc['id']})
        print(f'== {law} {ver}: n_train {len(yt)} n_val {len(yv)}')
        for r, d in zip(rungs, dec):
            print(f"  {r['id']:10s} c={r['complexity']:2d} sim={r['sim']:.4f} dev={r['dev']:.4f} {d['decision']:8s} {r['expr'][:90]}")
    released = history[-1]['released']
    # error bars: residual sd on the last frozen set, one widen-only factor fitted on dev
    rel = next(r for r in rungs if r['id'] == released)
    res_v = yv - lr.law_eval(rel, Xv); res_d = yd - lr.law_eval(rel, Xd)
    sigma = float(np.std(res_v[np.isfinite(res_v)]))
    k = lr.widen_factor(res_d[np.isfinite(res_d)], sigma)
    for r in rungs:
        rv = yv - lr.law_eval(r, Xv)
        r['sigma_sim'] = float(np.std(rv[np.isfinite(rv)])) if np.isfinite(rv).mean() > 0.99 else None
    out_bars = {'sigma_sim': sigma, 'widen_dev': k, 'sigma': sigma * k,
                'dev_cov68_raw': float(np.mean(np.abs(res_d) <= sigma)),
                'dev_cov68_widened': float(np.mean(np.abs(res_d) <= sigma * k))}
    print('error bars', out_bars)
    out = {'bars': out_bars,'law': law, 'title': L.TITLE, 'vars': L.VARS, 'var_info': L.VAR_INFO, 'target': L.TARGET,
           'margin': lr.MARGIN, 'dev_tol': lr.DEV_TOL,
           'rule': f'Keep a rung if its frozen-set R^2 beats the incumbent by more than {lr.MARGIN} AND its dev error '
                   f'is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; '
                   f'after a physics change every rung is re-scored on the new frozen set and every decision re-taken.',
           'rungs': rungs, 'history': history, 'released': released}
    json.dump(out, open(f'{ROOT}/results/{law}_ratchet.json', 'w'), indent=1, default=float)
    print('released', released)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
