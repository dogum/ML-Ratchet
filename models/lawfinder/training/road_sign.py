"""Road law, sign-checked rung (audit 25 Nov 2026, run 27 Nov 2026).

The released road law (p1-c23) gives a battery-electric car less energy per km on a steeper road. This rung asks
whether a law whose energy never falls with net grade can take its place under LawFinder's own ratchet.

Sign check: a law passes on a set of rows if, at every row, E(G + 0.25) >= E(G - 0.25) - 1e-9 (G = net grade, %);
every other input held. A law that does not read G passes.

Steps (all on simulations, except the dev error the ratchet already uses as its gate):
  1. Rebuild check: the p2 banks are rebuilt from their seeds (road_bank.py); every logged rung is re-scored on the
     rebuilt frozen set val_p2 and compared with the logged R^2. If the released law's score moves by more than
     1e-6 the script stops.
  2. Sign check of every logged rung (p1 and p2 fronts) on val_p2, with the share of falling rows overall and for
     battery-electric rows.
  3. A new p2 search on the train bank with the sign check as a hard constraint on the fitting rows (a law that fails
     it scores an infinite loss, so it never enters a population or the hall of fame); constants refitted on the
     train bank as before; any refitted law that fails the check on the train bank is dropped.
  4. The new front ('p2s-cN', complexity order) is appended to the logged ladder and the whole ladder walked under the
     unchanged rule (frozen R^2 beats the incumbent by more than 0.01, dev error no more than 5 % worse). The sign
     check is reported next to each decision; it is not a promotion rule.
  5. For the owner, a second walk in which failing the sign check on val_p2 rejects a rung (a rule change; nothing
     here is released or examined on it).

usage: python road_sign.py [seconds]   -> WORK/results/road_sign.json, WORK/results/road_search_p2s.json
"""
import json, os, sys, time
import numpy as np
import lawsearch as ls
import lawratchet as lr
import law_road as L
from _paths import ROOT, SRC

GI = L.VARS.index('G'); EI = L.VARS.index('e')
DG, TOL = 0.25, 1e-9


def falls(f, X, c):
    Xp = X.copy(); Xm = X.copy(); Xp[:, GI] += DG; Xm[:, GI] -= DG
    with np.errstate(all='ignore'):
        up, dn = f(Xp, c), f(Xm, c)
    return ~(up >= dn - TOL)          # NaN counts as a fall (the law is not defined there)


def law_falls(law, X):
    t = ls.from_json(law['tree'])
    return falls(lambda Z, c: ls.evaluate(t, Z, c), X, np.array(law['consts'], float))


class SignSearch(ls.Search):
    """lawsearch.Search with the sign check as a hard constraint on the fitting rows."""

    def _loss(self, t, c, X, y, w):
        loss = super()._loss(t, c, X, y, w)
        if np.isfinite(loss) and falls(lambda Z, cc: ls.evaluate(t, Z, cc), X, c).any():
            return np.inf
        return loss


def main(seconds=1200.0):
    R = json.load(open(f'{SRC}/results/road_ratchet.json'))
    va = L.load_bank('p2', 'val'); tr = L.load_bank('p2', 'train'); dev = L.load_dev()
    Xv, yv = L.X_of(va), L.target(va); Xt, yt = L.X_of(tr), L.target(tr); Xd, yd = L.X_of(dev), L.target(dev)
    ev = Xv[:, EI] > 0.5
    print(f'banks: train {len(yt)}, val {len(yv)} ({ev.sum()} BEV rows), dev {len(yd)}', flush=True)

    # 1. rebuild check
    rungs = [dict(r) for r in R['rungs']]
    worst = 0.0
    for r in rungs:
        sim, devv = lr.score(r, Xv, yv, Xd, yd, dev_err=L.dev_error)
        r['sim_rebuilt'], r['dev_rebuilt'] = sim, devv
        worst = max(worst, abs(sim - r['scores']['p2']['sim']) if np.isfinite(sim) else 0.0)
    rel = next(r for r in rungs if r['id'] == R['released'])
    d_rel = abs(rel['sim_rebuilt'] - rel['scores']['p2']['sim']); d_dev = abs(rel['dev_rebuilt'] - rel['scores']['p2']['dev'])
    print(f'rebuild check: released {rel["id"]} R2 {rel["sim_rebuilt"]:.6f} vs logged {rel["scores"]["p2"]["sim"]:.6f}; '
          f'dev {rel["dev_rebuilt"]:.6f} vs {rel["scores"]["p2"]["dev"]:.6f}; worst |dR2| over all rungs {worst:.2e}', flush=True)
    if d_rel > 1e-6 or d_dev > 1e-6:
        raise SystemExit('rebuilt frozen set or dev set does not reproduce the logged scores; stopping')

    # 2. sign check of every logged rung
    for r in rungs:
        f = law_falls(r, Xv)
        r['falls_share'] = float(f.mean()); r['falls_share_bev'] = float(f[ev].mean()); r['sign_ok'] = bool(not f.any())

    # 3. sign-constrained search on the train bank (seed 1, as the logged p2 search)
    fn = f'{ROOT}/results/road_search_p2s.json'
    if os.path.exists(fn):
        S = json.load(open(fn))
    else:
        t0 = time.time()
        log = open(f'{ROOT}/logs/road_search_p2s.log', 'a')
        s = SignSearch(L.VARS, seed=1)
        s.fit(Xt, yt, None, seconds=seconds, verbose=True, log=log)
        n = min(len(yt), 6000)
        idx = np.random.default_rng(2).permutation(len(yt))[:n]
        front = [lr.make_law(t, c, L.VARS) for (_, _, t, c) in s.front(Xt[idx], yt[idx], None, refit=True)]
        kept = []
        for f in front:
            ft = law_falls(f, Xt)
            if ft.any():
                print(f"  dropped c{f['complexity']} after refit: falls on {ft.mean():.2%} of train rows", flush=True)
                continue
            kept.append(f)
        S = {'law': 'road', 'version': 'p2s', 'n_train': int(len(yt)), 'seconds': time.time() - t0,
             'n_eval': int(s.n_eval), 'sign_check': f'E(G+{DG}) >= E(G-{DG}) - {TOL} on every fitting row', 'front': kept}
        json.dump(S, open(fn, 'w'), indent=1)
    new = []
    for f in S['front']:
        if f['complexity'] == 1:
            continue
        r = {'id': f'p2s-c{f["complexity"]}', 'version': 'p2s',
             'change': f'Physics p2, sign-checked search, complexity {f["complexity"]}', **f}
        r['sim_rebuilt'], r['dev_rebuilt'] = lr.score(r, Xv, yv, Xd, yd, dev_err=L.dev_error)
        fv = law_falls(r, Xv)
        r['falls_share'] = float(fv.mean()); r['falls_share_bev'] = float(fv[ev].mean()); r['sign_ok'] = bool(not fv.any())
        new.append(r)

    # 4. the unchanged rule over the logged ladder + the new rungs
    ladder = [dict(r, sim=r['sim_rebuilt'], dev=r['dev_rebuilt']) for r in rungs + new]
    dec, inc = lr.walk(ladder)
    # 5. information for the owner: the sign check as a hard gate
    gated = [dict(r) if r['sign_ok'] or r['id'] == 'r0' else dict(r, sim=-np.inf) for r in ladder]
    dec_g, inc_g = lr.walk(gated)

    # per-group dev error and BEV grade response of the interesting laws
    grp = np.where(Xd[:, EI] > 0.5, 'EV', np.where(Xd[:, L.VARS.index('h')] > 0.5, 'HEV', 'ICE'))

    def by_group(r):
        yh = lr.law_eval(r, Xd)
        return {g: float(np.median(np.abs(yh[grp == g] / yd[grp == g] - 1))) for g in ('ICE', 'HEV', 'EV') if (grp == g).any()}

    def grade_probe(r):
        # the audit's probe: 20 m/s, U=2, R=10, BEV row of the median BEV in val_p2 (M, D, T), K at the BEV median
        base = np.median(Xv[ev], axis=0); base[L.VARS.index('v')] = 20; base[L.VARS.index('U')] = 2
        base[L.VARS.index('R')] = 10; base[L.VARS.index('h')] = 0; base[EI] = 1
        X = np.tile(base, (3, 1)); X[:, GI] = [-2, 0, 2]
        return [float(x) for x in lr.law_eval(r, X)]

    out = {'date': '2026-11-27', 'released_before': R['released'], 'rule': R['rule'],
           'sign_check': f'E(G+{DG}) >= E(G-{DG}) - {TOL} at every val_p2 row, other inputs held',
           'rebuild_check': {'released_dR2': d_rel, 'released_ddev': d_dev, 'worst_dR2': worst},
           'search': {k: S[k] for k in ('seconds', 'n_eval', 'n_train', 'sign_check') if k in S},
           'rungs': [{'id': r['id'], 'complexity': r['complexity'], 'expr': r['expr'], 'sim': r['sim_rebuilt'],
                      'dev': r['dev_rebuilt'], 'sign_ok': r['sign_ok'], 'falls_share': r['falls_share'],
                      'falls_share_bev': r['falls_share_bev'], 'dev_by_group': by_group(r), 'bev_grade_probe': grade_probe(r),
                      'decision': d['decision'], 'why': d['why'], 'decision_if_sign_gate': g['decision']}
                     for r, d, g in zip(ladder, dec, dec_g)],
           'released': inc['id'], 'released_if_sign_gate': inc_g['id']}
    json.dump(out, open(f'{ROOT}/results/road_sign.json', 'w'), indent=1, default=float)
    for r in out['rungs']:
        print(f"{r['id']:9s} c={r['complexity']:2d} R2={r['sim']:.4f} dev={r['dev']:.4f} sign={'ok ' if r['sign_ok'] else 'NO '}"
              f"fall={r['falls_share']:.2f}/{r['falls_share_bev']:.2f} {r['decision']:8s} gate:{r['decision_if_sign_gate']:8s} "
              f"probe={['%.3f' % x for x in r['bev_grade_probe']]} {r['expr'][:70]}")
    print('released (unchanged rule):', out['released'], '| if the sign check were a gate:', out['released_if_sign_gate'])


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0)
