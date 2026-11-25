"""Round 2, G: the label-free gap detector, on DEV only (PREREG section 4). Never reads the exam.

python gap.py <X dev_preds.parquet> <member dev_preds.parquet> x4  -> round2/gap/gap_dev.json

spread of a dev unit = SD over the four family entries of their log point predictions (at the scoring unit, after the
harness's own unit aggregation: score.real_units); error = |log(pred_X / y)|.
Per set: Spearman rho(spread, error) with a cluster bootstrap 95 % CI (2,000 resamples, default_rng(20261123)).
Enrichment: units of the 16 D16 sets, each weighted 1 / (units in its set); the top 10 % of weighted spread flagged;
enrichment = flagged weight share in the five shared-bias sets / their share of all weight (5 / 16); cluster bootstrap
CI (clusters resampled within each set). Pass: enrichment CI lower bound > 1.0 AND median per-set Spearman > 0.10.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import score as S  # noqa: E402

SHARED = ['VED_EV', 'VED_PHEV', 'Mersin_diesel', 'Mersin_cng', 'ICCT']
SETS = S.D16 + ['ZTBus_legs']
NB = 2000


def units(path):
    return S.real_units(pd.read_parquet(path))


def groups_of(cl):
    codes, inv = np.unique(cl, return_inverse=True)
    order = np.argsort(inv, kind='stable'); st = np.searchsorted(inv[order], np.arange(len(codes) + 1))
    return [order[st[i]:st[i + 1]] for i in range(len(codes))]


def weighted_top(spread, w, q=0.10):
    o = np.argsort(-spread); cw = np.cumsum(w[o]) / w.sum()
    flag = np.zeros(len(spread), bool); flag[o[cw <= q]] = True
    if not flag.any():
        flag[o[0]] = True
    return flag


def main(xp, mps):
    UX = units(xp); UM = [units(p) for p in mps]
    rng = np.random.default_rng(20261123)
    per, pool = {}, []
    for sk in SETS:
        if sk not in UX:
            continue
        u = UX[sk].set_index('unit')
        L = []
        for U in UM:
            m = U[sk].set_index('unit').reindex(u.index)
            L.append(np.log(m.pred.values))
        L = np.stack(L, 1)
        spread = np.nanstd(L, 1, ddof=1)
        err = np.abs(np.log(u.pred.values / u.y.values))
        cl = u['cluster'].values if 'cluster' in u else np.arange(len(u))
        ok = np.isfinite(spread) & np.isfinite(err)
        spread, err, cl = spread[ok], err[ok], cl[ok]
        rho = float(spearmanr(spread, err).correlation) if len(err) > 2 else float('nan')
        G = groups_of(cl); bs = []
        for _ in range(NB):
            ix = np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))])
            if len(np.unique(spread[ix])) > 1 and len(np.unique(err[ix])) > 1:
                bs.append(spearmanr(spread[ix], err[ix]).correlation)
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 50 else (np.nan, np.nan))
        per[sk] = {'n': int(len(err)), 'n_clusters': len(G), 'spearman': rho, 'ci': [float(lo), float(hi)],
                   'median_spread': float(np.median(spread)), 'median_abs_log_err': float(np.median(err)),
                   'shared_bias_set': sk in SHARED}
        if sk in S.D16:
            pool.append(pd.DataFrame({'set': sk, 'spread': spread, 'err': err, 'cluster': cl.astype(str)}))
    D = pd.concat(pool, ignore_index=True)
    D['w'] = 1.0 / D.groupby('set')['set'].transform('size')
    D['shared'] = D.set.isin(SHARED)

    def enrich(d):
        f = weighted_top(d.spread.values, d.w.values)
        fw = d.w.values[f]
        return float((fw * d.shared.values[f]).sum() / fw.sum()) / float((d.w * d.shared).sum() / d.w.sum())

    e0 = enrich(D)
    sets = {s: (d.reset_index(drop=True), groups_of(d.cluster.values)) for s, d in D.groupby('set')}
    eb = []
    for _ in range(NB):
        parts = []
        for s, (d, G) in sets.items():
            ix = np.concatenate([G[g] for g in rng.integers(0, len(G), len(G))])
            dd = d.iloc[ix].copy(); dd['w'] = 1.0 / len(dd)
            parts.append(dd)
        eb.append(enrich(pd.concat(parts, ignore_index=True)))
    lo, hi = np.percentile(eb, [2.5, 97.5])
    f = weighted_top(D.spread.values, D.w.values)
    flagged = D[f].groupby('set').w.sum() / D[f].w.sum()
    rhos = [per[s]['spearman'] for s in S.D16 if s in per and np.isfinite(per[s]['spearman'])]
    med_rho = float(np.median(rhos))
    out = {'per_set': per, 'median_spearman_D16': med_rho,
           'enrichment': {'value': e0, 'ci': [float(lo), float(hi)], 'base_share_shared': 5 / 16,
                          'flagged_weight_by_set': {k: float(v) for k, v in flagged.items()}},
           'pass_marks': 'enrichment CI lower bound > 1.0 AND median per-set Spearman > 0.10 (PREREG section 4)',
           'verdict': 'flags the shared misses on dev' if (lo > 1.0 and med_rho > 0.10) else 'does not flag the shared misses on dev',
           'inputs': {'x': os.path.relpath(xp, P.R2), 'members': [os.path.relpath(m, P.R2) for m in mps]}}
    json.dump(out, open(os.path.join(HERE, 'gap_dev.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_set'}, indent=1, default=float))
    for s, v in per.items():
        print(f"{s:16s} n {v['n']:6d} rho {v['spearman']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] spread {v['median_spread']:.3f} err {v['median_abs_log_err']:.3f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
