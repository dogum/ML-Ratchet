"""THE EXAM (run once, at the end): FireSize on the FPA-FOD exam (655,869 ignitions, 2012-2020; main = 2012-2019,
recent = 2020), against B's baselines (proc/baselines/firesize_exam.json, same records).
usage: exam.py <model npz>   -> results/exam.json, results/exam_preds.parquet"""
import sys, json, os, time
from common import *
import supp, net
import features as F
from firedata import metrics as M

OUT = f'{ROOT}/results'
os.makedirs(OUT, exist_ok=True)


def main():
    mp = sys.argv[1]
    if os.path.exists(f'{OUT}/exam.json'):
        raise SystemExit('exam already run: results/exam.json exists (the exam is run once)')
    t0 = time.time()
    import hashlib, datetime
    sha = hashlib.sha256(open(mp, 'rb').read()).hexdigest()
    with open(f'{OUT}/exam.started', 'a') as fh:      # audit trail: every start of this script is recorded
        fh.write(f'{datetime.datetime.utcnow().isoformat()}Z model={mp} sha256={sha}\n')
    W = dict(np.load(mp, allow_pickle=False))
    W1 = dict(W, temp=1.0)      # sharp (pre-widening) distribution: point estimate = its median (decided on dev gate)
    feats = list(W['feats'])
    d = fod_records('exam')
    d = F.fod_gridmet(d)
    print('exam records', len(d), flush=True)
    parts, meds = [], []
    for s in range(0, len(d), 100000):
        dd = d.iloc[s:s + 100000]
        X = F.fod_features(dd)
        if any(f.startswith('ros') for f in feats):
            import physfeat as PF
            X = pd.concat([X, PF.phys_features(dd.x.values, dd.y.values, dd.tmmx.values, dd.rmin.values, dd.vs.values,
                                               dd.tmmx_m1.values, dd.rmin_m1.values, dd.vs_m1.values)], axis=1)
        Xv = X[feats].values.astype(np.float64)
        P = net.forward_np(W, Xv)
        parts.append(P.astype(np.float32))
        meds.append(supp.median_log10(net.forward_np(W1, Xv)))
        print(s, f'{time.time() - t0:.0f}s', flush=True)
    P = np.concatenate(parts)
    pe = P[:, supp.ESC_BIN:].sum(1)
    pl = P[:, supp.LARGE_BIN:].sum(1)
    med = 10 ** np.concatenate(meds)                 # sharp median (T = 1); probabilities use the widened T
    cdf = np.cumsum(P, 1)
    # 80 % central interval from the binned distribution (bin edges), coverage of the observed class
    b_obs = np.searchsorted(supp.EDGES, np.log10(d.size_ha.values), side='right')
    lo = np.argmax(cdf >= 0.1, 1); hi = np.argmax(cdf >= 0.9, 1)
    cover80 = (b_obs >= lo) & (b_obs <= hi)
    pit_lo = np.where(b_obs > 0, cdf[np.arange(len(P)), np.maximum(b_obs - 1, 0)], 0.0)
    res = dict(model=os.path.basename(mp), model_sha256=sha, temp=float(W.get('temp', 1.0)), point='median of T=1 distribution',
               n=int(len(d)), wall_s=None, periods={})
    masks = dict(all=np.ones(len(d), bool), main_2012_2019=(d.exam_period == 'main').values,
                 recent_2020=(d.exam_period == 'recent').values, mtbs_matched=d.MTBS_ID.notna().values,
                 early_2012_2016=(d.FIRE_YEAR <= 2016).values, late_2017_2019=((d.FIRE_YEAR >= 2017) & (d.FIRE_YEAR <= 2019)).values)
    for rg in REGIONS:
        masks[f'region_{rg}'] = (d.region == rg).values
    for k, m in masks.items():
        r = dict(size=M.size_scores(d.size_ha.values[m], med[m]),
                 escape=M.binary_scores(d.escape.values[m], pe[m]) if d.escape.values[m].std() > 0 else None,
                 large=M.binary_scores(d.large.values[m], pl[m]) if d.large.values[m].std() > 0 else None,
                 mean_p_escape=float(pe[m].mean()), cover80=float(cover80[m].mean()))
        res['periods'][k] = r
        print(k, json.dumps({a: ({b: round(c, 4) for b, c in v.items()} if isinstance(v, dict) else v)
                             for a, v in r.items()}), flush=True)
    # reliability of P(escape)
    qb = np.quantile(pe, np.linspace(0, 1, 21))
    ib = np.clip(np.searchsorted(qb, pe, side='right') - 1, 0, 19)
    res['reliability'] = [dict(p_mean=float(pe[ib == i].mean()), obs=float(d.escape.values[ib == i].mean()),
                               n=int((ib == i).sum())) for i in range(20)]
    res['baselines'] = json.load(open(f'{FP}/baselines/firesize_exam.json'))['results']
    res['wall_s'] = time.time() - t0
    json.dump(res, open(f'{OUT}/exam.json', 'w'), indent=1, default=float)
    pd.DataFrame(dict(FOD_ID=d.FOD_ID.values, p_escape=pe.astype(np.float32), p_large=pl.astype(np.float32),
                      med_ha=med.astype(np.float32), **{f'p{k}': P[:, k] for k in range(supp.NB)})).to_parquet(
        f'{OUT}/exam_preds.parquet', compression='zstd')
    print('DONE', res['wall_s'], flush=True)


if __name__ == '__main__':
    main()
