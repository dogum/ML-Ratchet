"""FINAL EXAM (run once, for the chosen rung only): BDG2 exam building-meters with q_ok, history local 2016 ->
forecast local 2017 with actual 2017 weather. Never used for choosing anything.

    python3 exam.py <run>

1. Widen factors (one per mode x meter) fitted on DEV only: the smallest k >= 1 with mean dev coverage of the 90 %
   interval >= 0.90 (widen-only rule).
2. Exam metrics per building-meter: CV(RMSE), NMBE (ASHRAE G14, p = 0), coverage, for cold, warm (full 2016 history)
   warm-8w and warm-2w (only the last 8 / 2 weeks of 2016).
3. BuildingData baselines on the same building-meters (bd.baselines('load_bdg2'), fit on 2016 by agent B) and two
   cold-start references computed here: dev median intensity x dev median time-of-week shape ('dumb') per use x meter.
4. Breakdowns by meter, use, climate zone, site. -> exam/exam_<run>.json, exam/exam_units_<run>.csv
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E, lcnet as N

MODES = (('cold', 'full'), ('warm', 'full'), ('warm', 'w8'), ('warm', 'w2'))
OUT = f'{C.ROOT}/exam'


def fit_widen(dev_units, nets):
    W = {}
    df = E.eval_units(dev_units, nets, modes=MODES, keep_series=True)
    for mode, v in MODES:
        key = mode if v == 'full' else f'{mode}_{v}'
        for m, g in df.groupby('meter'):
            k = 1.0
            for k in np.concatenate([np.arange(1.0, 3.0, 0.05), np.arange(3.0, 30.01, 0.25)]):
                cov = np.mean([E.cover(r['y'], r[f'{key}_mu'], r[f'{key}_sd'], k) for _, r in g.iterrows()])
                if cov >= 0.90: break
            W[f'{mode}_{v}_{m}'] = float(round(k, 2))
    return W


def dumb_cold(dev_units):
    """Dev median intensity (2017 target year of dev; fine: dev) x median TOW shape per (use, meter), with fallbacks."""
    I = {}; S = {}
    for u in dev_units:
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t]; mu = np.nanmean(y)
        sh = pd.Series(y / mu).groupby(f['tow'][t]).mean().reindex(range(168)).fillna(1.0).values
        for key in ((u['use'], u['meter']), ('*', u['meter'])):
            I.setdefault(key, []).append(mu); S.setdefault(key, []).append(sh)
    return {k: np.median(v) for k, v in I.items()}, {k: np.median(np.array(v), 0) for k, v in S.items()}


def main(run):
    os.makedirs(OUT, exist_ok=True)
    nets = {m: N.load(f'{C.ROOT}/ckpt/{run}_{m}.pt')[0] for m in ('cold', 'warm')}
    t0 = time.time()
    dev = D.real_units('dev')
    for u in dev: D.prepare(u, ('full', 'w8', 'w2'))
    wf = f'{C.ROOT}/data/widen_{run}.json'
    W = json.load(open(wf)) if os.path.exists(wf) else fit_widen(dev, nets)
    print('widen factors (dev):', W, flush=True)
    dI, dS = dumb_cold(dev)
    ex = D.real_units('exam')
    for u in ex: D.prepare(u, ('full', 'w8', 'w2'))
    print(len(ex), 'exam building-meters', round(time.time() - t0), 's', flush=True)
    df = E.eval_units(ex, nets, modes=MODES, widen=W)
    for u, i in zip(ex, range(len(ex))):
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t]
        key = (u['use'], u['meter']) if (u['use'], u['meter']) in dI else ('*', u['meter'])
        dumb = dI[key] * dS[key][f['tow'][t]]
        cv, nm, _ = C.g14(y, dumb)
        df.loc[i, 'dumb_cv'] = cv; df.loc[i, 'dumb_nmbe'] = nm
        h = u['H']['full']; tw = C.towt_predict(h['coef'], f['T'][t], f['tow'][t]) * h['S']
        cv, nm, _ = C.g14(y, tw); df.loc[i, 'towt_lc_cv'] = cv; df.loc[i, 'towt_lc_nmbe'] = nm
    df['building_id'] = [u['bid'] for u in ex]; df['cz'] = [u['cz'] for u in ex]
    bl = C.bd.baselines('load_bdg2')
    bl = bl[bl.split == 'exam'].rename(columns=lambda c: c.replace('_cvrmse', '_cv'))
    df = df.merge(bl, on=['building_id', 'meter'], how='left', suffixes=('', '_bl'))
    df.drop(columns=[c for c in df.columns if c.endswith('_n')], errors='ignore').to_csv(f'{OUT}/exam_units_{run}.csv', index=False)
    meth = {'LoadCast cold': 'cold', 'LoadCast warm (2016)': 'warm', 'LoadCast warm (last 8 weeks)': 'warm_w8', 'LoadCast warm (last 2 weeks)': 'warm_w2',
            'dumb cold (dev median)': 'dumb', 'persistence 1 h*': 'persistence_1h', 'persistence 24 h*': 'persistence_24h',
            'same hour last week*': 'same_hour_last_week', 'TOWT (B)': 'towt', 'TOWT (LoadCast ridge)': 'towt_lc',
            'change-point hourly': 'g14_changepoint', 'change-point daily': 'g14_changepoint_daily',
            'change-point monthly': 'g14_changepoint_monthly', 'LoadCast cold, daily': 'cold_daily',
            'LoadCast warm (2016), daily': 'warm_daily', 'LoadCast cold, monthly': 'cold_monthly',
            'LoadCast warm (2016), monthly': 'warm_monthly'}
    res = dict(run=run, widen=W, n=len(df), note='* uses the actual 2017 meter readings of the previous hour / day / '
               'week (an operational nowcast, not a forecast from 2016). Daily and monthly rows are scored on daily / '
               'monthly totals. LoadCast and dumb use p = 0; B\'s fitted baselines use n - p.', by_meter={}, breakdowns={})
    for m, g in df.groupby('meter'):
        T = {}
        for name, k in meth.items():
            if f'{k}_cv' not in g: continue
            c = g[f'{k}_cv']; nm = g[f'{k}_nmbe']
            T[name] = dict(n=int(c.notna().sum()), cv_med=float(c.median()), cv_p25=float(c.quantile(.25)), cv_p75=float(c.quantile(.75)),
                           nmbe_med=float(nm.median()), abs_nmbe_med=float(nm.abs().median()),
                           g14_ok=float(((c <= 30) & (nm.abs() <= 10)).mean()))
            if f'{k}_cov' in g: T[name]['coverage90_mean'] = float(g[f'{k}_cov'].mean()); T[name]['coverage90_med'] = float(g[f'{k}_cov'].median())
        res['by_meter'][m] = T
    for by in ('use', 'cz', 'site'):
        B = {}
        for (m, k), g in df.groupby(['meter', by]):
            B.setdefault(m, {})[str(k)] = dict(n=len(g), cold_cv=float(g.cold_cv.median()), warm_cv=float(g.warm_cv.median()),
                                                towt_cv=float(g.towt_cv.median()) if 'towt_cv' in g else None,
                                                dumb_cv=float(g.dumb_cv.median()),
                                                cold_cov=float(g.cold_cov.mean()), warm_cov=float(g.warm_cov.mean()),
                                                warm_beats_towt=float((g.warm_cv < g.towt_cv).mean()) if 'towt_cv' in g else None)
        res['breakdowns'][by] = B
    C.jdump(res, f'{OUT}/exam_{run}.json')
    for m, T in res['by_meter'].items():
        print(m)
        for name, r in T.items():
            print(f"   {name:30s} n {r['n']:4d} CV {r['cv_med']:6.1f} [{r['cv_p25']:5.1f}-{r['cv_p75']:5.1f}] NMBE {r['nmbe_med']:6.1f} G14ok {r['g14_ok']:.2f}"
                  + (f" cov90 {r['coverage90_mean']:.2f}" if 'coverage90_mean' in r else ''))
    print(round(time.time() - t0), 's')


if __name__ == '__main__':
    main(sys.argv[1])
