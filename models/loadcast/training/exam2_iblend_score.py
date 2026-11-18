"""LoadCast exam 2 (I-BLEND, IIIT Delhi): score v1.0 once, as split_exam2_iblend.json says. 17 Nov 2026.

    python exam2_iblend.py build --check     # first: the processed buildings must reproduce the frozen hashes
    python exam2_iblend_score.py             # -> training/exam2_iblend_results.json

Model: the released loadcast.py / loadcast_weights.npz with its shipped widening, through LoadCast.forecast (no refit).
Baselines, on the same good forecast-year hours:
  cold (and warm 8 / 2 weeks): dumb dev median intensity x median time-of-week shape (BDG2 dev, electricity, per use,
    exam.dumb_cold); dev TOWT-shape guess (rescore_seen.shape_guess); the dev-calibrated BuildingSim twin is NOT run
    (the integrity-pass builder, rescore_seen._sim_one, takes a BDG2 building id and BDG2 site weather only).
  warm, a year: TOWT and change-point fitted on the history year (BuildingData baselines.load_one, n - p as its code
    computes) and LoadCast's ridge TOWT on the same history (towt_lc, p = 0 as in exam.py).
  warm 8 / 2 weeks: the ridge TOWT on the same weeks.
  daily / monthly: every hourly prediction aggregated as evaluate.daily_monthly does (local days with >= 20 good hours,
    local months of good hours; p = 0), plus BuildingData's daily / monthly change-point models (n - p) for the year rows.
Interval: 95 % percentile bootstrap over the exam buildings, paired, 2,000 resamples, seed 20261117, on
median(LoadCast) - median(best baseline); best = lowest median on the full exam row.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402
import exam2_iblend as XB  # noqa: E402
import lc_common as C  # noqa: E402
import dataset as D  # noqa: E402
import exam as EX  # noqa: E402
import rescore_seen as R  # noqa: E402

sys.path.insert(0, os.path.join(_P.REPO, 'shared', 'building', 'data', 'scripts'))
import baselines as BL  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import loadcast as LCM  # noqa: E402

OUT = os.path.join(HERE, 'exam2_iblend_results.json')
NB, SEED = 2000, XB.SEED
METER = 'electricity'
WEEKS = {'warm_w8': 8 * 168, 'warm_w2': 2 * 168}


def daily_monthly(y, p, dkey, mon):
    ok = np.isfinite(y)
    d = pd.DataFrame(dict(y=np.where(ok, y, np.nan), p=np.where(ok, p, np.nan), d=dkey, m=mon))
    g = d.groupby('d')
    cnt = g.y.count()
    dd = g[['y', 'p']].mean()[cnt >= 20]
    mm = d.groupby('m')[['y', 'p']].mean()
    return C.g14(dd.y.values, dd.p.values)[:2], C.g14(mm.y.values, mm.p.values)[:2]


def main():
    S = json.load(open(XB.SPLIT, encoding='utf-8'))
    lc = LCM.LoadCast()
    # ---- dev guesses (BDG2 dev, electricity)
    dev = D.real_units('dev', meters=[METER])
    for u in dev:
        D.prepare(u, ('full',))
    dI, dS = EX.dumb_cold(dev)
    SH = R.shape_guess(dev)
    print(len(dev), 'dev electricity meters for the dev guesses', flush=True)

    recs = []
    for b, u in S['units'].items():
        fr = pd.read_parquet(os.path.join(XB.PROC, f'{b}.parquet'))
        idx = pd.DatetimeIndex(fr.time_utc)
        h = XB.hash_series(idx, [fr.kWh, fr.T_out, fr.dewpoint, fr.GHI, fr.wind_speed])
        if h != u['sha256']:
            raise SystemExit(f'{b}: processed hash {h} != frozen {u["sha256"]}')
        W = pd.DataFrame({c: fr[c].to_numpy(float) for c in ('T_out', 'dewpoint', 'GHI', 'wind_speed')}, index=idx)
        f = lc._frame(W, XB.TZ)
        role = fr.role.to_numpy()
        kwh = fr.kWh.to_numpy(float)
        hist = pd.Series(np.where(role == 'history', kwh, np.nan), index=idx)
        hsel = np.flatnonzero(role == 'history')
        tsel = np.flatnonzero(role == 'forecast')
        y = kwh[tsel]
        fy = u['forecast_year']
        start, end = pd.Timestamp(f'{fy}-01-01'), pd.Timestamp(f'{fy + 1}-01-01')
        unit = u['area_m2'] * C.IREF[METER] / 1000.0
        args = dict(tz=XB.TZ, lat=XB.MODEL_LAT, use=u['use'], area_m2=u['area_m2'], meter=METER,
                    meters_present=['electricity'], start=start, end=end)
        P, cov = {}, {}
        histories = {'cold': None, 'warm_year': hist.iloc[hsel]}
        for k, L in WEEKS.items():
            histories[k] = hist.iloc[hsel[-L:]]
        for k, hs in histories.items():
            fc = lc.forecast(W, history=hs, **args)
            assert len(fc) == len(tsel) and (fc.index == idx[tsel]).all()
            P[k] = fc.kW.to_numpy()
            ok = np.isfinite(y)
            cov[k] = float(np.mean((y[ok] >= fc.lo90.to_numpy()[ok]) & (y[ok] <= fc.hi90.to_numpy()[ok])))
            cov[k + '_widen'] = fc.attrs['widen']
            if hs is not None:     # LoadCast's ridge TOWT on the same history (the warm network's anchor)
                hh = hs.reindex(idx).values.astype(float) / unit
                hi = np.flatnonzero((idx >= hs.index.min()) & (idx <= hs.index.max()))
                H = LCM.history_encode(hh[hi], f['T'][hi], f['tow'][hi], f['dcode'][hi], f['hol'][hi], f['hour'][hi], 1.0)
                tag = 'towt_lc' if k == 'warm_year' else f'towt_lc_{k.split("_")[1]}'
                P[tag] = LCM.towt_predict(H['coef'], f['T'][tsel], f['tow'][tsel]) * H['S'] * unit
        tow = f['tow'][tsel]
        key = (u['use'], METER) if (u['use'], METER) in dI else ('*', METER)
        P['dumb'] = dI[key] * dS[key][tow] * unit
        P['shape'] = dI[key] * LCM.towt_predict(SH[METER], f['T'][tsel], tow) * unit
        loc = idx.tz_convert(XB.TZ)
        dkey = f['dcode'][tsel]
        mon = loc.month.to_numpy()[tsel]
        r = dict(building=b, use=u['use'], dumb_key='|'.join(key), n_good=int(np.isfinite(y).sum()))
        for name, p in P.items():
            cv, nm, _ = C.g14(y, p)
            (dcv, dnm), (mcv, mnm) = daily_monthly(y, p, dkey, mon)
            r.update({f'{name}_hourly_cv': cv, f'{name}_hourly_nmbe': nm, f'{name}_daily_cv': dcv,
                      f'{name}_daily_nmbe': dnm, f'{name}_monthly_cv': mcv, f'{name}_monthly_nmbe': mnm})
        for k, v in cov.items():
            r[f'{k}_cov90' if not k.endswith('_widen') else k] = v
        # BuildingData baselines on the history year (local 2016 -> 2017), its own code and n - p
        ys = pd.Series(np.where(np.isin(role, ['history', 'forecast']), kwh, np.nan), index=idx.tz_convert(XB.TZ))
        Ts = pd.Series(fr.T_out.to_numpy(float), index=idx.tz_convert(XB.TZ))
        assert u['history_year'] == 2016 and fy == 2017      # load_one is written for 2016 -> 2017
        bl = BL.load_one(ys, Ts)
        r['cp_kind'] = bl.pop('cp_kind', None)
        bl.pop('cp_r2_2016', None)
        for k, (cv, nm, n) in bl.items():
            nm_ = {'towt': 'towt_bd_hourly', 'g14_changepoint': 'cp_hourly', 'g14_changepoint_daily': 'cp_daily',
                   'g14_changepoint_monthly': 'cp_monthly'}.get(k, f'nowcast_{k}')
            r[f'{nm_}_cv'], r[f'{nm_}_nmbe'], r[f'{nm_}_n'] = cv, nm, n
        # BuildingData TOWT aggregated to days / months (p = 0): its hourly predictions on the same hours
        tw = (pd.Series(y, index=idx[tsel]))
        hmask = np.isin(role, ['history']) & np.isfinite(kwh) & np.isfinite(fr.T_out.to_numpy(float))
        tow_all = f['tow']
        Xh = BL.towt_X(fr.T_out.to_numpy(float)[hmask], tow_all[hmask])
        c_, *_ = np.linalg.lstsq(Xh, kwh[hmask], rcond=None)
        ptw = BL.towt_X(fr.T_out.to_numpy(float)[tsel], tow_all[tsel]) @ c_
        cv_chk = BL.g14(y, ptw, Xh.shape[1])[0]
        r['towt_bd_hourly_cv_check'] = cv_chk
        (dcv, dnm), (mcv, mnm) = daily_monthly(y, ptw, dkey, mon)
        r.update(towt_bd_daily_cv=dcv, towt_bd_daily_nmbe=dnm, towt_bd_monthly_cv=mcv, towt_bd_monthly_nmbe=mnm)
        del tw
        recs.append(r)
        print(b, {k: round(v, 1) for k, v in r.items() if k.endswith('_cv') and isinstance(v, float)}, flush=True)
    G = pd.DataFrame(recs)
    G.to_csv(os.path.join(_P.WORK, 'exam2', 'iblend_scored_buildings.csv'), index=False)

    COLD = ['dumb', 'shape']
    ROWS = [('cold_hourly', 'cold_hourly', [f'{b}_hourly' for b in COLD]),
            ('warm_year_hourly', 'warm_year_hourly', ['towt_bd_hourly', 'cp_hourly', 'towt_lc_hourly']),
            ('warm_w8_hourly', 'warm_w8_hourly', ['towt_lc_w8_hourly'] + [f'{b}_hourly' for b in COLD]),
            ('warm_w2_hourly', 'warm_w2_hourly', ['towt_lc_w2_hourly'] + [f'{b}_hourly' for b in COLD]),
            ('cold_daily', 'cold_daily', [f'{b}_daily' for b in COLD]),
            ('cold_monthly', 'cold_monthly', [f'{b}_monthly' for b in COLD]),
            ('warm_year_daily', 'warm_year_daily', ['towt_bd_daily', 'cp_daily', 'towt_lc_daily']),
            ('warm_year_monthly', 'warm_year_monthly', ['towt_bd_monthly', 'cp_monthly', 'towt_lc_monthly'])]
    rng = np.random.default_rng(SEED)
    n = len(G)
    draws = rng.integers(0, n, (NB, n))
    res = dict(exam='LoadCast exam 2, I-BLEND (IIIT Delhi campus), scored once on 17 Nov 2026',
               spec=os.path.basename(XB.SPLIT), split_sha256=S['split_sha256'], n_buildings=n,
               model='LoadCast v1.0 (released loadcast_weights.npz, shipped widening)',
               interval=dict(resamples=NB, seed=SEED, unit='building', level=0.95),
               g14_p='LoadCast, dev guesses, towt_lc and all daily / monthly aggregates of hourly predictions p = 0; '
                     'BuildingData TOWT (p = 174) and change-point models (p by kind) n - p, as load_one computes',
               not_run_as_specified=['Dev-calibrated BuildingSim twin baseline: not run. The integrity-pass twin '
                                     'builder (rescore_seen._sim_one) takes a BDG2 building id (bdg2_spec) and a BDG2 '
                                     'site\'s weather (lcsim.simulate_site); it does not accept a non-BDG2 spec.'],
               dev_guess=dict(n_dev_meters=len(dev), keys={b: k for b, k in zip(G.building, G.dumb_key)}),
               rows={}, buildings=json.loads(G.to_json(orient='records')))
    for rid, mcol, bases in ROWS:
        meds = {bb: float(G[f'{bb}_cv'].median()) for bb in bases}
        best = min(meds, key=meds.get)
        a, bvals = G[f'{mcol}_cv'].to_numpy(float), G[f'{best}_cv'].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(bvals)
        diffs = []
        for ii in draws:
            jj = ii[ok[ii]]
            diffs.append(np.median(a[jj]) - np.median(bvals[jj]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        mode = rid.rsplit('_', 1)[0]
        row = dict(n=int(ok.sum()), model=float(np.median(a[ok])), model_nmbe=float(G[f'{mcol}_nmbe'].median()),
                   baselines=meds, best=best, best_nmbe=float(G[f'{best}_nmbe'].median()),
                   diff=float(np.median(a[ok]) - np.median(bvals[ok])), interval=[float(lo), float(hi)],
                   verdict='win' if hi < 0 else ('loss' if lo > 0 else 'tie'),
                   per_building={bn: dict(model=float(x), best=float(z)) for bn, x, z in zip(G.building, a, bvals)})
        if rid.endswith('_hourly'):
            row['coverage90_mean'] = float(G[f'{mode}_cov90'].mean())
        res['rows'][rid] = row
        print(f"{rid:18s} n {row['n']} model {row['model']:6.1f} best {best:18s} {meds[best]:6.1f} diff {row['diff']:+6.1f} "
              f"[{lo:+6.1f}, {hi:+6.1f}] {row['verdict']}", flush=True)
    res['nowcast_info'] = {k: float(G[f'nowcast_{k}_cv'].median()) for k in
                           ('persistence_1h', 'persistence_24h', 'same_hour_last_week')}
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(res, fh, indent=1, default=float)
        fh.write('\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
