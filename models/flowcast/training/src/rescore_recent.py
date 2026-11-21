"""Reproduction re-score of the RELEASED FlowCast on the recent exam (WY2016-23, NWIS flow) and on the USGS annual
instantaneous flood peaks (main and recent), 18 Nov 2026. Seen: the exam was opened at release; nothing here
changes weights, widening or any decision. Companion of rescore_main.py (same helpers, same bootstrap).

usage: python training/src/rescore_recent.py run      -> WORK/results/rescore_recent.parquet, rescore_recent_nowcast.parquet,
                                                         WORK/preds/rescore_{val,recent}.npz (mean, q50, q95 per basin)
       python training/src/rescore_recent.py summary  -> WORK/results/rescore_recent_2026-11-18.json (per basin)
                                                         WORK/results/rescore_peaks_2026-11-18.json (per basin, peaks)
                                                         WORK/results/rescore_recent_summary_2026-11-18.json
       (the three JSONs were then copied by hand into training/results/ as the committed record)

Recent flow: as evalreal.py exam (period 'recent', REC = 2015-10-01..2023-09-30, wd.streamflow(b, 'nwis'), at least
365 observed days), through the released flowcast.py. Nowcast: as evalnow.py (NWIS flow over warm-up and window).
Flood peaks: as exam_report.py: NWIS annual instantaneous peaks without codes 5/6/7/C, partial dates or missing
values, water years 1990-99 (main) and 2016-23 (recent), against FlowCast's annual maximum daily flow (water years
with >= 330 days) × 1.38 (WaterData's dev ratio, peak_summary.json); per basin the median |% error| and median
% error, then the median over basins. Baselines per basin from baselines/peak_scores.parquet (same protocol).
Intervals: paired bootstrap over basins of the difference in medians, 10,000 resamples, default_rng(20261118).
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rescore_main import (boot, exam_ids, score, verdict, fcm, wd, M, MODEL, ROOT, WATER_DATA, VAL, NBOOT, SEED,  # noqa
                          LEADS, TAG, np, pd)
from common import REC  # noqa: E402

BL = f'{WATER_DATA}/data/proc/baselines'
K90 = fcm.FlowCast().k[1]     # 1.08, the released 90 % band factor


def window(s0, s1):
    w0 = (pd.Timestamp(s0) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    e1 = (pd.Timestamp(s1) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return w0, e1


def run():
    m = fcm.FlowCast()
    n = fcm.FlowCast(os.path.join(MODEL, 'flowcast_nowcast_weights.npz'))
    rows, now, preds = [], [], {'val': {}, 'recent': {}}
    for k, b in enumerate(exam_ids()):
        sp = wd.spec(b)
        for per, (s0, s1), src in (('val', VAL, 'caravan'), ('recent', REC, 'nwis')):
            try:
                obs = wd.streamflow(b, src).Q_mm
            except FileNotFoundError:
                continue
            o = obs.loc[s0:s1]
            if o.notna().sum() < 365:
                continue
            w0, e1 = window(s0, s1)
            f = wd.forcing(b, w0, e1)
            out = m.predict(f, sp, s0, s1)
            preds[per][b] = np.stack([out.Q.values, out.q50.values, out.q95.values], 1).astype(np.float32)
            if per == 'recent':
                idx = pd.date_range(s0, s1)
                r = score(obs, idx, out)
                r.update({'basin': b})
                rows.append(r)
                for lead in LEADS:
                    q = n.predict(f, sp, s0, s1, qobs=obs, lead=lead).Q
                    pers = obs.shift(lead, freq='D').reindex(idx)
                    r1, r2 = M.summary(o, q), M.summary(o, pers)
                    now.append({'basin': b, 'lead': lead, 'nse': r1['nse'], 'kge': r1['kge'],
                                'nse_persist': r2['nse'], 'kge_persist': r2['kge']})
        if k % 50 == 0:
            print(k, b, flush=True)
    pd.DataFrame(rows).to_parquet(f'{ROOT}/results/rescore_recent.parquet')
    pd.DataFrame(now).to_parquet(f'{ROOT}/results/rescore_recent_nowcast.parquet')
    for per in preds:
        np.savez_compressed(f'{ROOT}/preds/rescore_{per}.npz', **preds[per])


def ann_max(s, minn=330):
    s = s.dropna()
    w = s.index.year + (s.index.month >= 10).astype(int)
    g = s.groupby(w)
    c = g.count()
    return g.max()[c >= minn]


def peak_table():
    """exam_report.py's flood-peak table, from the re-scored predictions."""
    with open(f'{BL}/peak_summary.json') as f:
        R = json.load(f)['ratio_inst_over_daily_dev']
    pk = pd.read_parquet(f'{WATER_DATA}/data/proc/nwis/peaks.parquet')
    bad = pk.peak_cd.str.contains(r'(?:^|,)(?:5|6|7|C)(?:,|$)', regex=True) | pk.date_partial | pk.peak_cms.isna()
    pk = pk[~bad]
    rows = []
    for per, (s0, s1), (y0, y1) in [('val', VAL, (1990, 1999)), ('recent', REC, (2016, 2023))]:
        P = np.load(f'{ROOT}/preds/rescore_{per}.npz')
        idx = pd.date_range(s0, s1)
        for b in P.files:
            p = pk[pk.gauge_id == b].groupby('wy').peak_cms.max().loc[y0:y1]
            if len(p) == 0:
                continue
            area = wd.spec(b)['area_km2']
            x = P[b].astype(np.float64)
            # evalreal.py stored the RAW pinball q95 (widening was applied only when scoring coverage), so the
            # shipped 'q95' peak row is unwidened: q95_raw = q50 + (q95 - q50) / k90 (exact: the upper side is never
            # clipped). 'q95_released' is the widened q95 flowcast.py returns.
            ests = (('mean', x[:, 0]), ('q95', x[:, 1] + (x[:, 2] - x[:, 1]) / K90), ('q95_released', x[:, 2]))
            for name, col in ests:
                e = ann_max(pd.Series(col * area / 86.4, index=idx)) * R
                j = pd.concat([p, e], axis=1, keys=['o', 'e']).dropna()
                j = j[j.o > 0]
                for y, r in j.iterrows():
                    rows.append({'basin': b, 'wy': int(y), 'period': per, 'method': name, 'obs': r.o, 'est': r.e,
                                 'pct_err': 100 * (r.e - r.o) / r.o})
    return pd.DataFrame(rows), R


def per_basin_peaks(t):
    g = t.groupby('basin').pct_err
    return pd.DataFrame({'abs': g.apply(lambda x: x.abs().median()), 'bias': g.median(), 'n': g.size()})


def summary():
    rng = np.random.default_rng(SEED)
    med = lambda x, axis=None: np.nanmedian(x, axis=axis)  # noqa: E731
    medabs = lambda x, axis=None: np.nanmedian(np.abs(x), axis=axis)  # noqa: E731
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                           'exam_r1b_exam_cal.json')) as f:
        E = json.load(f)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                           'nowcast_summary.json')) as f:
        N = json.load(f)
    fc = pd.read_parquet(f'{ROOT}/results/rescore_recent.parquet').set_index('basin')
    nw = pd.read_parquet(f'{ROOT}/results/rescore_recent_nowcast.parquet')
    bl = pd.read_parquet(f'{BL}/flow_scores.parquet')
    bl = bl[(bl.split == 'exam') & (bl.period == 'recent')]
    a = wd.attributes('camels')
    S = {'what': 'Seen reproduction re-score, recent exam WY2016-23 (NWIS) and flood peaks. Paired bootstrap over '
                 f'basins of median(FlowCast) - median(baseline) on shared basins; {NBOOT} resamples, '
                 f'numpy default_rng({SEED}); win/loss only if the 95 % interval excludes 0',
         'claimed': {k: E['flow']['recent/flowcast'][k] for k in ('n', 'nse', 'kge', 'fhv', 'flv', 'peak_abs_err_daily',
                                                                 'timing_d', 'frac_nse_lt0', 'nse_531', 'cov90', 'cov50',
                                                                 'cov90_pooled_mean', 'width90_rel')},
         'flowcast': {'n': int(len(fc)), 'nse': float(fc.nse.median()), 'kge': float(fc.kge.median()),
                      'fhv': float(fc.fhv.median()), 'flv': float(fc.flv.median()),
                      'peak_abs_err_daily': float(fc.peak_abs_err_pct_median.median()),
                      'timing_d': float(fc.peak_timing_days.median()), 'frac_nse_lt0': float((fc.nse < 0).mean()),
                      'nse_531': float(fc[a.loc[fc.index, 'in_531'].values].nse.median()),
                      'cov90': float(fc.cov90.median()), 'cov50': float(fc.cov50.median()),
                      'cov90_pooled_mean': float(fc.cov90.mean()), 'width90_rel': float(fc.width90_rel.median())},
         'vs': {}, 'strata': {}, 'nowcast': {}, 'peaks': {}}
    metrics = [('nse', 'higher', med), ('kge', 'higher', med), ('fhv', 'lower', medabs),
               ('peak_abs_err_pct_median', 'lower', med), ('peak_timing_days', 'lower', med)]
    for meth in ('nn_dev', 'nn_k12', 'gr4j', 'nwm', 'clim', 'persist1'):
        g = bl[bl.method == meth].set_index('basin')
        common = g.index.intersection(fc.index)
        S['vs'][meth] = {'n_common': int(len(common)), 'wins_nse': int((fc.loc[common].nse > g.loc[common].nse).sum()),
                         'claimed_wins_frac': E['flow'][f'recent/{meth}']['flowcast_wins_frac']}
        for col, better, stat in metrics:
            x, y = fc.loc[common, col].values, g.loc[common, col].values
            ok = np.isfinite(x) & np.isfinite(y)
            d, lo, hi = boot(x[ok], y[ok], stat, rng)
            S['vs'][meth][col] = {'n': int(ok.sum()), 'flowcast': float(stat(x[ok])), 'baseline': float(stat(y[ok])),
                                  'diff': d, 'ci95': [lo, hi], 'verdict': verdict(lo, hi, better)}
        c531 = common[a.loc[common, 'in_531'].values]
        d, lo, hi = boot(fc.loc[c531, 'nse'].values, g.loc[c531, 'nse'].values, med, rng)
        S['vs'][meth]['nse_531'] = {'n': int(len(c531)), 'flowcast': float(fc.loc[c531, 'nse'].median()),
                                    'baseline': float(g.loc[c531, 'nse'].median()), 'diff': d, 'ci95': [lo, hi],
                                    'verdict': verdict(lo, hi, 'higher')}
    aa = a.loc[fc.index]
    strata = {'snow': aa.frac_snow > 0.3, 'arid': aa.aridity_FAO_PM > 1, 'humid': aa.aridity_FAO_PM < 0.6,
              'regulated': aa.reg_class.isin(['mid', 'high']), 'large': aa['area'] > 2000}
    gd = bl[bl.method == 'nn_dev'].set_index('basin')
    for sname, mask in strata.items():
        ids = fc.index[mask.values].intersection(gd.index)
        d, lo, hi = boot(fc.loc[ids, 'nse'].values, gd.loc[ids, 'nse'].values, med, rng)
        key = [k for k in E['strata'] if k.startswith('recent/' + sname)][0]
        S['strata'][sname] = {'n': int(len(ids)), 'flowcast': float(fc.loc[ids, 'nse'].median()),
                              'claimed': E['strata'][key]['flowcast'], 'nn_dev': float(gd.loc[ids, 'nse'].median()),
                              'diff': d, 'ci95': [lo, hi], 'verdict': verdict(lo, hi, 'higher')}
    for L in LEADS:
        g = nw[nw.lead == L].set_index('basin')
        row = {'n': int(len(g)), 'nse': float(g.nse.median()), 'kge': float(g.kge.median()),
               'nse_persist': float(g.nse_persist.median()), 'kge_persist': float(g.kge_persist.median()),
               'wins_vs_persist': float((g.nse > g.nse_persist).mean()), 'claimed': N['exam'][f'recent/{L}']}
        x, y = g.nse.values, g.nse_persist.values
        ok = np.isfinite(x) & np.isfinite(y)
        d, lo, hi = boot(x[ok], y[ok], med, rng)
        row['vs_persist'] = {'n': int(ok.sum()), 'diff': d, 'ci95': [lo, hi], 'verdict': verdict(lo, hi, 'higher')}
        for meth in ('gr4j', 'nwm'):
            b2 = bl[bl.method == meth].set_index('basin')
            common = g.index.intersection(b2.index)
            d, lo, hi = boot(g.loc[common, 'nse'].values, b2.loc[common, 'nse'].values, med, rng)
            row[f'vs_{meth}'] = {'n': int(len(common)), 'nowcast': float(g.loc[common, 'nse'].median()),
                                 'baseline': float(b2.loc[common, 'nse'].median()), 'diff': d, 'ci95': [lo, hi],
                                 'verdict': verdict(lo, hi, 'higher')}
        S['nowcast'][f'lead{L}'] = row
    # flood peaks
    T, R = peak_table()
    T.to_parquet(f'{ROOT}/results/rescore_peaks.parquet')
    PB = pd.read_parquet(f'{BL}/peak_scores.parquet')
    S['peaks']['ratio_inst_over_daily'] = R
    peak_rows = []
    for per in ('val', 'recent'):
        for est in ('mean', 'q95', 'q95_released'):
            t = T[(T.period == per) & (T.method == est)]
            pbf = per_basin_peaks(t)
            S['peaks'][f'{per}/{est}'] = {'n_basins': int(len(pbf)), 'n_peaks': int(len(t)),
                                          'median_abs_pct_err': float(pbf['abs'].median()),
                                          'median_bias_pct': float(pbf['bias'].median()),
                                          'claimed': E['peaks'].get(f'{per}/{est}')}
            for b, r in pbf.iterrows():
                peak_rows.append([b, per, est, int(r.n), round(float(r['abs']), 3), round(float(r['bias']), 3)])
        f = per_basin_peaks(T[(T.period == per) & (T.method == 'mean')])
        for meth in sorted(PB[PB.period == per].method.unique()):
            g = per_basin_peaks(PB[(PB.period == per) & (PB.method == meth)])
            common = f.index.intersection(g.index)
            d, lo, hi = boot(f.loc[common, 'abs'].values, g.loc[common, 'abs'].values, med, rng)
            db, lob, hib = boot(f.loc[common, 'bias'].values, g.loc[common, 'bias'].values, medabs, rng)
            S['peaks'][f'{per}/mean_vs_{meth}'] = {
                'n': int(len(common)), 'flowcast_abs': float(f.loc[common, 'abs'].median()),
                'baseline_abs': float(g.loc[common, 'abs'].median()), 'diff_abs': d, 'ci95_abs': [lo, hi],
                'verdict_abs': verdict(lo, hi, 'lower'), 'flowcast_bias': float(f.loc[common, 'bias'].median()),
                'baseline_bias': float(g.loc[common, 'bias'].median()), 'diff_abs_bias': db, 'ci95_abs_bias': [lob, hib],
                'verdict_abs_bias': verdict(lob, hib, 'lower')}
    rows = []
    for b, r in fc.iterrows():
        g = nw[nw.basin == b].set_index('lead')
        vals = [r[c] for c in ('nse', 'kge', 'fhv', 'flv', 'peak_abs_err_pct_median', 'peak_timing_days', 'cov90', 'cov50')]
        vals += [g.loc[L, 'nse'] for L in LEADS]
        rows.append([b] + [None if not np.isfinite(v) else round(float(v), 4) for v in vals])
    with open(f'{ROOT}/results/rescore_recent_{TAG}.json', 'w') as f:
        json.dump({'what': 'Released FlowCast re-scored on the recent exam WY2016-23 against NWIS daily flow with '
                           'waterdata.metrics (seen reproduction, 18 Nov 2026); columns as rescore_main_2026-11-18.json',
                   'n': len(rows), 'columns': ['basin', 'nse', 'kge', 'fhv', 'flv', 'peak_abs_err_pct_median',
                                               'peak_timing_days', 'cov90', 'cov50'] + [f'nowcast_nse_lead{L}' for L in LEADS],
                   'rows': rows}, f, separators=(',', ':'))
    with open(f'{ROOT}/results/rescore_peaks_{TAG}.json', 'w') as f:
        json.dump({'what': 'FlowCast annual max daily x 1.38 vs NWIS annual instantaneous peaks, per basin (estimate: mean, raw q95 as '
                           'shipped, q95_released = widened): number of '
                           'peaks, median |% error|, median % error (seen reproduction, 18 Nov 2026)',
                   'columns': ['basin', 'period', 'estimate', 'n_peaks', 'median_abs_pct_err', 'median_pct_err'],
                   'rows': peak_rows}, f, separators=(',', ':'))
    with open(f'{ROOT}/results/rescore_recent_summary_{TAG}.json', 'w') as f:
        json.dump(S, f, indent=1)
    print(json.dumps({'flowcast': S['flowcast'], 'claimed': S['claimed']}, indent=1))


if __name__ == '__main__':
    {'run': run, 'summary': summary}[sys.argv[1]]()
