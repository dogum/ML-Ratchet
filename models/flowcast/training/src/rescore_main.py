"""Reproduction re-score of the RELEASED FlowCast on the main exam (WY1990-99), 18 Nov 2026. Seen: the exam was
opened at release; nothing here changes weights, widening or any decision.

usage: python training/src/rescore_main.py run        -> WORK/results/rescore_main.parquet (+ nowcast parquet)
       python training/src/rescore_main.py summary    -> WORK/results/rescore_main_2026-11-18.json (per basin)
                                                         WORK/results/rescore_main_summary_2026-11-18.json
       (both JSONs were then copied by hand into training/results/ as the committed record)

Same path as the shipped exam (evalreal.py exam with k = 1.0, 1.08, then exam_report.py) but through the released
numpy module: flowcast.py's predict() on wd.forcing(b, 1988-10-01, 1999-10-01) and wd.spec(b) equals the torch
checkpoint to 1.7e-7 (export_check.json), and its widening is the released (1.0, 1.08). Scores are
waterdata.metrics.summary against Caravan Q (as evalreal.score), plus the band coverages. A basin is scored when it
has at least 365 observed days in the window (as evalreal). The nowcast weights are scored as evalnow.py does
(observed Caravan flow over warm-up and window, leads 1/3/7, persistence = obs shifted by the lead).

Intervals: paired bootstrap over basins of the difference in medians (FlowCast − baseline) on the basins both
have, 10,000 resamples, numpy default_rng(20261118); per-basin baseline scores from
$ML_RATCHET_WATER/data/proc/baselines/flow_scores.parquet (split exam, period val).
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa: F401,F403  (paths, VAL, WORK, MODEL, WATER_DATA; puts shared/water on sys.path)
import waterdata as wd
from waterdata import metrics as M

sys.path.insert(0, MODEL)
import flowcast as fcm  # noqa: E402

TAG = '2026-11-18'
SEED, NBOOT = 20261118, 10000
LEADS = (1, 3, 7)
S0, S1 = VAL
WARM0 = (pd.Timestamp(S0) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
END1 = (pd.Timestamp(S1) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def exam_ids():
    """The 619 exam basins, checked against splits.json itself."""
    ids = list(wd.basins('camels', split='exam'))
    with open(f'{WATER_DATA}/data/proc/splits.json', encoding='utf-8') as f:
        sp = json.load(f)
    want = sp['camels_us']['exam'] if 'camels_us' in sp else None
    if isinstance(want, dict):
        want = want.get('basins') or want.get('ids')
    assert want is not None and sorted(ids) == sorted(want) and len(ids) == 619, 'exam basins differ from splits.json'
    return ids


def score(o, idx, out):
    r = M.summary(o.reindex(idx), out.Q)
    oo = o.reindex(idx)
    v = oo.notna().values
    ov = oo.values[v]
    q = out[['q05', 'q25', 'q50', 'q75', 'q95']].values
    r['cov90'] = float(np.mean((ov >= q[v, 0]) & (ov <= q[v, 4])))
    r['cov50'] = float(np.mean((ov >= q[v, 1]) & (ov <= q[v, 3])))
    r['width90_rel'] = float(np.mean(q[v, 4] - q[v, 0]) / max(np.mean(ov), 1e-6))
    return r


def run():
    m = fcm.FlowCast()
    n = fcm.FlowCast(os.path.join(MODEL, 'flowcast_nowcast_weights.npz'))
    rows, now = [], []
    idx = pd.date_range(S0, S1)
    for k, b in enumerate(exam_ids()):
        obs = wd.streamflow(b, 'caravan').Q_mm
        o = obs.loc[S0:S1]
        if o.notna().sum() < 365:
            continue
        f = wd.forcing(b, WARM0, END1)
        sp = wd.spec(b)
        out = m.predict(f, sp, S0, S1)
        r = score(obs, idx, out)
        r.update({'basin': b})
        rows.append(r)
        for lead in LEADS:
            q = n.predict(f, sp, S0, S1, qobs=obs, lead=lead).Q
            pers = obs.shift(lead, freq='D').reindex(idx)
            r1, r2 = M.summary(o, q), M.summary(o, pers)
            now.append({'basin': b, 'lead': lead, 'nse': r1['nse'], 'kge': r1['kge'],
                        'nse_persist': r2['nse'], 'kge_persist': r2['kge']})
        if k % 50 == 0:
            print(k, b, round(r['nse'], 3), flush=True)
    pd.DataFrame(rows).to_parquet(f'{ROOT}/results/rescore_main.parquet')
    pd.DataFrame(now).to_parquet(f'{ROOT}/results/rescore_main_nowcast.parquet')


def boot(a, b, stat, rng):
    """Paired bootstrap of stat(a) - stat(b) over units; returns diff and 95 % percentile interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = stat(a) - stat(b)
    i = rng.integers(0, len(a), (NBOOT, len(a)))
    ds = stat(a[i], axis=1) - stat(b[i], axis=1)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def verdict(lo, hi, better):
    if better == 'higher':
        return 'win' if lo > 0 else 'loss' if hi < 0 else 'tie'
    return 'win' if hi < 0 else 'loss' if lo > 0 else 'tie'


def summary():
    fc = pd.read_parquet(f'{ROOT}/results/rescore_main.parquet').set_index('basin')
    nw = pd.read_parquet(f'{ROOT}/results/rescore_main_nowcast.parquet')
    bl = pd.read_parquet(f'{WATER_DATA}/data/proc/baselines/flow_scores.parquet')
    bl = bl[(bl.split == 'exam') & (bl.period == 'val')]
    a = wd.attributes('camels')
    per = fc[['nse', 'kge', 'fhv', 'flv', 'peak_abs_err_pct_median', 'peak_timing_days', 'cov90', 'cov50']].copy()
    cols = ['nse', 'kge', 'fhv', 'flv', 'peak_abs_err_pct_median', 'peak_timing_days', 'cov90', 'cov50']
    rows = []
    for b, r in per.iterrows():
        g = nw[nw.basin == b].set_index('lead')
        vals = [r[c] for c in cols] + [g.loc[L, 'nse'] for L in LEADS]
        rows.append([b] + [None if not np.isfinite(v) else round(float(v), 4) for v in vals])
    with open(f'{ROOT}/results/rescore_main_{TAG}.json', 'w') as f:
        json.dump({'what': 'Released FlowCast (r1b, k_widen 1.0/1.08) re-scored on the main exam WY1990-99 against '
                           'Caravan Q with waterdata.metrics (seen reproduction, 18 Nov 2026). fhv/flv %; '
                           'peak_abs_err_pct_median = median abs % error of annual daily peaks; peak_timing_days; '
                           'cov90/cov50 = share of days inside the 90/50 % band; nowcast_nse_leadL = released nowcast '
                           'weights at lead L days',
                   'n': len(rows), 'columns': ['basin'] + cols + [f'nowcast_nse_lead{L}' for L in LEADS],
                   'rows': rows}, f, separators=(',', ':'))

    rng = np.random.default_rng(SEED)
    med = lambda x, axis=None: np.nanmedian(x, axis=axis)
    medabs = lambda x, axis=None: np.nanmedian(np.abs(x), axis=axis)
    metrics = [('nse', 'higher', med, 'median NSE'), ('kge', 'higher', med, 'median KGE'),
               ('fhv', 'closer', medabs, 'median |FHV| %'), ('peak_abs_err_pct_median', 'lower', med, 'median annual daily peak abs error %'),
               ('peak_timing_days', 'lower', med, 'median peak timing error, days')]
    S = {'what': 'Paired bootstrap over basins of median(FlowCast) - median(baseline) on the basins both have; '
                 f'{NBOOT} resamples, numpy default_rng({SEED}); verdict win/loss only if the 95 % interval excludes 0',
         'claimed': {'nse': 0.3411705791096633, 'kge': 0.3666337887637192, 'fhv': -30.938536293888315,
                     'peak_abs_err_daily': 52.227386474609375, 'cov90': 0.9731653888280394, 'cov50': 0.5553121577217963,
                     'nse_531': 0.3570562389938964, 'frac_nse_lt0': 0.1421647819063005},
         'flowcast': {'n': int(len(fc)), 'nse': float(fc.nse.median()), 'kge': float(fc.kge.median()),
                      'fhv': float(fc.fhv.median()), 'flv': float(fc.flv.median()),
                      'peak_abs_err_daily': float(fc.peak_abs_err_pct_median.median()),
                      'timing_d': float(fc.peak_timing_days.median()), 'cov90': float(fc.cov90.median()),
                      'cov50': float(fc.cov50.median()), 'cov90_pooled_mean': float(fc.cov90.mean()),
                      'width90_rel': float(fc.width90_rel.median()), 'frac_nse_lt0': float((fc.nse < 0).mean()),
                      'nse_531': float(fc[a.loc[fc.index, 'in_531'].values].nse.median())},
         'vs': {}, 'nowcast': {}, 'strata': {}}
    for meth in ('nn_dev', 'nn_k12', 'gr4j', 'nwm', 'clim', 'persist1'):
        g = bl[bl.method == meth].set_index('basin')
        common = g.index.intersection(fc.index)
        S['vs'][meth] = {'n_common': int(len(common)), 'wins_nse': int((fc.loc[common].nse > g.loc[common].nse).sum())}
        for col, better, stat, label in metrics:
            x, y = fc.loc[common, col].values, g.loc[common, col].values
            ok = np.isfinite(x) & np.isfinite(y)
            d, lo, hi = boot(x[ok], y[ok], stat, rng)
            S['vs'][meth][col] = {'label': label, 'n': int(ok.sum()), 'flowcast': float(stat(x[ok])), 'baseline': float(stat(y[ok])),
                                  'diff': d, 'ci95': [lo, hi], 'verdict': verdict(lo, hi, 'higher' if better == 'higher' else 'lower')}
        # 531 subset NSE
        c531 = common[a.loc[common, 'in_531'].values]
        x, y = fc.loc[c531, 'nse'].values, g.loc[c531, 'nse'].values
        d, lo, hi = boot(x, y, med, rng)
        S['vs'][meth]['nse_531'] = {'n': int(len(c531)), 'flowcast': float(np.median(x)), 'baseline': float(np.median(y)),
                                    'diff': d, 'ci95': [lo, hi], 'verdict': verdict(lo, hi, 'higher')}
    # strata (paired vs U-dev)
    aa = a.loc[fc.index]
    strata = {'snow': aa.frac_snow > 0.3, 'arid': aa.aridity_FAO_PM > 1, 'humid': aa.aridity_FAO_PM < 0.6,
              'regulated': aa.reg_class.isin(['mid', 'high']), 'large': aa['area'] > 2000}
    gd = bl[bl.method == 'nn_dev'].set_index('basin')
    for sname, mask in strata.items():
        ids = fc.index[mask.values].intersection(gd.index)
        d, lo, hi = boot(fc.loc[ids, 'nse'].values, gd.loc[ids, 'nse'].values, med, rng)
        S['strata'][sname] = {'n': int(len(ids)), 'flowcast': float(fc.loc[ids, 'nse'].median()),
                              'nn_dev': float(gd.loc[ids, 'nse'].median()), 'diff': d, 'ci95': [lo, hi],
                              'verdict': verdict(lo, hi, 'higher')}
    # nowcast: vs persistence at the same lead (own computation), GR4J and NWM (per-basin baseline scores)
    for L in LEADS:
        g = nw[nw.lead == L].set_index('basin')
        row = {'n': int(len(g)), 'nse': float(g.nse.median()), 'kge': float(g.kge.median()),
               'nse_persist': float(g.nse_persist.median()), 'kge_persist': float(g.kge_persist.median()),
               'wins_vs_persist': float((g.nse > g.nse_persist).mean())}
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
    with open(f'{ROOT}/results/rescore_main_summary_{TAG}.json', 'w') as f:
        json.dump(S, f, indent=1)
    print(json.dumps({k: S[k] for k in ('flowcast',)}, indent=1))


if __name__ == '__main__':
    {'run': run, 'summary': summary}[sys.argv[1]]()
