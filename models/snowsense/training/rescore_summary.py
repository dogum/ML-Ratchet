"""Medians, paired bootstrap intervals and verdicts for the SnowSense reproduction re-score (added on import, seen exam).

    python training/rescore_summary.py        (reads training/rescore_exam_stations.csv; numpy, pandas, scipy only)

Paired bootstrap over stations: 10,000 resamples of the 621 exam stations with replacement (seed 20261118), the same
draw for SnowSense and every baseline. Statistic: difference in medians (SnowSense minus baseline; NaN-skipping
medians, as the exam's summaries are). For the bias rows the statistic is |median bias| (SnowSense minus baseline):
closer to zero is better, and the row reports the median bias the shipped card reports. 95 % percentile interval.
A baseline is beaten (or beats SnowSense) only if the interval excludes zero; a row is a win only if SnowSense beats
every baseline on it, a loss if any baseline beats it, else a tie. Win fractions with exact Clopper-Pearson
intervals are kept as a cross-check. Writes training/rescore_summary.json.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SRC = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(SRC, 'rescore_exam_stations.csv'))
SEED, B = 20261118, 10000
N = len(T)
IDX = np.random.default_rng(SEED).integers(0, N, size=(B, N))
NAMES = {'eb_default': 'Energy balance, WaterSim defaults (weather only)',
         'dd_default': 'Degree-day, WaterSim defaults (weather only)',
         'eb_devtuned': 'Energy balance, dev-tuned (added on import, seen exam)',
         'dd_devtuned': 'Degree-day, dev-tuned (added on import, seen exam)',
         'clim_shipped': "Climatology, station's own WY2001-2010 SWE (WaterData)",
         'ddcal_shipped': "Degree-day calibrated per station (WaterData; reads the station's SWE)"}
WEATHER = ['eb_default', 'dd_default', 'eb_devtuned', 'dd_devtuned', 'clim_shipped']
CEILING = ['ddcal_shipped']
METRICS = {'nse': 'higher', 'rmse_mm': 'lower', 'peak_swe_abs_err_pct': 'lower', 'meltout_mae_days': 'lower',
           'peak_swe_bias_pct': 'closer', 'meltout_bias_days': 'closer'}


def stat(a, better):
    m = np.nanmedian(a, axis=-1)
    return np.abs(m) if better == 'closer' else m


def compare(sub, metric, base, better):
    """Point difference, 95 % interval, and who wins, for SnowSense minus `base` on the stations in `sub`."""
    a, b = sub[f'net_{metric}'].values, sub[f'{base}_{metric}'].values
    idx = IDX[:, :len(a)] if len(a) == N else np.random.default_rng(SEED).integers(0, len(a), size=(B, len(a)))
    d = stat(a[idx], better) - stat(b[idx], better)
    lo, hi = np.percentile(d, [2.5, 97.5])
    point = float(stat(a, better) - stat(b, better))
    good = (lo > 0) if better == 'higher' else (hi < 0)
    bad = (hi < 0) if better == 'higher' else (lo > 0)
    return {'baseline': base, 'median_model': float(np.nanmedian(a)), 'median_baseline': float(np.nanmedian(b)),
            'diff': point, 'ci95': [float(lo), float(hi)], 'result': 'better' if good else 'worse' if bad else 'tie'}


def verdict(cs):
    r = [c['result'] for c in cs]
    return 'loss' if 'worse' in r else 'win' if all(x == 'better' for x in r) else 'tie'


rows = []
for metric, better in METRICS.items():
    for label, bases in (('weather-only physics and climatology', WEATHER), ('gauged ceiling', CEILING)):
        cs = [compare(T, metric, b, better) for b in bases]
        rows.append({'metric': metric, 'better': better, 'against': label, 'comparisons': cs, 'verdict': verdict(cs)})

below = {}
for name in ['net'] + WEATHER + CEILING:
    below[name] = float((T[f'{name}_nse'] < 0.5).mean())
bd = {}
for base in WEATHER + CEILING:
    a, b = (T['net_nse'].values < 0.5), (T[f'{base}_nse'].values < 0.5)
    d = a[IDX].mean(1) - b[IDX].mean(1)
    bd[base] = {'diff': float(a.mean() - b.mean()), 'ci95': [float(x) for x in np.percentile(d, [2.5, 97.5])]}

cp = {}
for base in WEATHER + CEILING:
    m = T['net_nse'].notna() & T[f'{base}_nse'].notna()
    x, n = int((T.loc[m, 'net_nse'] > T.loc[m, f'{base}_nse']).sum()), int(m.sum())
    ci = binomtest(x, n).proportion_ci(0.95, method='exact')
    cp[base] = {'wins': x, 'n': n, 'frac': x / n, 'ci95': [ci.low, ci.high]}

groups = {}
for col, key in (('region', 'by_region'), ('band', 'by_elev')):
    if col == 'band':
        T['band'] = pd.cut(T.elev_m, [0, 1500, 2200, 2800, 5000],
                           labels=['<1500 m', '1500-2200 m', '2200-2800 m', '>2800 m']).astype(str)
    groups[key] = {}
    for g, sub in T.groupby(col):
        groups[key][g] = {'n': int(len(sub)), 'net': float(sub.net_nse.median()),
                          **{b: compare(sub, 'nse', b, 'higher') for b in ('eb_default', 'eb_devtuned', 'ddcal_shipped')}}

med = {name: {c: float(T[f'{name}_{c}'].median()) for c in list(METRICS) + ['n_years']}
       for name in ['net', 'eb_default', 'dd_default', 'eb_devtuned', 'dd_devtuned', 'clim_shipped', 'ddcal_shipped',
                    'clim_rescored', 'ddcal_rescored']}
out = {'what': 'SnowSense v1.0 reproduction re-score on the rebuilt SNOTEL exam (18 Nov 2026, added on import, seen '
               'exam). Paired bootstrap over stations, 10,000 resamples, seed 20261118; percentile 95 % intervals of '
               'SnowSense minus baseline (difference in medians; |median| for bias rows).',
       'n_stations': N, 'baselines': NAMES, 'medians': med, 'rows': rows, 'share_nse_below_0.5': below,
       'share_below_diff': bd, 'win_fraction_clopper_pearson': cp, **groups}
json.dump(out, open(os.path.join(SRC, 'rescore_summary.json'), 'w'), indent=1)
for r in rows:
    print(r['metric'], r['against'], r['verdict'],
          [(c['baseline'], round(c['diff'], 3), [round(v, 3) for v in c['ci95']]) for c in r['comparisons']])
print(pd.DataFrame(med).T.round(3))
