"""Common scoring for the River Bake-off (every arm and every reference row goes through these functions).

Per basin-period (real data), from waterdata.metrics (the repo's shared/water/waterdata, vendored):
  nse, kge (+ r, alpha, beta), fhv (% bias of the top 2 % flows), flv, peak_bias_pct and peak_abs_err_pct_median
  (annual-maximum daily flow), peak_timing_days;
plus, from the quantiles:
  cov50, cov90   share of observed days inside the 25-75 % and 5-95 % bands
  crps           mean over days of 2 x the mean pinball loss at the 5 quantile levels (a 5-point approximation of the
                 CRPS, mm/day; for a point forecast it equals the MAE)
  crps_rel       crps / mean observed flow;  width90_rel  mean (q95 - q05) / mean observed flow
  mae, n_days, n_cross (days where the arm's quantiles crossed; they are sorted before scoring)
Frozen sim val (800 runs x 730 scored days): nse, kge, fhv, cov50, cov90, crps_rel per run.
Summaries: medians over units; paired bootstrap over units of the difference in medians (10,000 resamples,
default_rng(20261118)); per-class medians (snow / arid / humid / regulated / large).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from waterdata import metrics as M  # noqa: E402

OUT_KEYS = ['mean'] + C.QKEYS
HIGHER = {'nse': True, 'kge': True, 'cov50': None, 'cov90': None, 'crps': False, 'crps_rel': False,
          'fhv': 'abs', 'peak_abs_err_pct_median': False, 'peak_timing_days': False, 'mae': False}


class OutputError(ValueError):
    pass


def check_output(out, n_target):
    """Validate an arm's output and return float64 arrays: mean (n,), Q (n, 5) sorted, n_cross."""
    if not isinstance(out, dict):
        raise OutputError('predict must return a dict')
    miss = [k for k in OUT_KEYS if k not in out]
    if miss:
        raise OutputError(f'missing keys {miss}')
    arr = {}
    for k in OUT_KEYS:
        a = np.asarray(out[k], np.float64).reshape(-1)
        if a.shape != (n_target,):
            raise OutputError(f'{k} has shape {a.shape}, expected ({n_target},)')
        if not np.isfinite(a).all():
            raise OutputError(f'{k} has {int((~np.isfinite(a)).sum())} non-finite values')
        arr[k] = a
    Q = np.stack([arr[k] for k in C.QKEYS], 1)
    n_cross = int((np.diff(Q, axis=1) < -1e-9).any(1).sum())
    if n_cross:
        Q = np.sort(Q, 1)
    return arr['mean'], Q, n_cross


def pinball(y, Q, taus=C.QLEVELS):
    e = y[:, None] - Q
    t = np.asarray(taus)[None, :]
    return np.maximum(t * e, (t - 1) * e)


def crps_q(y, Q):
    """5-quantile CRPS approximation per day: 2 x mean pinball loss."""
    return 2.0 * pinball(y, Q).mean(1)


def interval_scores(y, mean, Q):
    v = np.isfinite(y)
    if v.sum() == 0:
        return {k: np.nan for k in ('cov50', 'cov90', 'crps', 'crps_rel', 'width90_rel', 'mae')}
    yv, Qv, mv = y[v], Q[v], mean[v]
    mo = max(float(np.mean(yv)), 1e-6)
    c = float(np.mean(crps_q(yv, Qv)))
    return {'cov50': float(np.mean((yv >= Qv[:, 1]) & (yv <= Qv[:, 3]))),
            'cov90': float(np.mean((yv >= Qv[:, 0]) & (yv <= Qv[:, 4]))),
            'crps': c, 'crps_rel': c / mo, 'width90_rel': float(np.mean(Qv[:, 4] - Qv[:, 0]) / mo),
            'mae': float(np.mean(np.abs(mv - yv)))}


def score_real(obs, mean, Q=None, dates=None, n_cross=0):
    """obs: pd.Series over the target days (NaN = missing) or an array with `dates`. Returns a dict of metrics.
    Q None (a point reference such as NWM): interval metrics use Q = mean (crps = mae, coverage ~ 0)."""
    if not isinstance(obs, pd.Series):
        obs = pd.Series(np.asarray(obs, float), index=pd.DatetimeIndex(dates))
    mean = np.asarray(mean, np.float64)
    sim = pd.Series(mean, index=obs.index)
    r = M.summary(obs, sim)
    y = obs.values.astype(np.float64)
    if Q is None:
        Q = np.repeat(mean[:, None], 5, 1)
        point = True
    else:
        point = False
    ok = np.isfinite(mean)                    # references may have gaps (NWM ends Jan 2023)
    yy = np.where(ok, y, np.nan)
    r.update(interval_scores(yy, np.where(ok, mean, 0.0), np.where(ok[:, None], Q, 0.0)))
    if point:
        r['cov50'] = r['cov90'] = r['width90_rel'] = np.nan
    r['n_cross'] = n_cross
    return r


def simval_scores(y, mean, Q):
    """Frozen sim val: per run (730 concatenated days)."""
    y = np.asarray(y, np.float64)
    r = {'nse': M.nse(y, mean), 'kge': M.kge(y, mean), 'fhv': M.fhv(y, mean)}
    r.update(interval_scores(y, mean, Q))
    return r


# ------------------------------------------------------------------ summaries
def med(x, axis=None):
    return np.nanmedian(x, axis=axis)


def medabs(x, axis=None):
    return np.nanmedian(np.abs(x), axis=axis)


def summarize(df, cols=('nse', 'kge', 'fhv', 'flv', 'peak_bias_pct', 'peak_abs_err_pct_median', 'peak_timing_days',
                        'cov50', 'cov90', 'crps', 'crps_rel', 'width90_rel', 'mae')):
    out = {'n': int(len(df))}
    for c in cols:
        if c in df:
            v = df[c].astype(float).values
            out[c] = float(np.nanmedian(v)) if np.isfinite(v).any() else None
    if 'fhv' in df:
        out['fhv_abs'] = float(np.nanmedian(np.abs(df['fhv'].astype(float))))
    if 'nse' in df:
        out['frac_nse_lt0'] = float(np.mean(df['nse'].astype(float) < 0))
    if 'cov90' in df and df['cov90'].notna().any():
        out['cov90_mean'] = float(df['cov90'].mean())
        out['cov50_mean'] = float(df['cov50'].mean())
    return out


def by_class(df, classes, col='nse'):
    """classes: dict unit -> list of class labels. Median of col per class."""
    out = {}
    for c in C.CLASSES:
        ids = [u for u in df.index if c in classes.get(u, [])]
        out[c] = {'n': len(ids), col: float(np.nanmedian(df.loc[ids, col])) if ids else None}
    return out


def paired_bootstrap(a, b, stat=med, nboot=C.NBOOT, rng=None):
    """Difference stat(a) - stat(b) on paired units, with a 95 % percentile interval."""
    rng = rng if rng is not None else np.random.default_rng(C.SEED)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return {'n': int(len(a)), 'diff': None, 'ci95': None}
    d = float(stat(a) - stat(b))
    ds = np.empty(nboot)
    for s in range(0, nboot, 1000):                       # chunks keep memory small
        i = rng.integers(0, len(a), (min(1000, nboot - s), len(a)))
        ds[s:s + len(i)] = stat(a[i], axis=1) - stat(b[i], axis=1)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {'n': int(len(a)), 'a': float(stat(a)), 'b': float(stat(b)), 'diff': d, 'ci95': [float(lo), float(hi)]}


def verdict(res, better='higher'):
    if res.get('ci95') is None:
        return 'n/a'
    lo, hi = res['ci95']
    if better == 'higher':
        return 'win' if lo > 0 else 'loss' if hi < 0 else 'tie'
    return 'win' if hi < 0 else 'loss' if lo > 0 else 'tie'


def compare(dfa, dfb, metrics=(('nse', 'higher', med), ('kge', 'higher', med), ('crps_rel', 'lower', med),
                               ('fhv', 'lower', medabs), ('peak_abs_err_pct_median', 'lower', med)),
            rng=None, classes=None):
    """Paired comparison of two per-unit score tables (index = unit) on their common units."""
    rng = rng if rng is not None else np.random.default_rng(C.SEED)
    common = dfa.index.intersection(dfb.index)
    out = {'n_common': int(len(common))}
    for col, better, stat in metrics:
        if col not in dfa or col not in dfb:
            continue
        r = paired_bootstrap(dfa.loc[common, col].values, dfb.loc[common, col].values, stat, rng=rng)
        r['verdict'] = verdict(r, better)
        out[col] = r
    if 'nse' in dfa and 'nse' in dfb and len(common):
        out['share_units_a_better_nse'] = float(np.mean(dfa.loc[common, 'nse'].values > dfb.loc[common, 'nse'].values))
    if classes is not None:
        out['by_class_nse'] = {}
        for c in C.CLASSES:
            ids = [u for u in common if c in classes.get(u, [])]
            r = paired_bootstrap(dfa.loc[ids, 'nse'].values, dfb.loc[ids, 'nse'].values, med, rng=rng)
            r['verdict'] = verdict(r, 'higher')
            out['by_class_nse'][c] = r
    return out
