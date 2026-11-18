"""Streamflow / snow skill metrics (numpy only). All functions ignore days where obs or sim is NaN.

nse      Nash-Sutcliffe efficiency (Nash & Sutcliffe 1970)
kge      Kling-Gupta efficiency 2009 (r, alpha = sd ratio, beta = mean ratio); kge_parts returns the components
fhv      %bias of the high-flow segment: flows above the 98th percentile of the flow-duration curve (Yilmaz et al. 2008)
flv      %bias of the low-flow segment: log flows below the 70th exceedance percentile, relative to the minimum (Yilmaz 2008)
peak_error  mean relative error of annual-maximum daily flow (water years) in %, and median absolute
peak_timing mean absolute timing error (days) of matched peaks: each observed peak (scipy-free prominence rule)
            is matched to the sim maximum within +-window days (Kratzert et al. 2021 / neuralhydrology style)
summary  dict of all of the above
"""
import numpy as np
import pandas as pd


def _clean(obs, sim):
    o = np.asarray(obs, dtype=float); s = np.asarray(sim, dtype=float)
    m = np.isfinite(o) & np.isfinite(s)
    return o[m], s[m]


def nse(obs, sim):
    o, s = _clean(obs, sim)
    if o.size < 2 or np.var(o) == 0:
        return np.nan
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def kge_parts(obs, sim):
    o, s = _clean(obs, sim)
    if o.size < 2 or o.std() == 0 or o.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    a = s.std() / o.std(); b = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)), float(r), float(a), float(b)


def kge(obs, sim):
    return kge_parts(obs, sim)[0]


def fhv(obs, sim, h=0.02):
    o, s = _clean(obs, sim)
    if o.size < 10:
        return np.nan
    o = np.sort(o)[::-1]; s = np.sort(s)[::-1]
    n = max(1, int(np.round(h * o.size)))
    return float(100 * (s[:n].sum() - o[:n].sum()) / o[:n].sum()) if o[:n].sum() > 0 else np.nan


def flv(obs, sim, l=0.3, eps=1e-6):
    o, s = _clean(obs, sim)
    if o.size < 10:
        return np.nan
    o = np.sort(np.maximum(o, eps)); s = np.sort(np.maximum(s, eps))
    n = max(2, int(np.round(l * o.size)))
    o, s = np.log(o[:n]), np.log(s[:n])
    qo = np.sum(o - o.min()); qs = np.sum(s - s.min())
    return float(-100 * (qs - qo) / qo) if qo > 0 else np.nan


def annual_peaks(q, wy_start_month=10):
    q = pd.Series(q).dropna()
    wy = q.index.year + (q.index.month >= wy_start_month).astype(int)
    g = q.groupby(wy)
    cnt = g.count()
    ok = cnt[cnt >= 330].index
    return g.max().loc[ok], g.idxmax().loc[ok]


def peak_error(obs, sim):
    """obs, sim: pd.Series with a DatetimeIndex. Annual maximum daily flow per water year (>=330 valid days)."""
    o = pd.Series(obs); s = pd.Series(sim).reindex(o.index)
    m = o.notna() & s.notna()
    po, _ = annual_peaks(o[m]); ps, _ = annual_peaks(s[m])
    j = pd.concat([po, ps], axis=1, keys=['o', 's']).dropna()
    j = j[j.o > 0]
    if len(j) == 0:
        return np.nan, np.nan
    rel = 100 * (j.s - j.o) / j.o
    return float(rel.mean()), float(rel.abs().median())


def peak_timing(obs, sim, window=3, n_per_year=3):
    """Mean |timing error| in days for the largest n_per_year separated peaks per year (separation 2*window+1 days)."""
    o = pd.Series(obs); s = pd.Series(sim).reindex(o.index)
    m = o.notna() & s.notna()
    o, s = o[m], s[m]
    if len(o) < 365:
        return np.nan
    errs = []
    for _, oy in o.groupby(o.index.year):
        x = oy.copy()
        for _ in range(n_per_year):
            if x.dropna().empty:
                break
            t = x.idxmax()
            lo, hi = t - pd.Timedelta(days=window), t + pd.Timedelta(days=window)
            sw = s.loc[lo:hi]
            if len(sw):
                errs.append(abs((sw.idxmax() - t).days))
            x.loc[t - pd.Timedelta(days=2 * window):t + pd.Timedelta(days=2 * window)] = np.nan
    return float(np.mean(errs)) if errs else np.nan


def summary(obs, sim):
    o = pd.Series(obs); s = pd.Series(sim).reindex(o.index)
    k, r, a, b = kge_parts(o, s)
    pe, pea = peak_error(o, s)
    oo, ss = _clean(o, s)
    return {'nse': nse(o, s), 'kge': k, 'r': r, 'alpha': a, 'beta': b, 'fhv': fhv(o, s), 'flv': flv(o, s),
            'peak_bias_pct': pe, 'peak_abs_err_pct_median': pea, 'peak_timing_days': peak_timing(o, s),
            'n_days': int(oo.size)}


# --- snow ---
def swe_metrics(obs, sim):
    """SWE skill on days with obs: NSE, RMSE (mm), bias in peak SWE (%), error in melt-out date (days) per water year."""
    o = pd.Series(obs); s = pd.Series(sim).reindex(o.index)
    m = o.notna() & s.notna()
    o, s = o[m], s[m]
    out = {'nse': nse(o, s), 'rmse_mm': float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else np.nan}
    wy = o.index.year + (o.index.month >= 10).astype(int)
    pk, mo = [], []
    for y, oy in o.groupby(wy):
        if len(oy) < 300 or oy.max() < 25:
            continue
        sy = s.loc[oy.index]
        pk.append(100 * (sy.max() - oy.max()) / oy.max())
        mo.append((melt_out(sy) - melt_out(oy)).days if melt_out(sy) is not None and melt_out(oy) is not None else np.nan)
    out['peak_swe_bias_pct'] = float(np.mean(pk)) if pk else np.nan
    out['peak_swe_abs_err_pct'] = float(np.mean(np.abs(pk))) if pk else np.nan
    mo = [x for x in mo if np.isfinite(x)]
    out['meltout_mae_days'] = float(np.mean(np.abs(mo))) if mo else np.nan
    out['meltout_bias_days'] = float(np.mean(mo)) if mo else np.nan
    out['n_years'] = len(pk)
    return out


def melt_out(swe, thresh=5.0):
    """First day after the annual peak when SWE drops below thresh mm."""
    swe = pd.Series(swe).dropna()
    if swe.empty:
        return None
    t = swe.idxmax()
    after = swe.loc[t:]
    below = after[after < thresh]
    return below.index[0] if len(below) else None
