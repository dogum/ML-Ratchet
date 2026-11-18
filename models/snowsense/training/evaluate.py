"""Score a SnowSense checkpoint.

python3 evaluate.py <run> sim <valtag>      frozen simulated validation: median per-series NSE (the ratchet metric)
python3 evaluate.py <run> dev|exam [k.json] real stations, WY2011-2025, B's protocol; daily and annual coverage

Estimates: SWE = 100 max(q50, 0)^2 (median in z = sqrt(SWE)/10 space). Annual peak SWE, peak date and melt-out (first day after the
peak below 5 mm, as waterdata.metrics.melt_out) from the median trajectory; their intervals from the comonotone
quantile trajectories (q16/q84, q05/q95; one error draw shared by the whole year), widened by factors
fitted on dev only (calib_sigma.py, widen-only).
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
from common import *
from model import SnowNet

torch.set_num_threads(1)


def load(run):
    s = torch.load(f'{ROOT}/ckpt/{run}.pt', weights_only=False)
    net = SnowNet(hidden=s.get('hidden', 64), head=s.get('head', 'gauss'))
    net.load_state_dict(s['net'])
    net.eval()
    return net


def predict(net, X, bs=256):
    """Quantiles (n, T, 5) in z = sqrt(SWE)/10 space: q05, q16, q50, q84, q95."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net.quantiles(torch.tensor(np.asarray(X[i:i + bs], np.float32))).numpy())
    return np.concatenate(out)


def band(Q, j, k=1.0):
    """SWE (mm) of quantile j (0..4), its distance from the median widened by k."""
    z = Q[..., 2] + k * (Q[..., j] - Q[..., 2])
    return 100.0 * np.maximum(z, 0) ** 2


def melt_out_idx(s, thresh=5.0):
    i = int(np.argmax(s))
    b = np.where(s[i:] < thresh)[0]
    return i + int(b[0]) if len(b) else None


def annual(Q, kp=1.0, km=1.0):
    """Annual products for one WY slice (Q: (days, 5))."""
    s = band(Q, 2)
    out = {'peak': float(s.max()), 'peak_day': int(np.argmax(s)), 'melt': melt_out_idx(s)}
    for tag, (jl, jh) in (('68', (1, 3)), ('90', (0, 4))):
        out[f'peak_lo{tag}'], out[f'peak_hi{tag}'] = float(band(Q, jl, kp).max()), float(band(Q, jh, kp).max())
        out[f'melt_lo{tag}'], out[f'melt_hi{tag}'] = melt_out_idx(band(Q, jl, km)), melt_out_idx(band(Q, jh, km))
    return out


def sim_score(run, valtag):
    d = np.load(f'{ROOT}/banks/{valtag}.npz')
    meta = pd.read_parquet(f'{ROOT}/banks/{valtag}_meta.parquet')
    net = load(run)
    Q = predict(net, d['X'])
    Y = d['Y'].astype(np.float32)
    S = band(Q, 2)
    sl = slice(WARM, WARM + 365)
    meta['g'] = meta['run'].astype(str) + '_' + meta['k'].astype(str)
    nses, pk, mo = [], [], []
    for g, ix in meta.groupby('g').indices.items():
        y, s = Y[ix, sl].ravel(), S[ix, sl].ravel()
        m = np.isfinite(y)
        v = np.var(y[m])
        if v > 25.0:
            nses.append(1 - np.mean((s[m] - y[m]) ** 2) / v)
        for i in ix:
            yy, ss = Y[i, sl], S[i, sl]
            if np.nanmax(yy) >= 25:
                pk.append(abs(ss.max() - yy.max()) / yy.max() * 100)
                a, b = melt_out_idx(np.nan_to_num(yy)), melt_out_idx(ss)
                if a is not None and b is not None:
                    mo.append(abs(a - b))
    yy = Y[:, sl]
    snowy = (yy > 1) | (S[:, sl] > 1)
    cov = float(((yy >= band(Q[:, sl], 1)) & (yy <= band(Q[:, sl], 3)))[snowy].mean())
    return {'nse_median': float(np.median(nses)), 'n_series': len(nses), 'peak_abs_err_pct_median': float(np.median(pk)),
            'meltout_mae_days': float(np.mean(mo)), 'cov68_daily': cov}


def real_score(run, split, k=None, tag=None):
    from evalset import station_set
    k = k or {'daily': 1.0, 'peak': 1.0, 'melt': 1.0}
    net = load(run)
    rows, yrows = [], []
    for sid, lat, elev, ok in station_set(split):
        df = load_station(sid)
        X, D, Y, wys = station_windows(sid, list(range(2011, 2026)), lat, elev, df)
        if not len(X):
            continue
        Q = predict(net, X)
        med = stitch(band(Q, 2), D, wys)
        r = score(df['SWE_eod'], med, ok)
        lo, hi = stitch(band(Q, 1, k['daily']), D, wys), stitch(band(Q, 3, k['daily']), D, wys)
        lo9, hi9 = stitch(band(Q, 0, k['daily']), D, wys), stitch(band(Q, 4, k['daily']), D, wys)
        o = df['SWE_eod'].reindex(med.index)
        m = o.notna() & np.isin(wy_of(med.index), ok) & ((o > 1) | (med > 1))
        r['cov68_daily'] = float(((o >= lo) & (o <= hi))[m].mean())
        r['cov90_daily'] = float(((o >= lo9) & (o <= hi9))[m].mean())
        r.update(station_id=sid, method=run)
        rows.append(r)
        for j, y in enumerate(wys):
            if y not in ok:
                continue
            wmask = (wy_of(D[j]) == y)
            oy = df['SWE_eod'].reindex(D[j][wmask])
            if oy.notna().sum() < 300 or oy.max() < 25:
                continue
            a = annual(Q[j][wmask], k['peak'], k['melt'])
            a.update(station_id=sid, wy=y, obs_peak=float(oy.max()), obs_peak_day=int(np.nanargmax(oy.values)),
                     obs_melt=melt_out_idx(np.nan_to_num(oy.values)))
            yrows.append(a)
    sc = pd.DataFrame(rows)
    yr = pd.DataFrame(yrows)
    tag = tag or f'{run}_{split}'
    sc.to_parquet(f'{ROOT}/results/{tag}.parquet')
    yr.to_parquet(f'{ROOT}/results/{tag}_years.parquet')
    return summarize(sc, yr)


def summarize(sc, yr):
    s = {'n': int(len(sc))}
    for c in ('nse', 'rmse_mm', 'peak_swe_abs_err_pct', 'peak_swe_bias_pct', 'meltout_mae_days', 'meltout_bias_days',
              'cov68_daily', 'cov90_daily'):
        s[c] = float(sc[c].median())
    if len(yr):
        s['peak_cov68'] = float(((yr.obs_peak >= yr.peak_lo68) & (yr.obs_peak <= yr.peak_hi68)).mean())
        s['peak_cov90'] = float(((yr.obs_peak >= yr.peak_lo90) & (yr.obs_peak <= yr.peak_hi90)).mean())
        m = yr['obs_melt'].notna() & yr['melt'].notna()
        y = yr[m]
        lo = y.melt_lo68.fillna(-1e9)
        hi = y.melt_hi68.fillna(1e9)
        s['melt_cov68'] = float(((y.obs_melt >= lo) & (y.obs_melt <= hi)).mean())
        lo = y.melt_lo90.fillna(-1e9)
        hi = y.melt_hi90.fillna(1e9)
        s['melt_cov90'] = float(((y.obs_melt >= lo) & (y.obs_melt <= hi)).mean())
        s['peak_day_mae'] = float(np.abs(yr['peak_day'] - yr['obs_peak_day']).median())
        s['melt_mae_years'] = float(np.abs(y['melt'] - y['obs_melt']).median())
        s['n_station_years'] = int(len(yr))
    return s


if __name__ == '__main__':
    run, what = sys.argv[1], sys.argv[2]
    if what == 'sim':
        r = sim_score(run, sys.argv[3])
    else:
        k = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
        r = real_score(run, what, k)
    print(json.dumps({'run': run, 'what': what, **r}))
