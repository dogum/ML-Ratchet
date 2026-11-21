"""SnowSense shared helpers: paths, thread limits, SNOTEL loading, water-year windows, input features, scoring.

Inputs are the SNOTEL station's own daily P and T (T_mean, T_min, T_max), plus latitude and elevation. Never SWE.
A water-year window runs from 1 Aug of the previous year (61 warm-up days) to 1 Oct, L = 427 days; days inside the
water year (1 Oct .. 30 Sep) are scored. SNOTEL SWE is start-of-day, so the target for day D is obs SWE(D+1),
exactly as in B's baselines (waterdata/metrics.swe_metrics).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK, MODEL, SNOW_SCORES   # puts shared/water and shared/water/watersim on sys.path

os.environ.setdefault('NUMBA_CACHE_DIR', f'{WORK}/.numba')
ROOT = WORK   # the old sandbox workspace root (water/snowsense): results/, banks/, ckpt/, release/, web/

L = 427          # window length (days)
WARM = 61        # 1 Aug .. 30 Sep warm-up
NCH = 10         # input channels
SEED = 20261118


def wd():
    import waterdata
    return waterdata


def wy_of(idx):
    return idx.year + (idx.month >= 10).astype(int)


def load_station(sid, start=None, end=None):
    """Daily frame with P (gaps -> 0, flagged), T_mean/T_min/T_max (gaps interpolated, flagged), SWE (raw start-of-day)
    and SWE_eod (= SWE of the next day, the end-of-day target)."""
    d = wd().snotel(sid).asfreq('D')
    out = pd.DataFrame(index=d.index)
    out['mP'] = d['P'].isna().astype(np.float32)
    out['P'] = d['P'].clip(lower=0).fillna(0.0)
    tm = d['T_mean'].copy()
    tn, tx = d['T_min'].copy(), d['T_max'].copy()
    tm = tm.fillna((tn + tx) / 2)
    out['mT'] = tm.isna().astype(np.float32)
    for c, s in (('T_mean', tm), ('T_min', tn), ('T_max', tx)):
        out[c] = s
    # fill T_min/T_max from T_mean where only they are missing, then interpolate everything
    out['T_min'] = out['T_min'].fillna(out['T_mean'] - 5.0)
    out['T_max'] = out['T_max'].fillna(out['T_mean'] + 5.0)
    for c in ('T_mean', 'T_min', 'T_max'):
        out[c] = out[c].interpolate(limit_direction='both')
    out['T_min'] = np.minimum(out['T_min'], out['T_mean'])
    out['T_max'] = np.maximum(out['T_max'], out['T_mean'])
    out['SWE'] = d['SWE']
    out['SWE_eod'] = d['SWE'].shift(-1)
    return out.loc[start:end]


def window_start(wy):
    return pd.Timestamp(f'{wy - 1}-08-01')


def features(P, Tm, Tn, Tx, mP, mT, dates, lat, elev_m):
    """(n, NCH) float32 network inputs for one window (or any run of days)."""
    doy = pd.DatetimeIndex(dates).dayofyear.values
    ang = 2 * np.pi * (doy - 1) / 365.25
    f = np.empty((len(P), NCH), np.float32)
    f[:, 0] = np.log1p(np.maximum(P, 0)) / 2.0
    f[:, 1] = Tm / 10.0
    f[:, 2] = Tn / 10.0
    f[:, 3] = Tx / 10.0
    f[:, 4] = np.sin(ang)
    f[:, 5] = np.cos(ang)
    f[:, 6] = mP
    f[:, 7] = mT
    f[:, 8] = (elev_m - 2200.0) / 800.0
    f[:, 9] = (lat - 43.0) / 8.0
    return f


def station_windows(sid, wys, lat, elev_m, df=None):
    """Input windows (n_wy, L, NCH), date index list, and the end-of-day SWE targets (n_wy, L) (NaN = no obs)."""
    if df is None:
        df = load_station(sid)
    X, Y, D, keep = [], [], [], []
    for y in wys:
        s = window_start(y)
        idx = pd.date_range(s, periods=L, freq='D')
        w = df.reindex(idx)
        if w['T_mean'].isna().all():
            continue
        Tm = w['T_mean'].interpolate(limit_direction='both').values
        Tn = w['T_min'].interpolate(limit_direction='both').values
        Tx = w['T_max'].interpolate(limit_direction='both').values
        mT = np.where(w['mT'].isna(), 1.0, w['mT'].values)
        mP = np.where(w['mP'].isna(), 1.0, w['mP'].values)
        P = w['P'].fillna(0.0).values
        X.append(features(P, Tm, Tn, Tx, mP, mT, idx, lat, elev_m))
        Y.append(w['SWE_eod'].values.astype(np.float32))
        D.append(idx)
        keep.append(y)
    if not X:
        return np.zeros((0, L, NCH), np.float32), [], np.zeros((0, L), np.float32), []
    return np.stack(X), D, np.stack(Y), keep


def stitch(pred, dates, wys):
    """Join per-window predictions (n, L) into one daily Series, keeping only each window's own water year."""
    parts = []
    for p, idx, y in zip(pred, dates, wys):
        s = pd.Series(p, index=idx)
        parts.append(s[(wy_of(idx) == y)])
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def score(obs_eod, sim, wys_ok):
    """B's protocol: swe_metrics on obs SWE(D+1) restricted to the given complete water years."""
    from waterdata import metrics as M
    o = obs_eod[np.isin(wy_of(obs_eod.index), list(wys_ok))].dropna()
    return M.swe_metrics(o, sim)


def sim_metrics(Y, S, mask):
    """Per-series NSE for simulated validation: Y, S (n, T) arrays, mask (n, T) bool."""
    out = []
    for y, s, m in zip(Y, S, mask):
        y, s = y[m], s[m]
        v = np.var(y)
        out.append(1 - np.mean((s - y) ** 2) / v if v > 1.0 else np.nan)
    return np.array(out)
