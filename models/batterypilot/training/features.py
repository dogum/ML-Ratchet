"""Causal features for BatteryPilot (numpy; shipped in the release).

At decision hour t (the offer for hour t is submitted before it starts) the policy may use:
- real-time prices of hours < t (settled), the day-ahead prices of today and, once they clear (14:00 local), of
  tomorrow; the BA's day-ahead demand forecast for the next 24 h; the last observed temperature; the calendar;
- the battery (duration, degradation cost).  SOC enters through the value curve, not the features.
Prices are scaled by ps = median day-ahead price of the past 7 days (>= $5), and squashed with asinh.
"""
import numpy as np

H_DA = 36      # day-ahead horizon read (hours from t)
H_RT = 24      # real-time history (hours before t)
H_DF = 24      # demand forecast horizon
PUBLISH_HOUR = 14
N_FEAT = 2 * H_DA + H_RT + 6 + H_DF + 1 + 5 + 3


def zq(p, ps):
    return np.arcsinh(p / ps)


def price_scale(da, t_idx, hist=168):
    """Median DA over the previous `hist` hours (causal), floor $5."""
    da = np.asarray(da, float)
    out = np.empty(len(t_idx))
    for i, t in enumerate(t_idx):
        w = da[max(0, t - hist):t]
        out[i] = np.nanmedian(w) if len(w) and np.isfinite(w).any() else (da[t] if np.isfinite(da[t]) else 25.0)
    return np.maximum(out, 5.0)


def rolling_scale(da, hist=168):
    """Causal rolling median of DA for every hour (fast via daily blocks)."""
    da = np.asarray(da, float)
    n = len(da)
    out = np.empty(n)
    # recompute once per 24 h (exact median of the trailing week at each day boundary), hold within the day
    last = 25.0
    for t in range(n):
        if t % 24 == 0:
            w = da[max(0, t - hist):t]
            if len(w) >= 24 and np.isfinite(w).any():
                last = float(np.nanmedian(w))
            elif np.isfinite(da[t:t + 24]).any():
                last = float(np.nanmedian(da[t:t + 24]))
        out[t] = last
    return np.maximum(out, 5.0)


def da_known_mask(hour, day, t, offsets):
    """1 where the DA price of hour t+o is known at decision time t (same local day, or the next day after the
    14:00 publication)."""
    tgt = t[:, None] + offsets[None, :]
    tgt = np.clip(tgt, 0, len(day) - 1)
    dd = day[tgt] - day[t][:, None]
    known = (dd <= 0) | ((dd == 1) & (hour[t][:, None] >= PUBLISH_HOUR))
    return known


def build(path, t_idx, duration_h, c_lin, ps=None):
    """Feature matrix (len(t_idx), N_FEAT) for decision hours t_idx of a path dict/DataFrame with arrays
    rt, da, df, T, hour, day, dow, doy.  Hours outside the path are padded with edge values."""
    rt = np.asarray(path['rt'], float); da = np.asarray(path['da'], float)
    df = np.asarray(path['df'], float); T = np.asarray(path['T'], float)
    hour = np.asarray(path['hour']); day = np.asarray(path['day'])
    dow = np.asarray(path['dow']); doy = np.asarray(path['doy'])
    n = len(rt)
    t = np.asarray(t_idx)
    if ps is None:
        ps = rolling_scale(da)[t]
    ps = np.asarray(ps, float)
    if np.ndim(ps) == 0:
        ps = np.full(len(t), float(ps))
    cl = np.broadcast_to(np.asarray(c_lin, float), (len(t),)) if np.ndim(c_lin) == 0 else np.asarray(c_lin)[t] \
        if np.asarray(c_lin).shape == (n,) else np.asarray(c_lin, float)
    o_da = np.arange(H_DA)
    idx = np.clip(t[:, None] + o_da[None, :], 0, n - 1)
    known = da_known_mask(hour, day, t, o_da)
    zda = np.where(known, zq(da[idx], ps[:, None]), 0.0)
    o_rt = np.arange(-H_RT, 0)
    ridx = np.clip(t[:, None] + o_rt[None, :], 0, n - 1)
    seen = ridx < t[:, None]       # t = 0 only: the clipped index is the current hour (see batterypilot.build)
    zrt = np.where(seen, zq(rt[ridx], ps[:, None]), zq(da[ridx], ps[:, None]))
    o6 = np.arange(-6, 0)
    i6 = np.clip(t[:, None] + o6[None, :], 0, n - 1)
    sp6 = np.where(i6 < t[:, None], zq(rt[i6], ps[:, None]) - zq(da[i6], ps[:, None]), 0.0)
    # demand forecast relative to the trailing 30-day max of the forecast (tightness proxy)
    dfi = np.clip(t[:, None] + np.arange(H_DF)[None, :], 0, n - 1)
    if 'dmax' in path:
        dmax = np.asarray(path['dmax'], float)[t]
    else:
        dmax = _rollmax(df, 720)[t]
    dfn = df[dfi] / np.maximum(dmax[:, None], 1.0) - 0.8
    Tl = (T[np.clip(t - 1, 0, n - 1)] - 20.0) / 10.0
    hh = 2 * np.pi * hour[t] / 24.0; dy = 2 * np.pi * doy[t] / 365.25
    cal = np.c_[np.sin(hh), np.cos(hh), np.sin(dy), np.cos(dy), (dow[t] >= 5).astype(float)]
    bat = np.c_[np.log2(np.full(len(t), duration_h, float)), cl / ps, np.log(ps / 30.0)]
    X = np.concatenate([zda, known.astype(float), zrt, sp6, dfn, Tl[:, None], cal, bat], 1)
    return np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32), ps


def _rollmax(x, w):
    """Trailing max over the last w+1 values (inclusive), numpy only."""
    x = np.asarray(x, float)
    xp = np.r_[np.full(w, -np.inf), x]
    return np.lib.stride_tricks.sliding_window_view(xp, w + 1).max(1)
