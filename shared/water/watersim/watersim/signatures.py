"""Hydrological signatures computed from a streamflow series the same way for simulated and real basins.

baseflow filters: Lyne & Hollick (1979) three-pass filter as specified by Ladson et al. (2013) (alpha = 0.925,
reflected 30-day padding); Eckhardt (2005) two-parameter filter (a = 0.98, BFImax = 0.8 for perennial porous aquifers).
recession: Brutsaert & Nieber (1977) -dQ/dt = a Q^b on recession days (>= 5 consecutive declining days, the first
2 days after the peak dropped; Vogel & Kroll 1992 style), fitted by least squares in log space; plus a daily
recession constant k = median(Q[t+1]/Q[t]) on the same days.
CAMELS signatures (Addor et al. 2018, Table 3 definitions): q_mean, runoff_ratio, stream_elas, slope_fdc,
baseflow_index, hfd_mean, q5, q95, high_q_freq/dur, low_q_freq/dur, zero_q_freq.
"""
import numpy as np
import pandas as pd


def lyne_hollick(q, alpha=0.925, passes=3, pad=30):
    q = np.asarray(q, float)
    n = len(q)
    if n < 3:
        return q.copy()
    p = min(pad, n - 1)
    x = np.concatenate([q[p:0:-1], q, q[-2:-p - 2:-1]])
    b = x.copy()
    for k in range(passes):
        y = b[::-1] if k % 2 == 1 else b
        qf = np.zeros_like(y)
        out = np.empty_like(y)
        qf[0] = y[0]
        out[0] = y[0]
        for t in range(1, len(y)):
            qf[t] = alpha * qf[t - 1] + (1 + alpha) / 2 * (y[t] - y[t - 1])
            if qf[t] < 0:
                qf[t] = 0.0
            out[t] = y[t] - qf[t]
            if out[t] > y[t]:
                out[t] = y[t]
            if out[t] < 0:
                out[t] = 0.0
        b = out[::-1] if k % 2 == 1 else out
    return b[p:p + n]


def eckhardt(q, a=0.98, bfimax=0.8):
    q = np.asarray(q, float)
    b = np.empty_like(q)
    b[0] = bfimax * q[0]
    for t in range(1, len(q)):
        b[t] = ((1 - bfimax) * a * b[t - 1] + (1 - a) * bfimax * q[t]) / (1 - a * bfimax)
        if b[t] > q[t]:
            b[t] = q[t]
    return b


def bfi(q, method='lyne_hollick'):
    q = np.asarray(q, float)
    m = np.isfinite(q)
    if m.sum() < 30:
        return np.nan
    qq = pd.Series(q).interpolate(limit_direction='both').values
    b = lyne_hollick(qq) if method == 'lyne_hollick' else eckhardt(qq)
    return float(b[m].sum() / qq[m].sum()) if qq[m].sum() > 0 else np.nan


def recession(q, min_len=5, drop=2, min_q=1e-4):
    """Brutsaert-Nieber recession fit. Returns {'bn_a', 'bn_b', 'k_daily', 'n_points', 'n_segments'}."""
    q = np.asarray(q, float)
    dq = np.diff(q)
    dec = (dq < 0) & np.isfinite(dq) & (q[1:] > min_q) & (q[:-1] > min_q)
    xs, ys, ks = [], [], []
    nseg = 0
    i = 0
    n = len(dec)
    while i < n:
        if dec[i]:
            j = i
            while j < n and dec[j]:
                j += 1
            if j - i >= min_len:
                nseg += 1
                for t in range(i + drop, j):
                    xs.append(0.5 * (q[t] + q[t + 1]))
                    ys.append(-(q[t + 1] - q[t]))
                    ks.append(q[t + 1] / q[t])
            i = j
        else:
            i += 1
    if len(xs) < 10:
        return {'bn_a': np.nan, 'bn_b': np.nan, 'k_daily': np.nan, 'n_points': len(xs), 'n_segments': nseg}
    lx, ly = np.log(xs), np.log(ys)
    bb, la = np.polyfit(lx, ly, 1)
    return {'bn_a': float(np.exp(la)), 'bn_b': float(bb), 'k_daily': float(np.median(ks)), 'n_points': len(xs),
            'n_segments': nseg}


def _runs(mask):
    """Mean length of True runs."""
    lens, c = [], 0
    for v in mask:
        if v:
            c += 1
        elif c:
            lens.append(c)
            c = 0
    if c:
        lens.append(c)
    return float(np.mean(lens)) if lens else 0.0


def camels_signatures(q, p=None, index=None):
    """CAMELS-style signatures of a daily streamflow series q (mm/day); p (mm/day) for runoff ratio and elasticity."""
    q = np.asarray(q, float)
    m = np.isfinite(q)
    out = {}
    qm = np.nanmean(q)
    out['q_mean'] = float(qm)
    if p is not None:
        p = np.asarray(p, float)
        mm = m & np.isfinite(p)
        out['runoff_ratio'] = float(q[mm].sum() / p[mm].sum()) if p[mm].sum() > 0 else np.nan
        if index is not None and mm.sum() > 730:
            s = pd.DataFrame({'q': q, 'p': p}, index=index)
            wy = s.index.year + (s.index.month >= 10)
            a = s.groupby(wy).agg(['mean', 'count'])
            a = a[(a[('q', 'count')] > 330)]
            dq = a[('q', 'mean')] - a[('q', 'mean')].mean()
            dp = a[('p', 'mean')] - a[('p', 'mean')].mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                e = (dq / dp) * (a[('p', 'mean')].mean() / a[('q', 'mean')].mean())
            out['stream_elas'] = float(np.nanmedian(e.replace([np.inf, -np.inf], np.nan))) if len(e) else np.nan
    qs = q[m]
    if len(qs) == 0:
        return out
    q33, q66 = np.percentile(qs, [33, 66])
    out['slope_fdc'] = float((np.log(q66 + 1e-6) - np.log(q33 + 1e-6)) / (0.66 - 0.33)) if q33 > 0 else np.nan
    out['baseflow_index'] = bfi(q)
    out['q5'] = float(np.percentile(qs, 5))
    out['q95'] = float(np.percentile(qs, 95))
    med = np.median(qs)
    nyears = len(qs) / 365.25
    hi = q > 9 * med
    lo = q < 0.2 * qm
    out['high_q_freq'] = float(np.sum(hi[m]) / nyears)
    out['high_q_dur'] = _runs(hi & m)
    out['low_q_freq'] = float(np.sum(lo[m]) / nyears)
    out['low_q_dur'] = _runs(lo & m)
    out['zero_q_freq'] = float(np.mean(qs == 0))
    if index is not None:
        s = pd.Series(q, index=index)
        wy = s.index.year + (s.index.month >= 10)
        hfd = []
        for y, g in s.groupby(wy):
            if g.notna().sum() < 330:
                continue
            c = g.fillna(0).cumsum()
            if c.iloc[-1] <= 0:
                continue
            k = int(np.searchsorted(c.values, c.iloc[-1] / 2))
            hfd.append(k)
        out['hfd_mean'] = float(np.mean(hfd)) if hfd else np.nan
    return out


def richards_baker(q):
    q = np.asarray(q, float)
    q = q[np.isfinite(q)]
    return float(np.sum(np.abs(np.diff(q))) / np.sum(q[1:])) if q.sum() > 0 else np.nan
