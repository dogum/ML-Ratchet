"""Hydrological signatures (classical 'detective' estimators). numpy/pandas only.

baseflow_lh        Lyne & Hollick (1979) digital filter, alpha=0.925, 3 passes (forward, backward, forward) (Ladson et al. 2013)
baseflow_eckhardt  Eckhardt (2005) two-parameter filter, a=0.98, BFImax=0.80 (perennial porous aquifer default)
recession          Brutsaert & Nieber (1977): fit -dQ/dt = a Q^b on recession days (>= 5 days of decline, first 2 days
                   after each peak dropped, dQ/dt = Q(t+1)-Q(t), Q = mean of the two days); OLS in log space.
                   Also K_lin (days): median Q/(-dQ/dt), the linear-reservoir time constant.
budyko_fu          Fu (1981) curve E/P = 1 + phi - (1 + phi^w)^(1/w), w = 2.6 (Zhang et al. 2004 global value).
signatures         dict of all signatures for one basin and period.
"""
import numpy as np
import pandas as pd


def baseflow_lh(q, alpha=0.925, passes=3):
    q = np.asarray(q, float)
    b = q.copy()
    for p in range(passes):
        x = b if p % 2 == 0 else b[::-1]
        out = np.empty_like(x); qf = 0.0; out[0] = x[0]
        for t in range(1, x.size):
            qf = alpha * qf + (1 + alpha) / 2 * (x[t] - x[t - 1])
            qf = max(qf, 0.0)
            out[t] = x[t] - min(qf, x[t])
        b = out if p % 2 == 0 else out[::-1]
    return b


def baseflow_eckhardt(q, a=0.98, bfimax=0.8):
    q = np.asarray(q, float)
    b = np.empty_like(q); b[0] = q[0] * bfimax
    for t in range(1, q.size):
        b[t] = min(q[t], ((1 - bfimax) * a * b[t - 1] + (1 - a) * bfimax * q[t]) / (1 - a * bfimax))
    return b


def _contiguous(q):
    """Longest-gap-free handling: fill short gaps (<=3 d) linearly, return array and mask of filled-valid."""
    s = pd.Series(q).interpolate(limit=3, limit_area='inside')
    return s


def bfi(q, method='lh'):
    s = _contiguous(q).dropna()
    if len(s) < 365:
        return np.nan
    b = baseflow_lh(s.values) if method == 'lh' else baseflow_eckhardt(s.values)
    return float(b.sum() / s.values.sum()) if s.values.sum() > 0 else np.nan


def recession(q, min_len=5, skip=2):
    s = pd.Series(q)
    v = s.values
    dq = v[1:] - v[:-1]; qm = 0.5 * (v[1:] + v[:-1])
    dec = (dq < 0) & np.isfinite(dq) & (qm > 0)
    xs, ys = [], []
    i = 0
    n = dec.size
    while i < n:
        if dec[i]:
            j = i
            while j < n and dec[j]:
                j += 1
            if j - i >= min_len:
                xs.append(qm[i + skip:j]); ys.append(-dq[i + skip:j])
            i = j
        else:
            i += 1
    if not xs:
        return np.nan, np.nan, np.nan
    x = np.concatenate(xs); y = np.concatenate(ys)
    m = (x > 0) & (y > 0)
    if m.sum() < 20:
        return np.nan, np.nan, np.nan
    bb, la = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(np.exp(la)), float(bb), float(np.median(x[m] / y[m]))


def budyko_fu(aridity, w=2.6):
    phi = np.asarray(aridity, float)
    return 1 + phi - (1 + phi ** w) ** (1 / w)


def signatures(q, P=None, PET=None, T=None):
    q = pd.Series(q).astype(float)
    qq = q.dropna()
    out = {}
    if len(qq) < 365:
        return out
    out['q_mean'] = float(qq.mean())
    out['q5'] = float(qq.quantile(0.05)); out['q95'] = float(qq.quantile(0.95))
    med = qq.median()
    out['high_q_freq'] = float((qq > 9 * med).mean() * 365.25)
    out['low_q_freq'] = float((qq < 0.2 * qq.mean()).mean() * 365.25)
    out['zero_q_freq'] = float((qq <= 0).mean())
    q33, q66 = np.log(max(qq.quantile(0.67), 1e-6)), np.log(max(qq.quantile(0.34), 1e-6))
    out['slope_fdc'] = float((q66 - q33) / (0.66 - 0.33))
    out['bfi_lh'] = bfi(q, 'lh'); out['bfi_eckhardt'] = bfi(q, 'eck')
    a, b, k = recession(q.values)
    out['rec_a'] = a; out['rec_b'] = b; out['rec_k_days'] = k
    wy = qq.index.year + (qq.index.month >= 10).astype(int)
    hfd = []
    for y, g in qq.groupby(wy):
        if len(g) >= 330:
            c = g.cumsum(); hfd.append(int(np.searchsorted(c.values, c.values[-1] / 2)))
    out['hfd_mean'] = float(np.mean(hfd)) if hfd else np.nan
    if P is not None:
        P = pd.Series(P).reindex(qq.index)
        out['runoff_ratio'] = float(qq.sum() / P.sum()) if P.sum() > 0 else np.nan
        ann = pd.DataFrame({'q': qq, 'p': P}).groupby(wy).sum()
        ann = ann[ann.index.isin([y for y, g in qq.groupby(wy) if len(g) >= 330])]
        if len(ann) >= 3 and ann.p.std() > 0:
            dq = ann.q - ann.q.mean(); dp = ann.p - ann.p.mean()
            out['elasticity_p'] = float(np.median(dq / dp.where(dp.abs() > 1e-9)) * ann.p.mean() / ann.q.mean())
        if PET is not None:
            PET = pd.Series(PET).reindex(qq.index)
            out['aridity'] = float(PET.sum() / P.sum())
        if T is not None:
            T = pd.Series(T).reindex(qq.index)
            out['frac_snow'] = float(P[T < 0].sum() / P.sum())
    return out
