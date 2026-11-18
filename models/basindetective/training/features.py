"""BasinDetective features and signatures.

sigs()       exact, fast (numba) port of waterdata.signatures.signatures (B's classical signature code), so the
             simulated labels and the real truth are computed the same way. verify_sigs() checks the port.
featurize()  one feature vector per basin from: calibration-period forcing (P, T_mean, PET) + calibration-period Q,
             and target-period forcing (no Q). Everything a gauge record and a forcing file can give.
"""
import common  # noqa: F401  (threads, numba cache)
import numpy as np
import pandas as pd
import numba as nb

from common import EPSQ


@nb.njit(cache=True)
def _lh(q, alpha, passes):
    b = q.copy()
    n = q.size
    for p in range(passes):
        if p % 2 == 0:
            x = b.copy()
        else:
            x = b[::-1].copy()
        out = np.empty(n)
        qf = 0.0
        out[0] = x[0]
        for t in range(1, n):
            qf = alpha * qf + (1 + alpha) / 2 * (x[t] - x[t - 1])
            if qf < 0.0:
                qf = 0.0
            out[t] = x[t] - min(qf, x[t])
        if p % 2 == 0:
            b = out
        else:
            b = out[::-1].copy()
    return b


@nb.njit(cache=True)
def _eck(q, a, bfimax):
    n = q.size
    b = np.empty(n)
    b[0] = q[0] * bfimax
    for t in range(1, n):
        b[t] = min(q[t], ((1 - bfimax) * a * b[t - 1] + (1 - a) * bfimax * q[t]) / (1 - a * bfimax))
    return b


@nb.njit(cache=True)
def _rec_pairs(v, min_len, skip):
    n = v.size - 1
    dq = v[1:] - v[:-1]
    qm = 0.5 * (v[1:] + v[:-1])
    dec = np.zeros(n, np.bool_)
    for i in range(n):
        dec[i] = (dq[i] < 0) and np.isfinite(dq[i]) and (qm[i] > 0)
    xs = np.empty(n)
    ys = np.empty(n)
    m = 0
    i = 0
    while i < n:
        if dec[i]:
            j = i
            while j < n and dec[j]:
                j += 1
            if j - i >= min_len:
                for k in range(i + skip, j):
                    xs[m] = qm[k]
                    ys[m] = -dq[k]
                    m += 1
            i = j
        else:
            i += 1
    return xs[:m], ys[:m]


def _interp3(q):
    """pandas interpolate(limit=3, limit_area='inside') then dropna, as in waterdata.signatures.bfi."""
    s = pd.Series(q)
    if not np.isnan(q).any():
        return q
    return s.interpolate(limit=3, limit_area='inside').dropna().values


def bfi_fast(q, method='lh'):
    s = _interp3(np.asarray(q, float))
    if s.size < 365:
        return np.nan
    b = _lh(s, 0.925, 3) if method == 'lh' else _eck(s, 0.98, 0.8)
    tot = s.sum()
    return float(b.sum() / tot) if tot > 0 else np.nan


def recession_fast(v):
    x, y = _rec_pairs(np.asarray(v, float), 5, 2)
    m = (x > 0) & (y > 0)
    if x.size == 0 or m.sum() < 20:
        return np.nan, np.nan, np.nan
    bb, la = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(np.exp(la)), float(bb), float(np.median(x[m] / y[m]))


def water_year(idx):
    return np.asarray(idx.year + (idx.month >= 10).astype(int))


def sigs(q, P=None, PET=None, T=None):
    """Same outputs as waterdata.signatures.signatures (q: pd.Series with DatetimeIndex, NaN = missing)."""
    q = pd.Series(q).astype(float)
    ok = np.isfinite(q.values)
    qq = q.values[ok]
    idx = q.index[ok]
    out = {}
    if qq.size < 365:
        return out
    out['q_mean'] = float(qq.mean())
    out['q5'] = float(np.quantile(qq, 0.05))
    out['q95'] = float(np.quantile(qq, 0.95))
    med = np.median(qq)
    out['high_q_freq'] = float((qq > 9 * med).mean() * 365.25)
    out['low_q_freq'] = float((qq < 0.2 * qq.mean()).mean() * 365.25)
    out['zero_q_freq'] = float((qq <= 0).mean())
    q33 = np.log(max(np.quantile(qq, 0.67), 1e-6))
    q66 = np.log(max(np.quantile(qq, 0.34), 1e-6))
    out['slope_fdc'] = float((q66 - q33) / (0.66 - 0.33))
    out['bfi_lh'] = bfi_fast(q.values, 'lh')
    out['bfi_eckhardt'] = bfi_fast(q.values, 'eck')
    a, b, k = recession_fast(q.values)
    out['rec_a'] = a
    out['rec_b'] = b
    out['rec_k_days'] = k
    wy = water_year(idx)
    hfd = []
    full = []
    for y in np.unique(wy):
        g = qq[wy == y]
        if g.size >= 330:
            c = np.cumsum(g)
            hfd.append(int(np.searchsorted(c, c[-1] / 2)))
            full.append(y)
    out['hfd_mean'] = float(np.mean(hfd)) if hfd else np.nan
    if P is not None:
        Pv = pd.Series(P).reindex(idx).values
        ps = np.nansum(Pv)
        out['runoff_ratio'] = float(qq.sum() / ps) if ps > 0 else np.nan
        if len(full) >= 3:
            aq = np.array([qq[wy == y].sum() for y in full])
            ap = np.array([np.nansum(Pv[wy == y]) for y in full])
            if ap.std() > 0:
                dq = aq - aq.mean()
                dp = ap - ap.mean()
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = np.where(np.abs(dp) > 1e-9, dq / dp, np.nan)
                out['elasticity_p'] = float(np.nanmedian(r) * ap.mean() / aq.mean())
        if PET is not None:
            PETv = pd.Series(PET).reindex(idx).values
            out['aridity'] = float(np.nansum(PETv) / ps)
        if T is not None:
            Tv = pd.Series(T).reindex(idx).values
            out['frac_snow'] = float(np.nansum(Pv[Tv < 0]) / ps)
    return out


# ----------------------------------------------------------------------------------------------- features
FDC_P = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]   # exceedance probabilities


def _climate(df):
    P = df['P'].values.astype(float)
    T = df['T_mean'].values.astype(float)
    E = df['PET'].values.astype(float)
    pm = max(P.mean(), 1e-3)
    em = max(E.mean(), 1e-3)
    mon = df.index.month.values
    Pm = np.array([P[mon == m].mean() for m in range(1, 13)]) / pm
    Em = np.array([E[mon == m].mean() for m in range(1, 13)]) / em
    Tm = np.array([T[mon == m].mean() for m in range(1, 13)])
    wy = water_year(df.index)
    ys = np.unique(wy)
    ap = np.array([P[wy == y].sum() for y in ys])
    c = {
        'logP': np.log(pm), 'logPET': np.log(em), 'arid': em / pm, 'Tmean': T.mean(),
        'fsnow': P[T < 0].sum() / max(P.sum(), 1e-9),
        'fwarm': P[T > 10].sum() / max(P.sum(), 1e-9),
        'wetfrac': (P > 1.0).mean(), 'p99rel': np.quantile(P, 0.99) / pm,
        'pseas': float(np.corrcoef(Pm, Em)[0, 1]) if Pm.std() > 0 and Em.std() > 0 else 0.0,
        'pcv': ap.std() / max(ap.mean(), 1e-9) if ys.size > 1 else 0.0,
    }
    return c, Pm, Em, Tm


def featurize(cal, qcal, tgt):
    """cal, tgt: DataFrames with P, T_mean, PET (daily DatetimeIndex). qcal: Series on cal's index (NaN = missing).
    Returns (feature vector float32, dict of calibration signatures, dict of feature names->values)."""
    f = {}
    cc, Pm, Em, Tm = _climate(cal)
    tc, tPm, tEm, tTm = _climate(tgt)
    for k, v in cc.items():
        f[f'c_{k}'] = v
    for k, v in tc.items():
        f[f't_{k}'] = v
        if k not in ('wetfrac',):
            f[f'dt_{k}'] = v - cc[k]
    f['dt_wetfrac'] = tc['wetfrac'] - cc['wetfrac']
    for m in range(12):
        f[f'c_Pm{m}'] = Pm[m]
        f[f'c_Em{m}'] = Em[m]
        f[f'c_Tm{m}'] = Tm[m]
        f[f'dt_Pm{m}'] = tPm[m] * np.exp(tc['logP'] - cc['logP']) - Pm[m]
        f[f'dt_Tm{m}'] = tTm[m] - Tm[m]
    q = pd.Series(np.asarray(qcal, float), index=cal.index)
    s = sigs(q, cal['P'], cal['PET'], cal['T_mean'])
    qv = q.values
    ok = np.isfinite(qv)
    qq = qv[ok]
    qm = max(qq.mean(), 1e-4)
    f['q_missing'] = 1 - ok.mean()
    f['q_logmean'] = np.log(qm + EPSQ)
    f['q_rr'] = s.get('runoff_ratio', np.nan)
    f['q_logrr'] = np.log(max(s.get('runoff_ratio', np.nan), 1e-3))
    f['q_bfi_lh'] = s['bfi_lh']
    f['q_bfi_eck'] = s['bfi_eckhardt']
    f['q_logreck'] = np.log(np.clip(s['rec_k_days'], 0.3, 3000)) if np.isfinite(s['rec_k_days']) else np.nan
    f['q_recb'] = np.clip(s['rec_b'], -1, 5) if np.isfinite(s['rec_b']) else np.nan
    f['q_slopefdc'] = np.clip(s['slope_fdc'], -1, 30)
    f['q_hfd'] = s['hfd_mean'] / 365.0
    f['q_logq5'] = np.log(s['q5'] + EPSQ)
    f['q_logq95'] = np.log(s['q95'] + EPSQ)
    f['q_highfreq'] = s['high_q_freq'] / 365.25
    f['q_lowfreq'] = s['low_q_freq'] / 365.25
    f['q_zerofreq'] = s['zero_q_freq']
    f['q_elast'] = np.clip(s.get('elasticity_p', np.nan), -2, 6)
    lqv = np.log(qq + EPSQ * 0.1 + 1e-4)
    for p in FDC_P:
        f[f'q_fdc{int(p * 100):02d}'] = np.log(np.quantile(qq, 1 - p) + 1e-4) - np.log(qm)
    mon = cal.index.month.values[ok]
    for m in range(12):
        sel = mon == m + 1
        f[f'q_m{m}'] = qq[sel].mean() / qm if sel.any() else np.nan
    # day-to-day dynamics (on gap-filled series for correlations)
    qf = pd.Series(qv).interpolate(limit_area='inside').bfill().ffill().values
    lqf = np.log(qf + 1e-3)
    P = cal['P'].values

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    for k in (0, 1, 2, 3, 5, 10, 30):
        f[f'q_pq{k}'] = corr(P[:P.size - k], qf[k:])
    for k in (1, 3, 10, 30, 90):
        f[f'q_ac{k}'] = corr(lqf[:-k], lqf[k:])
    dq = np.diff(qf)
    f['q_flash'] = np.abs(dq).sum() / max(qf.sum(), 1e-9)
    f['q_flat'] = (np.abs(dq) < 0.01 * qf[1:] + 1e-6).mean()
    f['q_rise'] = (dq > 0).mean()
    # 7-day minimum flow relative to mean; std of weekly flow vs weekly input
    q7 = pd.Series(qf).rolling(7).mean().values[6:]
    p7 = pd.Series(P).rolling(7).mean().values[6:]
    f['q_min7'] = np.log(q7.min() + 1e-3) - np.log(qm)
    f['q_cv7_over_p'] = (q7.std() / qm) / max(p7.std() / max(P.mean(), 1e-3), 1e-3)
    wy = water_year(cal.index)
    ys = np.unique(wy)
    aq = np.array([np.nanmean(qv[wy == y]) for y in ys])
    ap = np.array([P[wy == y].mean() for y in ys])
    f['q_anncv'] = np.nanstd(aq) / qm
    f['q_annpcorr'] = corr(np.nan_to_num(aq, nan=np.nanmean(aq)), ap) if ys.size > 2 else 0.0
    # seasonal-cycle damping: amplitude of monthly Q regime vs monthly (P - PET)+ regime
    qmon = np.array([f[f'q_m{m}'] for m in range(12)])
    wmon = np.maximum(Pm * np.exp(cc['logP']) - Em * np.exp(cc['logPET']) * 0.5, 0) + 1e-3
    f['q_seasamp'] = np.nanstd(qmon)
    f['q_seasdamp'] = np.nanstd(qmon) / max(np.std(wmon / wmon.mean()), 1e-3)
    names = list(f.keys())
    x = np.array([f[n] for n in names], np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    return x.astype(np.float32), s, names


def target_sigs(q, P, PET, T):
    return sigs(q, P, PET, T)


def delta_labels(s_cal, s_tgt):
    """Out-of-period change targets from two signature dicts (target minus calibration, transformed)."""
    g = lambda d, k: d.get(k, np.nan)
    return {
        'd_logq': np.log(g(s_tgt, 'q_mean') + EPSQ) - np.log(g(s_cal, 'q_mean') + EPSQ),
        'd_rr': g(s_tgt, 'runoff_ratio') - g(s_cal, 'runoff_ratio'),
        'd_bfi': g(s_tgt, 'bfi_lh') - g(s_cal, 'bfi_lh'),
        'd_logq5': np.log(g(s_tgt, 'q5') + EPSQ) - np.log(g(s_cal, 'q5') + EPSQ),
        'd_logq95': np.log(g(s_tgt, 'q95') + EPSQ) - np.log(g(s_cal, 'q95') + EPSQ),
        'd_hfd': g(s_tgt, 'hfd_mean') - g(s_cal, 'hfd_mean'),
    }


def verify_sigs(n=8):
    """Compare sigs() with B's waterdata.signatures.signatures on real dev basins (cal period)."""
    import waterdata as wd
    from waterdata.signatures import signatures
    from common import CAL
    worst = 0.0
    for b in wd.basins('camels', 'dev')[:n]:
        f = wd.forcing(b, *CAL)
        q = wd.streamflow(b, 'caravan').Q_mm.loc[CAL[0]:CAL[1]]
        A = signatures(q, f.P, f.PET, f.T_mean)
        B = sigs(q, f.P, f.PET, f.T_mean)
        for k in A:
            a, bb = A[k], B[k]
            if np.isnan(a) and np.isnan(bb):
                continue
            worst = max(worst, abs(a - bb) / max(abs(a), 1e-9))
    return worst


if __name__ == '__main__':
    import time
    t = time.time()
    print('max relative difference vs waterdata.signatures:', verify_sigs(52), f'{time.time() - t:.1f}s')
