"""Features for the three input granularities (hourly, monthly, annual) and the meter-quirk emulator.

Every feature is computed the same way for simulated and real buildings. Inputs:
  meters: dict channel -> hourly kW array (NaN = missing); channels C.CHANNELS (elec, fuel, dheat, dcool)
  T, GHI: hourly outdoor temperature (degC) and global horizontal irradiance (W/m2) on the same hours
  loc: local-time DatetimeIndex of those hours (hour-beginning)
Scale-free shape features use each series divided by its own mean; magnitude enters only through log10(mean kW).
"""
import numpy as np

import common as C

TB = np.arange(-21, 37, 3.0)          # daily-temperature bin edges (19 bins)
NB = len(TB) - 1
TAUS = [0, 12, 24, 48, 96, 192]       # EWMA time constants (h) for the lag/mass features
HB = 18.33                            # degree-day base (65 F, as RECS/CBECS)


# ----------------------------------------------------------------------------- helpers
def _ewma(x, tau):
    if tau == 0:
        return x.copy()
    a = 1.0 - np.exp(-1.0 / tau)
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):          # numba-free; 8760 steps ~ 3 ms
        acc = acc + a * (x[i] - acc)
        y[i] = acc
    return y


def _ewma_fast(x, tau):
    if tau == 0:
        return x.copy()
    from scipy.signal import lfilter
    a = 1.0 - np.exp(-1.0 / tau)
    y, _ = lfilter([a], [1, -(1 - a)], x, zi=[x[0] * (1 - a)])
    return y


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return 0.0
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _fill_bins(v):
    """Fill NaN bins by linear interpolation / edge hold; all-NaN -> 0."""
    m = np.isfinite(v)
    if not m.any():
        return np.zeros_like(v)
    x = np.arange(len(v))
    return np.interp(x, x[m], v[m])


def prism(Td, Ed, w=None):
    """5-parameter change-point fit E = b + h*max(Tbh-T,0) + c*max(T-Tbc,0), h,c >= 0 (grid over balance points).
    Returns base, h, c, Tbh, Tbc, r2."""
    m = np.isfinite(Td) & np.isfinite(Ed)
    Td, Ed = Td[m], Ed[m]
    if len(Td) < 8 or np.std(Ed) == 0:
        return np.array([np.nanmean(Ed) if len(Ed) else 0, 0, 0, 15.0, 20.0, 0.0])
    best = None
    sst = ((Ed - Ed.mean()) ** 2).sum() + 1e-12
    for tbh in np.arange(4, 23, 1.5):
        xh = np.maximum(tbh - Td, 0)
        for tbc in np.arange(max(tbh, 10), 29, 1.5):
            xc = np.maximum(Td - tbc, 0)
            X = np.c_[np.ones_like(Td), xh, xc]
            coef, *_ = np.linalg.lstsq(X, Ed, rcond=None)
            if coef[1] < 0 or coef[2] < 0:
                cols = [0] + [k for k in (1, 2) if coef[k] >= 0]
                c2, *_ = np.linalg.lstsq(X[:, cols], Ed, rcond=None)
                coef = np.zeros(3)
                coef[cols] = c2
                coef = np.maximum(coef, [-np.inf, 0, 0])
            sse = ((X @ coef - Ed) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, coef, tbh, tbc)
    sse, coef, tbh, tbc = best
    return np.array([coef[0], coef[1], coef[2], tbh if coef[1] > 0 else 15.0, tbc if coef[2] > 0 else 20.0,
                     1 - sse / sst])


# ----------------------------------------------------------------------------- hourly
N_W_H = 8 + NB
N_C_H = 3 + 2 * NB + 168 + 48 + 6 + 12 + 10 + 8


def daily_T(T, loc):
    import pandas as pd
    day = (loc.normalize() if hasattr(loc, 'normalize') else loc)
    return pd.Series(T).groupby(np.asarray(day)).mean()


def feat_hourly(meters, T, GHI, loc, lat):
    T = np.asarray(T, float)
    GHI = np.asarray(GHI, float)
    dow = np.asarray(loc.dayofweek)
    hr = np.asarray(loc.hour)
    dkey = np.asarray(loc.normalize().asi8)
    ud, dinv = np.unique(dkey, return_inverse=True)
    nd = len(ud)
    cnt = np.bincount(dinv, minlength=nd)
    Td = np.bincount(dinv, weights=np.nan_to_num(T), minlength=nd) / np.maximum(cnt, 1)
    dd_wd = np.bincount(dinv, weights=(dow < 5).astype(float), minlength=nd) / np.maximum(cnt, 1) > 0.5
    span = nd / 365.0
    hdd = np.maximum(HB - Td, 0).sum() / nd
    cdd = np.maximum(Td - HB, 0).sum() / nd
    occ = np.histogram(np.clip(Td, TB[0] + 1e-6, TB[-1] - 1e-6), TB)[0] / nd
    w = [np.nanmean(T) / 10, np.nanstd(T) / 10, hdd / 10, cdd / 10, np.nanmean(GHI) / 200, lat / 90, span,
         np.log10(max(nd, 1)) / 3]
    feats = [np.array(w, float), occ]
    # EWMA of hourly T -> daily, for the lag (thermal mass) features
    Tf = np.nan_to_num(T, nan=np.nanmean(T))
    Tew = [np.bincount(dinv, weights=_ewma_fast(Tf, tau), minlength=nd) / np.maximum(cnt, 1) for tau in TAUS]
    how = dow * 24 + hr
    for ch in C.CHANNELS:
        x = meters.get(ch)
        if x is None or np.all(~np.isfinite(x)) or np.nansum(np.abs(x)) <= 0:
            feats.append(np.zeros(N_C_H))
            continue
        x = np.asarray(x, float)
        v = np.isfinite(x)
        mu = np.nanmean(x)
        if not np.isfinite(mu) or mu <= 0:
            feats.append(np.zeros(N_C_H))
            continue
        xn = x / mu
        f = [1.0, v.mean(), np.log10(mu)]
        # daily means (days with >= 12 valid hours)
        dv = np.bincount(dinv, weights=v.astype(float), minlength=nd)
        Ed = np.bincount(dinv, weights=np.where(v, xn, 0), minlength=nd) / np.maximum(dv, 1)
        Ed[dv < 12] = np.nan
        for sel in (dd_wd, ~dd_wd):
            b = np.digitize(Td[sel], TB) - 1
            e = Ed[sel]
            ok = np.isfinite(e) & (b >= 0) & (b < NB)
            s = np.bincount(b[ok], weights=e[ok], minlength=NB)
            n = np.bincount(b[ok], minlength=NB)
            f.extend(_fill_bins(np.where(n > 0, s / np.maximum(n, 1), np.nan)))
        # hour-of-week profile
        s = np.bincount(how[v], weights=xn[v], minlength=168)
        n = np.bincount(how[v], minlength=168)
        pw = _fill_bins(np.where(n > 0, s / np.maximum(n, 1), np.nan))
        f.extend(pw)
        # weekday hourly profile on cold and hot days
        Th = Td[dinv]
        for sel in ((Th < 5) & (dow < 5) & v, (Th > 22) & (dow < 5) & v):
            s = np.bincount(hr[sel], weights=xn[sel], minlength=24)
            n = np.bincount(hr[sel], minlength=24)
            p = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            f.extend(p if np.isfinite(p).all() else pw[:24] * 0 + np.nanmean(pw[:120].reshape(5, 24), 0))
        # PRISM on daily
        pr = prism(Td, Ed)
        f.extend([pr[0], pr[1] * 10, pr[2] * 10, pr[3] / 10, pr[4] / 10, pr[5]])
        # lag / mass: correlation of daily load with EWMA-smoothed T, heating and cooling seasons
        for sel in (Td < 10, Td > 20):
            for tw in Tew:
                f.append(_corr(Ed[sel], tw[sel]) if sel.sum() > 10 else 0.0)
        # hourly lag correlations of deviations from the hour-of-week mean (winter, summer)
        dev = xn - pw[how]
        cs = np.concatenate([[0.0], np.cumsum(Tf)])             # centred 24-h mean, window [i-12, i+11]
        ii = np.arange(len(Tf))
        a0, b0 = np.maximum(ii - 12, 0), np.minimum(ii + 12, len(Tf))
        Tdev = Tf - (cs[b0] - cs[a0]) / (b0 - a0)
        for sel in (Th < 5, Th > 22):
            for lag in (0, 2, 4, 8, 16):
                a = dev[lag:]
                b2 = Tdev[:len(Tdev) - lag]
                ss = sel[lag:] & v[lag:]
                f.append(_corr(a[ss], b2[ss]) if ss.sum() > 48 else 0.0)
        # stats
        xv = xn[v]
        night = np.nanmean(np.where((hr <= 4) & (dow < 5) & v, xn, np.nan))
        day_ = np.nanmean(np.where((hr >= 10) & (hr <= 15) & (dow < 5) & v, xn, np.nan))
        wkd = np.nanmean(Ed[dd_wd]) if np.isfinite(Ed[dd_wd]).any() else 1.0
        wke = np.nanmean(Ed[~dd_wd]) if np.isfinite(Ed[~dd_wd]).any() else wkd
        f.extend([np.percentile(xv, 1), np.percentile(xv, 50), np.percentile(xv, 99) / 3,
                  night / max(day_, 1e-3) if np.isfinite(night) and np.isfinite(day_) else 1.0,
                  wke / max(wkd, 1e-3), np.nanstd(Ed), (xv <= 1e-6).mean(),
                  np.nanmean(np.abs(np.diff(xn))) if v.sum() > 2 else 0.0])
        f = np.nan_to_num(np.asarray(f, float), nan=0.0, posinf=0.0, neginf=0.0)
        feats.append(np.clip(f, -20, 20))
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------- monthly
N_W_M = 12 + 12 + 4
N_C_M = 3 + 12 + 6


def monthly_from_hourly(meters, T, loc, shift_days=0):
    """Monthly mean kW per channel (billing windows shifted by shift_days) and monthly mean T, HDD, CDD per day."""
    if shift_days:
        import pandas as pd
    t = loc - pd.Timedelta(days=shift_days) if shift_days else loc
    mon = np.asarray(t.month) - 1
    day = np.asarray(loc.normalize().asi8)
    ud, dinv = np.unique(day, return_inverse=True)
    cnt = np.bincount(dinv)
    Td = np.bincount(dinv, weights=np.nan_to_num(np.asarray(T, float), nan=np.nanmean(T))) / np.maximum(cnt, 1)
    dmon = np.bincount(dinv, weights=mon) / np.maximum(cnt, 1)
    dmon = np.round(dmon).astype(int)
    nd = np.bincount(dmon, minlength=12)
    Tm = np.bincount(dmon, weights=Td, minlength=12) / np.maximum(nd, 1)
    Hm = np.bincount(dmon, weights=np.maximum(HB - Td, 0), minlength=12) / np.maximum(nd, 1)
    Cm = np.bincount(dmon, weights=np.maximum(Td - HB, 0), minlength=12) / np.maximum(nd, 1)
    out = {}
    for ch, x in meters.items():
        x = np.asarray(x, float)
        v = np.isfinite(x)
        s = np.bincount(mon[v], weights=x[v], minlength=12)
        n = np.bincount(mon[v], minlength=12)
        mm = np.where(n > 24 * 5, s / np.maximum(n, 1), np.nan)
        out[ch] = mm
    return out, Tm, Hm, Cm, nd


def feat_monthly(mm, Tm, Hm, Cm, nd, lat):
    """mm: channel -> 12 monthly mean kW (NaN = missing month). Tm/Hm/Cm: monthly mean T and HDD/CDD per day."""
    ok = nd > 0
    w = [np.nanmean(Tm[ok]) / 10, np.nanstd(Tm[ok]) / 10, np.sum(Hm * nd) / max(nd.sum(), 1) / 10,
         np.sum(Cm * nd) / max(nd.sum(), 1) / 10]
    feats = [Tm / 10, Hm / 10, np.array(w)]
    feats[0] = np.where(ok, feats[0], 0)
    feats[1] = np.where(ok, feats[1], 0)
    for ch in C.CHANNELS:
        x = mm.get(ch)
        if x is None or not np.isfinite(x).any() or np.nansum(x) <= 0:
            feats.append(np.zeros(N_C_M))
            continue
        x = np.asarray(x, float)
        mu = np.nanmean(x)
        if mu <= 0:
            feats.append(np.zeros(N_C_M))
            continue
        xn = x / mu
        f = [1.0, np.isfinite(x).mean(), np.log10(mu)]
        f.extend(np.nan_to_num(xn, nan=0.0))
        pr = prism(Tm, xn)
        f.extend([pr[0], pr[1] * 10, pr[2] * 10, pr[3] / 10, pr[4] / 10, pr[5]])
        feats.append(np.clip(np.nan_to_num(np.asarray(f, float)), -20, 20))
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------- annual
N_A = 4 + 4 * 2


def feat_annual(kwh, hdd, cdd):
    """kwh: channel -> annual kWh (0/None = not used). hdd, cdd: annual degree days, base 18.33 degC."""
    f = [hdd / 3000, cdd / 1500, np.log10(hdd + 50), np.log10(cdd + 50)]
    for ch in C.CHANNELS:
        e = kwh.get(ch)
        if e is None or not np.isfinite(e) or e <= 0:
            f.extend([0.0, 0.0])
        else:
            f.extend([1.0, np.log10(e) - 5])
    return np.asarray(f, np.float32)


# ----------------------------------------------------------------------------- quirks (simulated meters -> real-like)
def quirk_hourly(meters, rng, n):
    """Real meter problems seen in BDG2 (DATA.md): gaps, zero runs, spikes, level shifts, daily-spread readings,
    clock offsets, extra unmodelled load on shared meters, noise. Returns a new dict and the span mask."""
    import pandas as pd
    out = {}
    # partial record: 25 % of samples see 30-330 days
    keep = np.ones(n, bool)
    if rng.random() < 0.25:
        L = int(rng.uniform(30, 330) * 24)
        s0 = rng.integers(0, max(n - L, 1))
        keep[:] = False
        keep[s0:s0 + L] = True
    chans = [c for c in meters if meters[c] is not None]
    drop = set()
    for c in chans:
        p = dict(elec=0.03, fuel=0.3, dheat=0.1, dcool=0.1)[c]
        if rng.random() < p and len(chans) - len(drop) > 1:
            drop.add(c)
    for c in chans:
        if c in drop:
            continue
        x = np.asarray(meters[c], float).copy()
        mu = max(np.nanmean(x), 1e-9)
        # noise: AR(1) multiplicative hourly + daily random walk-ish factor
        sh = rng.uniform(0.01, 0.15)
        e = rng.normal(0, sh, n)
        e = pd.Series(e).ewm(alpha=rng.uniform(0.3, 1.0)).mean().to_numpy()
        sd = rng.uniform(0.0, 0.10)
        dn = np.repeat(rng.normal(0, sd, n // 24 + 2), 24)[:n]
        x = x * np.exp(e + dn) + rng.normal(0, 0.01, n) * mu
        # unmodelled extra load (other spaces on the meter, process, losses)
        if rng.random() < 0.3:
            x = x + rng.uniform(0, 0.3) * mu
        # meter-to-building scale mismatch (sub-metering, shared meters)
        x = x * np.exp(rng.normal(0, 0.12))
        # level shift
        if rng.random() < 0.08:
            k = rng.integers(n // 10, n - n // 10)
            x[k:] *= rng.uniform(0.6, 1.5)
        # spikes
        if rng.random() < 0.3:
            k = rng.integers(0, n, rng.integers(1, 10))
            x[k] *= rng.uniform(3, 10, len(k))
        # zero runs
        if rng.random() < 0.1:
            k = rng.integers(0, n - 200)
            x[k:k + rng.integers(24, 200)] = 0.0
        # gaps
        if rng.random() < 0.6:
            for _ in range(rng.integers(1, 6)):
                k = rng.integers(0, n)
                x[k:k + int(rng.exponential(24 * 7))] = np.nan
            x[rng.random(n) < rng.uniform(0, 0.03)] = np.nan
        # daily totals spread over 24 h (Gator) / clock offset (standard vs daylight time stamps)
        if rng.random() < 0.05:
            d = np.nanmean(x[:n // 24 * 24].reshape(-1, 24), 1)
            x[:n // 24 * 24] = np.repeat(d, 24)
        if rng.random() < 0.2:
            x = np.roll(x, rng.choice([-1, 1]))
        x = np.maximum(x, 0)
        x[~keep] = np.nan
        out[c] = x
    return out, keep


def quirk_monthly(mm, rng):
    out = {}
    for c, x in mm.items():
        x = np.asarray(x, float) * np.exp(rng.normal(0, rng.uniform(0.01, 0.06), 12)) * np.exp(rng.normal(0, 0.1))
        if rng.random() < 0.3:
            x[rng.integers(0, 12, rng.integers(1, 3))] = np.nan
        out[c] = x
    return out


def quirk_annual(kwh, rng):
    out = {}
    for c, e in kwh.items():
        if e is None or e <= 0:
            continue
        if c != 'elec' and rng.random() < dict(fuel=0.05, dheat=0.1, dcool=0.3)[c]:
            continue                      # fuel not reported / district chilled water not in the survey
        out[c] = float(e) * np.exp(rng.normal(0, rng.uniform(0.03, 0.15)))
    return out
