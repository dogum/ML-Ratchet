"""Feature vectors shared by the simulator and the real-data banks (identical code path for both).

Times: target day D = 24 hours of local standard time (LST); the forecast is issued at 10:00 LST on D-1, so the input
history is the 168 h before issue (hist index j -> LST hour (10 + j) % 24), followed by a 14 h gap, then D.
Demand target: log(y / prior_total) per hour (prior = building-stock physics with the region's descriptor params).
Wind / solar targets: capacity factor y / capacity online at issue.
"""
import numpy as np

COMP = ('base', 'light', 'heat', 'cool', 'btm')


def _lst_profile(r):
    """hist [168] -> mean by LST hour (target order 0..23)."""
    prof = np.zeros(24); cnt = np.zeros(24)
    h = (10 + np.arange(168)) % 24
    ok = np.isfinite(r)
    np.add.at(prof, h[ok], r[ok]); np.add.at(cnt, h[ok], 1)
    return np.where(cnt > 0, prof / np.maximum(cnt, 1), 0.0)


def _daily(x, fn=np.nanmean):
    with np.errstate(all='ignore'):
        v = np.array([fn(x[k * 24:(k + 1) * 24]) for k in range(7)])
    return np.nan_to_num(v)


def demand_x(yh, ph, pt, wt, cal, desc, fc):
    """yh [168] actual demand (NaN ok); ph/pt dicts of prior components [168]/[24]; wt dict T, q, ghi [24] input
    weather of D; cal dict daytype (0/1/2), dst (0/1), doy; desc [5]; fc 0/1 (forecast weather)."""
    with np.errstate(all='ignore'):
        rho = yh / ph['total'] - 1
    rho = np.where(np.isfinite(rho), np.clip(rho, -0.5, 0.5), np.nan)
    s = np.nanmean(ph['total'])
    hs = ph['heat'] / ph['total']; cs = ph['cool'] / ph['total']
    x = [_daily(rho), _lst_profile(rho), np.nan_to_num(rho[14:38]), np.nan_to_num(rho[144:168]),
         _daily(hs), _daily(cs), _daily(ph['T']) / 10]
    for c in COMP:
        x.append(pt[c] / s)
    x += [pt['total'] / s - 1, wt['T'] / 10, wt['q'] / 10, wt['ghi'] / 1000]
    dt = np.zeros(3); dt[int(cal['daytype'])] = 1
    x += [dt, [cal['dst'], np.sin(2 * np.pi * cal['doy'] / 365.25), np.cos(2 * np.pi * cal['doy'] / 365.25)],
          np.asarray(desc, float), [fc, np.mean(~np.isfinite(rho))],
          np.nan_to_num(rho)]                      # v2 features (appended): the full 168 h ratio history
    return np.concatenate([np.ravel(a) for a in x]).astype(np.float32)


def demand_y(y, pt_total):
    with np.errstate(all='ignore'):
        return np.log(y / pt_total).astype(np.float32)


def wind_x(ah, ph, pt, pt_lo, pt_hi, ws, cap, doy, fc):
    """ah/ph [168] actual/prior MW (prior already x dev scale); pt, pt_lo, pt_hi [24] prior for D at the input wind,
    x0.9, x1.1; ws [24] capacity-weighted 50 m wind of D (m/s); cap MW online at issue."""
    c = max(cap, 1.0)
    ah = np.where(np.isfinite(ah), np.maximum(ah, 0), np.nan)    # EIA-930 reports small negative output (station load)
    a = ah / c; p = ph / c
    ok = np.isfinite(a)
    ratio = np.clip(np.nansum(np.where(ok, ah, 0)) / max(np.nansum(np.where(ok, ph, 0)), 1e-6), 0, 3) if ok.any() else 1.0
    with np.errstate(all='ignore'):
        e = np.where(ok, a - p, np.nan)
    x = [pt / c, (pt_hi - pt_lo) / c, _daily(a), _daily(p), np.nan_to_num(a[144:168]), p[144:168],
         np.nan_to_num(e[144:168]), [ratio, np.nanstd(e) if ok.sum() > 2 else 0.0], ws / 10,
         [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), fc, 1 - ok.mean()],
         np.nan_to_num(e)]                         # v2 features (appended): full 168 h CF error history
    return np.nan_to_num(np.concatenate([np.ravel(v) for v in x])).astype(np.float32)


def solar_x(ah, ph, pt, pt_lo, pt_hi, clr, kt, cap, doy, fc):
    """as wind_x; clr [24] clear-sky GHI (kW/m2) and kt [24] clearness of D at population points."""
    c = max(cap, 1.0)
    ah = np.where(np.isfinite(ah), np.maximum(ah, 0), np.nan)    # negative night-time output (station load) -> 0
    a = ah / c; p = ph / c
    ok = np.isfinite(a)
    with np.errstate(all='ignore'):
        dr = np.array([np.nansum(np.where(ok, ah, 0)[k * 24:(k + 1) * 24]) / max(np.nansum(np.where(ok, ph, 0)[k * 24:(k + 1) * 24]), 1e-6)
                       for k in range(7)])
    dr = np.clip(np.nan_to_num(dr, nan=1.0), 0, 3)
    ratio = np.clip(np.nansum(np.where(ok, ah, 0)) / max(np.nansum(np.where(ok, ph, 0)), 1e-6), 0, 3) if ok.any() else 1.0
    x = [pt / c, (pt_hi - pt_lo) / c, dr, [ratio], np.nan_to_num(a[144:168]), p[144:168], clr, kt,
         [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), fc, 1 - ok.mean()],
         np.nan_to_num(a - p)]                     # v2 features (appended): full 168 h CF error history
    return np.nan_to_num(np.concatenate([np.ravel(v) for v in x])).astype(np.float32)
