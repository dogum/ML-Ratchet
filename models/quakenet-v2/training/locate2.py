"""Classical ways to combine several stations, used as baselines and as the physics anchor QuakeNet v2 corrects.

combine_v12  QuakeNet v1.2 run per station, combined classically: epicentre by weighted least squares on the sphere
             from each station's distance and back-azimuth readings (with v1.2's dev-widened error bars); Mw and log depth
             as inverse-variance means; origin time as the mean of (P pick - ak135 P time) at that location and depth.
tt_locate    a travel-time locator: P picks (STA/LTA) and S picks (horizontal STA/LTA + AIC), grid search over the globe
             (2 deg, then 0.25 deg around the best three) and 27 depths with ak135 P and S times; origin time solved for;
             Huber misfit (P 2 s, S 10 s).
centroid     the dumb baseline: the stations' centroid; dev-median depth and Mw; origin time from the centroid.
All take per-station arrays (station lat/lon in degrees, readings, pick times in s on any common clock)."""
import numpy as np
from scipy.optimize import least_squares
import tt as TT
V12_FACTOR = dict(mw=1.0, dist=1.6404, logz=2.4165, baz=1.1398)     # v1.2's widen-only factors (sigma_calibration.json)
KM = TT.KM_PER_DEG

def local_offset(lat0, lon0, lat, lon):
    """(north, east) km of point(s) from (lat0, lon0) along the great circle"""
    d = TT.dist(lat0, lon0, lat, lon) * KM; a = np.radians(TT.azimuth(lat0, lon0, lat, lon)); return d * np.cos(a), d * np.sin(a)

def from_offset(lat0, lon0, n, e):
    return TT.destination(lat0, lon0, np.degrees(np.arctan2(e, n)) % 360, np.hypot(n, e) / KM)

def combine_v12(slat, slon, R, tpick=None, calibrated=True):
    """R: dict of arrays from sim2.decode (raw sigmas); returns dict"""
    f = V12_FACTOR if calibrated else dict(mw=1, dist=1, logz=1, baz=1)
    sd = R['dist_sig'] * f['dist']; sb = np.degrees(1 / np.sqrt(np.maximum(R['kappa'], 1e-3))) * f['baz']; sb = np.maximum(sb, 0.5)
    c0 = [TT.destination(a, b, z, d) for a, b, z, d in zip(slat, slon, R['baz'], R['dist'])]
    def res(p):
        return np.concatenate([(TT.dist(p[0], p[1], slat, slon) - R['dist']) / sd, TT.angdiff(TT.azimuth(slat, slon, p[0], p[1]), R['baz']) / sb])
    u = TT.unit(np.array([c[0] for c in c0]), np.array([c[1] for c in c0])); w = 1 / (sd * sb); m = TT.latlon((u * w[:, None]).sum(0))
    starts = c0 + [m]; cost = [np.sum(res(np.array(s, float)) ** 2) for s in starts]; best = None
    for i in np.argsort(cost)[:2]:
        s = starts[i]
        r = least_squares(res, np.array(s, float), x_scale=[1.0, 1.0], method='lm' if 2 * len(slat) >= 2 else 'trf', max_nfev=200)
        if best is None or r.cost < best.cost: best = r
    lat, lon = float(np.clip(best.x[0], -90, 90)), float((best.x[1] + 180) % 360 - 180)
    J = best.jac; cov = np.linalg.pinv(J.T @ J); s2 = max(1.0, 2 * best.cost / max(len(best.fun) - 2, 1)) if len(slat) > 2 else 1.0   # deg^2
    cl = np.cos(np.radians(lat)); C = np.array([[cov[0, 0], cov[0, 1] * cl], [cov[0, 1] * cl, cov[1, 1] * cl * cl]]) * KM * KM * s2  # (north, east) km^2
    wm = 1 / (R['mw_sig'] * f['mw']) ** 2; mw = float((R['mw'] * wm).sum() / wm.sum()); mw_sig = float(1 / np.sqrt(wm.sum()))
    wz = 1 / (R['logz_sig'] * f['logz']) ** 2; lz = float((R['logz'] * wz).sum() / wz.sum()); lz_sig = float(1 / np.sqrt(wz.sum()))
    out = dict(lat=lat, lon=lon, cov=C, mw=mw, mw_sig=mw_sig, logz=lz, logz_sig=lz_sig, depth=10 ** lz)
    if tpick is not None:
        D = TT.dist(lat, lon, slat, slon); r = np.asarray(tpick) - TT.tt('P', D, 10 ** lz); t0 = float(np.mean(r))
        p = np.abs(TT.slowness(D, 10 ** lz)).mean(); epi = np.sqrt(np.trace(C) / 2) / KM
        out.update(t0=t0, t0_sig=float(np.sqrt(np.var(r) / len(r) + (p * epi) ** 2 + (0.12 * 10 ** lz * 2.303 * lz_sig) ** 2 + 1.0)))
    return out

_G = None
def _grid():
    global _G
    if _G is None:
        la = np.arange(-88, 88.1, 2.0); lats, lons = [], []
        for a in la:
            n = max(1, int(round(180 * np.cos(np.radians(a)))));  lo = np.linspace(-180, 180, n, endpoint=False); lats += [a] * n; lons += list(lo)
        _G = (np.array(lats), np.array(lons))
    return _G

ZS = TT.DEPTHS
def _misfit(glat, glon, slat, slon, tp, ts, zlist, sp=2.0, ss=10.0):
    D = TT.dist(glat[:, None], glon[:, None], slat[None], slon[None]); ok = ((D >= 15) & (D <= 100)).all(1); best = (np.inf, None)
    hub = lambda r: np.where(np.abs(r) < 3, 0.5 * r * r, 3 * np.abs(r) - 4.5)
    hasS = np.isfinite(ts); out = []
    for z in zlist:
        RP = tp[None] - TT.tt('P', D, z); RS = ts[None] - TT.tt('S', D, z)
        wP = np.ones_like(RP) / sp ** 2; wS = np.where(hasS[None], 1 / ss ** 2, 0) * np.ones_like(RS)
        t0 = ((RP * wP).sum(1) + np.nan_to_num(RS * wS).sum(1)) / (wP.sum(1) + wS.sum(1))
        c = hub((RP - t0[:, None]) / sp).sum(1) + np.where(hasS[None], hub(np.nan_to_num(RS - t0[:, None]) / ss), 0).sum(1)
        c[~ok] = np.inf; out.append((c, t0))
    return out

def tt_locate(slat, slon, tp, ts):
    slat, slon, tp = np.asarray(slat, float), np.asarray(slon, float), np.asarray(tp, float); ts = np.array([np.nan if v is None else v for v in ts], float)
    gl, go = _grid(); zc = [10, 33, 70, 120, 200, 300, 400, 500, 600]; M = _misfit(gl, go, slat, slon, tp, ts, zc)
    C = np.stack([m[0] for m in M]); flat = np.argsort(C.ravel())[:3]; best = (np.inf,)
    for f in flat:
        zi, gi = np.unravel_index(f, C.shape)
        if not np.isfinite(C[zi, gi]): continue
        a = np.arange(-3, 3.01, 0.25); A, B = np.meshgrid(gl[gi] + a, a, indexing='ij'); lat = np.clip(A.ravel(), -89.9, 89.9)
        lon = ((go[gi] + B.ravel() / max(np.cos(np.radians(gl[gi])), 0.2)) + 180) % 360 - 180
        for z, (c, t0) in zip(ZS, _misfit(lat, lon, slat, slon, tp, ts, ZS)):
            j = int(np.argmin(c))
            if c[j] < best[0]: best = (c[j], lat[j], lon[j], z, t0[j])
    if not np.isfinite(best[0]): return dict(lat=np.nan, lon=np.nan, depth=np.nan, t0=np.nan)
    return dict(lat=float(best[1]), lon=float(best[2]), depth=float(best[3]), t0=float(best[4]), cost=float(best[0]))

def centroid(slat, slon, tpick, z_med, mw_med):
    lat, lon = TT.latlon(TT.unit(np.asarray(slat), np.asarray(slon)).sum(0)); D = TT.dist(lat, lon, slat, slon)
    return dict(lat=float(lat), lon=float(lon), depth=z_med, mw=mw_med, t0=float(np.mean(np.asarray(tpick) - TT.tt('P', D, z_med))))
