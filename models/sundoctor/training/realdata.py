"""Real data -> SunDoctor's input grid: every day resampled onto 96 steps of local *solar* time, so solar noon is always
step 48 whatever the longitude, time zone or daylight-saving rule."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, pandas as pd, sys
sys.path.insert(0, _P.SRC)
import pvsim as S

SYS2107 = dict(lat=38.996306, lon=-122.134111, tilt=25.0, azimuth=180.0, pdc_kw=37.2, tz='US/Pacific', n_strings=6)

def to_solar_grid(t_utc, series, lon):
    """t_utc: DatetimeIndex (UTC) of interval *centers*; series: (n,) or (n, k). Returns day list and (days, 96[, k])."""
    doy = t_utc.dayofyear.values; ut = (t_utc - t_utc.normalize()).total_seconds().values / 3600.0
    tsol_abs = (t_utc.normalize().values.astype('datetime64[s]').astype(np.int64) / 3600.0) + ut + lon / 15 + S.eot_minutes(doy) / 60
    day0 = np.floor(tsol_abs.min() / 24) * 24; n_days = int(np.ceil((tsol_abs.max() - day0) / 24))
    grid = day0 + np.arange(n_days)[:, None] * 24 + S.TSOL[None]           # absolute solar hours
    x = np.asarray(series, np.float64); x2 = x.reshape(len(x), -1); out = np.full((n_days * 96, x2.shape[1]), np.nan)
    g = grid.ravel()
    for j in range(x2.shape[1]):
        ok = np.isfinite(x2[:, j]); o = np.interp(g, tsol_abs[ok], x2[ok, j], left=np.nan, right=np.nan)
        # do not bridge gaps longer than 30 min
        idx = np.searchsorted(tsol_abs[ok], g); lo = np.clip(idx - 1, 0, ok.sum() - 1); hi = np.clip(idx, 0, ok.sum() - 1)
        gap = tsol_abs[ok][hi] - tsol_abs[ok][lo]; o[gap > 0.51] = np.nan
        out[:, j] = o
    days = pd.to_datetime((day0 / 24 + np.arange(n_days)) * 86400, unit='s')      # solar date
    return days, out.reshape(n_days, 96, -1).squeeze()

def geometry(days, lat, tilt, azimuth):
    doy = days.dayofyear.values[:, None]
    cz, el, az = S.sun(lat, doy, S.TSOL[None]); ca = S.cos_aoi(tilt, azimuth, cz, az)
    g, dn, dh = S.ineichen(cz, doy, 3.0); b, sk, gn = S.haydavies(tilt, ca, cz, dn, dh, g, doy)
    return dict(cz=np.maximum(cz, 0), ca=np.maximum(ca, 0), poa_clear=b + sk + gn, el=el, az=az)

def load_2107():
    z = np.load(_P.WORK + '/data/sys2107_15min.npz')
    # interval centers, local clock. PVDAQ labels here sit ~3.5 min after the interval center: clear-day POA is symmetric
    # about solar noon only with this shift (power and irradiance are mutually aligned to <2.5 min)
    t_local = pd.to_datetime(z['t'], unit='m') + pd.Timedelta(minutes=-3.5)
    t_utc = t_local.tz_localize(SYS2107['tz'], ambiguous='NaT', nonexistent='NaT').tz_convert('UTC').tz_localize(None)
    good = ~pd.isna(t_utc); t_utc = t_utc[good]
    pac = z['pac'][good]; poa = z['poa'][good]; ta = z['tamb'][good]
    pac = np.where((pac >= -0.5) & (pac <= 31.0), pac, np.nan)          # drop spikes beyond the inverter's 30 kW maximum
    poa = np.where((poa >= -5) & (poa <= 1600), poa, np.nan)
    days, P = to_solar_grid(pd.DatetimeIndex(t_utc), pac, SYS2107['lon'])
    _, G = to_solar_grid(pd.DatetimeIndex(t_utc), poa, SYS2107['lon'])
    _, T = to_solar_grid(pd.DatetimeIndex(t_utc), ta, SYS2107['lon'])
    return days, P / SYS2107['pdc_kw'], G / 1000.0, T, geometry(days, SYS2107['lat'], SYS2107['tilt'], SYS2107['azimuth'])

def sibling_truth(P, G):
    """Answer key: per inverter-day fraction of the array working, from the 23 siblings. Uses unclipped daylight only
    (fleet healthy level, the 80th percentile of the 24, between 0.08 and 0.70 of DC nameplate) and compares morning
    with afternoon: days where they disagree by more than 0.06 (e.g. a string that only reconnects once dew has dried)
    are marked ambiguous (NaN), as are days with fewer than 4 usable steps on either side of solar noon."""
    ref = np.nanpercentile(P, 80, axis=2)
    ok = (ref > 0.08) & (ref < 0.70)
    r = np.where(ok[..., None], P / np.maximum(ref[..., None], 1e-3), np.nan)
    am, pm = r[:, :48], r[:, 48:]
    with np.errstate(all='ignore'):
        fa = np.nanmedian(am, 1); fp = np.nanmedian(pm, 1); fr = np.nanmedian(r, 1)
        na = np.sum(np.isfinite(am), 1); npm = np.sum(np.isfinite(pm), 1)
    fr[(na < 4) | (npm < 4) | (np.abs(fa - fp) > 0.06)] = np.nan
    return fr, na + npm
