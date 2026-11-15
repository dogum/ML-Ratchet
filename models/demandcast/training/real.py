"""Real-BA regions (2018-2025) and real forecast-weather errors (Open-Meteo previous runs)."""
import json, os, numpy as np, pandas as pd
from common import *
import region as RG, renew as R
from _paths import CALIB


def _calib(name):
    """work/data/calib/<name> if the calibration was re-run, else the released one in models/demandcast/calib/."""
    f = f'{DATA}/calib/{name}'
    return f if os.path.exists(f) else os.path.join(CALIB, name)

_CACHE = {}


def desc_vec(p):
    return np.array([p['e_res'], p['e_hp'], p['ua'], p['mu_c'] / 10 - 2.3, p['btm']], np.float32)


def real_region(ba, calib='dev'):
    key = (ba, calib)
    if key in _CACHE: return _CACHE[key]
    W = load_bundle(ba)
    g = RG.Region(); g.ba = ba; g.off = 0
    g.cal = RG.calendar_arrays(HOURS, TZ[ba], LST[ba])
    g.wx = dict(T=W['T'].astype(float), q=W['q'].astype(float), RH=W['RH'].astype(float), ghi=W['ghi'].astype(float),
                ghi_clr=W['ghi_clr'].astype(float), btm_cf=W['btm_cf'].astype(float))
    g.W = {k: W[k] for k in ('pT', 'pRH', 'pws10', 'pws50', 'pps', 'pghi', 'pcf_track', 'pcf_fixed', 'plat', 'plon', 'pkind', 'pw')}
    cal = json.load(open(_calib(f'demand_{calib}.json')))[ba]['params']
    g.pdesc = cal; g.desc = desc_vec(cal)
    d = gd.eia930(ba, columns=['d', 'df', 'q_ok', 'ng_WND', 'ng_SUN']).reindex(HOURS)
    g.yd = np.where(d.q_ok.fillna(False).values.astype(bool), d.d.values, np.nan).astype(float)
    g.df = d.df.values.astype(float)
    g.yw = d.ng_WND.values.astype(float); g.ys = d.ng_SUN.values.astype(float)
    rk = json.load(open(_calib(f'renew_{calib}.json')))[ba]
    g.wg, g.wcap = R.wind_groups(ba); g.wk = rk['wind']['k']
    g.sg, g.scap = R.solar_groups(ba)
    g.has_solar = rk['solar'] is not None
    g.sk = rk['solar']['k'] if g.has_solar else 1.0
    # dev-calibrated timestamp fix: CISO and PJM solar in EIA-930 lead the physics by one hour on dev years
    lag = json.load(open(_calib('renew_dev.json')))[ba]['solar']
    g.solar_lag = int(lag.get('lag', 0)) if lag else 0
    if g.solar_lag: g.ys = np.roll(g.ys, -g.solar_lag)
    g.finalize()
    _CACHE[key] = g
    return g


def om_dir():
    """Open-Meteo previous-runs cache: $DC_OPENMETEO if set, else work/data/openmeteo/ (the release's place) if it
    exists, else the grid data root's raw/openmeteo/ (`mlratchet data grid --only forecastwx`). Added 14 Nov 2026."""
    if os.environ.get('DC_OPENMETEO'):
        return os.environ['DC_OPENMETEO']
    if os.path.isdir(f'{DATA}/openmeteo'):
        return f'{DATA}/openmeteo'
    return os.path.join(GRID, 'data', 'raw', 'openmeteo')


def om_series(pid, year, var):
    f = f'{om_dir()}/{pid}_{year}.json'
    if not os.path.exists(f): return None
    h = json.load(open(f))
    if var not in h or var + '_previous_day1' not in h: return None
    a = np.array([np.nan if v is None else v for v in h[var]], float)
    b = np.array([np.nan if v is None else v for v in h[var + '_previous_day1']], float)
    return a, b


class Forecasts:
    """Per-BA forecast-error arrays over HOURS: T (K, pop-weighted), rh (relative), ghi mult per point, v mult per point.
    NaN where no forecast is archived."""
    def __init__(self, ba):
        pts = gd.weather_points(ba)['points']
        n = len(pts)
        self.T = np.full(NH, np.nan); self.rh = np.full(NH, np.nan)
        self.ghi = np.full((n, NH), np.nan); self.v = np.full((n, NH), np.nan)
        wsum = np.zeros(NH); Tacc = np.zeros(NH); rhacc = np.zeros(NH); rws = np.zeros(NH)
        for k, p in enumerate(pts):
            for y in (2022, 2023, 2024, 2025):
                i0 = hidx(pd.Timestamp(f'{y}-01-01', tz='UTC')); i1 = hidx(pd.Timestamp(f'{y + 1}-01-01', tz='UTC')) if y < 2025 else NH
                if p['kind'] == 'pop':
                    r = om_series(p['pid'], y, 'temperature_2m')
                    if r is not None:
                        e = r[1] - r[0]; ok = np.isfinite(e)
                        Tacc[i0:i1][ok[:i1 - i0]] += p['w'] * e[:i1 - i0][ok[:i1 - i0]]; wsum[i0:i1][ok[:i1 - i0]] += p['w']
                    r = om_series(p['pid'], y, 'relative_humidity_2m')
                    if r is not None:
                        e = (r[1] - r[0]) / np.maximum(r[0], 5); ok = np.isfinite(e)
                        rhacc[i0:i1][ok[:i1 - i0]] += p['w'] * e[:i1 - i0][ok[:i1 - i0]]; rws[i0:i1][ok[:i1 - i0]] += p['w']
                r = om_series(p['pid'], y, 'shortwave_radiation')
                if r is not None:
                    a, b = r
                    m = np.where(a + b > 20, (b + 10) / (a + 10), 1.0)
                    self.ghi[k, i0:i1] = np.clip(m, 0.1, 5)[:i1 - i0]
                r = om_series(p['pid'], y, 'wind_speed_80m')
                if r is not None:
                    a, b = r
                    self.v[k, i0:i1] = np.clip((b + 0.5) / (a + 0.5), 0.2, 4)[:i1 - i0]
        with np.errstate(all='ignore'):
            self.T = np.where(wsum > 0.5, Tacc / wsum, np.nan)
            self.rh = np.where(rws > 0.5, rhacc / rws, np.nan)
        # points without their own radiation / wind forecast use the BA mean of points that have one
        for arr in (self.ghi, self.v):
            with np.errstate(all='ignore'):
                mean = np.nanmean(arr, 0)
            for k in range(n):
                arr[k] = np.where(np.isfinite(arr[k]), arr[k], mean)

    def fc(self, iss, ie):
        """Forecast-error dict for hours [iss, ie) or None if temperature forecast missing. Variables without an
        archived forecast get zero error (that part of the input is actual weather)."""
        T = self.T[iss:ie]
        if not np.isfinite(T).all(): return None
        rh = np.nan_to_num(self.rh[iss:ie]); g = np.nan_to_num(self.ghi[:, iss:ie], nan=1.0); v = np.nan_to_num(self.v[:, iss:ie], nan=1.0)
        bits = 1 + 2 * bool(np.isfinite(self.rh[iss:ie]).all()) + 4 * bool(np.isfinite(self.ghi[:, iss:ie]).all()) \
            + 8 * bool(np.isfinite(self.v[:, iss:ie]).all())
        return dict(T=T, rh=rh, ghi=g, v=v, full=bits == 15, bits=bits)
