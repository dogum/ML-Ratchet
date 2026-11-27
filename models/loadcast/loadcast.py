"""LoadCast v1.0.1: hourly building energy per meter from weather, calendar and metadata, with 90% error bars.
numpy / pandas / scipy only. Trained only on BuildingSim simulations (see MODEL_CARD.md).

    import loadcast as lc
    m = lc.LoadCast()                                   # loads loadcast_weights.npz next to this file
    fc = m.forecast(weather, tz='America/New_York', lat=40.2, use='education', area_m2=8000, meter='electricity',
                    meters_present=['electricity', 'chilledwater', 'steam'], start='2017-01-01', end='2018-01-01',
                    history=None)                       # history: hourly kWh Series (UTC) -> warm mode
    fc[['kW', 'lo90', 'hi90']]

weather: hourly UTC-indexed DataFrame with T_out (degC), dewpoint (degC), GHI (W/m2), wind_speed (m/s). It must cover
the history window (warm mode) and the forecast window, plus 7 days before each (rolling features). The annual
climate summary comes from the forecast window's local calendar year(s) present in the frame: give at least a year.
"""
import functools, os
import warnings
import numpy as np, pandas as pd

METERS = ['electricity', 'chilledwater', 'steam', 'hotwater', 'gas']
IREF = {'electricity': 15.0, 'chilledwater': 15.0, 'steam': 15.0, 'hotwater': 10.0, 'gas': 10.0}
USES = ['office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse', 'food', 'lab', 'other']
HVAC_CLASSES = ['unknown', 'district_all', 'district_heat', 'district_chw', 'gas_rtu', 'gas_vav', 'all_electric']
SCHED_CLASSES = ['unknown', 'standard', 'extended', '24x7']
COUNTRY = {'Europe/London': 'UK', 'Europe/Dublin': 'IE', 'America/Toronto': 'CA'}
KNOTS = np.array([4.4, 12.8, 18.3, 26.7, 32.2])
STATIC_NAMES = ['use_office', 'use_education', 'use_lodging', 'use_retail', 'use_healthcare', 'use_assembly', 'use_warehouse', 'use_food', 'use_lab', 'use_other', 'logA', 'yb', 'yb_missing', 'm_electricity', 'm_chilledwater', 'm_steam', 'm_hotwater', 'm_gas', 'has_electricity', 'has_chilledwater', 'has_steam', 'has_hotwater', 'has_gas', 'hv_unknown', 'hv_district_all', 'hv_district_heat', 'hv_district_chw', 'hv_gas_rtu', 'hv_gas_vav', 'hv_all_electric', 'sc_unknown', 'sc_standard', 'sc_extended', 'sc_24x7', 'Tmean', 'HDD', 'CDD', 'abslat']
WX_NAMES = ['T', 'T24', 'Tew72', 'T168', 'Tdmax', 'dew', 'lat_pos', 'ghi', 'ghi24', 'wind']
CAL_NAMES = ['h_s1', 'h_c1', 'h_s2', 'h_c2', 'dow0', 'dow1', 'dow2', 'dow3', 'dow4', 'dow5', 'dow6', 'hol', 'xmas', 'doy_s', 'doy_c', 'summer']
HIST_NAMES = ['towt_hat', 'cp_hat', 'prof', 'logI', 'logweeks', 'zero_frac', 'cp_b', 'cp_h', 'cp_c', 'hTmean', 'hTmin', 'hTmax', 'we_ratio', 'night_ratio', 'recent', 'hol_ratio', 'daily_cv', 'T_above', 'T_below']
QMIN = 0.1
Z90 = 1.6449

def country_of_tz(tz):
    return COUNTRY.get(tz, 'US')


def _easter(y):
    a = y % 19; b = y // 100; c = y % 100; d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31; da = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, mo, da)


def _nth_dow(y, mo, dow, n):
    d = pd.Timestamp(y, mo, 1); off = (dow - d.dayofweek) % 7
    return d + pd.Timedelta(days=off + 7 * (n - 1))


def _last_dow(y, mo, dow):
    d = pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0); off = (d.dayofweek - dow) % 7
    return d - pd.Timedelta(days=off)


def _observed(d):      # weekend fixed-date holidays move to Monday (UK/IE/CA substitute days)
    return d + pd.Timedelta(days={5: 2, 6: 1}.get(d.dayofweek, 0))


@functools.lru_cache(maxsize=64)
def holiday_dates(country, year):
    y = year; D = []
    if country == 'US':
        D = [pd.Timestamp(y, 1, 1), _nth_dow(y, 1, 0, 3), _nth_dow(y, 2, 0, 3), _last_dow(y, 5, 0), pd.Timestamp(y, 7, 4),
             _nth_dow(y, 9, 0, 1), _nth_dow(y, 11, 3, 4), _nth_dow(y, 11, 3, 4) + pd.Timedelta(days=1),
             pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 31)]
    elif country == 'CA':
        e = _easter(y)
        vic = pd.Timestamp(y, 5, 24); vic = vic - pd.Timedelta(days=(vic.dayofweek - 0) % 7)
        D = [_observed(pd.Timestamp(y, 1, 1)), _nth_dow(y, 2, 0, 3), e - pd.Timedelta(days=2), vic,
             _observed(pd.Timestamp(y, 7, 1)), _nth_dow(y, 8, 0, 1), _nth_dow(y, 9, 0, 1), _nth_dow(y, 10, 0, 2),
             pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 31)]
    elif country == 'UK':
        e = _easter(y)
        D = [_observed(pd.Timestamp(y, 1, 1)), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1), _nth_dow(y, 5, 0, 1),
             _last_dow(y, 5, 0), _last_dow(y, 8, 0), pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25),
             pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 27), pd.Timestamp(y, 12, 31)]
    elif country == 'IE':
        e = _easter(y)
        D = [_observed(pd.Timestamp(y, 1, 1)), _observed(pd.Timestamp(y, 3, 17)), e + pd.Timedelta(days=1),
             _nth_dow(y, 5, 0, 1), _nth_dow(y, 6, 0, 1), _nth_dow(y, 8, 0, 1), _last_dow(y, 10, 0),
             pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 27)]
    return frozenset(d.normalize() for d in D)


def is_holiday(days, country):
    """days: DatetimeIndex of local dates (naive). -> bool array."""
    days = pd.DatetimeIndex(days).normalize()
    out = np.zeros(len(days), bool)
    for y in np.unique(days.year):
        hs = holiday_dates(country, int(y))
        out |= np.isin(days.values, np.array(sorted(hs), dtype='datetime64[ns]'))
    return out


def _rolling_mean_past(x, n):
    c = np.cumsum(np.r_[0.0, x])
    idx = np.arange(1, len(x) + 1); lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def _ewm(x, tau):
    a = 1.0 / tau; out = np.empty(len(x)); s = x[0]
    for i, v in enumerate(x):
        s += a * (v - s); out[i] = s
    return out


def weather_features(T, dew, ghi, wind, day_codes):
    """T, dew, ghi, wind: hourly arrays (NaN-free). day_codes: integer local-day id per hour (for the daily max).
    Returns (n, 10) float32. Same formulas in JS (the LC object in lab/index.html; site/src/instruments/loadcast/net.ts)."""
    T = np.asarray(T, float); dew = np.asarray(dew, float); ghi = np.asarray(ghi, float); wind = np.asarray(wind, float)
    T24 = _rolling_mean_past(T, 24); T168 = _rolling_mean_past(T, 168); Tew = _ewm(T, 72.0)
    dmax = pd.Series(T).groupby(np.asarray(day_codes)).transform('max').values
    g24 = _rolling_mean_past(ghi, 24)
    X = np.column_stack([T / 10, T24 / 10, Tew / 10, T168 / 10, dmax / 10, dew / 10, np.maximum(dew - 12, 0) / 5,
                         ghi / 500, g24 / 200, wind / 5])
    return X.astype(np.float32)


def calendar_features(hour, dow, holiday, month, mday, doy):
    h = np.asarray(hour, float); n = len(h)
    X = np.zeros((n, len(CAL_NAMES)), np.float32)
    X[:, 0] = np.sin(2 * np.pi * h / 24); X[:, 1] = np.cos(2 * np.pi * h / 24)
    X[:, 2] = np.sin(4 * np.pi * h / 24); X[:, 3] = np.cos(4 * np.pi * h / 24)
    X[np.arange(n), 4 + np.asarray(dow)] = 1
    X[:, 11] = np.asarray(holiday, float)
    month = np.asarray(month); mday = np.asarray(mday)
    X[:, 12] = ((month == 12) & (mday >= 23)) | ((month == 1) & (mday <= 2))
    X[:, 13] = np.sin(2 * np.pi * np.asarray(doy) / 365.25); X[:, 14] = np.cos(2 * np.pi * np.asarray(doy) / 365.25)
    X[:, 15] = ((month == 6) & (mday >= 20)) | (month == 7) | ((month == 8) & (mday <= 20))
    return X


def annual_climate(T):
    """Target-period climate summary from hourly T: mean, HDD18, CDD18 (per year-equivalent)."""
    T = np.asarray(T, float); n_days = len(T) / 24.0
    Td = T[: int(n_days) * 24].reshape(-1, 24).mean(1) if len(T) >= 24 else T
    hdd = np.clip(18 - Td, 0, None).sum() * 365.0 / max(len(Td), 1)
    cdd = np.clip(Td - 18, 0, None).sum() * 365.0 / max(len(Td), 1)
    return np.array([T.mean() / 10, hdd / 3000, cdd / 1000], np.float32)


def meters_of(row_meters):
    """BDG2 meta 'meters' field (comma string or dict) -> list of load meters present."""
    if isinstance(row_meters, dict): ms = list(row_meters)
    else: ms = str(row_meters or '').split(',')
    return [m for m in METERS if m in ms]


def hvac_class_from_meters(meters):
    ms = set(meters)
    chw = 'chilledwater' in ms; heat = bool(ms & {'steam', 'hotwater'})
    if chw and heat: return 'district_all'
    if heat: return 'district_heat'
    if chw: return 'district_chw'
    return 'unknown'


def static_features(use, area, year_built, meter, meters_present, hvac_class, sched_class, clim, lat):
    x = np.zeros(len(STATIC_NAMES), np.float32)
    u = use if use in USES else 'other'
    x[USES.index(u)] = 1
    x[10] = (np.log10(max(area, 50.0)) - 3.8) / 0.5
    if year_built is None or not np.isfinite(year_built): x[12] = 1
    else: x[11] = (year_built - 1970) / 30
    x[13 + METERS.index(meter)] = 1
    for m in meters_present:
        if m in METERS: x[18 + METERS.index(m)] = 1
    x[23 + HVAC_CLASSES.index(hvac_class)] = 1
    x[30 + SCHED_CLASSES.index(sched_class)] = 1
    x[34:37] = clim
    x[37] = abs(lat) / 50
    return x


def towt_X(T, tow):
    seg = [np.minimum(T, KNOTS[0])]
    for a, b in zip(KNOTS[:-1], KNOTS[1:]): seg.append(np.clip(T, a, b) - a)
    seg.append(np.maximum(T - KNOTS[-1], 0))
    X = np.zeros((len(T), 168 + len(seg)))
    X[np.arange(len(T)), tow] = 1
    X[:, 168:] = np.column_stack(seg)
    return X


def towt_fit(y, T, tow, ridge=1.0):
    """TOWT (time-of-week + 6-segment piecewise-linear temperature), ridge-regularised toward a flat profile so it
    works on a few weeks of history too. Returns 174 coefficients (NaN rows ignored)."""
    ok = np.isfinite(y) & np.isfinite(T)
    X = towt_X(T[ok], tow[ok]); yy = y[ok]
    if len(yy) < 48:
        c = np.zeros(174); c[:168] = np.nanmean(y) if np.isfinite(y).any() else 0.0; return c
    ym = yy.mean()
    A = X.T @ X + ridge * np.eye(174); b = X.T @ (yy - ym)
    c = np.linalg.solve(A, b); c[:168] += ym
    return c


def towt_predict(c, T, tow):
    T = np.asarray(T, float)
    seg = [np.minimum(T, KNOTS[0])]
    for a, b in zip(KNOTS[:-1], KNOTS[1:]): seg.append(np.clip(T, a, b) - a)
    seg.append(np.maximum(T - KNOTS[-1], 0))
    return c[np.asarray(tow)] + np.column_stack(seg) @ c[168:]


def cp_fit(yd, Td):
    """Daily 5-parameter-like fit with fixed balance points 15 / 18 C: y = b + h max(15-T,0) + c max(T-18,0), h,c >= 0."""
    ok = np.isfinite(yd) & np.isfinite(Td)
    if ok.sum() < 5: return np.array([np.nanmean(yd) if np.isfinite(yd).any() else 0.0, 0.0, 0.0])
    X = np.column_stack([np.ones(ok.sum()), np.maximum(15 - Td[ok], 0), np.maximum(Td[ok] - 18, 0)])
    from scipy.optimize import nnls
    # intercept may be anything >= 0, slopes >= 0
    c, _ = nnls(X, yd[ok])
    return c


def history_encode(y, T, tow, dcode, is_hol, hour, scale_ref):
    """y, T, tow, dcode (local day id), is_hol, hour: arrays over the HISTORY window (y may contain NaN).
    scale_ref: the cold-start scale (A * IREF / 1000 in kW, or 1 if y is already in those units).
    Returns dict: S (history mean, units of y), coef (174 TOWT on y/S), cp (3), prof (168, /S), scal (14)."""
    y = np.asarray(y, float); T = np.asarray(T, float)
    S = float(np.nanmean(y)) if np.isfinite(y).any() else np.nan
    if not np.isfinite(S) or S <= 0:
        S = max(float(np.nanmax(y)) if np.isfinite(y).any() else 0.0, 1e-3)
    yn = y / S
    coef = towt_fit(yn, T, tow, ridge=2.0)
    dser = pd.DataFrame(dict(y=yn, T=T, d=dcode))
    g = dser.groupby('d'); cnt = g.y.count(); yd = g.y.mean()[cnt >= 18]; Td = g.T.mean()[cnt >= 18]
    cp = cp_fit(yd.values, Td.values)
    prof = pd.Series(yn).groupby(np.asarray(tow)).mean().reindex(range(168)).values
    prof = np.where(np.isfinite(prof), prof, np.nanmean(yn) if np.isfinite(yn).any() else 1.0)
    n_ok = np.isfinite(y).sum()
    weeks = max(n_ok / 168.0, 0.1)
    zero_frac = float(np.mean(yn[np.isfinite(yn)] < 0.02)) if n_ok else 0.0
    wk = (np.asarray(tow) // 24) < 5
    we = np.nanmean(yn[~wk]) / max(np.nanmean(yn[wk]), 1e-3) if (~wk).any() and wk.any() else 1.0
    hr = np.asarray(hour)
    night = (hr < 5) | (hr >= 22); dayh = (hr >= 10) & (hr < 16)
    nr = np.nanmean(yn[night & wk]) / max(np.nanmean(yn[dayh & wk]), 1e-3)
    rec = np.nanmean(yn[-24 * 28:]) if np.isfinite(yn[-24 * 28:]).any() else 1.0
    hol = np.asarray(is_hol, bool)
    hr_ = (np.nanmean(yn[hol & wk]) / max(np.nanmean(yn[(~hol) & wk]), 1e-3)) if (hol & wk & np.isfinite(yn)).any() else 1.0
    dcv = float(np.nanstd(yd.values) / max(np.nanmean(yd.values), 1e-3)) if len(yd) > 2 else 0.0
    Tf = T[np.isfinite(T)]
    sc = np.array([np.log(S / scale_ref), np.log(weeks) / 4, zero_frac, cp[0], cp[1] * 10, cp[2] * 10,
                   Tf.mean() / 10, np.percentile(Tf, 2) / 10, np.percentile(Tf, 98) / 10, we, nr, rec, hr_, dcv], float)
    sc = np.nan_to_num(sc, nan=1.0)
    return dict(S=S, coef=coef.astype(np.float32), cp=cp.astype(np.float32), prof=prof.astype(np.float32),
                scal=sc.astype(np.float32))


def history_hourly(h, Tt, tow_t, Tday_t, T24_t):
    """Per target hour history features: towt_hat, cp_hat, prof, then broadcast scalars, then range indicators."""
    towt = towt_predict(h['coef'], Tt, tow_t)
    cp = h['cp']; cph = cp[0] + cp[1] * np.maximum(15 - Tday_t, 0) + cp[2] * np.maximum(Tday_t - 18, 0)
    prof = h['prof'][tow_t]
    n = len(Tt)
    sc = np.broadcast_to(h['scal'], (n, len(h['scal'])))
    above = np.maximum(T24_t - h['scal'][8] * 10, 0) / 10; below = np.maximum(h['scal'][7] * 10 - T24_t, 0) / 10
    return np.column_stack([towt, cph, prof, sc, above, below]).astype(np.float32)


def g14(y, yh, p=0):
    """ASHRAE Guideline 14 CV(RMSE)% and NMBE% (same as BuildingData baselines; p = 0 for LoadCast)."""
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    ok = np.isfinite(y) & np.isfinite(yh); y, yh = y[ok], yh[ok]; n = len(y)
    if n <= p + 1 or y.mean() <= 0: return np.nan, np.nan, n
    ym = y.mean()
    return 100 * np.sqrt(((y - yh) ** 2).sum() / (n - p)) / ym, 100 * (y - yh).sum() / ((n - p) * ym), n


def np_forward(W, x, base):
    h = (x - W['mu_x']) / W['sd_x']
    for i in sorted(int(k.split('.')[1]) for k in W if k.startswith('body.') and k.endswith('.weight')):
        h = h @ W[f'body.{i}.weight'].T + W[f'body.{i}.bias']
        h = h / (1.0 + np.exp(-np.clip(h, -60, 60)))      # SiLU
    o = h @ W['head.weight'].T + W['head.bias']
    return base + o[:, 0], np.clip(o[:, 1], -7, 4)


class LoadCast:
    def __init__(self, weights=None):
        weights = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loadcast_weights.npz')
        z = np.load(weights, allow_pickle=False)
        self.W = {}
        for mode in ('cold', 'warm'):
            self.W[mode] = {k[len(mode) + 1:]: z[k].astype(np.float64) for k in z.files if k.startswith(mode + '/')}
        self.widen = {str(k): float(v) for k, v in zip(z['widen_keys'], z['widen_vals'])}

    @staticmethod
    def _frame(weather, tz):
        w = weather.sort_index()
        loc = w.index.tz_convert(tz); naive = loc.tz_localize(None); days = naive.normalize()
        T = w['T_out'].interpolate(limit_direction='both').values
        dew = np.minimum(w['dewpoint'].interpolate(limit_direction='both').values, T)
        ghi = w['GHI'].fillna(0).clip(lower=0).values
        wind = w['wind_speed'].interpolate(limit_direction='both').fillna(3).values
        dcode = ((days - days[0]) / pd.Timedelta(days=1)).astype(int).values
        hol = is_holiday(days, country_of_tz(tz))
        wx = weather_features(T, dew, ghi, wind, dcode)
        cal = calendar_features(loc.hour.values, loc.dayofweek.values, hol, loc.month.values, loc.day.values,
                                loc.dayofyear.values)
        Tday = pd.Series(T).groupby(dcode).transform('mean').values
        return dict(idx=w.index, T=T, Tday=Tday, T24=wx[:, 1] * 10, wx=wx, cal=cal, dcode=dcode, hol=hol,
                    hour=loc.hour.values, tow=loc.dayofweek.values * 24 + loc.hour.values, year=loc.year.values)

    def forecast(self, weather, tz, lat, use, area_m2, meter, meters_present=None, start=None, end=None,
                 history=None, year_built=None, hvac='unknown', sched='unknown', calibrated=True, frame=None,
                 assume_electricity=True):
        """Returns a DataFrame (UTC index over [start, end)) with kW, sigma, lo90, hi90 and mode ('cold'/'warm').

        Every simulated training building had an electricity meter (has_electricity: mean 1, sd 0.001), so a
        meters_present list without 'electricity' put that input at z = -1000 in v1.0 (audit, 25 Nov 2026: a steam or
        chilled water forecast rose 24-50x in every hour). v1.0.1 (27 Nov 2026) adds 'electricity' to the list by
        default (assume_electricity=True): on the 19 exam building-meters this touches, median cold CV(RMSE) fell from
        597 % to 114 % and warm (a year) from 51 % to 35 %. assume_electricity=False gives v1.0's behaviour, with a
        warning."""
        f = frame or self._frame(weather, tz)
        idx = f['idx']
        t0 = pd.Timestamp(start) if start is not None else idx[0]
        t1 = pd.Timestamp(end) if end is not None else idx[-1] + pd.Timedelta(hours=1)
        if t0.tzinfo is None: t0 = t0.tz_localize(tz).tz_convert('UTC')
        if t1.tzinfo is None: t1 = t1.tz_localize(tz).tz_convert('UTC')
        tsel = np.flatnonzero((idx >= t0) & (idx < t1))
        yrs = np.unique(f['year'][tsel])
        clim = annual_climate(f['T'][np.isin(f['year'], yrs)])
        meters_present = list(meters_present or [meter])
        if 'electricity' not in meters_present:
            if assume_electricity:
                meters_present.append('electricity')
            else:
                warnings.warn("LoadCast: meters_present has no 'electricity' and assume_electricity=False (v1.0 "
                              "behaviour); every training building had one, so the network is far outside its training "
                              "data (24-50x too high in the audit).", stacklevel=2)
        if hvac == 'unknown': hvac = hvac_class_from_meters(meters_present)
        st = static_features(use, area_m2, year_built, meter, meters_present, hvac, sched, clim, lat)
        unit = area_m2 * IREF[meter] / 1000.0           # kW per model unit
        X = [np.broadcast_to(st, (len(tsel), len(st))), f['wx'][tsel], f['cal'][tsel]]
        if history is None:
            mode = 'cold'; base = np.zeros(len(tsel)); Q = 1.0
        else:
            mode = 'warm'
            h = history.reindex(idx).values.astype(float) / unit
            hi = np.flatnonzero((idx >= history.index.min()) & (idx <= history.index.max()))
            H = history_encode(h[hi], f['T'][hi], f['tow'][hi], f['dcode'][hi], f['hol'][hi], f['hour'][hi], 1.0)
            hh = history_hourly(H, f['T'][tsel], f['tow'][tsel], f['Tday'][tsel], f['T24'][tsel])
            X.append(hh); Q = max(H['S'], QMIN); base = hh[:, 0] * H['S'] / Q
        X = np.concatenate(X, 1)
        mu, ls = np_forward(self.W[mode], X, base)
        weeks = None if history is None else len(history.dropna()) / 168.0
        variant = 'full' if mode == 'cold' or weeks > 20 else ('w8' if weeks > 4 else 'w2')
        k = self.widen.get(f'{mode}_{variant}_{meter}', 1.0) if calibrated else 1.0
        kW = np.clip(mu, 0, None) * Q * unit; sd = np.exp(ls) * Q * unit * k
        out = pd.DataFrame(dict(kW=kW, sigma=sd, lo90=np.maximum(kW - Z90 * sd, 0), hi90=kW + Z90 * sd), index=idx[tsel])
        out.attrs['mode'] = mode; out.attrs['widen'] = k
        return out
