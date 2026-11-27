"""Weather: EPW reader, contract checks, resampling, solar geometry and surface irradiance (pvlib), sky, ground and
mains-water temperatures.

Contract (BUILDING_BRIEF): a UTC-indexed DataFrame with T_out [C], RH [%], dewpoint [C], GHI, DNI, DHI [W/m2],
wind_speed [m/s], wind_dir [deg], pressure [Pa], optional cloud [0-1] and IR_horiz [W/m2]; plus meta
{lat, lon, elev, tz, source}. Convention used here: a row stamped t describes the interval [t, t + step):
irradiance is the interval mean, temperatures/humidity/wind are interval means (EPW instants are averaged
pairwise when read). Missing DNI/DHI are split from GHI with the Erbs model; a missing dewpoint comes from RH.
"""
import hashlib
import math
import numpy as np
import pandas as pd
import pvlib

from . import psychro as ps

SIGMA = 5.670374e-8

# US standard-time offsets -> a representative IANA zone (EPW files carry only the offset)
_TZ_BY_OFFSET = {-5: 'America/New_York', -6: 'America/Chicago', -7: 'America/Denver', -8: 'America/Los_Angeles',
                 -9: 'America/Anchorage', -10: 'Pacific/Honolulu', -4: 'America/Puerto_Rico', 0: 'UTC',
                 1: 'Europe/Paris', 9: 'Asia/Tokyo'}
_NO_DST_STATES = {'AZ', 'HI', 'PR'}

CONTRACT_COLUMNS = ['T_out', 'RH', 'dewpoint', 'GHI', 'DNI', 'DHI', 'wind_speed', 'wind_dir', 'pressure']


def read_epw(path, year=2019):
    """Read an EnergyPlus weather file. Returns (DataFrame in the contract, meta). The (TMY) year is rebased to
    `year` so the calendar (weekdays, holidays) is a real one; Feb 29 never appears in TMY files."""
    with open(path, 'r', encoding='latin-1') as f:
        lines = f.read().splitlines()
    loc = lines[0].split(',')
    lat, lon, tzoff, elev = float(loc[6]), float(loc[7]), float(loc[8]), float(loc[9])
    state = loc[2].strip()
    rows = [l.split(',') for l in lines[8:] if l and l[0].isdigit()]
    a = np.array([[r[1], r[2], r[3], r[6], r[7], r[8], r[9], r[12], r[13], r[14], r[15], r[20], r[21], r[22], r[23]]
                  for r in rows], dtype=float)
    month, day, hour = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int)
    # hour-ending h (local standard time) -> interval start h-1
    t_local = pd.to_datetime(dict(year=np.full(len(a), year), month=month, day=day)) + pd.to_timedelta(hour - 1, 'h')
    t_utc = pd.DatetimeIndex(t_local) - pd.Timedelta(hours=tzoff)
    t_utc = t_utc.tz_localize('UTC')

    def inst_to_interval(x):            # mean of the instants bounding the interval [h-1, h]
        prev = np.r_[x[-1], x[:-1]]
        return 0.5 * (x + prev)

    T = inst_to_interval(a[:, 3])
    Td = inst_to_interval(a[:, 4])
    P = inst_to_interval(np.where(a[:, 6] > 0, a[:, 6], np.nan))
    P = np.where(np.isfinite(P), P, ps.pressure_from_elevation(elev))
    W = ps.w_from_dewpoint(Td, P)
    RH = np.clip(ps.rh_from_w(T, W, P), 1, 100)
    ws = inst_to_interval(np.clip(a[:, 12], 0, 40))
    wd = a[:, 11]
    df = pd.DataFrame(dict(T_out=T, RH=RH, dewpoint=Td, GHI=np.clip(a[:, 8], 0, None), DNI=np.clip(a[:, 9], 0, None),
                           DHI=np.clip(a[:, 10], 0, None), wind_speed=ws, wind_dir=wd, pressure=P,
                           cloud=np.clip(a[:, 14], 0, 10) / 10.0, IR_horiz=a[:, 7]), index=t_utc)
    df = df.sort_index()
    tz = _TZ_BY_OFFSET.get(int(round(tzoff)), 'UTC')
    if state in _NO_DST_STATES:
        tz = {'AZ': 'America/Phoenix', 'HI': 'Pacific/Honolulu', 'PR': 'America/Puerto_Rico'}[state]
    meta = dict(lat=lat, lon=lon, elev=elev, tz=tz, tz_offset_std=tzoff, source='EPW ' + str(path).split('/')[-1],
                name=loc[1], state=state, wmo=loc[5])
    return df, meta


def rebase_year(df, year):
    """Move a single-year (typical-year) frame to another calendar year, keeping month/day/hour; drops Feb 29."""
    idx = df.index
    keep = ~((idx.month == 2) & (idx.day == 29))
    df = df[keep]
    idx = df.index
    new = pd.to_datetime(dict(year=np.full(len(idx), year), month=idx.month, day=idx.day, hour=idx.hour,
                              minute=idx.minute)).values
    out = df.copy()
    out.index = pd.DatetimeIndex(new).tz_localize('UTC')
    return out.sort_index()


def check_weather(df, meta):
    """Fill derivable columns and validate units (raises on obviously wrong units)."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    if 'pressure' not in df or df['pressure'].isna().all():
        df['pressure'] = float(ps.pressure_from_elevation(meta.get('elev', 0.0)))
    df['pressure'] = df['pressure'].fillna(df['pressure'].median())
    if df['pressure'].median() < 2000:          # given in hPa
        df['pressure'] = df['pressure'] * 100.0
    df['T_out'] = df['T_out'].interpolate(limit_direction='both')
    if 'dewpoint' not in df or df['dewpoint'].isna().all():
        if 'RH' in df:
            w = ps.w_from_rh(df['T_out'].values, df['RH'].clip(1, 100).values, df['pressure'].values)
            df['dewpoint'] = ps.dewpoint_from_w(w, df['pressure'].values)
        else:
            df['dewpoint'] = df['T_out'] - 8.0
    df['dewpoint'] = np.minimum(df['dewpoint'].interpolate(limit_direction='both'), df['T_out'])
    if 'RH' not in df or df['RH'].isna().any():
        w = ps.w_from_dewpoint(df['dewpoint'].values, df['pressure'].values)
        df['RH'] = np.clip(ps.rh_from_w(df['T_out'].values, w, df['pressure'].values), 1, 100)
    for c in ('wind_speed', 'wind_dir'):
        if c not in df:
            df[c] = 3.0 if c == 'wind_speed' else 180.0
        df[c] = df[c].interpolate(limit_direction='both').fillna(3.0 if c == 'wind_speed' else 180.0)
    if 'GHI' not in df:
        raise ValueError('weather needs GHI [W/m2]')
    df['GHI'] = df['GHI'].interpolate(limit_direction='both').clip(lower=0)
    if df['T_out'].abs().max() > 80:
        raise ValueError('T_out looks like Kelvin or Fahrenheit; contract is deg C')
    return df


def _pad_cyclic(w, start, end):
    """A typical-year frame (one calendar year) is wrapped around its ends so runs can start/end anywhere in
    that year (e.g. a UTC year whose first local hours fall on the previous day)."""
    span = w.index[-1] - w.index[0]
    if not (pd.Timedelta(days=360) < span < pd.Timedelta(days=367)):
        return w
    if w.index[0] <= start and w.index[-1] >= end - pd.Timedelta(hours=1):
        return w
    yr = pd.DateOffset(years=1)
    head = w[w.index >= w.index[-1] - pd.Timedelta(days=3)].copy()
    head.index = head.index - yr
    tail = w[w.index <= w.index[0] + pd.Timedelta(days=3)].copy()
    tail.index = tail.index + yr
    out = pd.concat([head, w, tail])
    return out[~out.index.duplicated()].sort_index()


def _infer_step(idx):
    if len(idx) < 2:
        return pd.Timedelta('1h')
    return pd.Timedelta(np.median(np.diff(idx.asi8)), 'ns')


def resample(df, times, dt):
    """Weather on the simulation grid `times` (interval starts, step dt). Means for coarser steps, linear
    interpolation of interval centres (state-like variables) and step-hold (irradiance) for finer steps."""
    step_w = _infer_step(df.index)
    if step_w == dt and df.index.equals(times):
        return df.copy()
    if dt >= step_w:
        r = df.resample(dt, origin=times[0]).mean()
        r = r.reindex(times)
        return r.interpolate(limit_direction='both')
    # finer: centres
    tc_w = df.index.asi8 + step_w.value // 2
    tc_s = times.asi8 + dt.value // 2
    out = {}
    for c in df.columns:
        v = df[c].values.astype(float)
        if c in ('GHI', 'DNI', 'DHI', 'cloud'):
            j = np.clip(np.searchsorted(df.index.asi8, times.asi8, side='right') - 1, 0, len(v) - 1)
            out[c] = v[j]
        elif c == 'wind_dir':
            s, co = np.sin(np.radians(v)), np.cos(np.radians(v))
            out[c] = np.degrees(np.arctan2(np.interp(tc_s, tc_w, s), np.interp(tc_s, tc_w, co))) % 360
        else:
            out[c] = np.interp(tc_s, tc_w, v)
    return pd.DataFrame(out, index=times)


def kusuda_ground(times_doy, T_mean, amp, day_min, depth=1.0, alpha=0.0465):
    """Kusuda & Achenbach (1965) undisturbed ground temperature at depth [m]; alpha soil diffusivity m2/day."""
    d = depth * math.sqrt(math.pi / (365.0 * alpha))
    return T_mean - amp * math.exp(-d) * np.cos(2 * math.pi / 365.0 * (times_doy - day_min - depth / 2.0
                                                                        * math.sqrt(365.0 / (math.pi * alpha))))


def mains_water_temp(doy, T_annual_C, dT_monthly_C, lat):
    """Burch & Christensen (2007) mains water temperature correlation (as in EnergyPlus WaterMainsTemperature
    CORRELATION and ResStock/HPXML), deg C. doy: day of year array."""
    Tavg = T_annual_C * 9 / 5 + 32
    dTmax = dT_monthly_C * 9 / 5
    ratio = 0.4 + 0.01 * (Tavg - 44)
    lag = 35 - 1.0 * (Tavg - 44)
    shift = 0.0 if lat >= 0 else 182.5
    Tm = (Tavg + 6) + ratio * (dTmax / 2) * np.sin(np.radians(0.986 * (doy - 15 - lag - shift) - 90))
    return (np.maximum(Tm, 33.0) - 32) * 5 / 9


def sky_temperature(T, Td, cloud, ir_horiz=None):
    """Sky temperature [C]: from horizontal IR if available, else Clark & Allen (1978) clear-sky emissivity with
    the cloud correction used by EnergyPlus."""
    if ir_horiz is not None:
        ir = np.asarray(ir_horiz, float)
        ok = (ir > 50) & (ir < 700)
        if ok.mean() > 0.9:
            Ts = (np.where(ok, ir, np.nan) / SIGMA) ** 0.25 - 273.15
            return pd.Series(Ts).interpolate(limit_direction='both').values
    N = np.clip(np.asarray(cloud, float), 0, 1) * 10
    eps = (0.787 + 0.764 * np.log((np.asarray(Td) + 273.15) / 273.0)) * (1 + 0.0224 * N - 0.0035 * N ** 2 +
                                                                          0.00028 * N ** 3)
    eps = np.clip(eps, 0.5, 1.0)
    return (eps * (np.asarray(T) + 273.15) ** 4) ** 0.25 - 273.15


SURFACES = ('N', 'E', 'S', 'W')


def prepare(weather, meta, start=None, end=None, dt='1h', orientation=0.0, pv=None, albedo=0.2, cache=True,
            holidays=None):
    """Everything the time loop needs from weather, as numpy arrays on the simulation grid.

    orientation: azimuth rotation of the building (deg, clockwise): facade 'S' faces 180 + orientation.
    pv: list of (tilt, azimuth) pairs to transpose for PV arrays.
    holidays: None (v1.0: US calendar), or a country code from holidays.resolve ('CA', 'UK', 'IE'; opt-in).
    Returns dict with times (UTC), local calendar, T_out, W_out, RH, pressure, wind, T_wb, T_sky, T_ground,
    T_mains, solar per surface (beam/diffuse/ground and incidence angle), sun position.
    """
    dt = pd.Timedelta(dt)
    key = None
    if cache:
        # Every column is hashed. Until 27 Nov 2026 only T_out was, so two frames that differed in wind, humidity or
        # irradiance shared a cache entry (audit: a warm 2 m/s cache hid an 8 m/s run in FaultFinder and ThermoPilot).
        h = hashlib.md5()
        for c in sorted(map(str, weather.columns)):
            col = weather[c]
            h.update(c.encode())
            h.update(np.ascontiguousarray(col.values, dtype=float).tobytes() if np.issubdtype(col.dtype, np.number)
                     else str(col.tolist()).encode())
        h.update(str((len(weather), weather.index[0], weather.index[-1], tuple(sorted(meta.items())), start, end,
                      dt, orientation, None if pv is None else tuple(map(tuple, pv)), albedo)).encode())
        if holidays is not None:
            h.update(str(holidays).encode())
        key = h.hexdigest()
        if key in _CACHE:
            return _CACHE[key]
    w = check_weather(weather, meta)
    start = pd.Timestamp(start if start is not None else w.index[0])
    end = pd.Timestamp(end if end is not None else w.index[-1] + _infer_step(w.index))
    start = start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
    end = end.tz_localize('UTC') if end.tz is None else end.tz_convert('UTC')
    times = pd.date_range(start, end, freq=dt, inclusive='left')
    w = _pad_cyclic(w, start, end)
    wr = resample(w, times, dt)
    n = len(times)
    lat, lon, elev = float(meta['lat']), float(meta['lon']), float(meta.get('elev', 0.0) or 0.0)
    P = wr['pressure'].values
    T = wr['T_out'].values
    Td = np.minimum(wr['dewpoint'].values, T)
    W = ps.w_from_dewpoint(Td, P)
    RH = np.clip(ps.rh_from_w(T, W, P), 1, 100)
    # sun at interval midpoints
    tmid = times + dt / 2
    sp = pvlib.solarposition.get_solarposition(tmid, lat, lon, altitude=elev, method='nrel_numpy')
    zen = sp['apparent_zenith'].values
    azi = sp['azimuth'].values
    cosz = np.cos(np.radians(zen))
    ghi = wr['GHI'].values.copy()
    ghi[zen >= 90] = 0.0
    dni_extra = pvlib.irradiance.get_extra_radiation(tmid).values
    if 'DNI' in wr and 'DHI' in wr and np.isfinite(wr['DNI'].values).all() and np.isfinite(wr['DHI'].values).all() \
            and wr['DNI'].values.max() > 0:
        dni = np.clip(wr['DNI'].values, 0, None)
        dhi = np.clip(wr['DHI'].values, 0, None)
        dni[zen >= 90] = 0
        dhi[zen >= 90] = 0
        # step-hold of hourly values at sunrise/sunset can make beam on the horizontal exceed GHI; keep consistent
        dhi = np.minimum(dhi, ghi + 1e-9)
        dni = np.where(cosz > 0.02, np.minimum(dni, np.maximum(ghi - dhi, 0) / np.maximum(cosz, 0.02)), 0.0)
        ghi = dhi + dni * np.maximum(cosz, 0)
    else:
        e = pvlib.irradiance.erbs(ghi, zen, tmid)
        dni, dhi = np.nan_to_num(e['dni'].values), np.nan_to_num(e['dhi'].values)
    airmass = pvlib.atmosphere.get_relative_airmass(np.minimum(zen, 89.9))
    surf = {}

    def transpose(tilt, az):
        tot = pvlib.irradiance.get_total_irradiance(tilt, az, zen, azi, dni, ghi, dhi, dni_extra=dni_extra,
                                                    airmass=airmass, albedo=albedo, model='perez')
        aoi = pvlib.irradiance.aoi(tilt, az, zen, azi)
        beam = np.nan_to_num(np.asarray(tot['poa_direct'], float))
        sky = np.nan_to_num(np.asarray(tot['poa_sky_diffuse'], float))
        grd = np.nan_to_num(np.asarray(tot['poa_ground_diffuse'], float))
        return dict(beam=np.clip(beam, 0, None), sky=np.clip(sky, 0, None), ground=np.clip(grd, 0, None),
                    aoi=np.asarray(aoi), tilt=tilt, az=az)

    for k, base in zip(SURFACES, (0.0, 90.0, 180.0, 270.0)):
        surf[k] = transpose(90.0, (base + orientation) % 360)
    surf['H'] = dict(beam=dni * np.maximum(cosz, 0), sky=dhi, ground=np.zeros(n), aoi=zen, tilt=0.0, az=180.0)
    pvs = []
    for (tilt, az) in (pv or []):
        pvs.append(transpose(float(tilt), float(az)))
    # calendar in local civil time
    tz = meta.get('tz', 'UTC')
    loc = times.tz_convert(tz)
    cal = _calendar(loc, holidays)
    # climate summaries from the whole weather frame (not just the run window)
    wd = w['T_out']
    T_ann = float(wd.mean())
    mon = wd.groupby(wd.index.month).mean()
    amp = float((mon.max() - mon.min()) / 2) if len(mon) > 1 else 10.0
    day_min = float(wd.resample('D').mean().rolling(15, center=True, min_periods=1).mean().idxmin().dayofyear) \
        if len(wd) > 48 else 15.0
    doy = cal['doy'] + cal['hour'] / 24.0
    T_g = kusuda_ground(doy, T_ann, amp, day_min, depth=1.0)
    T_g_deep = kusuda_ground(doy, T_ann, amp, day_min, depth=3.0)
    T_mains = mains_water_temp(doy, T_ann, 2 * amp, lat)
    T_sky = sky_temperature(T, Td, wr['cloud'].values if 'cloud' in wr else _cloud_from_kt(ghi, zen, dni_extra, times),
                            wr['IR_horiz'].values if 'IR_horiz' in wr else None)
    T_wb = ps.wetbulb(T, W, P)
    hdd = float(np.clip(18.0 - wd.resample('D').mean(), 0, None).sum())
    cdd = float(np.clip(wd.resample('D').mean() - 18.0, 0, None).sum())
    out = dict(times=times, dt_s=dt.total_seconds(), n=n, lat=lat, lon=lon, elev=elev, tz=tz,
               T_out=T, W_out=W, RH=RH, dewpoint=Td, P=P, wind=wr['wind_speed'].values.clip(0, 40),
               wind_dir=wr['wind_dir'].values, T_wb=T_wb, T_sky=T_sky, T_ground=T_g, T_ground_deep=T_g_deep,
               T_mains=T_mains, ghi=ghi, dni=dni, dhi=dhi, zenith=zen, azimuth=azi, surf=surf, pv=pvs, cal=cal,
               climate=dict(T_annual=T_ann, T_amp=amp, HDD18=hdd, CDD18=cdd,
                            T_heat_design=float(np.percentile(wd.values, 0.4)),
                            T_cool_design=float(np.percentile(wd.values, 99.6)),
                            T_wb_cool_design=float(np.percentile(ps.wetbulb(wd.values, ps.w_from_dewpoint(
                                np.minimum(w['dewpoint'].values, wd.values), w['pressure'].values),
                                w['pressure'].values), 99.6))))
    if key is not None:
        if CACHE_SIZE[0] > 0:
            if len(_CACHE) > CACHE_SIZE[0]:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = out
    return out


_CACHE = {}
CACHE_SIZE = [24]      # v1.0 keeps up to 25 prepared frames (evicts the oldest past 24); set_cache_size() bounds it


def set_cache_size(n=24):
    """Bounded-memory mode for long bank runs (LoadCast's processes ran out of memory; it restarted them every
    LC_MAXPARTS parts). n = most prepared weather frames kept beyond the newest (v1.0: 24); 0 = no cache. A frame
    of one year is 3.0 MB at 1 h and 11.8 MB at 15 min (Chicago TMYx), so the v1.0 cache holds at most ~75 / ~300 MB;
    the rest of LoadCast's growth was not traced. Returns the old value."""
    old = CACHE_SIZE[0]
    CACHE_SIZE[0] = max(int(n), 0)
    while len(_CACHE) > max(CACHE_SIZE[0], 0) + (1 if CACHE_SIZE[0] > 0 else 0):
        _CACHE.pop(next(iter(_CACHE)))
    return old


def clear_cache():
    _CACHE.clear()


def _cloud_from_kt(ghi, zen, dni_extra, times):
    """Cloud fraction from the daily clearness index when the weather has no cloud column (used only for sky
    temperature): kt_day = sum GHI / sum clear-sky-ish (0.75 x extraterrestrial on the horizontal)."""
    cosz = np.maximum(np.cos(np.radians(zen)), 0)
    ext = 0.75 * dni_extra * cosz
    d = pd.Series(ghi, index=times).resample('D').sum() / pd.Series(ext, index=times).resample('D').sum().clip(1)
    kt = d.reindex(times.floor('D')).values
    return np.clip(1.0 - np.nan_to_num(kt, nan=0.5), 0, 1)


US_HOL_FIXED = [(1, 1), (7, 4), (12, 25), (12, 24), (12, 31)]


def _holidays(d):
    hol = np.zeros(len(d), bool)
    for m, dd in US_HOL_FIXED:
        hol |= (d.month == m) & (d.day == dd)
    dow, day = d.dayofweek, d.day
    hol |= (d.month == 1) & (dow == 0) & (day >= 15) & (day <= 21)
    hol |= (d.month == 2) & (dow == 0) & (day >= 15) & (day <= 21)
    hol |= (d.month == 5) & (dow == 0) & (day > 24)
    hol |= (d.month == 9) & (dow == 0) & (day <= 7)
    hol |= (d.month == 11) & (((dow == 3) & (day >= 22) & (day <= 28)) | ((dow == 4) & (day >= 23) & (day <= 29)))
    return hol


def _calendar(loc, country=None):
    """Local civil calendar: hour (float), daytype 0 weekday / 1 Saturday / 2 Sunday-holiday, doy, month.
    country: None = the v1.0 US calendar; 'CA' / 'UK' / 'IE' = buildingsim.holidays (opt-in)."""
    if country is None:
        hol = _holidays(loc.normalize().tz_localize(None))
    else:
        from .holidays import is_holiday
        hol = is_holiday(loc.normalize().tz_localize(None), country)
    dtype = np.where(hol | (loc.dayofweek == 6), 2, np.where(loc.dayofweek == 5, 1, 0))
    return dict(hour=(loc.hour + loc.minute / 60.0).values.astype(float), daytype=dtype,
                doy=loc.dayofyear.values.astype(float), month=loc.month.values, dow=loc.dayofweek.values,
                holiday=hol)


def epw_path(city):
    """Path of one of the bundled TMYx files by a substring of its name (e.g. 'Denver', 'Miami')."""
    import glob
    import os
    d = os.path.join(os.environ.get('ML_RATCHET_BUILDING') or os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'buildingsim', 'refdata', 'weather', 'epw')   # $ML_RATCHET_BUILDING or shared/building
    m = sorted(glob.glob(os.path.join(d, f'*{city}*.epw')))
    if not m:
        raise FileNotFoundError(city)
    return m[0]


# IECC climate zone of the bundled TMYx cities (DOE prototype representative cities)
EPW_CLIMATE_ZONE = {'Miami': '1A', 'Houston': '2A', 'Phoenix': '2B', 'Atlanta': '3A', 'El.Paso': '3B',
                    'Los.Angeles': '3B', 'San.Francisco': '3C', 'Baltimore': '4A', 'New.York': '4A',
                    'Seattle': '4C', 'Chicago': '5A', 'Denver': '5B', 'Boulder': '5B', 'Minneapolis': '6A',
                    'Helena': '6B', 'Duluth': '7', 'Fairbanks': '8'}
