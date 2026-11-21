"""Weather in the shared contract, wind conversions, solar radiation, daily-to-hourly disaggregation and a stochastic
hourly fire-weather generator with a diurnal cycle (overnight RH recovery, afternoon wind peak, night slow-down).

Contract columns (hourly, UTC DatetimeIndex): T (C), RH (%), wind_speed (m/s at 10 m), wind_dir (deg, FROM),
precip (mm per hour), srad (W/m2, optional), cloud (0-1, optional).

Wind heights: the contract wind is at 10 m. Rothermel needs the 20-ft (6.1 m) wind above the vegetation, then a
wind adjustment factor to midflame. firesim uses U20ft = U10m / 1.15 (Turner & Lawson 1978; Lawson & Armitage 2008),
the standard conversion in US fire behaviour practice.
"""
import numpy as np
import pandas as pd

U10_TO_U20 = 1.0 / 1.15


def solar_elevation(index_utc, lat, lon):
    """Solar elevation (radians) for a UTC DatetimeIndex (Spencer 1971 declination and equation of time)."""
    idx = pd.DatetimeIndex(index_utc)
    doy = idx.dayofyear.to_numpy()
    hr = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    tst = hr * 60.0 + eot + 4.0 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    s = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.arcsin(np.clip(s, -1, 1))


def clear_sky_ghi(index_utc, lat, lon):
    """Clear-sky global horizontal irradiance (W/m2), Kasten & Czeplak (1980): 910 sin(h) - 30, at the hour's middle."""
    idx = pd.DatetimeIndex(index_utc) + pd.Timedelta(minutes=30)
    h = solar_elevation(idx, lat, lon)
    return np.maximum(910.0 * np.sin(h) - 30.0, 0.0)


def ensure_srad(w, lat=None, lon=None):
    """Return a copy with 'srad'; estimated from clear sky and 'cloud' (Kasten & Czeplak cloud factor) if missing."""
    w = w.copy()
    if 'srad' in w and w['srad'].notna().all():
        return w
    lat = lat if lat is not None else w.attrs.get('lat', 40.0)
    lon = lon if lon is not None else w.attrs.get('lon', -115.0)
    cs = clear_sky_ghi(w.index, lat, lon)
    c = w['cloud'].fillna(0.3).to_numpy(float) if 'cloud' in w else np.full(len(w), 0.2)
    est = cs * (1.0 - 0.75 * np.clip(c, 0, 1) ** 3.4)
    if 'srad' in w:
        w['srad'] = w['srad'].fillna(pd.Series(est, index=w.index))
    else:
        w['srad'] = est
    return w


def check_weather(w):
    """Validate a contract weather frame; returns a cleaned hourly copy (UTC index, filled gaps)."""
    if not isinstance(w, pd.DataFrame):
        raise TypeError('weather must be a pandas DataFrame with a UTC DatetimeIndex')
    need = ['T', 'RH', 'wind_speed', 'wind_dir']
    miss = [c for c in need if c not in w]
    if miss:
        raise ValueError(f'weather lacks columns {miss}')
    w = w.copy()
    idx = pd.DatetimeIndex(w.index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    w.index = idx.tz_convert('UTC')
    w = w[~w.index.duplicated()].sort_index()
    if 'precip' not in w:
        w['precip'] = 0.0
    full = pd.date_range(w.index[0].floor('h'), w.index[-1].ceil('h'), freq='h')
    w = w.reindex(w.index.union(full))
    # wind direction interpolated as vector
    u = -w['wind_speed'] * np.sin(np.radians(w['wind_dir']))
    v = -w['wind_speed'] * np.cos(np.radians(w['wind_dir']))
    w = w.interpolate(method='time', limit_direction='both')
    u = u.interpolate(method='time', limit_direction='both')
    v = v.interpolate(method='time', limit_direction='both')
    w['wind_dir'] = (np.degrees(np.arctan2(-u, -v)) % 360.0)
    w = w.reindex(full)
    w['precip'] = w['precip'].fillna(0.0).clip(lower=0)
    w['RH'] = w['RH'].clip(1, 100)
    w['wind_speed'] = w['wind_speed'].clip(lower=0)
    w.attrs.update(getattr(w, 'attrs', {}))
    return w


def hourly_from_daily(daily, lat, lon, t_peak_lst=15.0, wind_amp=0.45, rh_from='minmax'):
    """Disaggregate daily weather (gridMET-like) to hourly in the contract.

    daily columns (any of): tmin, tmax (C) | tmmn, tmmx (K); rmin, rmax (%); vs or wind_speed (m/s at 10 m, daily mean);
    th or wind_dir (deg from); pr or precip (mm/day); srad (W/m2 daily mean, optional).
    Temperature: sinusoid between Tmin at sunrise and Tmax at t_peak (LST), exponential night cooling (Cesaraccio
    et al. 2001 style). RH: daily min at the Tmax time and max at the Tmin time, via vapour pressure held constant
    between the two (overnight recovery). Wind: daily mean times 1 + wind_amp cos(2 pi (t - 15)/24) (afternoon
    peak, night slow-down). Precipitation spread evenly over 14-20 LST.
    """
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    tmin = d['tmin'] if 'tmin' in d else d['tmmn'] - 273.15
    tmax = d['tmax'] if 'tmax' in d else d['tmmx'] - 273.15
    rmin = d['rmin'] if 'rmin' in d else pd.Series(20.0, index=d.index)
    rmax = d['rmax'] if 'rmax' in d else pd.Series(80.0, index=d.index)
    ws = d['vs'] if 'vs' in d else d.get('wind_speed', pd.Series(3.0, index=d.index))
    wd = d['th'] if 'th' in d else d.get('wind_dir', pd.Series(225.0, index=d.index))
    pr = d['pr'] if 'pr' in d else d.get('precip', pd.Series(0.0, index=d.index))
    off = lon / 15.0
    rows = []
    for i, day in enumerate(d.index):
        tn, tx = float(tmin.iloc[i]), float(tmax.iloc[i])
        tx_prev = float(tmax.iloc[i - 1]) if i > 0 else tx
        tn_next = float(tmin.iloc[i + 1]) if i + 1 < len(d) else tn
        for h in range(24):
            lst = h + 0.5
            if lst < 6.0:            # before sunrise: yesterday's evening decay toward today's Tmin
                T = tn + (tx_prev - tn) * np.exp(-(lst + 24.0 - t_peak_lst) / 4.0)
            elif lst <= t_peak_lst:  # morning rise to the afternoon peak
                T = tn + (tx - tn) * np.sin(np.pi / 2 * (lst - 6.0) / (t_peak_lst - 6.0))
            else:                    # evening decay toward tomorrow's Tmin
                T = tn_next + (tx - tn_next) * np.exp(-(lst - t_peak_lst) / 4.0)
            # RH: interpolate between rmax (at Tmin) and rmin (at Tmax) through the temperature
            frac = np.clip((T - tn) / max(tx - tn, 0.1), 0, 1)
            rh = float(rmax.iloc[i]) + (float(rmin.iloc[i]) - float(rmax.iloc[i])) * frac
            u = float(ws.iloc[i]) * (1.0 + wind_amp * np.cos(2 * np.pi * (lst - 15.0) / 24.0))
            p = float(pr.iloc[i]) / 6.0 if 14 <= lst < 20 else 0.0
            t_utc = pd.Timestamp(day) + pd.Timedelta(hours=h - off)
            rows.append((t_utc, T, rh, max(u, 0.0), float(wd.iloc[i]), p))
    out = pd.DataFrame(rows, columns=['time', 'T', 'RH', 'wind_speed', 'wind_dir', 'precip']).set_index('time')
    # Round to the hour with ties going up: for longitudes whose solar offset ends in half an hour (lon = -112.5),
    # pandas' round-half-to-even sent alternate hours onto the same timestamp and the dedup below dropped half the
    # series (PR #7 review, 21 Nov 2026). Other longitudes are unchanged.
    out.index = (pd.DatetimeIndex(out.index).tz_localize('UTC').round('s') + pd.Timedelta(seconds=1)).round('h')
    out = out[~out.index.duplicated()]
    if 'srad' in d:
        cs = clear_sky_ghi(out.index, lat, lon)
        dm = pd.Series(cs, index=out.index).resample('D').mean()
        ratio = (d['srad'].reindex(dm.index.tz_localize(None).normalize()).to_numpy() /
                 np.maximum(dm.to_numpy(), 1.0))
        r = pd.Series(np.clip(ratio, 0, 1.2), index=dm.index).reindex(out.index, method='ffill')
        out['srad'] = cs * r.to_numpy()
    out.attrs['lat'] = lat
    out.attrs['lon'] = lon
    return out


# ------------------------------------------------------------------------------------------------ generator
CLIMATES = {
    # mean Tmax/Tmin (C), diurnal amplitude, mean dewpoint depression in afternoon, wind mean (m/s at 10 m),
    # prevailing wind (deg from), wet-day probability, wet-day mean (mm), dry-wind event prob per day
    'west_dry_summer': dict(tmax=31.0, tmin=12.0, td=0.0, wind=3.5, wind_dir=230.0, p_wet=0.06, wet_mm=4.0,
                            p_event=0.06, rh_min=15.0, rh_max=65.0),
    'california_autumn': dict(tmax=27.0, tmin=10.0, td=-2.0, wind=3.0, wind_dir=250.0, p_wet=0.04, wet_mm=6.0,
                              p_event=0.12, rh_min=18.0, rh_max=70.0, event_dir=45.0),
    'southwest_spring': dict(tmax=26.0, tmin=6.0, td=-8.0, wind=4.5, wind_dir=240.0, p_wet=0.05, wet_mm=3.0,
                             p_event=0.10, rh_min=8.0, rh_max=45.0),
    'northern_rockies_summer': dict(tmax=28.0, tmin=8.0, td=-2.0, wind=3.0, wind_dir=240.0, p_wet=0.12,
                                    wet_mm=4.0, p_event=0.05, rh_min=18.0, rh_max=75.0),
    'southeast_spring': dict(tmax=25.0, tmin=10.0, td=5.0, wind=3.5, wind_dir=220.0, p_wet=0.25, wet_mm=10.0,
                             p_event=0.04, rh_min=30.0, rh_max=95.0),
}


def generate_weather(start, hours=72, climate='west_dry_summer', lat=40.0, lon=-115.0, rng=None, kind='parametric',
                     **over):
    """Stochastic hourly fire weather in the contract format.

    Daily synoptic anomalies (AR(1), lag-1 0.7) drive Tmax/Tmin, afternoon RH and wind. Within a day: temperature
    and RH follow the diurnal shapes of hourly_from_daily (RH recovers overnight), wind peaks in the afternoon and
    slows at night, and its direction swings by up to +-40 degrees (diurnal upslope/downslope). Dry-wind events
    (foehn/Santa Ana-like: strong wind from `event_dir`, very low RH, poor overnight recovery) occur with
    probability p_event per day and last 1-3 days. Wet days follow a Markov chain; rain falls in the afternoon.
    Returns a DataFrame; df.attrs holds the climate, the event days and the latent anomalies (labels).
    kind='real': real HRRR hours from B's dev-fire packages instead (realdata.random_weather; exam-guarded).
    """
    if kind == 'real':
        from . import realdata
        return realdata.random_weather(start=start, hours=hours, rng=rng, **over)
    rng = np.random.default_rng(rng)
    c = dict(CLIMATES[climate] if isinstance(climate, str) else climate)
    c.update(over)
    start = pd.Timestamp(start)
    if start.tz is None:
        start = start.tz_localize('UTC')
    ndays = int(np.ceil(hours / 24.0)) + 2
    a = np.zeros((ndays, 3))
    for i in range(1, ndays):
        a[i] = 0.7 * a[i - 1] + np.sqrt(1 - 0.49) * rng.standard_normal(3)
    event = np.zeros(ndays, bool)
    i = 0
    while i < ndays:
        if rng.random() < c.get('p_event', 0.0):
            L = rng.integers(1, 4)
            event[i:i + L] = True
            i += L
        i += 1
    wet = np.zeros(ndays, bool)
    for i in range(ndays):
        p = c['p_wet'] * (2.5 if i > 0 and wet[i - 1] else 1.0)
        wet[i] = (rng.random() < p) and not event[i]
    tmax = c['tmax'] + 3.0 * a[:, 0] + np.where(event, 2.0, 0.0) - np.where(wet, 5.0, 0.0)
    tmin = c['tmin'] + 2.5 * a[:, 0] + np.where(event, 4.0, 0.0) - np.where(wet, 1.0, 0.0)
    rmin = np.clip(c['rh_min'] * np.exp(0.35 * a[:, 1]) * np.where(event, 0.45, 1.0) + np.where(wet, 35.0, 0.0),
                   3, 95)
    rmax = np.clip(c['rh_max'] * np.exp(0.15 * a[:, 1]) * np.where(event, 0.45, 1.0) + np.where(wet, 25.0, 0.0),
                   rmin + 5, 100)
    wind = c['wind'] * np.exp(0.35 * a[:, 2]) * np.where(event, 2.6, 1.0)
    wdir = np.where(event, c.get('event_dir', (c['wind_dir'] + 180.0) % 360.0),
                    c['wind_dir'] + 25.0 * rng.standard_normal(ndays)) % 360.0
    pr = np.where(wet, rng.exponential(c['wet_mm'], ndays), 0.0)
    day0 = (start - pd.Timedelta(days=1)).tz_convert(None).normalize()
    days = pd.date_range(day0, periods=ndays, freq='D')
    daily = pd.DataFrame(dict(tmin=tmin, tmax=np.maximum(tmax, tmin + 3), rmin=rmin, rmax=rmax, vs=wind, th=wdir,
                              pr=pr), index=days)
    h = hourly_from_daily(daily, lat, lon, wind_amp=0.5)
    # events: weaker night slow-down (downslope winds persist) and diurnal direction swing otherwise
    lst = ((h.index.hour + lon / 15.0) % 24).to_numpy()
    dnorm = h.index.tz_convert(None).normalize()
    ev_h = pd.Series(event, index=days).reindex(dnorm, method='ffill').to_numpy().astype(bool)
    swing = 30.0 * np.sin(2 * np.pi * (lst - 9.0) / 24.0) * (~ev_h)
    h['wind_dir'] = (h['wind_dir'] + swing) % 360.0
    gust = np.exp(0.15 * rng.standard_normal(len(h)))
    h['wind_speed'] = np.where(ev_h, h['wind_speed'] * (0.8 + 0.2 * np.cos(2 * np.pi * (lst - 15) / 24)) / (
        1 + 0.5 * np.cos(2 * np.pi * (lst - 15) / 24)), h['wind_speed']) * gust
    h['cloud'] = np.clip(0.15 + 0.6 * pd.Series(wet, index=days).reindex(dnorm, method='ffill').to_numpy(), 0, 1)
    # keep the hour that contains a fractional start (PR #7 review, 21 Nov 2026); on-the-hour starts are unchanged
    h = h[(h.index >= start.floor('h')) & (h.index < start + pd.Timedelta(hours=hours))]
    h = ensure_srad(h, lat, lon)
    h.attrs = dict(lat=lat, lon=lon, climate=c, event_days=[str(d.date()) for d in days[event]],
                   wet_days=[str(d.date()) for d in days[wet]])
    return h
