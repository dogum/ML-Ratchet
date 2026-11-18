"""Stochastic daily weather generator (Richardson 1981 / WGEN family) for variety beyond the real forcing.

Precipitation occurrence: first-order two-state Markov chain with seasonally varying wet-day probability
pi(doy) and persistence d = p11 - p01 (p01 = pi (1 - d), p11 = pi (1 - d) + d).
Wet-day amounts: 0.1 mm + gamma with shape k (0.5-0.9, Wilks 1999), mean m(doy) = P(doy) / pi(doy); optional
year-to-year multiplicative factor exp(N(0, annual_sd)) for the inter-annual variability daily WGENs miss.
fit_climate() fits all of this to a real forcing frame.
Seasonal precipitation: P(doy) = P_mean (1 + A cos(2 pi (doy - phase)/365.25)), A in [-0.9, 0.9].
Temperature: T(doy) = T_ann + T_amp cos(2 pi (doy - doy_warm)/365.25) + AR(1) anomaly (rho 0.7, sigma larger in
winter), wet-day anomaly -dT_wet; diurnal range DTR shrinks on wet days.
Shortwave: clear-sky Rso (FAO-56) x transmissivity (dry ~0.72, wet ~0.40, + noise).
Humidity: dew point = T_min - dew depression (grows with aridity), vp = e_sat(T_dew).
Wind: lognormal AR(1) around a basin mean (m/s at 10 m).
"""
import numpy as np
import pandas as pd

from .pet import esat_kpa, clear_sky_radiation


def default_climate(spec, rng=None):
    """Climate parameters for a spec: from spec['climate'] when given, else derived from lat/elevation/p_mean."""
    c = dict(spec.get('climate') or {})
    lat = float(spec.get('lat', 40.0))
    z = float(spec.get('elev_mean_m', 500.0))
    rng = np.random.default_rng(rng)
    c.setdefault('T_ann', 27.5 - 0.0085 * lat ** 2 - 6.0 * z / 1000.0)
    c.setdefault('T_amp', 4.0 + 0.30 * abs(lat))
    c.setdefault('P_mean', float(spec.get('p_mean') or 2.8))
    c.setdefault('P_amp', 0.2)
    c.setdefault('P_phase_doy', 15.0 if lat >= 0 else 196.0)
    c.setdefault('wet_frac', float(np.clip(0.18 + 0.07 * c['P_mean'], 0.12, 0.7)))
    c.setdefault('persistence', 0.35)
    c.setdefault('gamma_shape', 0.7)
    c.setdefault('dtr', 11.0)
    c.setdefault('dew_depression', 3.0)
    c.setdefault('wind_mean', 3.0)
    c.setdefault('T_sigma', 2.5)
    return c


def generate_weather(spec, start='2000-01-01', end='2009-12-31', rng=None, climate=None):
    """Daily forcing DataFrame (contract columns, UTC daily index) from the stochastic generator."""
    rng = np.random.default_rng(rng)
    c = default_climate({**spec, 'climate': climate or spec.get('climate')}, rng)
    idx = pd.date_range(start, end, freq='D')
    n = len(idx)
    doy = idx.dayofyear.values.astype(float)
    lat = float(spec.get('lat', 40.0))
    z = float(spec.get('elev_mean_m', 500.0))
    w = 2 * np.pi / 365.25
    warm = c.get('T_warm_doy', 200.0 if lat >= 0 else 17.0)
    seas_p = np.clip(1.0 + c['P_amp'] * np.cos(w * (doy - c['P_phase_doy'])), 0.05, None)
    pi = np.clip(c['wet_frac'] * (0.6 + 0.4 * seas_p), 0.02, 0.9)
    d = c['persistence']
    p01 = pi * (1 - d)
    p11 = p01 + d
    wet = np.zeros(n, bool)
    u = rng.random(n)
    state = rng.random() < pi[0]
    for t in range(n):
        state = u[t] < (p11[t] if state else p01[t])
        wet[t] = state
    mean_amt = c['P_mean'] * seas_p / pi
    k = c['gamma_shape']
    ex = np.maximum(mean_amt - 0.1, 0.01)          # wet days carry at least 0.1 mm (the wet-day threshold)
    P = np.where(wet, 0.1 + rng.gamma(k, ex / k), 0.0)
    sa = c.get('annual_sd', 0.0)                   # inter-annual variability beyond the daily process
    if sa > 0:
        yrs = idx.year.values
        fac = {y: np.exp(rng.normal(0, sa) - 0.5 * sa ** 2) for y in np.unique(yrs)}
        P = P * np.array([fac[y] for y in yrs])
    T_clim = c['T_ann'] + c['T_amp'] * np.cos(w * (doy - warm))
    sig = c['T_sigma'] * (1.0 + 0.4 * np.cos(w * (doy - warm - 182.6))) / np.sqrt(1 + 0.4 ** 2 / 2)
    eps = rng.normal(size=n)
    an = np.zeros(n)
    for t in range(1, n):
        an[t] = 0.7 * an[t - 1] + np.sqrt(1 - 0.49) * eps[t]
    T = T_clim + sig * an - 1.0 * wet * (T_clim > 5)
    dtr = np.clip(c['dtr'] * (1 - 0.35 * wet) + rng.normal(0, 1.5, n), 2.0, 25.0)
    Tmax = T + dtr / 2
    Tmin = T - dtr / 2
    rso = clear_sky_radiation(lat, doy, z) / 0.0864  # W/m2
    tau = np.where(wet, 0.42, 0.74) + rng.normal(0, 0.06, n)
    srad = rso * np.clip(tau, 0.1, 0.8) / 0.75
    tdew = Tmin - np.clip(c['dew_depression'] * (1 - 0.6 * wet) + rng.normal(0, 1.0, n), 0, None)
    vp = esat_kpa(tdew) * 1000.0
    lw = np.zeros(n)
    e2 = rng.normal(size=n)
    for t in range(1, n):
        lw[t] = 0.6 * lw[t - 1] + 0.8 * e2[t]
    wind = c['wind_mean'] * np.exp(0.35 * lw - 0.06)
    df = pd.DataFrame({'P': P, 'T_mean': T, 'T_min': Tmin, 'T_max': Tmax, 'SRAD': srad, 'vp': vp, 'wind': wind},
                      index=idx)
    df.index.name = 'date'
    df.attrs['meta'] = {'source': 'watersim.weather (stochastic)', 'climate': c}
    return df


def fit_climate(df, lat=None):
    """Weather-generator parameters fitted to a real daily forcing frame (method of moments and first harmonics), so
    generate_weather(spec, climate=fit_climate(real)) makes stochastic variants of a real basin's climate."""
    d = df.copy()
    doy = d.index.dayofyear.values.astype(float)
    w = 2 * np.pi / 365.25
    P = d['P'].values.astype(float)
    wet = P > 0.1
    c = {}
    c['P_mean'] = float(np.nanmean(P))
    c['wet_frac'] = float(np.mean(wet))
    p01 = np.mean(wet[1:][~wet[:-1]]) if (~wet[:-1]).any() else 0.3
    p11 = np.mean(wet[1:][wet[:-1]]) if wet[:-1].any() else 0.5
    c['persistence'] = float(np.clip(p11 - p01, 0.0, 0.8))
    amt = P[wet] - 0.1
    c['gamma_shape'] = float(np.clip(amt.mean() ** 2 / max(amt.var(), 1e-9), 0.3, 1.5)) if len(amt) > 10 else 0.7
    ann = d['P'].groupby(d.index.year).agg(['sum', 'count'])
    ann = ann[ann['count'] > 360]['sum']
    if len(ann) >= 4:
        cv_real = ann.std() / ann.mean()
        cv_daily = np.sqrt(np.nanvar(P) / 365.25) / max(np.nanmean(P), 1e-9)
        c['annual_sd'] = float(np.sqrt(max(cv_real ** 2 - cv_daily ** 2, 0.0)))
    X = np.column_stack([np.ones(len(d)), np.cos(w * doy), np.sin(w * doy)])
    b, *_ = np.linalg.lstsq(X, P, rcond=None)
    amp = np.hypot(b[1], b[2])
    c['P_amp'] = float(np.clip(amp / max(b[0], 1e-9), 0.0, 0.9))
    c['P_phase_doy'] = float((np.arctan2(b[2], b[1]) / w) % 365.25)
    T = d['T_mean'].interpolate(limit_direction='both').values
    bt, *_ = np.linalg.lstsq(X, T, rcond=None)
    c['T_ann'] = float(bt[0])
    c['T_amp'] = float(np.hypot(bt[1], bt[2]))
    c['T_warm_doy'] = float((np.arctan2(bt[2], bt[1]) / w) % 365.25)
    c['T_sigma'] = float(np.std(T - X @ bt))
    if 'T_max' in d and 'T_min' in d:
        c['dtr'] = float(np.nanmean(d['T_max'] - d['T_min']))
    if 'vp' in d and d['vp'].notna().any() and 'T_min' in d:
        e = np.clip(d['vp'].values / 1000.0, 1e-3, None)
        tdew = 237.3 * np.log(e / 0.6108) / (17.27 - np.log(e / 0.6108))
        c['dew_depression'] = float(np.clip(np.nanmean(d['T_min'].values - tdew), 0.0, 15.0))
    if 'wind' in d and d['wind'].notna().any():
        c['wind_mean'] = float(np.nanmean(d['wind']))
    return c
