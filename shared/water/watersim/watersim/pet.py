"""Potential evapotranspiration from daily forcing.

Methods (all return mm/day, numpy arrays):
  penman_monteith   FAO-56 grass reference ET0 (Allen et al. 1998, eq. 6), full radiation budget.
  priestley_taylor  Priestley & Taylor (1972), alpha = 1.26, net radiation as in FAO-56.
  hargreaves        Hargreaves & Samani (1985), temperature + extraterrestrial radiation only (fallback).
  oudin             Oudin et al. (2005), temperature + extraterrestrial radiation (used with GR models).
  penman_open_water Penman (1948) open-water evaporation, Shuttleworth (1993) wind function (reservoirs).

Solar geometry, vapour pressure and net long-wave radiation follow FAO-56 chapter 3.
Units in: T in C, SRAD in W/m2 (daily mean), vp in Pa, wind in m/s at `wind_height` m, elevation in m, lat in degrees.
"""
import numpy as np
import pandas as pd

from .constants import GSC, SIGMA_MJ_DAY, W_TO_MJ_DAY

LAMBDA_FAO = 2.45  # MJ/kg used by FAO-56 for the mm conversion (1/lambda = 0.408)


def esat_kpa(T):
    """Saturation vapour pressure over water (kPa), FAO-56 eq. 11 (Tetens form)."""
    T = np.asarray(T, float)
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def esat_ice_kpa(T):
    """Saturation vapour pressure over ice (kPa), Buck (1981)."""
    T = np.asarray(T, float)
    return 0.61115 * np.exp(22.452 * T / (T + 272.55))


def slope_svp(T):
    """Slope of the saturation vapour pressure curve (kPa/K), FAO-56 eq. 13."""
    T = np.asarray(T, float)
    return 4098.0 * esat_kpa(T) / (T + 237.3) ** 2


def pressure_kpa(z):
    """Atmospheric pressure from elevation (kPa), FAO-56 eq. 7."""
    return 101.3 * ((293.0 - 0.0065 * np.asarray(z, float)) / 293.0) ** 5.26


def psychrometric(z):
    return 0.665e-3 * pressure_kpa(z)


def extraterrestrial_radiation(lat_deg, doy):
    """Daily extraterrestrial radiation Ra (MJ/m2/day), FAO-56 eq. 21; doy = day of year (1..366)."""
    phi = np.deg2rad(lat_deg)
    J = np.asarray(doy, float)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * J / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * J / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    ws = np.arccos(x)
    ra = 24 * 60 / np.pi * GSC * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return np.maximum(ra, 0.0)


def clear_sky_radiation(lat_deg, doy, z):
    """Clear-sky shortwave Rso (MJ/m2/day), FAO-56 eq. 37."""
    return (0.75 + 2e-5 * z) * extraterrestrial_radiation(lat_deg, doy)


def wind_2m(u, height=10.0):
    """Wind at 2 m from wind at `height` m over grass, FAO-56 eq. 47."""
    if height == 2.0:
        return np.asarray(u, float)
    return np.asarray(u, float) * 4.87 / np.log(67.8 * height - 5.42)


def _vp_kpa(T_min, vp):
    if vp is None:
        return esat_kpa(T_min)  # FAO-56 eq. 48: dew point ~ T_min
    v = np.asarray(vp, float) / 1000.0
    return np.where(np.isfinite(v), v, esat_kpa(T_min))


def _rs_mj(srad, lat, doy, z, T_max, T_min, krs=0.16):
    ra = extraterrestrial_radiation(lat, doy)
    est = krs * np.sqrt(np.maximum(T_max - T_min, 0.0)) * ra  # Hargreaves radiation formula, FAO-56 eq. 50
    if srad is None:
        return est
    rs = np.asarray(srad, float) * W_TO_MJ_DAY
    return np.where(np.isfinite(rs), rs, est)


def net_radiation(T_max, T_min, srad, vp, lat, doy, z, albedo=0.23):
    """Net radiation Rn (MJ/m2/day), FAO-56 eqs. 38-40."""
    T_max = np.asarray(T_max, float)
    T_min = np.asarray(T_min, float)
    rs = _rs_mj(srad, lat, doy, z, T_max, T_min)
    rso = clear_sky_radiation(lat, doy, z)
    ea = _vp_kpa(T_min, vp)
    ratio = np.clip(np.where(rso > 0, rs / np.maximum(rso, 1e-9), 0.5), 0.25, 1.0)
    rnl = SIGMA_MJ_DAY * ((T_max + 273.16) ** 4 + (T_min + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) \
        * (1.35 * ratio - 0.35)
    return (1 - albedo) * rs - rnl


def penman_monteith(T_mean, T_max, T_min, srad, vp, wind, lat, doy, z, wind_height=10.0, rn=None):
    """FAO-56 reference crop ET0 (mm/day), eq. 6, soil heat flux G = 0 at the daily step.
    rn: optional net radiation (MJ/m2/day) to use instead of the FAO-56 estimate (e.g. from reanalysis)."""
    T = np.asarray(T_mean, float)
    T_max = np.asarray(T_max, float)
    T_min = np.asarray(T_min, float)
    if rn is None:
        rn = net_radiation(T_max, T_min, srad, vp, lat, doy, z)
    es = (esat_kpa(T_max) + esat_kpa(T_min)) / 2
    ea = _vp_kpa(T_min, vp)
    u2 = wind_2m(np.where(np.isfinite(wind), wind, 2.0) if wind is not None else np.full_like(T, 2.0), wind_height)
    d = slope_svp(T)
    g = psychrometric(z)
    et0 = (0.408 * d * rn + g * 900.0 / (T + 273.0) * u2 * np.maximum(es - ea, 0.0)) / (d + g * (1 + 0.34 * u2))
    return np.maximum(et0, 0.0)


def priestley_taylor(T_mean, T_max, T_min, srad, vp, lat, doy, z, alpha=1.26, rn=None):
    T = np.asarray(T_mean, float)
    if rn is None:
        rn = net_radiation(T_max, T_min, srad, vp, lat, doy, z)
    d = slope_svp(T)
    g = psychrometric(z)
    return np.maximum(alpha * d / (d + g) * rn / LAMBDA_FAO, 0.0)


def hargreaves(T_mean, T_max, T_min, lat, doy):
    """Hargreaves & Samani (1985): 0.0023 Ra (T+17.8) sqrt(Tmax-Tmin), Ra as evaporation equivalent (FAO-56 eq. 52)."""
    ra = extraterrestrial_radiation(lat, doy)
    T = np.asarray(T_mean, float)
    return np.maximum(0.0023 * 0.408 * ra * (T + 17.8) * np.sqrt(np.maximum(np.asarray(T_max) - np.asarray(T_min), 0)), 0.0)


def oudin(T_mean, lat, doy):
    """Oudin et al. (2005): PE = Re/(lambda rho) (T+5)/100 if T > -5 else 0 (mm/day); lambda = 2.45 MJ/kg."""
    ra = extraterrestrial_radiation(lat, doy)
    T = np.asarray(T_mean, float)
    return np.where(T + 5 > 0, ra / LAMBDA_FAO * (T + 5) / 100.0, 0.0)


def penman_open_water(T_mean, T_max, T_min, srad, vp, wind, lat, doy, z, wind_height=10.0, albedo=0.08):
    """Penman (1948) open-water evaporation with the Shuttleworth (1993) wind function 6.43(1+0.536 u2) (mm/day)."""
    T = np.asarray(T_mean, float)
    rn = net_radiation(T_max, T_min, srad, vp, lat, doy, z, albedo=albedo)
    es = (esat_kpa(T_max) + esat_kpa(T_min)) / 2
    ea = _vp_kpa(T_min, vp)
    u2 = wind_2m(np.where(np.isfinite(wind), wind, 2.0) if wind is not None else np.full_like(T, 2.0), wind_height)
    d = slope_svp(T)
    g = psychrometric(z)
    lam = 2.501 - 0.002361 * T
    e = (d * rn + g * 6.43 * (1 + 0.536 * u2) * np.maximum(es - ea, 0)) / (lam * (d + g))
    return np.maximum(e, 0.0)


METHODS = ('penman_monteith', 'priestley_taylor', 'hargreaves', 'oudin')


def compute_pet(df, lat, z, method='penman_monteith', wind_height=10.0, srad_is_net=False):
    """PET (mm/day) for a forcing DataFrame with the contract columns. Falls back to Hargreaves when radiation,
    humidity and wind are all missing and the method needs them. Returns (pet array, method actually used).
    srad_is_net: SRAD is net shortwave (e.g. ERA5-Land); then Rn = SRAD + LRAD_net when LRAD_net is present,
    else Rn = SRAD - Rnl(FAO-56) with the incoming shortwave for Rnl estimated as SRAD / 0.77."""
    doy = df.index.dayofyear.values
    T = df['T_mean'].values.astype(float)
    tmax = df['T_max'].values.astype(float) if 'T_max' in df else T + 5
    tmin = df['T_min'].values.astype(float) if 'T_min' in df else T - 5
    srad = df['SRAD'].values.astype(float) if 'SRAD' in df else None
    vp = df['vp'].values.astype(float) if 'vp' in df else None
    if vp is None and 'RH' in df:
        vp = df['RH'].values.astype(float) / 100.0 * esat_kpa(T) * 1000.0
    wind = df['wind'].values.astype(float) if 'wind' in df else None
    if method == 'forcing':
        if 'PET' not in df or not np.isfinite(df['PET'].values).any():
            method = 'penman_monteith'
        else:
            p = df['PET'].values.astype(float)
            if np.isfinite(p).all():
                return np.maximum(p, 0.0), 'forcing'
            fill, _ = compute_pet(df, lat, z, 'penman_monteith', wind_height)
            return np.where(np.isfinite(p), np.maximum(p, 0), fill), 'forcing+pm_fill'
    if method in ('penman_monteith', 'priestley_taylor') and srad is None and vp is None and wind is None:
        method = 'hargreaves'
    rn = None
    if srad_is_net and srad is not None:
        if 'LRAD_net' in df and np.isfinite(df['LRAD_net'].values).mean() > 0.9:
            rn = (srad + df['LRAD_net'].values.astype(float)) * W_TO_MJ_DAY
            rn = np.where(np.isfinite(rn), rn, net_radiation(tmax, tmin, None, vp, lat, doy, z))
        else:
            rin = srad / 0.77
            rnl = net_radiation(tmax, tmin, rin, vp, lat, doy, z, albedo=0.0) - rin * W_TO_MJ_DAY
            rn = srad * W_TO_MJ_DAY + rnl
            rn = np.where(np.isfinite(rn), rn, net_radiation(tmax, tmin, None, vp, lat, doy, z))   # SRAD gaps
        method_tag = method + '(Rn from net radiation)'
    else:
        method_tag = method
    if method == 'penman_monteith':
        return penman_monteith(T, tmax, tmin, srad, vp, wind, lat, doy, z, wind_height, rn=rn), method_tag
    if method == 'priestley_taylor':
        return priestley_taylor(T, tmax, tmin, srad, vp, lat, doy, z, rn=rn), method_tag
    if method == 'hargreaves':
        return hargreaves(T, tmax, tmin, lat, doy), method
    if method == 'oudin':
        return oudin(T, lat, doy), method
    raise ValueError(f'unknown PET method {method!r}; choose from {METHODS + ("forcing",)}')


def disaggregate_daily(pet_daily, index, lat, lon):
    """Spread daily PET totals (Series indexed by UTC date) over sub-daily steps (DatetimeIndex, UTC) in proportion to
    the sine of the solar elevation at each step's midpoint (zero at night); totals per UTC date are preserved."""
    idx = pd.DatetimeIndex(index)
    step_h = float(np.median(np.diff(idx.values).astype('timedelta64[s]').astype(float))) / 3600.0
    mid = idx + pd.Timedelta(hours=step_h / 2)
    doy = mid.dayofyear.values
    phi = np.deg2rad(lat)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    solar_h = mid.hour.values + mid.minute.values / 60.0 + lon / 15.0
    omega = np.deg2rad((solar_h - 12.0) * 15.0)
    w = np.maximum(np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega), 0.0)
    day = idx.normalize()
    ws = pd.Series(w, index=idx).groupby(day).transform('sum').values
    n = pd.Series(1.0, index=idx).groupby(day).transform('sum').values
    frac = np.where(ws > 0, w / np.where(ws > 0, ws, 1), 1.0 / n)
    tot = pd.Series(pet_daily).reindex(day).values
    return np.nan_to_num(tot) * frac
