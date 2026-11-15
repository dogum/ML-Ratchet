"""Wind and solar power from weather.

Wind (per farm):
  hub-height wind  v_hub = v_ref (z_hub / z_ref)^alpha            (power law; or log law with roughness z0)
  air density      rho = p / (R_d T_v),  T_v virtual temperature   (humid air)
  density-equivalent wind (IEC 61400-12-1, pitch-regulated): v_eq = v_hub (rho / 1.225)^(1/3)
  turbine curve    P = min(P_r, 0.5 rho0 A Cp v_eq^3) between cut-in and cut-out, Cp(v) from a max-Cp plateau that
                   falls as v^-3 above rated; specific power SP = P_r / A [W/m^2] sets the rated speed
  farm curve       turbine curve convolved with a Gaussian spread of wind speeds across the farm
                   (sigma = 0.1 v + 0.5 m/s; Norgaard & Holttinen 2004 multi-turbine power curve)
  losses           wake (default 10 %), availability (97 %), electrical (2 %), high-wind hysteresis, low-temperature
                   shutdown below -30 C
Solar (pvlib): solar position -> (Erbs decomposition if only GHI) -> Perez transposition (fixed tilt or single-axis
  tracking with backtracking) -> SAPM cell temperature -> PVWatts DC (gamma -0.37 %/K) -> PVWatts inverter clipped at
  the AC rating (DC/AC ratio) -> system losses (default 14 %, PVWatts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RHO0 = 1.225
R_D = 287.05


def air_density(T_C, p_hPa=None, elev_m=0.0, rh=None):
    """Moist-air density [kg/m^3] from temperature, pressure (or ISA pressure at elevation) and relative humidity [0-1]."""
    T = np.asarray(T_C, float) + 273.15
    if p_hPa is None:
        p = 101325.0 * (1 - 2.25577e-5 * np.asarray(elev_m, float)) ** 5.25588
    else:
        p = np.asarray(p_hPa, float) * 100.0
    if rh is not None:
        es = 610.94 * np.exp(17.625 * (T - 273.15) / (T - 273.15 + 243.04))   # Magnus, Pa
        e = np.clip(np.asarray(rh, float), 0, 1) * es
        Tv = T / (1 - 0.378 * e / p)
    else:
        Tv = T
    return p / (R_D * Tv)


def hub_wind(ws_ref, z_ref=10.0, z_hub=90.0, alpha=None, z0=None):
    """Extrapolate wind speed to hub height: power law (alpha, default 1/7) or log law (roughness z0 [m])."""
    ws_ref = np.asarray(ws_ref, float)
    if z0 is not None:
        return ws_ref * np.log(z_hub / z0) / np.log(z_ref / z0)
    a = 1.0 / 7.0 if alpha is None else alpha
    return ws_ref * (z_hub / z_ref) ** a


def shear_exponent(ws_low, ws_high, z_low, z_high):
    """Shear exponent from two heights (e.g. NASA POWER 10 m and 50 m)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(np.maximum(ws_high, 0.1) / np.maximum(ws_low, 0.1)) / np.log(z_high / z_low)
    return np.clip(np.nan_to_num(a, nan=1 / 7), 0.0, 0.6)


def turbine_curve(v, specific_power=300.0, cp_max=0.46, v_in=3.0, v_out=25.0, rho=RHO0):
    """Normalised power (0-1 of rating) of a pitch-regulated turbine vs wind speed at density rho.

    P/P_r = min(1, 0.5 rho Cp v^3 / SP); a smooth knee near rated (softmin with exponent 8) mimics pitch control.
    """
    v = np.asarray(v, float)
    p = 0.5 * rho * cp_max * v ** 3 / specific_power
    k = 8.0
    p = p / (1 + p ** k) ** (1 / k)
    # cut-in ramp (Cp collapses at low tip-speed ratio) and cut-out
    p = np.where(v < v_in, 0.0, p * np.clip((v - v_in) / 1.0, 0, 1) ** 0.5)
    return np.where(v > v_out, 0.0, p)


def farm_curve(v, specific_power=300.0, cp_max=0.46, v_in=3.0, v_out=25.0, sigma_frac=0.10, sigma0=0.5, n=41):
    """Multi-turbine (farm) power curve: turbine curve averaged over a Gaussian spread of wind speeds."""
    v = np.asarray(v, float)
    z = np.linspace(-3, 3, n)
    w = np.exp(-0.5 * z ** 2); w /= w.sum()
    sig = sigma_frac * v + sigma0
    vv = np.maximum(v[..., None] + sig[..., None] * z, 0)
    return (turbine_curve(vv, specific_power, cp_max, v_in, v_out) * w).sum(-1)


def wind_power(ws_hub, capacity_MW=1.0, T_C=15.0, p_hPa=None, rh=None, elev_m=0.0, specific_power=300.0,
               cp_max=0.46, wake_loss=0.10, availability=0.97, electrical_loss=0.02, v_out=25.0, farm=True):
    """Farm output [MW] from hub-height wind speed and air state.  Returns array like ws_hub."""
    rho = air_density(T_C, p_hPa, elev_m, rh)
    v_eq = np.asarray(ws_hub, float) * (rho / RHO0) ** (1 / 3)
    f = farm_curve(v_eq, specific_power, cp_max, v_out=v_out) if farm else turbine_curve(v_eq, specific_power, cp_max,
                                                                                        v_out=v_out)
    f = f * (1 - wake_loss) * availability * (1 - electrical_loss)
    f = np.where(np.asarray(T_C) < -30, 0.0, f)
    return capacity_MW * f


def specific_power_from_spec(unit):
    """W/m^2 from a contract wind spec: `specific_power`, or rating / rotor area, or design wind class (B's
    design_wind_speed_mph): low-wind designs have low specific power."""
    if unit.get("specific_power"):
        return float(unit["specific_power"])
    if unit.get("rotor_diameter_m") and unit.get("turbine_MW"):
        return unit["turbine_MW"] * 1e6 / (np.pi * unit["rotor_diameter_m"] ** 2 / 4)
    od = unit.get("online_date")
    year = pd.Timestamp(od).year if od is not None and not pd.isna(od) else 2012
    # US fleet trend (LBNL Land-Based Wind Market Report 2023): ~400 W/m2 in 2000-2005 -> ~250 in 2020+
    return float(np.interp(year, [2000, 2008, 2014, 2020], [400, 340, 280, 245]))


# ---------------------------------------------------------------------- solar
def solar_power(times, lat, lon, ghi, dni=None, dhi=None, temp_air=20.0, wind_speed=1.0, capacity_MWac=1.0,
                dc_ac=1.3, tracking="single_axis", tilt=None, azimuth=180.0, gamma_pdc=-0.0037, losses=0.14,
                albedo=0.2, gcr=0.35, elevation=0.0):
    """AC output [MW] of a PV plant from irradiance (W/m^2, period-mean values labelled at the period start are
    shifted by half a step for solar position).  Uses pvlib."""
    import pvlib
    times = pd.DatetimeIndex(times)
    if times.tz is None:
        times = times.tz_localize("UTC")
    step = (times[1] - times[0]) if len(times) > 1 else pd.Timedelta("1h")
    tpos = times + step / 2
    sp = pvlib.solarposition.get_solarposition(tpos, lat, lon, altitude=elevation)
    zen = sp["apparent_zenith"].values
    ghi = np.asarray(ghi, float)
    if dni is None or dhi is None:
        dec = pvlib.irradiance.erbs(ghi, sp["zenith"].values, tpos)
        dni, dhi = np.asarray(dec["dni"]), np.asarray(dec["dhi"])
    dni = np.nan_to_num(np.asarray(dni, float)); dhi = np.nan_to_num(np.asarray(dhi, float))
    dni_extra = pvlib.irradiance.get_extra_radiation(tpos).values
    am = pvlib.atmosphere.get_relative_airmass(zen)
    if tracking in ("single_axis", "tracking", True):
        tr = pvlib.tracking.singleaxis(sp["apparent_zenith"], sp["azimuth"], axis_tilt=0, axis_azimuth=180,
                                       max_angle=60, backtrack=True, gcr=gcr)
        surf_tilt = np.nan_to_num(tr["surface_tilt"].values)
        surf_az = np.nan_to_num(tr["surface_azimuth"].values, nan=180.0)
    else:
        surf_tilt = np.full(len(times), abs(lat) * 0.87 if tilt is None else tilt)
        surf_az = np.full(len(times), azimuth)
    poa = pvlib.irradiance.get_total_irradiance(surf_tilt, surf_az, sp["apparent_zenith"].values, sp["azimuth"].values,
                                                dni, ghi, dhi, dni_extra=dni_extra, airmass=am, albedo=albedo,
                                                model="perez")
    poa_g = np.nan_to_num(np.asarray(poa["poa_global"], float)).clip(0)
    tcell = pvlib.temperature.sapm_cell(poa_g, np.asarray(temp_air, float) * np.ones(len(times)),
                                        np.asarray(wind_speed, float) * np.ones(len(times)),
                                        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"])
    pdc0 = capacity_MWac * dc_ac
    pdc = pvlib.pvsystem.pvwatts_dc(poa_g, tcell, pdc0, gamma_pdc) * (1 - losses)
    pac = pvlib.inverter.pvwatts(pdc, capacity_MWac / 0.96, eta_inv_nom=0.96)
    return np.clip(np.nan_to_num(pac), 0, capacity_MWac)


def fleet_profiles(units, weather_by_point, kind="wind", **kw):
    """Aggregate MW from many farms.  units: DataFrame with lat, lon, pmax_MW (+ hub_height_m, online/retired);
    weather_by_point: dict point_id -> DataFrame (hourly UTC) with columns ws10 [m/s], ws50 (optional), t2m [C],
    ps [hPa], rh [0-1] (wind) or ghi, dni, dhi, t2m, ws10 (solar), plus attrs lat/lon.  Each farm uses the nearest
    point.  Returns a Series of MW."""
    pts = list(weather_by_point)
    plat = np.array([weather_by_point[p].attrs["lat"] for p in pts])
    plon = np.array([weather_by_point[p].attrs["lon"] for p in pts])
    idx = weather_by_point[pts[0]].index
    total = np.zeros(len(idx))
    tn = idx.tz_localize(None) if idx.tz is not None else idx
    for _, u in units.iterrows():
        j = int(np.argmin((plat - u.lat) ** 2 + ((plon - u.lon) * np.cos(np.radians(u.lat))) ** 2))
        w = weather_by_point[pts[j]]
        on = np.ones(len(idx), bool)
        if "online" in u and not pd.isna(u.online):
            on &= tn >= u.online
        if "retired" in u and not pd.isna(u.retired):
            on &= tn < u.retired
        if kind == "wind":
            hub = u.get("hub_height_m") if not pd.isna(u.get("hub_height_m", np.nan)) else 80.0
            if "ws50" in w:
                a = shear_exponent(w.ws10.values, w.ws50.values, 10, 50)
                v = w.ws50.values * (hub / 50.0) ** a
            else:
                v = hub_wind(w.ws10.values, 10, hub, kw.get("alpha"))
            p = wind_power(v, u.pmax_MW, w.t2m.values, w.get("ps"), w.get("rh"),
                           specific_power=specific_power_from_spec(u), **{k: v_ for k, v_ in kw.items()
                                                                         if k in ("wake_loss", "availability")})
        else:
            p = solar_power(idx, u.lat, u.lon, w.ghi.values, w.get("dni"), w.get("dhi"), w.t2m.values,
                            w.get("ws10", 1.0), capacity_MWac=u.pmax_MW,
                            dc_ac=u.get("dc_ac") if not pd.isna(u.get("dc_ac", np.nan)) else 1.3,
                            tracking="single_axis" if str(u.get("tracking", "single_axis")).lower() not in
                            ("fixed", "fixed_tilt", "none", "nan") else "fixed")
        total += np.where(on, p, 0.0)
    return pd.Series(total, index=idx)
