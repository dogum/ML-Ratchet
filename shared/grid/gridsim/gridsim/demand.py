"""Weather-driven hourly electricity demand.

    L_t = sum_j b_j * calendar_j(t)                                  hour-of-week x (weekday/weekend/holiday) shape, trend
        + h_1 * HDH_t + c_1 * CDH_t + c_2 * CDH_t^2                  heating / cooling degree-hours on the building
                                                                     temperature T_b (exponential thermal lag tau)
        + c_h * CDH_t * (q_t - q_ref)                                 latent (humidity) cooling load, q specific humidity
        + s_1 * GHI_t                                                 solar gain / lighting offset
        + (diurnal x season interactions of HDH/CDH)
        - BTM_PV_t                                                    behind-the-meter solar (pvlib CF x BTM capacity)

T_b follows dT_b/dt = (T_out - T_b)/tau.  HDH = max(T_h - T_b, 0), CDH = max(T_b - T_c, 0).  Linear in the b, h, c, s
coefficients given (tau, T_h, T_c); those three are chosen by grid search on the training data (`fit_demand`).
A population-weighted temperature over several stations should be used for a whole BA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

US_HOLIDAYS_FIXED = [(1, 1), (7, 4), (12, 25), (12, 24), (12, 31)]


def _holidays(idx_local):
    d = idx_local.normalize()
    hol = np.zeros(len(d), bool)
    for m, dd in US_HOLIDAYS_FIXED:
        hol |= (d.month == m) & (d.day == dd)
    # Memorial (last Mon May), Labor (1st Mon Sep), Thanksgiving (4th Thu Nov) + Friday
    hol |= (d.month == 5) & (d.dayofweek == 0) & (d.day > 24)
    hol |= (d.month == 9) & (d.dayofweek == 0) & (d.day <= 7)
    hol |= (d.month == 11) & (((d.dayofweek == 3) & (d.day >= 22) & (d.day <= 28)) |
                              ((d.dayofweek == 4) & (d.day >= 23) & (d.day <= 29)))
    return hol


def building_temperature(T_out, tau_h=8.0):
    """First-order thermal lag of the outdoor temperature (hourly steps)."""
    T = np.asarray(T_out, float)
    a = 1 - np.exp(-1.0 / max(tau_h, 1e-3))
    out = np.empty_like(T)
    x = T[0]
    for i, v in enumerate(T):
        x += a * (v - x)
        out[i] = x
    return out


def specific_humidity(T_C, rh, p_hPa=1013.25):
    es = 6.1094 * np.exp(17.625 * np.asarray(T_C) / (np.asarray(T_C) + 243.04))
    e = np.clip(np.asarray(rh, float), 0, 1) * es
    return 0.622 * e / (np.asarray(p_hPa) - 0.378 * e)


def design_matrix(weather, tau_h=8.0, T_h=15.0, T_c=20.0, tz="America/Chicago", q_ref=0.010, year0=2019):
    """weather: hourly DataFrame (UTC index) with t2m [C] and optional rh [0-1], ghi [W/m^2]."""
    idx = weather.index
    loc = idx.tz_convert(tz) if idx.tz is not None else idx
    T = weather["t2m"].values.astype(float)
    Tb = building_temperature(T, tau_h)
    hdh = np.maximum(T_h - Tb, 0)
    cdh = np.maximum(Tb - T_c, 0)
    cols = {}
    hol = _holidays(loc)
    daytype = np.where(hol | (loc.dayofweek == 6), 2, np.where(loc.dayofweek == 5, 1, 0))
    for dt_ in range(3):
        for h in range(24):
            cols[f"cal_{dt_}_{h}"] = ((daytype == dt_) & (loc.hour == h)).astype(float)
    doy = loc.dayofyear.values / 365.25 * 2 * np.pi
    for k in (1, 2):
        cols[f"sin{k}"] = np.sin(k * doy); cols[f"cos{k}"] = np.cos(k * doy)
    cols["trend"] = (loc.year.values + loc.dayofyear.values / 365.25 - year0)
    day = ((loc.hour >= 7) & (loc.hour <= 22)).astype(float)
    cols["hdh"] = hdh; cols["hdh_day"] = hdh * day
    cols["cdh"] = cdh; cols["cdh2"] = cdh ** 2; cols["cdh_day"] = cdh * day
    cols["hdh_raw"] = np.maximum(T_h - T, 0)          # fast response (resistance heat)
    if "rh" in weather:
        q = specific_humidity(T, weather["rh"].values)
        cols["cdh_hum"] = cdh * (q - q_ref) * 100
    if "ghi" in weather:
        cols["ghi"] = weather["ghi"].values / 1000.0
    return pd.DataFrame(cols, index=idx)


def fit_demand(weather, load_MW, taus=(3, 6, 10, 16, 24), T_hs=(12, 15, 18), T_cs=(18, 21, 24), btm_MW=None,
               tz="America/Chicago", ridge=1e-3):
    """Least-squares fit (grid search over tau, T_h, T_c).  Returns params dict for `demand_model`."""
    y = pd.Series(load_MW).reindex(weather.index).astype(float)
    if btm_MW is not None:
        y = y + pd.Series(btm_MW).reindex(weather.index).fillna(0)
    ok = y.notna().values
    best = None
    for tau in taus:
        for th in T_hs:
            for tc in T_cs:
                if tc < th:
                    continue
                X = design_matrix(weather, tau, th, tc, tz)
                A = X.values[ok]
                coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y.values[ok])
                rmse = np.sqrt(np.mean((A @ coef - y.values[ok]) ** 2))
                if best is None or rmse < best[0]:
                    best = (rmse, tau, th, tc, coef, list(X.columns))
    rmse, tau, th, tc, coef, cols = best
    return {"tau_h": tau, "T_h": th, "T_c": tc, "coef": dict(zip(cols, map(float, coef))), "tz": tz,
            "train_rmse_MW": float(rmse), "train_mean_MW": float(y[ok].mean())}


def demand_model(weather, params, btm_MW=None):
    """Hourly demand [MW] (Series) from weather and fitted params; subtracts behind-the-meter PV if given."""
    X = design_matrix(weather, params["tau_h"], params["T_h"], params["T_c"], params.get("tz", "America/Chicago"))
    coef = params["coef"]
    yhat = sum(X[c].values * coef.get(c, 0.0) for c in X.columns)
    if btm_MW is not None:
        yhat = yhat - pd.Series(btm_MW).reindex(weather.index).fillna(0).values
    return pd.Series(yhat, index=weather.index)


def btm_pv(times, capacity_MW, lat, lon, ghi, temp_air=20.0):
    """Behind-the-meter PV output [MW]: fixed-tilt rooftop, DC/AC 1.15 (pvlib via renewables.solar_power)."""
    from .renewables import solar_power
    return pd.Series(solar_power(times, lat, lon, ghi, temp_air=temp_air, capacity_MWac=capacity_MW, dc_ac=1.15,
                                 tracking="fixed", tilt=20.0), index=times)
