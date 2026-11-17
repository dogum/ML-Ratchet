"""Moist-air psychrometrics (SI). ASHRAE Handbook Fundamentals 2017, ch. 1.

Saturation pressure: Hyland & Wexler (1983) as given in ASHRAE Fundamentals eq. 5 (over ice, -100..0 C) and
eq. 6 (over water, 0..200 C). Humidity ratio W = 0.621945 p_w / (p - p_w) (eq. 20). Enthalpy
h = 1.006 t + W (2501 + 1.86 t) kJ/kg dry air (eq. 32). Every function takes numpy arrays or floats; the `_s`
variants are scalar (math) versions for the time-step loop.  Checked against CoolProp HAPropsSI in
checks/check_psychro.py.
"""
import math
import numpy as np

P_ATM = 101325.0
CP_A = 1006.0        # J/kg K dry air
CP_V = 1860.0        # J/kg K water vapour
H_FG = 2501000.0     # J/kg latent heat at 0 C (enthalpy datum)
H_FG_ROOM = 2454000.0  # J/kg latent heat at ~20 C, used for latent gains and coil condensate
R_DA = 287.042       # J/kg K
CP_W = 4186.0        # J/kg K liquid water
RHO_W = 998.0        # kg/m3 liquid water

# Hyland-Wexler constants
_C1, _C2, _C3, _C4, _C5, _C6, _C7 = (-5.6745359e3, 6.3925247, -9.677843e-3, 6.2215701e-7, 2.0747825e-9,
                                     -9.484024e-13, 4.1635019)
_C8, _C9, _C10, _C11, _C12, _C13 = (-5.8002206e3, 1.3914993, -4.8640239e-2, 4.1764768e-5, -1.4452093e-8,
                                    6.5459673)


def p_ws(t):
    """Saturation vapour pressure [Pa] at dry-bulb t [C] (over ice below 0 C)."""
    T = np.asarray(t, float) + 273.15
    ice = T < 273.15
    lw = _C8 / T + _C9 + _C10 * T + _C11 * T ** 2 + _C12 * T ** 3 + _C13 * np.log(T)
    li = _C1 / T + _C2 + _C3 * T + _C4 * T ** 2 + _C5 * T ** 3 + _C6 * T ** 4 + _C7 * np.log(T)
    return np.exp(np.where(ice, li, lw))


def p_ws_s(t):
    T = t + 273.15
    if T < 273.15:
        return math.exp(_C1 / T + _C2 + _C3 * T + _C4 * T * T + _C5 * T ** 3 + _C6 * T ** 4 + _C7 * math.log(T))
    return math.exp(_C8 / T + _C9 + _C10 * T + _C11 * T * T + _C12 * T ** 3 + _C13 * math.log(T))


def w_from_pw(pw, p=P_ATM):
    return 0.621945 * pw / np.maximum(p - pw, 1.0)


def w_sat(t, p=P_ATM):
    return w_from_pw(p_ws(t), p)


def w_sat_s(t, p=P_ATM):
    pw = p_ws_s(t)
    return 0.621945 * pw / (p - pw)


def w_from_rh(t, rh, p=P_ATM):
    """rh in percent."""
    return w_from_pw(np.asarray(rh, float) / 100.0 * p_ws(t), p)


def pw_from_w(w, p=P_ATM):
    w = np.asarray(w, float)
    return p * w / (0.621945 + w)


def rh_from_w(t, w, p=P_ATM):
    return 100.0 * pw_from_w(w, p) / p_ws(t)


def rh_from_w_s(t, w, p=P_ATM):
    return 100.0 * (p * w / (0.621945 + w)) / p_ws_s(t)


def w_from_dewpoint(td, p=P_ATM):
    return w_from_pw(p_ws(td), p)


def dewpoint_from_w(w, p=P_ATM):
    """Dew/frost point [C] from humidity ratio: Newton iterations on p_ws(td) = p_w."""
    pw = np.maximum(pw_from_w(w, p), 1e-3)
    # initial guess: ASHRAE eq. 39 (Peppers)
    a = np.log(pw / 1000.0)
    td = 6.54 + 14.526 * a + 0.7389 * a ** 2 + 0.09486 * a ** 3 + 0.4569 * (pw / 1000.0) ** 0.1984
    td = np.where(td < 0, 6.09 + 12.608 * a + 0.4959 * a ** 2, td)
    for _ in range(6):
        f = p_ws(td)
        df = (p_ws(td + 0.01) - f) / 0.01
        td = td - (f - pw) / df
    return td


def enthalpy(t, w):
    """J/kg dry air."""
    return CP_A * t + w * (H_FG + CP_V * t)


def enthalpy_s(t, w):
    return CP_A * t + w * (H_FG + CP_V * t)


def t_from_hw(h, w):
    return (h - H_FG * w) / (CP_A + CP_V * w)


def t_sat_from_h_s(h, p=P_ATM):
    """Saturation temperature [C] whose saturated-air enthalpy equals h [J/kg] (scalar, Newton)."""
    t = (h / 1000.0 - 10.0) / 3.0  # rough start
    t = min(max(t, -40.0), 60.0)
    for _ in range(30):
        ws = w_sat_s(t, p)
        f = CP_A * t + ws * (H_FG + CP_V * t) - h
        ws2 = w_sat_s(t + 0.01, p)
        f2 = CP_A * (t + 0.01) + ws2 * (H_FG + CP_V * (t + 0.01)) - h
        d = (f2 - f) / 0.01
        step = f / d
        t -= step
        if abs(step) < 1e-6:
            break
    return t


def wetbulb(t, w, p=P_ATM):
    """Thermodynamic wet-bulb [C] (ASHRAE eq. 33/35 solved by Newton), vectorised."""
    t = np.asarray(t, float)
    w = np.asarray(w, float)
    tw = t - 0.33 * (t - dewpoint_from_w(w, p))   # start
    for _ in range(25):
        f = _wb_resid(tw, t, w, p)
        df = (_wb_resid(tw + 0.005, t, w, p) - f) / 0.005
        tw = tw - f / df
    return tw


def _wb_resid(tw, t, w, p):
    ws = w_sat(tw, p)
    above = tw >= 0
    w_calc = np.where(above,
                      ((2501.0 - 2.326 * tw) * ws - 1.006 * (t - tw)) / (2501.0 + 1.86 * t - 4.186 * tw),
                      ((2830.0 - 0.24 * tw) * ws - 1.006 * (t - tw)) / (2830.0 + 1.86 * t - 2.1 * tw))
    return w_calc - w


def wetbulb_s(t, w, p=P_ATM):
    """Scalar wet-bulb (a few Newton steps; accurate to ~0.01 K)."""
    tw = t - 0.4 * max(t - 0.0, 0.0) * 0.5
    for _ in range(20):
        f = _wb_resid_s(tw, t, w, p)
        df = (_wb_resid_s(tw + 0.005, t, w, p) - f) / 0.005
        st = f / df
        tw -= st
        if abs(st) < 1e-4:
            break
    return tw


def _wb_resid_s(tw, t, w, p):
    ws = w_sat_s(tw, p)
    if tw >= 0:
        wc = ((2501.0 - 2.326 * tw) * ws - 1.006 * (t - tw)) / (2501.0 + 1.86 * t - 4.186 * tw)
    else:
        wc = ((2830.0 - 0.24 * tw) * ws - 1.006 * (t - tw)) / (2830.0 + 1.86 * t - 2.1 * tw)
    return wc - w


def rho_moist(t, w, p=P_ATM):
    """Moist-air density [kg moist air / m3] (ideal gas, ASHRAE eq. 28 via specific volume)."""
    v = R_DA * (np.asarray(t, float) + 273.15) * (1 + 1.607858 * np.asarray(w, float)) / p   # m3/kg dry air
    return (1 + w) / v


def rho_dry_per_m3(t, w, p=P_ATM):
    """kg dry air per m3 of moist air."""
    return p / (R_DA * (np.asarray(t, float) + 273.15) * (1 + 1.607858 * np.asarray(w, float)))


def pressure_from_elevation(z):
    """Standard atmosphere pressure [Pa] (ASHRAE eq. 3)."""
    return 101325.0 * (1 - 2.25577e-5 * np.asarray(z, float)) ** 5.2559
