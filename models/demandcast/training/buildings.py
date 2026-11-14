"""Building-stock thermal demand model (the demand physics of DemandCast).

A region's hourly electricity demand is the sum of
  base      residential / commercial / industrial activity  x occupancy schedules from the calendar
            (DST-aware civil hour, weekday / Saturday / Sunday-holiday), plus a water-heating term that follows the
            30-day mean temperature (colder mains water in winter)
  lighting  share of base x occupancy x darkness (1 - daylight from GHI)
  heating   heat need of the stock delivered by resistance (COP 1), heat pumps (COP falls with outdoor T) and
            non-electric systems (fans/pumps only, 3 %)
  cooling   sensible + latent load over the air-conditioner COP (falls with outdoor T), x AC saturation
  - BTM PV  rooftop solar (pvlib capacity factor x installed MW)

Heat need of the stock (thermal units, MW_th) = UA x E_set[max(T_set - T_eff, 0)] and the cooling analogue, where
  T_eff = T_lag + internal gains/UA + window solar gains/UA,
  T_lag = sol-air temperature passed through the envelope's thermal mass: a mix of a light (tau1) and heavy (tau2)
          first-order RC response, dT_lag/dt = (T_sa - T_lag)/tau,   T_sa = T_out + a_sol x GHI (opaque surfaces),
  T_set ~ Normal(mu, sigma) over the building stock (thermostat spread), with a night/unoccupied setback.
  E[max(x, 0)] for x ~ N(m, s) = m Phi(m/s) + s phi(m/s)   (closed form; `ghinge`).
Latent cooling: ventilation air dried from q_out to q_in (g/kg) only where cooling is on:
  L = UA x lam x max(q_out - q_in, 0) x Phi((T_eff - mu_c)/sigma_c),  lam ~ 0.75 K per g/kg = (h_fg / c_p) x
  (ventilation share of UA ~ 0.3) / 1000 (h_fg 2,501 kJ/kg, c_p 1.006 kJ/kg K).

All parameters are in a dict (see PARAMS); levels are relative to the base load P0 (MW).
"""
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.stats import norm

PARAMS = dict(
    P0=30000.0,        # MW, mean base (non-weather) load
    trend=0.0,         # fractional growth per year (relative to t_ref)
    sh_res=0.38, sh_com=0.36,        # base shares (industrial = rest)
    res_m=0.25, res_e=0.40, res_mt=7.5, res_et=19.0,   # residential morning/evening bump amplitude and time (h)
    com_open=7.0, com_close=18.5, com_wkend=0.45,       # commercial schedule
    ind_wkend=0.92,
    wh=0.03,           # water heating / seasonal base sensitivity per 10 K of 30-day mean below 15 C
    light=0.06,        # lighting share of base
    ua=0.20,           # stock conductance UA / P0 [1/K]  (MW_th per K per MW of base)
    tau1=4.0, tau2=30.0, f1=0.5,     # envelope time constants (h) and light share
    a_sol=3.0,         # K of sol-air temperature rise per kW/m2 GHI
    g_int=3.0,         # internal gains / UA, K (occupied)
    g_win=3.0,         # window solar gains / UA, K per kW/m2
    mu_h=19.5, sd_h=2.0, setback=1.5,   # heating setpoint distribution, night setback (K)
    mu_c=23.5, sd_c=2.0, setup=1.0,     # cooling setpoint distribution, unoccupied set-up (K)
    q_in=9.0, lam=0.75,                 # indoor humidity target (g/kg), latent factor K per g/kg
    e_res=0.15, e_hp=0.10,              # share of heat need met by resistance / heat pumps
    ac=0.95,                            # AC saturation (share of cooling need actually served)
    btm=0.0,           # behind-the-meter PV MW_ac / P0
)

US_HOL_FIXED = [(1, 1), (7, 4), (12, 25), (12, 24), (12, 31)]


def holidays(d):
    """US holidays on local civil dates (DatetimeIndex normalised): fixed, MLK, Presidents, Memorial, Labor,
    Thanksgiving + Friday."""
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


def calendar(times_utc, tz):
    """Civil-time calendar: hour (float, DST-aware), daytype 0 weekday / 1 Saturday / 2 Sunday or holiday, doy."""
    loc = pd.DatetimeIndex(times_utc).tz_convert(tz)
    hol = holidays(loc.normalize().tz_localize(None))
    dt = np.where(hol | (loc.dayofweek == 6), 2, np.where(loc.dayofweek == 5, 1, 0))
    return dict(hour=loc.hour.values.astype(float), daytype=dt, doy=loc.dayofyear.values.astype(float),
                year=loc.year.values + loc.dayofyear.values / 365.25)


def _g(h, mu, s):
    d = (h - mu + 12) % 24 - 12          # circular distance
    return np.exp(-0.5 * (d / s) ** 2)


def occupancy(cal, p):
    """Residential, commercial, industrial activity (mean ~1) and an occupied fraction for setbacks."""
    h, dt = cal['hour'], cal['daytype']
    wk = dt > 0
    mt = p['res_mt'] + 1.5 * wk          # weekends: later, broader morning
    res = 0.62 + p['res_m'] * _g(h + 0.5, mt, 1.6 + 0.8 * wk) + p['res_e'] * _g(h + 0.5, p['res_et'], 2.6) \
        + 0.12 * _g(h + 0.5, 13.0, 3.0) * wk - 0.18 * _g(h + 0.5, 3.5, 2.0)
    s = 1 / (1 + np.exp(-(h + 0.5 - p['com_open']) / 0.8)) / (1 + np.exp(-(p['com_close'] - h - 0.5) / 1.0))
    amp = np.where(dt == 0, 1.0, np.where(dt == 1, p['com_wkend'], p['com_wkend'] * 0.8))
    com = 0.55 + 0.75 * s * amp
    ind = (0.93 + 0.12 * _g(h + 0.5, 12.0, 4.0)) * np.where(dt == 0, 1.0, p['ind_wkend'])
    occ = np.clip(0.35 + 0.65 * _g(h + 0.5, 14.0, 6.0), 0, 1)   # 1 = setpoints at day values, 0 = full setback
    return res / 1.0, com / 1.0, ind, occ


def rc_filter(x, tau, x0=None):
    a = 1 - np.exp(-1.0 / max(tau, 0.05))
    zi = [(1 - a) * (x[0] if x0 is None else x0)]
    y, _ = lfilter([a], [1, -(1 - a)], x, zi=zi)
    return y


def ghinge(m, s):
    z = m / s
    return m * norm.cdf(z) + s * norm.pdf(z)


def cop_hp(T):
    return np.clip(2.6 + 0.08 * T, 1.0, 4.5)


def cop_ac(T):
    return np.clip(4.2 - 0.07 * (T - 25.0), 2.0, 5.5)


def demand(wx, cal, p, t_ref=2020.0):
    """wx: dict of hourly arrays T [C], q [g/kg], ghi [W/m2], btm_cf [0-1]; cal from calendar(); p params.
    Returns dict of components (MW) and total."""
    T, q, ghi = wx['T'], wx['q'], wx['ghi'] / 1000.0
    res, com, ind, occ = occupancy(cal, p)
    sh_ind = max(1 - p['sh_res'] - p['sh_com'], 0.0)
    T30 = rc_filter(T, 240.0)
    season = 1 + p['wh'] * (15.0 - T30) / 10.0
    growth = 1 + p['trend'] * (cal['year'] - t_ref)
    P0 = p['P0'] * growth
    base = P0 * (p['sh_res'] * res + p['sh_com'] * com + sh_ind * ind) * season
    dark = 1 - np.clip(ghi / 0.25, 0, 1)
    light = P0 * p['light'] * (0.5 * res + 0.5 * com) * dark
    Tsa = T + p['a_sol'] * ghi
    Tlag = p['f1'] * rc_filter(Tsa, p['tau1']) + (1 - p['f1']) * rc_filter(Tsa, p['tau2'])
    Teff = Tlag + p['g_int'] * (0.6 + 0.4 * occ) + p['g_win'] * rc_filter(ghi, 2.0)
    mh = p['mu_h'] - p['setback'] * (1 - occ)
    mc = p['mu_c'] + p['setup'] * (1 - occ)
    UA = p['ua'] * P0
    heat_th = UA * ghinge(mh - Teff, p['sd_h'])
    cool_on = norm.cdf((Teff - mc) / p['sd_c'])
    cool_th = UA * (ghinge(Teff - mc, p['sd_c']) + p['lam'] * np.maximum(q - p['q_in'], 0) * cool_on)
    e_other = max(1 - p['e_res'] - p['e_hp'], 0.0)
    heat = heat_th * (p['e_res'] + p['e_hp'] / cop_hp(T) + 0.03 * e_other)
    cool = cool_th * p['ac'] / cop_ac(T)
    btm = p['btm'] * P0 * wx['btm_cf']
    total = base + light + heat + cool - btm
    return dict(base=base, light=light, heat=heat, cool=cool, btm=btm, total=total)
