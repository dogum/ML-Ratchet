"""Dead and live fuel moisture.

Dead fuels (hourly):
  * 'nelson' : a Nelson (2000)-structured stick model, re-derived here (the NFDRS4 C++ source was not reachable).
    Each timelag class is a wooden cylinder (radius 0.20, 0.64, 2.0, 3.3 cm for 1-, 10-, 100-, 1000-h, as in
    NFDRS 2016 / Carlson et al. 2007; density 0.40 g/cm3). Bound water diffuses radially (finite volumes, implicit
    Euler, 11 nodes), D(T) Arrhenius in stick temperature. The surface exchanges water with the air through a
    film coefficient toward the equilibrium moisture of the air next to the stick, at the stick's temperature: the
    stick is warmed above air temperature by solar radiation (energy balance with convection and longwave), which
    lowers the relative humidity it sees (this is what dries litter at midday). Rain wets the surface toward the
    maximum moisture (capped by the rain mass the stick catches). The equilibrium isotherm is Nelson's (1984) form
    EMC = a - b ln(-(R T / Mw) ln h), with (a, b) fitted to the Simard (1968) NFDRS EMC over RH 10-95 percent at
    21 C (desorption; no separate adsorption branch).
  * 'timelag': classic exponential timelag response toward Simard EMC (NFDRS-1978 spirit), with rain wetting.
Live fuels:
  * NFDRS 2016-style Growing Season Index (Jolly et al. 2005) mapped linearly to live herbaceous (30-250 %) and
    woody (60-200 %) moisture, or constant values.
Curing: Scott & Burgan (2005) load transfer from live herbaceous moisture is applied inside the Rothermel kernel.
"""
import math

import numpy as np

from ._nb import njit

RGAS = 8.314
MW = 18.015
STICK_RADIUS_CM = {'1h': 0.20, '10h': 0.64, '100h': 2.0, '1000h': 3.3}
NOMINAL_TAU_H = {'1h': 1.0, '10h': 10.0, '100h': 100.0, '1000h': 1000.0}


def simard_emc(t_c, rh):
    """Simard (1968) equilibrium moisture content (%), as used by NFDRS 1978 (T in F inside)."""
    T = np.asarray(t_c, float) * 1.8 + 32.0
    H = np.clip(np.asarray(rh, float), 0, 100)
    e1 = 0.03229 + 0.281073 * H - 0.000578 * H * T
    e2 = 2.22749 + 0.160107 * H - 0.01478 * T
    e3 = 21.0606 + 0.005565 * H * H - 0.00035 * H * T - 0.483199 * H
    return np.where(H < 10, e1, np.where(H < 50, e2, e3))


def _fit_isotherm():
    h = np.linspace(0.10, 0.95, 60)
    T = 294.15
    dg = -(RGAS * T / MW) * np.log(h)
    m = simard_emc(21.0, h * 100) / 100.0
    A = np.column_stack([np.ones_like(dg), -np.log(dg)])
    a, b = np.linalg.lstsq(A, m, rcond=None)[0]
    return float(a), float(b)


ISO_A, ISO_B = _fit_isotherm()


@njit(cache=True)
def emc_nelson(t_c, rh_frac, a, b):
    h = rh_frac
    if h < 0.005:
        h = 0.005
    if h > 0.995:
        h = 0.995
    dg = -(RGAS * (t_c + 273.15) / MW) * math.log(h)
    m = a - b * math.log(dg)
    if m < 0.005:
        m = 0.005
    return m


@njit(cache=True)
def esat(t_c):
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


@njit(cache=True)
def _tridiag(a, b, c, d, x):
    n = d.shape[0]
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]


@njit(cache=True)
def nelson_run(t_c, rh, srad, rain_mm, wind_ms, radius_cm, d_ref, film, w_max, rain_capture, absorb, iso_a, iso_b,
               e_act, nsub, nnode, w0, kappa_w=4.0, fsp=0.30, cap_boost=0.0):
    """Run one stick class through an hourly series. Returns average stick moisture (fraction) at the end of each
    hour. d_ref: bound-water diffusivity at 20 C (cm2/s). film: surface transfer coefficient (cm/s).
    absorb: fraction of global radiation absorbed per unit stick surface.
    Diffusivity grows with local moisture, D = D_T exp(kappa_w (w - 0.1)), and liquid (capillary) transport above
    fibre saturation adds a factor 1 + cap_boost (w - fsp)/fsp, so rain water soaks in instead of staying in the
    outer shell."""
    nt = t_c.shape[0]
    out = np.empty(nt)
    rr = radius_cm
    dr = rr / nnode
    # node radii (cell centres) and volumes per unit length (2 pi r dr)
    rc = np.empty(nnode)
    vol = np.empty(nnode)
    for i in range(nnode):
        rc[i] = (i + 0.5) * dr
        vol[i] = ((i + 1) * dr) ** 2 - (i * dr) ** 2
    w = np.full(nnode, w0)
    a = np.zeros(nnode)
    bb = np.zeros(nnode)
    c = np.zeros(nnode)
    d = np.zeros(nnode)
    dn = np.zeros(nnode)
    dt = 3600.0 / nsub
    diam_m = 2.0 * rr / 100.0
    rho = 0.40
    for k in range(nt):
        # stick temperature from a steady energy balance
        u = wind_ms[k] * 0.3
        if u < 0.1:
            u = 0.1
        hc = 4.03 * math.sqrt(u / diam_m)
        dT = absorb * srad[k] / (hc + 6.0)
        ts = t_c[k] + dT
        hs = rh[k] / 100.0 * esat(t_c[k]) / esat(ts)
        raining = rain_mm[k] > 0.05
        if raining:
            weq = w_max
            fc = film * 20.0
        else:
            weq = emc_nelson(ts, hs, iso_a, iso_b)
            fc = film
        D = d_ref * math.exp(-e_act / RGAS * (1.0 / (ts + 273.15) - 1.0 / 293.15))
        # rain uptake budget for this hour (g water per cm2 of stick surface -> fraction of stick mass)
        if raining:
            budget = rain_capture * rain_mm[k] * 0.1 / math.pi
            mass_per_area = rho * rr / 2.0
            wcap = budget / mass_per_area
        else:
            wcap = 1e9
        w_prev = w.copy()
        wstart = 0.0
        for i in range(nnode):
            wstart += w[i] * vol[i]
        wstart /= rr * rr
        for _s in range(nsub):
            for i in range(nnode):
                fw = math.exp(kappa_w * (w[i] - 0.1))
                if w[i] > fsp:
                    fw *= 1.0 + cap_boost * (w[i] - fsp) / fsp
                dn[i] = D * fw
            # implicit Euler for  d(w)/dt = (1/r) d/dr (r D dw/dr), flux BC at r = R: -D dw/dr = fc (w_s - weq)
            for i in range(nnode):
                rin = i * dr
                rout = (i + 1) * dr
                vi = 0.5 * vol[i]          # r dr  (area per 2 pi)
                kin = 0.5 * (dn[i] + dn[i - 1]) * rin / dr / vi if i > 0 else 0.0
                kout = 0.5 * (dn[i] + dn[i + 1]) * rout / dr / vi if i < nnode - 1 else 0.0
                a[i] = -kin * dt
                c[i] = -kout * dt
                bb[i] = 1.0 + (kin + kout) * dt
                d[i] = w[i]
                if i == nnode - 1:
                    # surface flux with half-cell resistance in series with the film
                    keff = 1.0 / (1.0 / fc + 0.5 * dr / dn[i])
                    g = keff * rout / vi * dt
                    bb[i] += g
                    d[i] += g * weq
            _tridiag(a, bb, c, d, w)
        wend = 0.0
        for i in range(nnode):
            wend += w[i] * vol[i]
        wend /= rr * rr
        if raining and wend - wstart > wcap:
            # the stick cannot gain more water than it caught: scale the hour's gain node by node
            f = wcap / (wend - wstart)
            for i in range(nnode):
                w[i] = w_prev[i] + f * (w[i] - w_prev[i])
            wend = wstart + wcap
        out[k] = wend
    return out


DEFAULT_NELSON = dict(film=2.0e-4, w_max=0.45, rain_capture=0.3, absorb=0.35, e_act=38000.0, nsub=4, nnode=11,
                      kappa_w=4.0, fsp=0.30, cap_boost=0.0)


_DREF_CACHE = {}


def diffusivity_for_timelag(radius_cm, tau_h, film, P=None):
    """Bound-water diffusivity at 20 C (cm2/s) for which the stick's response to a humidity step (20 C, no sun,
    RH 20 -> 80 %) reaches 1 - 1/e of the way in tau_h hours: Byram's definition of the timelag class."""
    from scipy.optimize import brentq
    P = dict(DEFAULT_NELSON, **(P or {}))
    key = (radius_cm, tau_h, film, P['w_max'], P['kappa_w'], P['fsp'], P['cap_boost'], P['e_act'], int(P['nsub']),
           int(P['nnode']))
    if key in _DREF_CACHE:
        return _DREF_CACHE[key]
    n = int(max(8 * tau_h, 12))
    T = np.full(n, 20.0)
    H = np.full(n, 80.0)
    z = np.zeros(n)
    U = np.full(n, 2.0)
    w0 = emc_nelson(20.0, 0.20, ISO_A, ISO_B)
    wf = emc_nelson(20.0, 0.80, ISO_A, ISO_B)

    def f(lD):
        m = nelson_run(T, H, z, z, U, radius_cm, float(np.exp(lD)), film, P['w_max'], P['rain_capture'], 0.0,
                       ISO_A, ISO_B, P['e_act'], int(P['nsub']), int(P['nnode']), float(w0), P['kappa_w'], P['fsp'],
                       P['cap_boost'])
        fr = np.r_[0.0, (m - w0) / (wf - w0)]
        j = int(np.argmax(fr >= 1 - np.exp(-1)))
        if fr[j] < 1 - np.exp(-1):
            return np.log(n) - np.log(tau_h)
        t = (j - 1) + (1 - np.exp(-1) - fr[j - 1]) / (fr[j] - fr[j - 1]) if j > 0 else 0.0
        return np.log(max(t, 1e-3)) - np.log(tau_h)
    D = float(np.exp(brentq(f, np.log(1e-9), np.log(1e-2), xtol=1e-4)))
    _DREF_CACHE[key] = D
    return D


def dead_moisture(weather, method='nelson', classes=('1h', '10h', '100h'), init=None, params=None, spinup_h=0):
    """Hourly dead fuel moisture (%) for a weather DataFrame with T (C), RH (%), precip (mm), wind_speed (m/s at 10 m),
    srad (W/m2; estimated if missing). Returns a DataFrame with columns m1h, m10h, m100h (and m1000h if asked).
    `params` overrides DEFAULT_NELSON and may hold per-class 'tau_<cls>' timelags (h) and 'd_ref_<cls>' (cm2/s)."""
    import pandas as pd
    from .weather import ensure_srad
    w = ensure_srad(weather)
    P = dict(DEFAULT_NELSON)
    P.update(params or {})
    T = w['T'].to_numpy(float)
    H = np.clip(w['RH'].to_numpy(float), 1, 100)
    S = w['srad'].to_numpy(float)
    Pr = w['precip'].fillna(0).to_numpy(float) if 'precip' in w else np.zeros(len(w))
    U = w['wind_speed'].to_numpy(float) if 'wind_speed' in w else np.full(len(w), 2.0)
    out = {}
    for cls in classes:
        name = 'm' + cls
        if method == 'given':
            out[name] = w[name].to_numpy(float)
            continue
        if method == 'nelson':
            r = STICK_RADIUS_CM[cls]
            tau = P.get('tau_' + cls, NOMINAL_TAU_H[cls])
            dref = P.get('d_ref_' + cls) or diffusivity_for_timelag(r, tau, P['film'], P)
            w0 = (init or {}).get(name, None)
            if w0 is None:
                w0 = emc_nelson(T[0], H[0] / 100.0, ISO_A, ISO_B)
            else:
                w0 = w0 / 100.0
            m = nelson_run(T, H, S, Pr, U, r, dref, P['film'], P['w_max'], P['rain_capture'], P['absorb'],
                           ISO_A, ISO_B, P['e_act'], int(P['nsub']), int(P['nnode']), float(w0), P['kappa_w'],
                           P['fsp'], P['cap_boost'])
            out[name] = 100.0 * m
        elif method == 'timelag':
            tau = P.get('tau_' + cls, NOMINAL_TAU_H[cls])
            emc = simard_emc(T, H)
            m = np.empty(len(T))
            cur = (init or {}).get(name, emc[0])
            a = 1.0 - math.exp(-1.0 / tau)
            for k in range(len(T)):
                tgt = 35.0 if Pr[k] > 0.05 else emc[k]
                aa = min(1.0, a * (5.0 if Pr[k] > 0.05 else 1.0))
                cur = cur + aa * (tgt - cur)
                m[k] = cur
            out[name] = m
        else:
            raise ValueError(method)
    df = pd.DataFrame(out, index=w.index)
    if spinup_h:
        df = df.iloc[spinup_h:]
    return df


def gsi(tmin_c, vpd_pa, daylength_h, window=21):
    """Growing Season Index (Jolly et al. 2005) with 21-day running mean."""
    def ramp(x, lo, hi):
        return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0, 1)
    itmin = ramp(tmin_c, -2.0, 5.0)
    ivpd = 1.0 - ramp(vpd_pa, 900.0, 4100.0)
    iph = ramp(daylength_h, 10.0, 11.0)
    g = itmin * ivpd * iph
    k = np.ones(window) / window
    gp = np.pad(g, (window - 1, 0), mode='edge')
    return np.convolve(gp, k, mode='valid')


def live_moisture_from_gsi(g, greenup=0.5, herb=(30.0, 250.0), woody=(60.0, 200.0)):
    """NFDRS 2016-style mapping of GSI to live herbaceous and woody moisture (%), linear above the green-up threshold."""
    f = np.clip((np.asarray(g, float) - greenup) / (1.0 - greenup), 0, 1)
    return herb[0] + f * (herb[1] - herb[0]), woody[0] + f * (woody[1] - woody[0])


def daylength_h(doy, lat_deg):
    decl = 0.409 * np.sin(2 * np.pi * np.asarray(doy) / 365.0 - 1.39)
    x = -np.tan(np.radians(lat_deg)) * np.tan(decl)
    return 24.0 / np.pi * np.arccos(np.clip(x, -1, 1))
