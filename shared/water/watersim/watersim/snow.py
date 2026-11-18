"""Snow models, one elevation band per call (the engine loops over bands).

degree_day  HBV-light snow routine (Bergstrom 1976; Seibert & Vis 2012): rain/snow partition over a temperature
            interval TT +- TTI/2, snowfall correction SFCF, melt CFMAX (T - TT), refreeze CFR CFMAX (TT - T), liquid
            water held up to CWH x ice, rain on snow joins the liquid store.
energy_balance  A daily single-layer energy-balance model in the spirit of Walter et al. (2005) and the Utah Energy
            Balance model (Tarboton & Luce 1996): net shortwave with an ageing albedo, Brutsaert (1975) clear-sky
            long-wave with the Unsworth & Monteith (1975) cloud correction, neutral bulk-transfer sensible and latent heat
            with a Richardson-number stability reduction, ground heat, rain heat, a cold-content store, refreeze,
            sublimation/deposition and liquid water holding.

Elevation bands: the forcing is a basin average at reference elevation z_ref. Band temperature T_b = T + TLAPSE dz
(TLAPSE in C/km, dz in km); band precipitation P_b = P m_b with m_b = max(0, 1 + PGRAD dz) normalised so that the
area-weighted mean of P_b equals P (the forcing already is the basin mean). Mass is conserved exactly.
"""
import math

from ._nb import njit
from .constants import LAMBDA_F, LAMBDA_S, LAMBDA_V, C_ICE, C_WATER, CP_AIR, SIGMA_MJ_DAY, VON_KARMAN

# name, default, lo, hi, unit, description
DD_PARAMS = [
    ('TT', 0.0, -2.5, 2.5, 'C', 'threshold temperature for snowfall and melt'),
    ('TTI', 2.0, 0.0, 4.0, 'C', 'width of the rain/snow transition interval'),
    ('CFMAX', 3.5, 1.0, 8.0, 'mm/C/day', 'degree-day melt factor'),
    ('SFCF', 1.0, 0.7, 1.4, '-', 'snowfall correction factor (gauge undercatch, sublimation)'),
    ('CFR', 0.05, 0.0, 0.1, '-', 'refreezing coefficient'),
    ('CWH', 0.1, 0.0, 0.2, '-', 'liquid water holding capacity of snow (fraction of ice)'),
    ('TLAPSE', -6.5, -9.0, -4.0, 'C/km', 'temperature lapse rate between elevation bands'),
    ('PGRAD', 0.3, 0.0, 1.0, '1/km', 'relative precipitation gradient with elevation (normalised)'),
]

EB_PARAMS = [
    ('TT', 0.5, -1.5, 2.5, 'C', 'centre of the rain/snow transition'),
    ('TTI', 2.0, 0.0, 4.0, 'C', 'width of the rain/snow transition interval'),
    ('SFCF', 1.0, 0.7, 1.4, '-', 'snowfall correction factor'),
    ('CWH', 0.05, 0.0, 0.2, '-', 'liquid water holding capacity'),
    ('ALB_MAX', 0.85, 0.75, 0.92, '-', 'fresh-snow albedo'),
    ('ALB_MIN', 0.45, 0.30, 0.60, '-', 'old/melting-snow albedo'),
    ('ALB_DECAY', 12.0, 3.0, 30.0, 'day', 'albedo e-folding age (melting snow ages 3x faster)'),
    ('Z0', 0.002, 0.0002, 0.02, 'm', 'aerodynamic roughness of the snow surface'),
    ('TURB', 1.0, 0.2, 3.0, '-', 'multiplier on the turbulent transfer coefficient'),
    ('GHEAT', 0.173, 0.0, 0.5, 'MJ/m2/day', 'ground heat flux into the pack (Walter et al. 2005: 173 kJ/m2/day)'),
    ('CANOPY', 1.0, 0.2, 1.0, '-', 'canopy shortwave transmissivity (1 = open)'),
    ('TLAPSE', -6.5, -9.0, -4.0, 'C/km', 'temperature lapse rate between elevation bands'),
    ('PGRAD', 0.3, 0.0, 1.0, '1/km', 'relative precipitation gradient with elevation (normalised)'),
]

DD_INDEX = {p[0]: i for i, p in enumerate(DD_PARAMS)}
EB_INDEX = {p[0]: i for i, p in enumerate(EB_PARAMS)}


@njit
def snow_fraction(T, TT, TTI):
    """Fraction of precipitation falling as snow: 1 below TT - TTI/2, 0 above TT + TTI/2, linear between."""
    if TTI <= 0.0:
        return 1.0 if T <= TT else 0.0
    f = (TT + 0.5 * TTI - T) / TTI
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


@njit
def dd_step(ice, liq, P, T, TT, TTI, CFMAX, SFCF, CFR, CWH, dt):
    """One degree-day step for one band. Returns (ice, liq, rain, snow, melt, refreeze, outflow).
    Mass: snow + rain = d(ice + liq) + outflow."""
    fs = snow_fraction(T, TT, TTI)
    snow = P * fs * SFCF
    rain = P * (1.0 - fs)
    ice += snow
    melt = 0.0
    refr = 0.0
    if T > TT:
        melt = CFMAX * (T - TT) * dt
        if melt > ice:
            melt = ice
        ice -= melt
        liq += melt
    else:
        refr = CFR * CFMAX * (TT - T) * dt
        if refr > liq:
            refr = liq
        liq -= refr
        ice += refr
    if ice > 0.0:
        liq += rain
        out = liq - CWH * ice
        if out < 0.0:
            out = 0.0
        liq -= out
    else:
        out = liq + rain
        liq = 0.0
        ice = 0.0
    return ice, liq, rain, snow, melt, refr, out


@njit
def esat_ice_kpa(T):
    return 0.61115 * math.exp(22.452 * T / (T + 272.55))


@njit
def eb_step(ice, liq, cc, age, P, T, Rs, ea, u10, Rso, Pa, prm, dt):
    """One daily energy-balance step for one band.
    State: ice, liq (mm), cc cold content (MJ/m2, >= 0), age (days since snowfall).
    Forcing: P (mm), T daily mean (C), Rs shortwave (MJ/m2/day), ea vapour pressure (kPa), u10 wind (m/s, 10 m),
    Rso clear-sky shortwave (MJ/m2/day), Pa pressure (kPa). prm in EB_PARAMS order.
    Returns (ice, liq, cc, age, rain, snow, melt, refreeze, outflow, sublimation_loss, Qnet).
    Mass: snow + rain = d(ice + liq) + outflow + sublimation_loss."""
    TT = prm[0]
    TTI = prm[1]
    SFCF = prm[2]
    CWH = prm[3]
    amax = prm[4]
    amin = prm[5]
    adec = prm[6]
    z0 = prm[7]
    turb = prm[8]
    gheat = prm[9]
    canopy = prm[10]
    fs = snow_fraction(T, TT, TTI)
    snow = P * fs * SFCF
    rain = P * (1.0 - fs)
    if ice + liq + snow <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, rain, snow, 0.0, 0.0, rain, 0.0, 0.0
    ice += snow
    # fresh snow resets the age (partially for small falls)
    if snow > 0.0:
        r = snow / 5.0
        if r > 1.0:
            r = 1.0
        age = age * (1.0 - r)
    Ts = T if T < 0.0 else 0.0
    cc += C_ICE * snow * (-Ts)       # new snow arrives at the surface temperature
    alb = amin + (amax - amin) * math.exp(-age / adec)
    swe = ice + liq
    alb_eff = 0.2 + (alb - 0.2) * (1.0 - math.exp(-swe / 15.0))  # shallow pack shows the ground
    sw = (1.0 - alb_eff) * Rs * canopy * dt
    TaK = T + 273.15
    TsK = Ts + 273.15
    eps_c = 1.24 * (ea * 10.0 / TaK) ** (1.0 / 7.0)
    if eps_c > 1.0:
        eps_c = 1.0
    if Rso > 0.0:
        cf = 1.0 - Rs / Rso
        if cf < 0.0:
            cf = 0.0
        if cf > 1.0:
            cf = 1.0
    else:
        cf = 0.5
    eps = (1.0 - 0.84 * cf) * eps_c + 0.84 * cf
    eps = canopy * eps + (1.0 - canopy) * 1.0
    lwin = eps * SIGMA_MJ_DAY * TaK ** 4 * dt
    lwout = 0.98 * SIGMA_MJ_DAY * TsK ** 4 * dt
    # turbulent fluxes, neutral bulk transfer at 2 m with stability reduction
    lz = math.log(2.0 / z0)
    u2 = u10 * lz / math.log(10.0 / z0)
    if u2 < 0.3:
        u2 = 0.3
    ra = lz * lz / (VON_KARMAN * VON_KARMAN * u2)
    ri = 9.81 * 2.0 * (T - Ts) / (TaK * u2 * u2)
    fst = 1.0 / (1.0 + 10.0 * ri) if ri > 0.0 else 1.0
    rho = Pa * 1000.0 / (287.05 * TaK)
    ch = turb * fst / ra * 86400.0 * dt   # m per step
    H = rho * CP_AIR * (T - Ts) * ch
    es_s = esat_ice_kpa(Ts)
    emass = rho * 0.622 * (ea - es_s) / Pa * ch    # kg/m2 per step, + = condensation/deposition
    lam = LAMBDA_V if liq > 0.0 else LAMBDA_S
    LE = lam * emass
    G = gheat * dt
    Qr = C_WATER * rain * (T if T > 0.0 else 0.0)
    Q = sw + lwin - lwout + H + LE + G + Qr
    liq += rain
    melt = 0.0
    refr = 0.0
    if Q >= 0.0:
        d = Q if Q < cc else cc
        cc -= d
        e = Q - d
        melt = e / LAMBDA_F
        if melt > ice:
            melt = ice
        ice -= melt
        liq += melt
    else:
        refr = -Q / LAMBDA_F
        if refr > liq:
            refr = liq
        liq -= refr
        ice += refr
        cc += -(Q + refr * LAMBDA_F)
    ccmax = C_ICE * ice * (-Ts)
    if cc > ccmax:
        cc = ccmax
    if cc > 0.0 and liq > 0.0:
        r2 = cc / LAMBDA_F
        if r2 > liq:
            r2 = liq
        liq -= r2
        ice += r2
        refr += r2
        cc -= r2 * LAMBDA_F
    # mass exchange with the atmosphere
    subl = 0.0
    if emass < 0.0:
        loss = -emass
        a = loss if loss < liq else liq
        liq -= a
        loss -= a
        b = loss if loss < ice else ice
        ice -= b
        subl = a + b
    else:
        ice += emass
        subl = -emass
    if ice > 0.0:
        out = liq - CWH * ice
        if out < 0.0:
            out = 0.0
        liq -= out
    else:
        out = liq
        liq = 0.0
        ice = 0.0
        cc = 0.0
    age += dt * (3.0 if T > 0.0 else 1.0)
    return ice, liq, cc, age, rain, snow, melt, refr, out, subl, Q
