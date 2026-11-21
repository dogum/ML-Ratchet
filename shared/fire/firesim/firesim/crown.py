"""Crown fire: Van Wagner (1977) initiation and active-spread criteria; crown spread rate from Rothermel (1991) or
Cruz, Alexander & Wakimoto (2005); crown fraction burned and passive-fire spread from Scott & Reinhardt (2001);
crown fire length-to-width ratio from Rothermel (1991). SI units throughout (m, m/min, kW/m, kJ/m2, kg/m3).
"""
import math

import numpy as np

from ._nb import njit

CROWN_HEAT = 18000.0     # kJ/kg canopy fuel heat content (Scott & Reinhardt 2001)
S0_ACTIVE = 3.0          # kg m^-2 min^-1 critical mass flow rate (Van Wagner 1977): R_active = 3 / CBD (m/min)


@njit(cache=True)
def van_wagner_i0(cbh_m, fmc_pct):
    """Critical surface fireline intensity for crown ignition (kW/m), Van Wagner (1977):
    I0 = (0.010 CBH (460 + 25.9 FMC))^1.5, CBH m, FMC foliar moisture %."""
    if cbh_m < 0.1:
        cbh_m = 0.1
    return (0.010 * cbh_m * (460.0 + 25.9 * fmc_pct)) ** 1.5


@njit(cache=True)
def r_active_critical(cbd):
    """Van Wagner (1977) critical spread rate for active crowning (m/min) = 3 / CBD."""
    if cbd <= 0.0:
        return 1e30
    return S0_ACTIVE / cbd


@njit(cache=True)
def cruz2005_active(u10_kmh, cbd, effm_pct):
    """Cruz et al. (2005) active crown fire ROS (m/min): 11.02 U10^0.90 CBD^0.19 exp(-0.17 EFFM)."""
    if u10_kmh <= 0.0 or cbd <= 0.0:
        return 0.0
    return 11.02 * u10_kmh ** 0.90 * cbd ** 0.19 * math.exp(-0.17 * effm_pct)


@njit(cache=True)
def cruz2005_passive(u10_kmh, cbd, effm_pct):
    """Cruz et al. (2005) passive crown fire ROS: CROS_A exp(-CAC), CAC = CROS_A / (3/CBD)."""
    ra = cruz2005_active(u10_kmh, cbd, effm_pct)
    cac = ra / r_active_critical(cbd)
    return ra * math.exp(-cac)


@njit(cache=True)
def crown_lw(u20_mph):
    """Rothermel (1991) crown fire length-to-width ratio: 1 + 0.125 U20 (mi/h)."""
    return 1.0 + 0.125 * max(u20_mph, 0.0)


@njit(cache=True)
def cfb_scott_reinhardt(r_surf, r_init, r_active_c):
    """Crown fraction burned (Scott & Reinhardt 2001): CFB = 1 - exp(-a (Rs - R'init)),
    a = -ln(0.1) / (0.9 (R'active - R'init)); R'active is the surface ROS at which the crown fire would be active
    (here: the head surface ROS scaled so that CFB reaches 0.9 at the active threshold)."""
    if r_surf <= r_init:
        return 0.0
    d = r_active_c - r_init
    if d <= 1e-9:
        return 1.0
    a = 2.302585093 / (0.9 * d)
    c = 1.0 - math.exp(-a * (r_surf - r_init))
    if c < 0.0:
        c = 0.0
    if c > 1.0:
        c = 1.0
    return c


@njit(cache=True)
def cfb_van_wagner(r_surf, r_init):
    """Van Wagner (1993) crown fraction burned: 1 - exp(-0.23 (R - R0)), R and R0 (critical surface ROS) in m/min."""
    if r_surf <= r_init:
        return 0.0
    return 1.0 - math.exp(-0.23 * (r_surf - r_init))


def crown_fire(surface, cbh, cbd, ch, fmc=100.0, u20_ms=0.0, u10_ms=None, m1_pct=6.0, r10_mmin=None,
               method='rothermel1991', cfb_method='van_wagner1993', r_sa_mmin=None):
    """Crown fire behaviour for one head fire (scalars). `surface` is a dict from behave() surface output with
    'ros_mmin', 'hpa_kjm2', 'fli_kwm'. r10_mmin: fuel model 10 ROS at WAF 0.4 (needed for rothermel1991).

    Returns dict: i0, r_init, r_active_crit, cros_active, cac, cfb, fire_type (1 surface, 2 passive, 3 active),
    ros_final, fli_final, flame_length (Thomas 1963 for crown), crown_lw.
    """
    rs = surface['ros_mmin']
    hpa = surface['hpa_kjm2']
    fli = surface['fli_kwm']
    i0 = van_wagner_i0(cbh, fmc) if cbd > 0 and ch > cbh else np.inf
    r_init = 60.0 * i0 / hpa if hpa > 0 else np.inf
    rac = r_active_critical(cbd)
    if u10_ms is None:
        u10_ms = u20_ms * 1.15
    if method == 'cruz2005':
        cros = cruz2005_active(u10_ms * 3.6, cbd, m1_pct)
    else:
        cros = 3.34 * (r10_mmin if r10_mmin is not None else 0.0)
    cac = cros / rac if np.isfinite(rac) else 0.0
    res = dict(i0_kwm=i0, r_init_mmin=r_init, r_active_crit_mmin=rac, cros_active_mmin=cros, cac=cac,
               crown_lw=crown_lw(u20_ms * 2.23694))
    cfl = cbd * max(ch - cbh, 0.0)
    if not (fli > i0):
        res.update(fire_type=1, cfb=0.0, ros_mmin=rs, fli_kwm=fli, flame_length_m=0.0775 * fli ** 0.46 if fli > 0 else 0)
        return res
    if cac >= 1.0:
        ftype = 3
    else:
        ftype = 2
    if cfb_method == 'scott_reinhardt':
        # R'SA: surface head ROS at the 20-ft wind where the crown fire becomes active (behave() finds it by root
        # search, the crowning index); without it fall back to the Van Wagner (1993) form
        if r_sa_mmin is None or not np.isfinite(r_sa_mmin):
            cfb = cfb_van_wagner(rs, r_init)
        else:
            cfb = cfb_scott_reinhardt(rs, r_init, max(r_sa_mmin, r_init + 1e-6))
    else:
        cfb = cfb_van_wagner(rs, r_init)
    if method == 'cruz2005':
        rf = cros if ftype == 3 else max(rs, cros * math.exp(-cac))
    else:
        rf = cros if ftype == 3 else rs + cfb * max(cros - rs, 0.0)
    fli_f = (hpa + cfb * cfl * CROWN_HEAT) * rf / 60.0
    fl = 0.0266 * fli_f ** (2.0 / 3.0)
    res.update(fire_type=ftype, cfb=cfb, ros_mmin=rf, fli_kwm=fli_f, flame_length_m=fl, cfl_kgm2=cfl)
    return res
