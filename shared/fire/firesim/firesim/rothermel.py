"""Rothermel (1972) surface fire spread with the Albini (1976) changes, as written out in Andrews (2018) RMRS-GTR-371
(Tables 6a-c and 26), plus the BehavePlus spread-direction and ellipse equations.

Internal units are the published English units (ft, lb, Btu, min). Public wrappers convert to SI.
All kernels are numba-compiled when numba is available.

Bundle (moisture-dependent, wind/slope-independent) layout, see `B_*` indices.
"""
import math

import numpy as np

from ._nb import njit
from .fuels import FM, RHO_P, S_T, S_E

# bundle indices
B_R0, B_IR, B_PHIS, B_C, B_B, B_E, B_BRATIO, B_HA, B_TR, B_SIGMA, B_IRD, B_IRL, B_MXL, B_BETA, B_RHOB, B_SINK, \
    B_XI, B_WN, B_CURED, B_MD, B_ML, B_ETAD, B_ETAL, B_WND, B_WNL = range(25)
NBUNDLE = 25

# unit conversions
FTMIN_TO_MMIN = 0.3048
MMIN_TO_FTMIN = 1.0 / 0.3048
MS_TO_FTMIN = 196.850394
FTMIN_TO_MS = 1.0 / 196.850394
MPH_TO_FTMIN = 88.0
BTUFTS_TO_KWM = 3.46141
BTUFT2_TO_KJM2 = 11.35653
BTUFT2MIN_TO_KWM2 = 11.35653 / 60.0
FT_TO_M = 0.3048


@njit(cache=True)
def _subclass(s):
    if s >= 1200.0:
        return 0
    if s >= 192.0:
        return 1
    if s >= 96.0:
        return 2
    if s >= 48.0:
        return 3
    if s >= 16.0:
        return 4
    return 5


@njit(cache=True)
def weights(w0, sav, f, g, fw):
    """Rothermel (1972) f_ij, category f_i (fw[0] dead, fw[1] live) and Albini (1976) g_ij size-subclass weights."""
    A = np.zeros(6)
    for j in range(6):
        f[j] = 0.0
        g[j] = 0.0
        if w0[j] > 0.0:
            A[j] = sav[j] * w0[j] / RHO_P
    Ad = A[0] + A[1] + A[2] + A[3]
    Al = A[4] + A[5]
    At = Ad + Al
    fw[0] = 0.0
    fw[1] = 0.0
    if At <= 0.0:
        return
    for j in range(4):
        if Ad > 0.0:
            f[j] = A[j] / Ad
    for j in range(4, 6):
        if Al > 0.0:
            f[j] = A[j] / Al
    fw[0] = Ad / At
    fw[1] = Al / At
    for j in range(6):
        lo = 0 if j < 4 else 4
        hi = 4 if j < 4 else 6
        cj = _subclass(sav[j])
        s = 0.0
        for k in range(lo, hi):
            if w0[k] > 0.0 and _subclass(sav[k]) == cj:
                s += f[k]
        g[j] = s if (sav[j] >= 16.0 and w0[j] > 0.0) else 0.0


@njit(cache=True)
def bundle_one(w0_in, sav, heat, depth, mx_dead, dynamic, m, curing, out):
    """Moisture-dependent part of the Rothermel model for one fuel bed.

    w0_in: (6,) oven-dry loads lb/ft2 (dead herb slot ignored; filled by curing transfer)
    sav: (6,), heat: (6,), depth ft, mx_dead fraction, dynamic bool
    m: (6,) moisture fractions [1h, 10h, 100h, dead herb (normally = 1h), live herb, live woody]
    curing: fraction transferred live herb -> dead herb; if < 0, computed from live herb moisture (Scott & Burgan 2005,
            T = 1.333 - 1.11 Mherb clipped to [0,1]; GTR-371 prints 1.33, but its
            worked GR2 example (24.1 / 6.5 ft/min) is reproduced only with 1.333, as in BehavePlus); ignored for static models.
    out: (NBUNDLE,) array written in place.
    """
    w0 = np.empty(6)
    for j in range(6):
        w0[j] = w0_in[j]
    w0[3] = 0.0
    T = 0.0
    if dynamic and w0[4] > 0.0:
        if curing >= 0.0:
            T = min(1.0, curing)
        else:
            T = -1.11 * m[4] + 1.333
            if T < 0.0:
                T = 0.0
            if T > 1.0:
                T = 1.0
        w0[3] = T * w0[4]
        w0[4] = w0[4] - w0[3]
    for k in range(NBUNDLE):
        out[k] = 0.0
    out[B_CURED] = T
    f = np.zeros(6)
    g = np.zeros(6)
    fw = np.zeros(2)
    weights(w0, sav, f, g, fw)
    fd = fw[0]
    fl = fw[1]
    if fd + fl <= 0.0 or depth <= 0.0:
        return
    Ad = fd
    Al = fl
    # net loads, characteristic values by category
    wn_d = 0.0
    wn_l = 0.0
    hd = 0.0
    hl = 0.0
    md = 0.0
    ml = 0.0
    sd = 0.0
    sl = 0.0
    for j in range(4):
        if w0[j] > 0.0:
            wn_d += g[j] * w0[j] * (1.0 - S_T)
            hd += f[j] * heat[j]
            md += f[j] * m[j]
            sd += f[j] * sav[j]
    for j in range(4, 6):
        if w0[j] > 0.0:
            wn_l += g[j] * w0[j] * (1.0 - S_T)
            hl += f[j] * heat[j]
            ml += f[j] * m[j]
            sl += f[j] * sav[j]
    etas = 0.174 * S_E ** -0.19
    if etas > 1.0:
        etas = 1.0
    sigma = fd * sd + fl * sl
    # live moisture of extinction (Albini 1976)
    num = 0.0
    den = 0.0
    mnum = 0.0
    for j in range(4):
        if w0[j] > 0.0:
            ex = w0[j] * math.exp(-138.0 / sav[j])
            num += ex
            mnum += m[j] * ex
    for j in range(4, 6):
        if w0[j] > 0.0:
            den += w0[j] * math.exp(-500.0 / sav[j])
    mx_live = mx_dead
    if den > 0.0:
        W = num / den
        mfdead = mnum / num if num > 0.0 else 0.0
        mx_live = 2.9 * W * (1.0 - mfdead / mx_dead) - 0.226
        if mx_live < mx_dead:
            mx_live = mx_dead
    # moisture damping
    eta_d = 0.0
    if Ad > 0.0:
        r = md / mx_dead
        if r > 1.0:
            r = 1.0
        eta_d = 1.0 - 2.59 * r + 5.11 * r * r - 3.52 * r * r * r
    eta_l = 0.0
    if Al > 0.0:
        r = ml / mx_live
        if r > 1.0:
            r = 1.0
        eta_l = 1.0 - 2.59 * r + 5.11 * r * r - 3.52 * r * r * r
    # bed
    wtot = 0.0
    for j in range(6):
        wtot += w0[j]
    rho_b = wtot / depth
    beta = rho_b / RHO_P
    beta_op = 3.348 * sigma ** -0.8189
    bratio = beta / beta_op
    s15 = sigma ** 1.5
    gmax = s15 / (495.0 + 0.0594 * s15)
    Aexp = 133.0 * sigma ** -0.7913
    gam = gmax * bratio ** Aexp * math.exp(Aexp * (1.0 - bratio))
    ird = gam * wn_d * hd * eta_d * etas
    irl = gam * wn_l * hl * eta_l * etas
    ir = ird + irl
    xi = math.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    # heat sink
    sink_d = 0.0
    sink_l = 0.0
    for j in range(4):
        if w0[j] > 0.0:
            sink_d += f[j] * math.exp(-138.0 / sav[j]) * (250.0 + 1116.0 * m[j])
    for j in range(4, 6):
        if w0[j] > 0.0:
            sink_l += f[j] * math.exp(-138.0 / sav[j]) * (250.0 + 1116.0 * m[j])
    sink = rho_b * (fd * sink_d + fl * sink_l)
    r0 = ir * xi / sink if sink > 0.0 else 0.0
    tr = 384.0 / sigma
    out[B_R0] = r0
    out[B_IR] = ir
    out[B_PHIS] = 5.275 * beta ** -0.3
    out[B_C] = 7.47 * math.exp(-0.133 * sigma ** 0.55)
    out[B_B] = 0.02526 * sigma ** 0.54
    out[B_E] = 0.715 * math.exp(-3.59e-4 * sigma)
    out[B_BRATIO] = bratio
    out[B_HA] = ir * tr
    out[B_TR] = tr
    out[B_SIGMA] = sigma
    out[B_IRD] = ird
    out[B_IRL] = irl
    out[B_MXL] = mx_live
    out[B_BETA] = beta
    out[B_RHOB] = rho_b
    out[B_SINK] = sink
    out[B_XI] = xi
    out[B_WN] = wn_d + wn_l
    out[B_MD] = md
    out[B_ML] = ml
    out[B_ETAD] = eta_d
    out[B_ETAL] = eta_l
    out[B_WND] = wn_d
    out[B_WNL] = wn_l


@njit(cache=True)
def bundles_for(rows, w0, sav, heat, depth, mx, dyn, m, curing, out):
    """Vectorised: bundles for fuel table rows `rows` (int array), one moisture vector m (6,), into out (n, NBUNDLE)."""
    for i in range(rows.shape[0]):
        r = rows[i]
        if r < 0:
            for k in range(NBUNDLE):
                out[i, k] = 0.0
            continue
        bundle_one(w0[r], sav[r], heat[r], depth[r], mx[r], dyn[r], m, curing, out[i])


# wind limit modes
WL_NONE, WL_ROTHERMEL, WL_ANDREWS = 0, 1, 2
# length-to-width modes
LW_BEHAVE, LW_ANDERSON = 0, 1


@njit(cache=True)
def lw_ratio(ue_mph, mode, lw_max):
    if ue_mph <= 1e-7:
        return 1.0
    if mode == LW_BEHAVE:
        z = 1.0 + 0.25 * ue_mph
    else:
        z = 0.936 * math.exp(0.2566 * ue_mph) + 0.461 * math.exp(-0.1548 * ue_mph) - 0.397
    if z < 1.0:
        z = 1.0
    if z > lw_max:
        z = lw_max
    return z


@njit(cache=True)
def combine(bd, u_mid_ftmin, wind_rel_upslope, tan_slope, wl_mode, lw_mode, lw_max, out):
    """Head fire under wind blowing toward `wind_rel_upslope` (radians from upslope, clockwise) on slope tan_slope.
    BehavePlus vector addition (Andrews 2018 Table 26).

    out[0] head ROS ft/min, out[1] direction of max spread relative to upslope (rad, clockwise),
    out[2] effective wind speed ft/min, out[3] L/W, out[4] eccentricity, out[5] phi_w, out[6] phi_s,
    out[7] 1 if the wind limit was applied.
    """
    r0 = bd[B_R0]
    for k in range(8):
        out[k] = 0.0
    out[3] = 1.0
    if r0 <= 0.0:
        return
    bratio = bd[B_BRATIO]
    C = bd[B_C]
    Bw = bd[B_B]
    E = bd[B_E]
    u = u_mid_ftmin if u_mid_ftmin > 0.0 else 0.0
    phiw = C * u ** Bw * bratio ** -E if u > 0.0 else 0.0
    phis = bd[B_PHIS] * tan_slope * tan_slope
    ds = r0 * phis
    dw = r0 * phiw
    x = ds + dw * math.cos(wind_rel_upslope)
    y = dw * math.sin(wind_rel_upslope)
    dh = math.sqrt(x * x + y * y)
    rh = r0 + dh
    alpha = math.atan2(y, x) if dh > 0.0 else 0.0
    phie = rh / r0 - 1.0
    ue = 0.0
    if phie > 0.0:
        ue = (phie * bratio ** E / C) ** (1.0 / Bw)
    limited = 0.0
    if wl_mode != WL_NONE:
        ir = bd[B_IR]
        ulim = 0.9 * ir if wl_mode == WL_ROTHERMEL else 96.8 * ir ** (1.0 / 3.0)
        if ue > ulim:
            ue = ulim
            phie = C * ue ** Bw * bratio ** -E
            rh = r0 * (1.0 + phie)
            limited = 1.0
    z = lw_ratio(ue / 88.0, lw_mode, lw_max)
    e = math.sqrt(z * z - 1.0) / z if z > 1.0 else 0.0
    out[0] = rh
    out[1] = alpha
    out[2] = ue
    out[3] = z
    out[4] = e
    out[5] = phiw
    out[6] = phis
    out[7] = limited


@njit(cache=True)
def ros_dir(rh, e, gamma):
    """ROS from the ignition point (rear focus) in direction gamma relative to max spread (Andrews 2018 Table 26)."""
    return rh * (1.0 - e) / (1.0 - e * math.cos(gamma))


@njit(cache=True)
def byram_flame_length_ft(i_btu):
    return 0.45 * i_btu ** 0.46 if i_btu > 0.0 else 0.0


def waf_unsheltered(depth_ft):
    """Albini & Baughman (1979) / Andrews (2012) unsheltered wind adjustment factor (20-ft -> midflame)."""
    h = np.asarray(depth_ft, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = 1.83 / np.log((20.0 + 0.36 * h) / (0.13 * h))
    return np.where(h > 0, w, 0.0)


def waf_sheltered(cc_pct, ch_m, cbh_m):
    """Andrews (2012) sheltered WAF: 0.555 / (sqrt(f H) ln((20 + 0.36 H)/(0.13 H))), H canopy height in ft,
    f = crown fill portion = (CC/100) * (crown ratio)/3."""
    H = np.asarray(ch_m, float) / FT_TO_M
    cr = np.clip((np.asarray(ch_m, float) - np.asarray(cbh_m, float)) / np.maximum(np.asarray(ch_m, float), 1e-6),
                 0.0, 1.0)
    f = np.asarray(cc_pct, float) / 100.0 * cr / 3.0
    with np.errstate(divide='ignore', invalid='ignore'):
        w = 0.555 / (np.sqrt(f * H) * np.log((20.0 + 0.36 * H) / (0.13 * H)))
    return np.where((f > 0) & (H > 0), w, np.nan), f


def waf(depth_ft, cc_pct=None, ch_m=None, cbh_m=None, threshold=0.05):
    """WAF per cell: sheltered (Andrews 2012) when the crown fill portion f > threshold (FlamMap uses 5 percent),
    otherwise unsheltered from the fuel bed depth. Returns (waf, sheltered_mask)."""
    un = waf_unsheltered(depth_ft)
    if cc_pct is None:
        return un, np.zeros(np.shape(un), bool)
    sh, f = waf_sheltered(cc_pct, ch_m, cbh_m)
    use = (f > threshold) & np.isfinite(sh)
    return np.where(use, np.minimum(sh, un), un), use
