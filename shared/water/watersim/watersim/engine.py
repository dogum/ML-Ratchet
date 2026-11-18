"""Numba engine: snow over elevation bands + one runoff model, for one run (all fluxes recorded) or a batch of
parameter sets / basins (selected outputs only). Both paths use the same step functions.

Forcing matrix F (T, 7): P (mm/step), T_mean, T_min, T_max (C), SRAD (W/m2), vp (Pa), wind (m/s).
Bands: frac (B), dz_km (B, band elevation minus forcing reference elevation), z_m (B, absolute elevation).
"""
import numpy as np

from ._nb import njit
from .snow import dd_step, eb_step
from .models import hbv, gr4j, sacsma, vic_lite

# record columns of the full run
REC = ['P', 'rain', 'snowfall', 'melt', 'refreeze', 'liquid_input', 'SWE', 'snow_liquid', 'sublimation', 'SCA',
       'PET', 'PET_soil', 'snow_energy', 'ET_soil', 'Q_model', 'Q_quick', 'Q_inter', 'Q_base', 'exchange',
       'soil', 'groundwater', 'routing_store', 'recharge', 'surface_runoff', 'infiltration_excess']
NREC = len(REC)
NF = 11
SNOW_DD = 0
SNOW_EB = 1
SNOW_NONE = 2


@njit
def _model_step(mid, s, p, w, pet, dt, uhord, uhbuf, f):
    if mid == 0:
        hbv.step(s, p, w, pet, dt, uhord, uhbuf, f)
    elif mid == 1:
        gr4j.step(s, p, w, pet, dt, uhord, uhbuf, f)
    elif mid == 2:
        sacsma.step(s, p, w, pet, dt, uhord, uhbuf, f)
    elif mid == 3:
        vic_lite.step(s, p, w, pet, dt, uhord, uhbuf, f)
    else:
        for i in range(NF):
            f[i] = 0.0
        f[1] = w
        f[2] = w


@njit
def run_core(mid, snow_kind, F, pet, rso, dt, bfrac, bdz, bz, sp, mp, s, snow, uhord, uhbuf, pet_snow_red, rec, swe_b,
             infil_cap, uhx):
    """Run one basin. s, snow (B, 4: ice, liq, cc, age), uhbuf and uhx are updated in place (final state).
    infil_cap: infiltration capacity (mm per step); liquid input above it is infiltration-excess (Hortonian) runoff
    that bypasses the soil and is routed with the model's quick-flow unit hydrograph (<= 0: off)."""
    T = F.shape[0]
    B = bfrac.shape[0]
    L = uhord.shape[1]
    f = np.zeros(NF)
    if snow_kind == SNOW_EB:
        tl = sp[11]
        pg = sp[12]
    elif snow_kind == SNOW_DD:
        tl = sp[6]
        pg = sp[7]
    else:
        tl = 0.0
        pg = 0.0
    pm = np.empty(B)
    norm = 0.0
    for b in range(B):
        m = 1.0 + pg * bdz[b]
        if m < 0.0:
            m = 0.0
        pm[b] = m
        norm += bfrac[b] * m
    for b in range(B):
        pm[b] = pm[b] / norm if norm > 0.0 else 1.0
    pa = np.empty(B)
    for b in range(B):
        pa[b] = 101.3 * ((293.0 - 0.0065 * bz[b]) / 293.0) ** 5.26
    for t in range(T):
        P = F[t, 0]
        Tm = F[t, 1]
        Rs = F[t, 4] * 0.0864
        ea = F[t, 5] / 1000.0
        u = F[t, 6]
        tot_rain = 0.0
        tot_snow = 0.0
        tot_melt = 0.0
        tot_refr = 0.0
        tot_out = 0.0
        tot_swe = 0.0
        tot_liq = 0.0
        tot_subl = 0.0
        tot_sca = 0.0
        tot_q = 0.0
        for b in range(B):
            fb = bfrac[b]
            if fb <= 0.0:
                continue
            Pb = P * pm[b]
            Tb = Tm + tl * bdz[b]
            if snow_kind == SNOW_DD:
                ice, liq, rain, sn, melt, refr, out = dd_step(snow[b, 0], snow[b, 1], Pb, Tb, sp[0], sp[1], sp[2],
                                                               sp[3], sp[4], sp[5], dt)
                snow[b, 0] = ice
                snow[b, 1] = liq
                subl = 0.0
                qn = 0.0
            elif snow_kind == SNOW_EB:
                ice, liq, cc, age, rain, sn, melt, refr, out, subl, qn = eb_step(
                    snow[b, 0], snow[b, 1], snow[b, 2], snow[b, 3], Pb, Tb, Rs, ea, u, rso[t], pa[b], sp, dt)
                snow[b, 0] = ice
                snow[b, 1] = liq
                snow[b, 2] = cc
                snow[b, 3] = age
            else:
                rain = Pb
                sn = 0.0
                melt = 0.0
                refr = 0.0
                out = Pb
                subl = 0.0
                qn = 0.0
                ice = 0.0
                liq = 0.0
            swe = snow[b, 0] + snow[b, 1]
            swe_b[t, b] = swe
            sca = swe / 10.0
            if sca > 1.0:
                sca = 1.0
            tot_rain += fb * rain
            tot_snow += fb * sn
            tot_melt += fb * melt
            tot_refr += fb * refr
            tot_out += fb * out
            tot_swe += fb * swe
            tot_liq += fb * snow[b, 1]
            tot_subl += fb * subl
            tot_sca += fb * sca
            tot_q += fb * qn
        pe = pet[t]
        pes = pe * (1.0 - tot_sca) if pet_snow_red else pe
        win = tot_out
        exc = 0.0
        if infil_cap > 0.0 and win > infil_cap:
            exc = win - infil_cap
            win = infil_cap
        _model_step(mid, s, mp, win, pes, dt, uhord, uhbuf, f)
        if infil_cap > 0.0:
            for k in range(L):
                uhx[k] += exc * uhord[0, k]
            ox = uhx[0]
            for k in range(L - 1):
                uhx[k] = uhx[k + 1]
            uhx[L - 1] = 0.0
            sx = 0.0
            for k in range(L):
                sx += uhx[k]
            f[1] += ox
            f[2] += ox
            f[8] += sx
            f[10] += exc
        rec[t, 24] = exc
        rec[t, 0] = tot_rain + tot_snow
        rec[t, 1] = tot_rain
        rec[t, 2] = tot_snow
        rec[t, 3] = tot_melt
        rec[t, 4] = tot_refr
        rec[t, 5] = tot_out
        rec[t, 6] = tot_swe
        rec[t, 7] = tot_liq
        rec[t, 8] = tot_subl
        rec[t, 9] = tot_sca
        rec[t, 10] = pe
        rec[t, 11] = pes
        rec[t, 12] = tot_q
        for i in range(NF):
            rec[t, 13 + i] = f[i]


@njit
def run_batch(mid, snow_kind, F, pet, rso, dt, bfrac, bdz, bz, sp, mp, s0, snow0, uhord, pet_snow_red,
              fidx, want_swe, want_et, want_comp, Qout, SWEout, ETout, Bout, infil):
    """Run N members. F (Nf, T, 7), pet/rso (Nf, T); bands (N, B); sp (N, nsp); mp (N, nmp); s0 (N, ns);
    snow0 (N, B, 4); uhord (N, 3, L); fidx (N,) forcing row per member. Outputs Qout (N, T) model outflow
    (before channel routing), optional SWEout, ETout (soil ET + sublimation), Bout (slow component)."""
    N = mp.shape[0]
    T = F.shape[1]
    B = bfrac.shape[1]
    L = uhord.shape[2]
    rec = np.zeros((T, NREC))
    swe_b = np.zeros((T, B))
    for i in range(N):
        s = s0[i].copy()
        snow = snow0[i].copy()
        uhbuf = np.zeros((3, L))
        uhx = np.zeros(L)
        k = fidx[i]
        run_core(mid, snow_kind, F[k], pet[k], rso[k], dt, bfrac[i], bdz[i], bz[i], sp[i], mp[i], s, snow,
                 uhord[i], uhbuf, pet_snow_red, rec, swe_b, infil[i], uhx)
        for t in range(T):
            Qout[i, t] = rec[t, 14]
        if want_swe:
            for t in range(T):
                SWEout[i, t] = rec[t, 6]
        if want_et:
            for t in range(T):
                ETout[i, t] = rec[t, 13] + rec[t, 8]
        if want_comp:
            for t in range(T):
                Bout[i, t] = rec[t, 17]
