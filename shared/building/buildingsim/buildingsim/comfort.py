"""Thermal comfort: Fanger PMV/PPD (ISO 7730:2005 Annex D algorithm, vectorised) and degree-hours outside a band.
Checked against pythermalcomfort (checks/check_comfort.py)."""
import numpy as np


def pmv_ppd(ta, tr, vel, rh, met, clo, wme=0.0):
    ta, tr, vel, rh, met, clo = (np.asarray(x, float) for x in np.broadcast_arrays(ta, tr, vel, rh, met, clo))
    pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (ta + 235.0))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w
    fcl = np.where(icl <= 0.078, 1.0 + 1.29 * icl, 1.05 + 0.645 * icl)
    hcf = 12.1 * np.sqrt(vel)
    taa = ta + 273.0
    tra = tr + 273.0
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = tcla / 50.0
    hc = hcf
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * np.abs(100.0 * xf - taa) ** 0.25
        hc = np.maximum(hcf, hcn)
        xn_new = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
        done = np.abs(xn_new - xf) < 0.00015
        xn = xn_new
        if done.all():
            break
    tcl = 100.0 * xn - 273.0
    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)
    hl2 = np.where(mw > 58.15, 0.42 * (mw - 58.15), 0.0)
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - ta)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - ta)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return pmv, ppd


def degree_hours(T, lo, hi, dt_h, mask=None):
    T = np.asarray(T, float)
    below = np.clip(lo - T, 0, None)
    above = np.clip(T - hi, 0, None)
    if mask is not None:
        below, above = below * mask, above * mask
    return float(below.sum() * dt_h), float(above.sum() * dt_h)
