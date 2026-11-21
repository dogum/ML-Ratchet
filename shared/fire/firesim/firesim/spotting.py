"""Spotting: Albini (1979) maximum spotting distance (surface-fire form as in BehavePlus, after Chase 1984 /
Morris 1987), the Albini flat-terrain transport formula and mountain-terrain correction, Schroeder (1969)
probability of ignition, and a stochastic ember model used by the spread solver.

Formulas (English units inside, as published):
  f  = 322 (0.474 U20)^-1.01                       (U20 = 20-ft wind, mi/h)
  zF = 1.055 sqrt(f I)                             firebrand lofting height (ft), I fireline intensity (Btu/ft/s)
  D_flat = 0.000718 U_H sqrt(H) [0.362 + sqrt(z/H)/2 ln(z/H)]   (mi), H cover height (ft; halved if open)
  drift  = 0.000278 U20 z^0.643                     (mi)
For torching and crown fires firesim uses the same plume relation with the crown fireline intensity (a documented
simplification of Albini 1979 torching-tree and Albini et al. 2012 crown-fire spotting, which need tree-species
tables that are not reproduced here).
"""
import math

import numpy as np

from ._nb import njit

MI_TO_M = 1609.344


@njit(cache=True)
def albini_flat_distance_mi(z_ft, cover_h_ft, u20_mph):
    if cover_h_ft < 1.0:
        cover_h_ft = 1.0
    if z_ft <= cover_h_ft or u20_mph <= 0.0:
        return 0.0
    r = z_ft / cover_h_ft
    return 0.000718 * u20_mph * math.sqrt(cover_h_ft) * (0.362 + math.sqrt(r) / 2.0 * math.log(r))


@njit(cache=True)
def firebrand_height_ft(fli_btu, u20_mph):
    if u20_mph <= 1e-7 or fli_btu <= 1e-7:
        return 0.0
    f = 322.0 * (0.474 * u20_mph) ** -1.01
    return 1.055 * math.sqrt(f * fli_btu)


@njit(cache=True)
def max_spot_distance_m(fli_kwm, u20_ms, cover_h_m, open_canopy):
    """Albini maximum spotting distance (m) over flat terrain from a source of fireline intensity fli (kW/m)."""
    i_btu = fli_kwm / 3.46141
    u20 = u20_ms * 2.23694
    h = cover_h_m / 0.3048
    if open_canopy:
        h = h / 2.0
    z = firebrand_height_ft(i_btu, u20)
    d = albini_flat_distance_mi(z, h, u20)
    if d <= 0.0:
        return 0.0
    drift = 0.000278 * u20 * z ** 0.643
    return (d + drift) * MI_TO_M


def mountain_distance(flat_m, ridge_valley_dist_m, ridge_valley_elev_m, location=0):
    """Albini (1979) mountain-terrain correction (BehavePlus); location 0 midslope windward, 1 valley bottom,
    2 midslope leeward, 3 ridge top."""
    if ridge_valley_elev_m <= 0 or ridge_valley_dist_m <= 0:
        return flat_m
    flat_mi = flat_m / MI_TO_M
    rvd = ridge_valley_dist_m / MI_TO_M
    rve = ridge_valley_elev_m / 0.3048
    a1 = flat_mi / rvd
    b1 = rve / (10.0 * math.pi) / 1000.0
    lf = [0.0, 1.0, 2.0, 3.0][location]  # windward midslope, valley, leeward midslope, ridge
    x = a1
    for _ in range(6):
        x = a1 - b1 * (math.cos(math.pi * x - lf * math.pi / 2.0) - math.cos(lf * math.pi / 2.0))
    return x * rvd * MI_TO_M


@njit(cache=True)
def p_ignition(m1_frac, t_air_c, shade_frac):
    """Schroeder (1969) probability of ignition of fine fuel by a firebrand (as in Rothermel 1983, BehavePlus).
    Fuel temperature = air + (25 - 20 shade) F. Heat of ignition Qig (cal/g)."""
    tf_f = t_air_c * 1.8 + 32.0 + 25.0 - 20.0 * shade_frac
    tf = (tf_f - 32.0) / 1.8
    m = m1_frac
    qig = 144.51 - 0.266 * tf - 0.00058 * tf * tf - tf * m + 18.54 * (1.0 - math.exp(-15.1 * m)) + 640.0 * m
    if qig > 400.0:
        qig = 400.0
    x = 0.1 * (400.0 - qig)
    p = 0.000048 * x ** 4.3 / 50.0
    if p > 1.0:
        p = 1.0
    if p < 0.0:
        p = 0.0
    return p
