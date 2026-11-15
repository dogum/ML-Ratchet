"""Baselines on the same windows the Detective sees.

  tech_default : GridSim / RTS-GMLC technology defaults (needs the technology: given from CAMPD metadata, an oracle)
  stats        : simple statistics of the output series itself (1st percentile of non-zero output for minimum load,
                 99th percentile of hour-to-hour changes for ramps, counted starts, spell lengths, ...)
  eia923       : EIA-923 annual average heat rate of the linked generators (net basis), flat curve
  dumb         : median of each quantity per technology over the dev units (dev years)
All return a dict with the pdcore.REG keys in natural units (NaN where the baseline has no opinion).
"""
import numpy as np
from pdcore import runs, HOT_H, WARM_H

TECH_DEFAULT = {   # gridsim.fleet.TECH_DEFAULTS (RTS-GMLC / EIA): pmin frac, ramp %/min, min up h, min down h, HR full
    'coal_steam': (0.40, 0.015, 8, 8, 10.5), 'gas_cc': (0.40, 0.030, 4, 4, 7.3),
    'gas_ct': (0.40, 0.100, 1, 1, 11.0), 'gas_steam': (0.30, 0.015, 6, 6, 11.2)}
HR_SHAPE_40 = {'coal_steam': 1.089, 'gas_cc': 1.09, 'gas_ct': 1.33, 'gas_steam': 1.07}   # HR(40 %)/HR(100 %)


def tech_default(tech):
    pf, rp, ut, dt, hr = TECH_DEFAULT[tech]
    f = max(pf, 0.4)
    shape = {'coal_steam': [(0.4, 1.089), (1.0, 1.0)], 'gas_steam': [(0.3, 1.12), (0.6, 1.02), (1.0, 1.0)],
             'gas_cc': [(0.4, 1.09), (1.0, 1.0)], 'gas_ct': [(0.4, 1.33), (1.0, 1.0)]}[tech]
    hm = hr * np.interp(f, [a for a, _ in shape], [b for _, b in shape])
    return dict(pmin_frac=pf, ramp_up=rp * 60, ramp_dn=rp * 60, starts_yr=np.nan, min_up=ut, med_up=np.nan,
                min_dn=dt, hot=np.nan, warm=np.nan, cold=np.nan, hr_full=hr, hr_min=hm, cf=np.nan)


def stats(x):
    x = np.asarray(x, float)
    known = np.isfinite(x)
    xv = np.where(known, x, 0.0)
    on = xv > 0
    out = {k: np.nan for k in ['pmin_frac', 'ramp_up', 'ramp_dn', 'starts_yr', 'min_up', 'med_up', 'min_dn', 'hot',
                               'warm', 'cold', 'hr_full', 'hr_min', 'cf']}
    if on.sum() < 24:
        return out
    pc = np.percentile(xv[on], 99.5)
    out['pmin_frac'] = float(np.percentile(xv[on], 1) / pc)
    both = on[1:] & on[:-1] & known[1:] & known[:-1]
    d = np.diff(xv)[both] / pc
    if len(d) > 10:
        out['ramp_up'] = float(max(np.percentile(d, 99), 0)); out['ramp_dn'] = float(max(-np.percentile(d, 1), 0))
    s, L, v = runs(on)
    st = np.flatnonzero(v & (s > 0))
    out['starts_yr'] = float(len(st) * 8760.0 / max(known.sum(), 1))
    up = L[1:-1][v[1:-1]]; dn = L[1:-1][~v[1:-1]]
    if len(up) >= 3:
        out['min_up'] = float(np.percentile(up, 5)); out['med_up'] = float(np.median(up))
    if len(dn) >= 3:
        out['min_dn'] = float(np.percentile(dn, 5))
    prev = np.array([L[i - 1] for i in st if i - 1 > 0])
    if len(prev) >= 3:
        out['hot'] = float(np.mean(prev < HOT_H)); out['warm'] = float(np.mean((prev >= HOT_H) & (prev < WARM_H)))
        out['cold'] = float(np.mean(prev >= WARM_H))
    out['cf'] = float(xv.sum() / (pc * max(known.sum(), 1)))
    return out


def eia923(hr):
    o = {k: np.nan for k in ['pmin_frac', 'ramp_up', 'ramp_dn', 'starts_yr', 'min_up', 'med_up', 'min_dn', 'hot',
                             'warm', 'cold', 'cf']}
    o['hr_full'] = hr if hr else np.nan
    o['hr_min'] = hr if hr else np.nan
    return o
