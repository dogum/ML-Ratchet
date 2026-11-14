"""Model-era prior (Phase D rung r7): how much less efficient an older vehicle is than a modern one with the same
engine size, class and transmission. Fitted on fueleconomy.gov DEV families only (rd.fueleconomy('dev'); the exam
families are never read). Unadjusted (lab) city and highway mpg, so it is the powertrain, not the label formula.

  log(fuel per mile) = a(group) + b log(displ) + c cyl + d boost + e(trans) + f(class) + g(drive) + s(year)
  s(year): piecewise linear, knots every 4 years 1984-2024, s(2021) = 0 (the generator's donor records are MY2018-24).
Hybrids (power-split, P2) and BEVs: the same trend form on their own records (few, so knots every 6 years from 2000).

python era.py -> training/era_prior.json   (era_factor(year, group) = fuel per mile relative to MY2021)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json
sys.path.insert(0, _P.VEHICLE)
import numpy as np, pandas as pd
import realdata as rd

REF = 2021


def hat(y, knots):
    """piecewise-linear basis (value at each knot), linear extrapolation clamped to the end knots."""
    y = np.clip(np.asarray(y, float), knots[0], knots[-1])
    B = np.zeros((len(y), len(knots)))
    for j in range(len(knots)):
        if j > 0:
            m = (y >= knots[j - 1]) & (y <= knots[j]); B[m, j] = (y[m] - knots[j - 1]) / (knots[j] - knots[j - 1])
        if j < len(knots) - 1:
            m = (y >= knots[j]) & (y <= knots[j + 1]); B[m, j] = np.maximum(B[m, j], (knots[j + 1] - y[m]) / (knots[j + 1] - knots[j]))
    return B


def clsgrp(v):
    v = str(v)
    if 'Pickup' in v: return 'pickup'
    if 'Sport Utility' in v or 'Special Purpose' in v: return 'suv'
    if 'Van' in v: return 'van'
    return 'car'


def fit(d, ycol, knots, covs=True):
    d = d[(d[ycol] > 0)].copy()
    y = np.log(1.0 / d[ycol].values)
    parts = [hat(d.year.values, knots)]
    names = [f'k{k}' for k in knots]
    if covs:
        dl = np.log(d.displ.fillna(d.displ.median()).clip(0.6, 8).values)
        parts += [dl[:, None], d.cylinders.fillna(4).values[:, None] / 4, d.boost.fillna(0).astype(float).values[:, None]]
        names += ['log_displ', 'cyl4', 'boost']
        for k in ('manual', 'CVT'):
            parts.append((d.trans_type == k).values[:, None].astype(float)); names.append('tr_' + k)
        for k in ('suv', 'pickup', 'van'):
            parts.append((d.VClass.map(clsgrp) == k).values[:, None].astype(float)); names.append('cls_' + k)
        parts.append(d.drive.astype(str).str.contains('4|All', regex=True).values[:, None].astype(float)); names.append('awd')
    X = np.hstack(parts)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s = beta[:len(knots)]
    s = s - np.interp(REF, knots, s)
    return dict(knots=list(map(float, knots)), s=list(map(float, s)), n=int(len(d)), resid_sd=float(np.std(res)),
                coef={n: float(b) for n, b in zip(names[len(knots):], beta[len(knots):])})


USE = {'conv': 'conv', 'hev': 'conv', 'bev': None}   # decided on the dev fit below (printed by main)


def era_factor(year, group, P):
    """fuel (energy) per mile of a MY `year` vehicle / the same vehicle in MY2021 (city and highway mean).
    Hybrids use the conventional engine trend: their own dev fit (168 records, 2004-2026) is not monotone and rests on a
    handful of 2004-2009 cars. BEVs: no era factor; their dev fit (140 records, no covariates) shows no trend that
    size does not explain (0.86-1.01, residual sd 0.19)."""
    if USE.get(group, group) is None:
        return 1.0
    e = P[USE.get(group, group)]
    return float(np.exp(0.5 * (np.interp(year, e['city']['knots'], e['city']['s']) + np.interp(year, e['hwy']['knots'], e['hwy']['s']))))


def main():
    f = rd.fueleconomy('dev')
    out = {'source': 'fueleconomy.gov vehicles.csv, dev families only (rd.fueleconomy("dev"))', 'ref_year': REF}
    g = f[(f.pt == 'gas') & f.atvType.isna() | (f.pt == 'gas') & (f.atvType == 'FFV')]
    kn = list(range(1984, 2025, 4)) + [2027]
    out['conv'] = {'city': fit(g, 'UCity', kn), 'hwy': fit(g, 'UHighway', kn)}
    h = f[(f.pt == 'hev') & (f.year >= 2004)]
    kh = [2004, 2010, 2016, 2022, 2027]
    out['hev'] = {'city': fit(h, 'UCity', kh), 'hwy': fit(h, 'UHighway', kh)}
    b = f[(f.pt == 'bev') & (f.year >= 2008)].copy()
    b['UCity_e'] = 1.0 / b.cityE.where(b.cityE > 0); b['UHwy_e'] = 1.0 / b.highwayE.where(b.highwayE > 0)
    kb = [2008, 2014, 2020, 2027]
    out['bev'] = {'city': fit(b, 'UCity_e', kb, covs=False), 'hwy': fit(b, 'UHwy_e', kb, covs=False)}
    json.dump(out, open(os.path.join(_P.SRC, 'era_prior.json'), 'w'), indent=1)
    for grp in ('conv', 'hev', 'bev'):
        e = out[grp]; raw = lambda y: float(np.exp(0.5 * (np.interp(y, e['city']['knots'], e['city']['s']) + np.interp(y, e['hwy']['knots'], e['hwy']['s']))))
        print(grp, 'fit', {y: round(raw(y), 3) for y in (2000, 2005, 2008, 2011, 2014, 2017, 2020, 2024)}, 'used', {y: round(era_factor(y, grp, out), 3) for y in (2000, 2005, 2008, 2011, 2014, 2017, 2020, 2024)},
              'n', out[grp]['city']['n'], 'resid sd', round(out[grp]['city']['resid_sd'], 3))


if __name__ == '__main__':
    main()
