"""Rung b2b (dev only): one reconciled water-yield prior for the ERA5-Land precipitation multiplier m.

1. m* per dev basin, exactly as models/flowcast/training/src/calib_water.py computes it: the attribute-prior WaterSim
   (no calibration, latent 0) for HBV, GR4J, SAC-SMA, VIC-lite x degree-day, P x m on a 15-point grid 0.4-2.5,
   m* where the mean simulated flow matches the observed mean (Caravan Q, WY2001-2014), median over the 4 models.
2. Three mean functions for log m*, each fitted with calib_water.py's robust least squares:
   M0 constant; M1 a + b frac_snow (BasinDetective's form); M2 a + b elev_km + c frac_snow + d log(P/3) (FlowCast's).
3. The form with the smallest leave-one-basin-out robust RMSE (1.4826 x median |LOO residual|) is chosen, refitted on
   all 52 basins, and given that LOO robust RMSE as its scatter sd. -> phys/b2b.json, results/calib_prior.json
usage: python calib_prior.py
"""
import json
import math
import os

import r2
import numpy as np
import pandas as pd
import waterdata as wd
import watersim as ws
from features import forcing_array, climate_stats, spec_attrs

DEVWIN = ('2000-10-01', '2014-09-30')
GRID = np.exp(np.linspace(np.log(0.4), np.log(2.5), 15))
OUT = os.path.join(r2.RESULTS, 'calib_prior.json')
CACHE = os.path.join(r2.WORK, 'calib_prior_mstar.parquet')


def m_star_table():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    rows = []
    for b in wd.basins('camels', split='dev'):
        spec = wd.spec(b)
        df = wd.forcing(b, '1999-10-01', DEVWIN[1])
        meta = wd.forcing_meta(b)
        q = wd.streamflow(b, 'caravan').Q_mm.loc[DEVWIN[0]:DEVWIN[1]]
        F = forcing_array(df.loc[DEVWIN[0]:])
        clim = climate_stats(F[:, 0], F[:, 1], F[:, 2])
        ms = []
        for model in ('hbv', 'gr4j', 'sacsma', 'vic_lite'):
            qs = []
            for m in GRID:
                d2 = df.copy(); d2['P'] = d2['P'] * m
                o = ws.simulate(spec, (d2, meta), DEVWIN[0], DEVWIN[1], model=model, snow='degree_day', labels=False,
                                irrigation=False, reservoirs=[])
                qs.append(o['ts']['Q'].reindex(q.index)[q.notna()].mean())
            qs = np.array(qs)
            obs = q[q.notna()].mean()
            if obs <= qs[0]:
                ms.append(GRID[0] * obs / max(qs[0], 1e-6))
            elif obs >= qs[-1]:
                ms.append(GRID[-1])
            else:
                ms.append(float(np.exp(np.interp(np.log(obs), np.log(qs), np.log(GRID)))))
        a = spec_attrs(spec)
        rows.append({'basin': b, 'm_star': float(np.median(ms)), 'elev_km': a['elev_mean_m'] / 1000,
                     'frac_snow': clim['frac_snow'], 'log_pw': math.log(clim['p_mean'] / 3.0)})
        print(b, round(rows[-1]['m_star'], 3), flush=True)
    d = pd.DataFrame(rows).set_index('basin')
    d.to_parquet(CACHE)
    return d


def robust_fit(X, y):
    """calib_water.py's robust least squares (3 passes, down-weight |r| > 2 robust sd)."""
    w = np.ones(len(y))
    for _ in range(3):
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r))
        w = np.where(np.abs(r) > 2 * s, 2 * s / np.abs(r), 1.0)
    return beta, s


def design(d, form):
    one = np.ones(len(d))
    if form == 'M0':
        return np.column_stack([one])
    if form == 'M1':
        return np.column_stack([one, d.frac_snow])
    return np.column_stack([one, d.elev_km, d.frac_snow, d.log_pw])


def main():
    d = m_star_table()
    y = np.log(np.clip(d.m_star.values, 0.4, 2.5))
    res = {'n': len(d), 'm_star_median': float(d.m_star.median()), 'forms': {}}
    for form in ('M0', 'M1', 'M2'):
        X = design(d, form)
        beta, s_in = robust_fit(X, y)
        loo = np.empty(len(y))
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            bi, _ = robust_fit(X[k], y[k])
            loo[i] = y[i] - X[i] @ bi
        res['forms'][form] = {'beta': beta.tolist(), 'insample_robust_sd': float(s_in),
                              'loo_robust_rmse': float(1.4826 * np.median(np.abs(loo))),
                              'loo_rmse': float(np.sqrt(np.mean(loo ** 2)))}
        print(form, res['forms'][form], flush=True)
    best = min(res['forms'], key=lambda f: res['forms'][f]['loo_robust_rmse'])
    b = res['forms'][best]
    beta = b['beta'] + [0.0] * (4 - len(b['beta']))
    if best == 'M1':
        coef = {'a': beta[0], 'b_elev': 0.0, 'b_snow': beta[1], 'b_wet': 0.0}
    elif best == 'M0':
        coef = {'a': beta[0], 'b_elev': 0.0, 'b_snow': 0.0, 'b_wet': 0.0}
    else:
        coef = {'a': beta[0], 'b_elev': beta[1], 'b_snow': beta[2], 'b_wet': beta[3]}
    coef = {k: round(v, 3) for k, v in coef.items()}
    sd = round(b['loo_robust_rmse'], 3)
    p1_check = res['forms']['M2']['beta']
    res['p1_reproduced'] = {'beta_here': p1_check, 'beta_p1': [-0.088, 0.058, 0.153, 0.397],
                            'max_abs_diff': float(np.max(np.abs(np.array(p1_check) - [-0.088, 0.058, 0.153, 0.397])))}
    res['chosen'] = {'form': best, **coef, 'sd': sd}
    no_change = best == 'M2' and abs(sd - 0.146) < 0.01
    res['no_change'] = bool(no_change)
    os.makedirs(r2.RESULTS, exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    if not no_change:
        p1 = json.load(open(os.path.join(r2.PHYS, 'b2a.json')))
        rec = {'version': 'b2b', 'parent': 'p1',
               'note': f'p1 with the reconciled water-yield prior (round2/calib_prior.py, dev only): form {best}, '
                       f'log m = {coef["a"]} + {coef["b_elev"]} elev_km + {coef["b_snow"]} frac_snow + '
                       f'{coef["b_wet"]} log(P/3) + N(0, {sd}) (sd = leave-one-basin-out robust RMSE)',
               'camels_share': 0.5, 'muskingum_share': 0.25,
               'undercatch': {**coef, 'sd': sd, 'lo': 0.4, 'hi': 2.5}, 'forcing_error': {}, 'obs_noise': {},
               'b2': {}}
        assert p1['undercatch']['lo'] == 0.4
        json.dump(rec, open(os.path.join(r2.PHYS, 'b2b.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('chosen', 'no_change', 'p1_reproduced')}, indent=1))


if __name__ == '__main__':
    main()
