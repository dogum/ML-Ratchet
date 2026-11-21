"""Recipe step 5 (calibrate the simulator on dev only): per-station calibration of both snow routines on the DEV
stations' own SWE, water years <= 2010 only (WY2011-2025 is the dev gate), with bounds wider than WaterSim's tables
(many of A's dev calibrations sat on a bound). Fits a multivariate normal in transformed space per routine
-> results/prior_<tag>.json, used by simbank.py for physics >= p3. Also writes per-station parameters
(results/calib_dev.json) for the twin test.

DE/rand/1/bin, population 32, 50 generations, objective NSE of end-of-day SWE on the good water years' WY days.
"""
import json
import sys
import numpy as np
import pandas as pd
from common import *
from simbank import station_record, good_wys
from watersim.simulate import prepare_run, simulate_batch

SPACES = {
    'degree_day': [('TT', -3.0, 4.0, 'lin'), ('TTI', 0.0, 4.0, 'lin'), ('CFMAX', 0.5, 12.0, 'log'),
                   ('SFCF', 0.5, 2.0, 'log')],
    'energy_balance': [('TT', -3.0, 4.0, 'lin'), ('SFCF', 0.5, 2.0, 'log'), ('ALB_MIN', 0.2, 0.7, 'lin'),
                       ('ALB_DECAY', 1.0, 40.0, 'log'), ('TURB', 0.05, 5.0, 'log'), ('GHEAT', 0.0, 1.0, 'lin')],
}
UNIFORM = {
    'degree_day': {'CFR': (0.0, 0.1), 'CWH': (0.0, 0.2)},
    'energy_balance': {'TTI': (0.0, 4.0), 'CWH': (0.0, 0.2), 'ALB_MAX': (0.75, 0.92), 'CANOPY': (0.7, 1.0)},
}


def calib_one(sid, snow, rng, last_wy=2010):
    df, wys, lat, elev = station_record(sid)
    df = load_station(sid, None, f'{last_wy}-10-01')
    wys = good_wys(df, last_wy)
    if len(wys) < 3:
        return None
    spec = {'id': sid, 'area_km2': 1.0, 'lat': lat, 'lon': -110.0, 'elev_mean_m': elev}
    f = df[['P', 'T_mean', 'T_min', 'T_max']]
    o = prepare_run(spec, f, 'none', None, snow, 'hargreaves')
    obs = df['SWE_eod'].values
    m = np.isin(wy_of(df.index), wys) & np.isfinite(obs)
    sp = SPACES[snow]
    lo = np.array([s[1] for s in sp]); hi = np.array([s[2] for s in sp])
    lg = np.array([s[3] == 'log' for s in sp])
    a = np.where(lg, np.log(lo), lo); b = np.where(lg, np.log(hi), hi)

    def dec(U):
        X = a + U * (b - a)
        return np.where(lg, np.exp(X), X)

    def ev(U):
        X = dec(U)
        pl = [dict(zip([s[0] for s in sp], x)) for x in X]
        S = simulate_batch(None, None, pl, model='none', snow=snow, start=f.index[0], end=f.index[-1], spinup='1y',
                           outputs=('SWE',), o=o)['SWE']
        y = obs[m]
        return 1 - ((S[:, m] - y) ** 2).mean(1) / y.var()

    NP, G = 32, 50
    U = rng.random((NP, len(sp)))
    fit = ev(U)
    for g in range(G):
        idx = np.array([rng.choice(NP, 3, replace=False) for _ in range(NP)])
        V = np.clip(U[idx[:, 0]] + 0.6 * (U[idx[:, 1]] - U[idx[:, 2]]), 0, 1)
        cr = rng.random(U.shape) < 0.8
        cr[np.arange(NP), rng.integers(0, len(sp), NP)] = True
        T = np.where(cr, V, U)
        ft = ev(T)
        better = ft > fit
        U[better], fit[better] = T[better], ft[better]
    i = int(np.argmax(fit))
    return {'params': dict(zip([s[0] for s in sp], dec(U[i:i + 1])[0].tolist())), 'nse': float(fit[i]),
            'n_wy': len(wys)}


def fit_prior(cal, snow, inflate=1.3):
    sp = SPACES[snow]
    rows = [c[snow]['params'] for c in cal.values() if c.get(snow) and c[snow]['nse'] > 0.5]
    Z = np.array([[np.log(r[s[0]]) if s[3] == 'log' else r[s[0]] for s in sp] for r in rows])
    return {'names': [s[0] for s in sp], 'transform': [s[3] for s in sp], 'lo': [s[1] for s in sp],
            'hi': [s[2] for s in sp], 'mean': Z.mean(0).tolist(), 'cov': np.cov(Z.T).tolist(), 'inflate': inflate,
            'uniform': UNIFORM[snow], 'n_stations': len(rows)}


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'p3'
    rng = np.random.default_rng(SEED)
    cal = {}
    for sid in sorted(wd().splits()['snotel']['dev']):
        cal[sid] = {}
        for snow in ('degree_day', 'energy_balance'):
            cal[sid][snow] = calib_one(sid, snow, rng)
        print(sid, {s: (v and round(v['nse'], 3)) for s, v in cal[sid].items()}, flush=True)
    json.dump(cal, open(f'{ROOT}/results/calib_dev.json', 'w'), indent=1)
    pr = {s: fit_prior(cal, s) for s in SPACES}
    json.dump(pr, open(f'{ROOT}/results/prior_{tag}.json', 'w'), indent=1)
    for s in SPACES:
        print(s, dict(zip(pr[s]['names'], np.round(pr[s]['mean'], 3))), 'sd', np.round(np.sqrt(np.diag(pr[s]['cov'])), 3))
