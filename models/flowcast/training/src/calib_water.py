"""Rung p1 physics measurement (dev only): how much water does each dev basin really yield, relative to what
WaterSim makes from ERA5-Land precipitation?

For each of the 52 dev basins (WY2001-2014, warm-up from 1999-10; all non-exam): run the attribute-prior
simulator (no calibration; latent = 0; HBV, GR4J, SAC-SMA, VIC-lite x degree-day) with P x m for a grid of m, and
find the multiplier m* whose mean simulated flow matches the observed mean flow (median over the 4 models).
Then regress log m* on elevation, snow fraction and wetness -> phys/p1.json 'undercatch'.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
import watersim as ws

GRID = np.exp(np.linspace(np.log(0.4), np.log(2.5), 15))
rows = []
for b in wd.basins('camels', split='dev'):
    spec = wd.spec(b)
    df = wd.forcing(b, '1999-10-01', DEVWIN[1])
    meta = wd.forcing_meta(b)
    q = wd.streamflow(b, 'caravan').Q_mm.loc[DEVWIN[0]:DEVWIN[1]]
    qm = q.mean()
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
    rows.append({'basin': b, 'm_star': float(np.median(ms)), 'm_models': ms, 'q_obs': qm, 'p_era5': clim['p_mean'],
                 'rr_obs': qm / clim['p_mean'], 'elev_km': a['elev_mean_m'] / 1000, 'frac_snow': clim['frac_snow'],
                 'aridity': clim['aridity'], 'log_pw': math.log(clim['p_mean'] / 3.0), 'lat': spec['lat'],
                 'lon': spec['lon'], 'reg': a['reservoir_storage_frac']})
    print(b, round(rows[-1]['m_star'], 3), round(rows[-1]['rr_obs'], 3), flush=True)
d = pd.DataFrame(rows)
d.to_parquet(f'{ROOT}/results/calib_water.parquet')
y = np.log(np.clip(d.m_star.values, 0.4, 2.5))
X = np.column_stack([np.ones(len(d)), d.elev_km, d.frac_snow, d.log_pw])
# robust (Huber-ish) least squares: two passes, down-weight |r| > 2 sd
w = np.ones(len(y))
for _ in range(3):
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    r = y - X @ beta
    s = 1.4826 * np.median(np.abs(r))
    w = np.where(np.abs(r) > 2 * s, 2 * s / np.abs(r), 1.0)
print('log m* = %.3f + %.3f elev_km + %.3f frac_snow + %.3f log(p/3)' % tuple(beta), 'resid robust sd %.3f' % s)
print('median m*', d.m_star.median(), 'IQR', d.m_star.quantile([.25, .75]).values)
jdump({'beta': beta.tolist(), 'resid_sd': s, 'n': len(d), 'median_m': d.m_star.median()},
      f'{ROOT}/results/calib_water.json')
