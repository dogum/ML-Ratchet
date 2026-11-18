"""Reservoir module on B's DEV reservoirs (ResOpsUS, splits['reservoirs']['dev'], 15 dams; dev time is free):
(1) mass-balance replay: start each water year from the observed storage, apply observed inflow and observed release
    (no evaporation), compare the simulated storage path with the observed one (error after 1 year / capacity);
(2) storage-elevation geometry: fit the power law z = z0 + H (S/C)^(1/b) to observed (storage, elevation) pairs;
(3) generic policies driven by observed inflow (SOP with demand = mean observed release, a rule curve fitted to the
    observed mean seasonal storage, Hanasaki 2006): NSE of release and storage vs observed. These are the classical
    reservoir baselines in their simplest form; ReservoirPilot must beat them."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from _common import write
import waterdata as wd
from watersim.reservoir import Reservoir, DT_HM3
from watersim import metrics as M

meta = wd.reservoir_meta('dev')
rows = []
for _, m in meta.iterrows():
    d = wd.reservoir(m.res_id)
    rec = {'res_id': m.res_id, 'name': m.dam_name, 'main_use': m.main_use, 'cap_hm3': float(m.cap_hm3),
           'mb_rel_resid_B': float(m.mb_rel_resid) if pd.notna(m.mb_rel_resid) else None}
    C = float(max(m.cap_hm3, d['storage_hm3'].max())) if pd.notna(m.cap_hm3) else float(d['storage_hm3'].max())
    depth = float(m.depth_m) if pd.notna(m.depth_m) and m.depth_m > 0 else 10 * C ** 0.25 + 5
    # (2) geometry
    g = d[['storage_hm3', 'elevation_m']].dropna()
    g = g[(g.storage_hm3 > 0)]
    if len(g) > 365 and g.elevation_m.std() > 0.1:
        try:
            f = lambda s, z0, H, b: z0 + H * (s / C) ** (1 / b)
            p, _ = curve_fit(f, g.storage_hm3.values, g.elevation_m.values,
                             p0=[g.elevation_m.min() - depth / 2, depth, 2.5], bounds=([-1e4, 0.1, 1.0], [1e4, 1e3, 6.0]),
                             maxfev=20000)
            pred = f(g.storage_hm3.values, *p)
            rec['geometry'] = {'b': float(p[2]), 'H_m': float(p[1]),
                               'r2': float(1 - np.sum((pred - g.elevation_m) ** 2) / np.sum((g.elevation_m - g.elevation_m.mean()) ** 2))}
        except Exception as e:
            rec['geometry'] = {'error': repr(e)}
    a3 = d[['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
    if len(a3) < 3 * 365:
        rows.append(rec)
        continue
    a3 = a3.asfreq('D').interpolate(limit=5).dropna()
    # (1) replay by water year
    errs = []
    wy = a3.index.year + (a3.index.month >= 10)
    for y, gg in a3.groupby(wy):
        if len(gg) < 360:
            continue
        R = Reservoir(C, depth_m=depth, dead_frac=0.0, init_frac=0.0)
        R.reset(storage_hm3=float(gg.storage_hm3.iloc[0] - (gg.inflow_cms.iloc[0] - gg.release_cms.iloc[0]) * DT_HM3))
        S = []
        for t in range(len(gg)):
            S.append(R.step(gg.inflow_cms.iloc[t], gg.release_cms.iloc[t], 0.0)['storage_hm3'])
        errs.append(abs(S[-1] - gg.storage_hm3.iloc[-1]) / C)
    rec['replay_end_of_year_abs_err_frac_cap_median'] = float(np.median(errs)) if errs else None
    # (3) generic policies on the whole simultaneous record
    I = a3.inflow_cms.values
    idx = a3.index
    Dm = float(a3.release_cms.mean())
    seas = (a3.storage_hm3 / C).groupby(a3.index.dayofyear).mean()
    pol = {}
    for name, policy, pp in (
            ('sop', 'sop', {}),
            ('rule_curve', 'rule_curve', {'s_lo': float(seas.min()), 's_hi': float(seas.max()),
                                          'doy_full': float(seas.idxmax()), 't_adjust_days': 30.0}),
            ('hanasaki', 'hanasaki', {'demand_mean_m3s': Dm})):
        R = Reservoir(C, depth_m=depth, dead_frac=0.05, init_frac=float(a3.storage_hm3.iloc[0] / C), policy=policy,
                      policy_params=pp, evap='none')
        o = R.run(I, idx, np.zeros(len(I)), demand_m3s=Dm)
        out_flow = o['release_m3s'] + o['spill_m3s']
        pol[name] = {'NSE_release': M.nse(a3.release_cms.values, out_flow),
                     'NSE_storage': M.nse(a3.storage_hm3.values, o['storage_hm3']),
                     'KGE_release': M.kge(a3.release_cms.values, out_flow)}
    pol['inflow_as_release'] = {'NSE_release': M.nse(a3.release_cms.values, I)}
    rec['policies'] = pol
    rec['n_days'] = len(a3)
    rows.append(rec)
    print(m.res_id, m.main_use, rec.get('replay_end_of_year_abs_err_frac_cap_median'), {k: round(v['NSE_release'], 2) for k, v in pol.items()}, rec.get('geometry'))


def med(path):
    v = []
    for r in rows:
        x = r
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        if x is not None and np.isfinite(x):
            v.append(x)
    return (float(np.median(v)), len(v)) if v else (None, 0)


summary = {'replay_end_of_year_err_frac_cap': med(['replay_end_of_year_abs_err_frac_cap_median']),
           'geometry_b': med(['geometry', 'b']), 'geometry_r2': med(['geometry', 'r2'])}
for p in ('sop', 'rule_curve', 'hanasaki', 'inflow_as_release'):
    summary[p] = {k: med(['policies', p, k]) for k in ('NSE_release', 'NSE_storage', 'KGE_release')}
write('reservoir_dev.json', {'what': __doc__.strip().split('\n')[0], 'n_reservoirs': len(rows), 'summary_median_n': summary,
                             'reservoirs': rows})
print(summary)
