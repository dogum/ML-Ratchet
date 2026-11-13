"""RoadData's EPA wide table maps test procedures to cycles wrongly: 'HWFE' (highway) tests are labelled FTP and the
main FTP-75 procedure 'Federal fuel 2-day exhaust (w/can load)' is labelled HWY. We do NOT edit data/; this rebuilds the
per-vehicle FTP / HWY / US06 mpg from the long test table with the procedure names, and refits RoadData's log-linear
baseline (dev-fitted, same features) on the corrected labels. -> data/epa_fixed.parquet, data/epa_baseline_fixed.json"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json
sys.path.insert(0, _P.VEHICLE)
import numpy as np, pandas as pd
import realdata as rd
FTP = {'Federal fuel 2-day exhaust (w/can load)', 'Federal fuel 3-day exhaust', 'CVS 75 and later (w/o can. load)'}
t = rd.epa_testcars(long=True)
t = t[(t.fe_mpg > 3) & (t.fe_mpg < 300)]
cyc = np.where(t.test_proc.isin(FTP), 'FTP', np.where(t.test_proc == 'HWFE', 'HWY', np.where(t.test_proc == 'US06', 'US06', None)))
t = t.assign(cyc=cyc); t = t[t.cyc.notna() & (t.fuel == 'gasoline')]
w = t.groupby(['epa_vehicle_id', 'cyc']).fe_mpg.median().unstack()
w.columns = [f'{c}_mpg_fix' for c in w.columns]
w.to_parquet(_P.WORK + '/data/epa_fixed.parquet')
v = rd.epa_testcars().join(w)
print('medians conv', v[v.architecture == 'conv'][['FTP_mpg_fix', 'HWY_mpg_fix', 'US06_mpg_fix']].median().to_dict())
e = v.assign(F50=v.A_N + v.B_N_per_mps * 22.352 + v.C_N_per_mps2 * 22.352 ** 2)
res = {}
for arch, cyc_, ycol in [('conv', 'FTP', 'FTP_mpg_fix'), ('conv', 'HWY', 'HWY_mpg_fix'), ('conv', 'US06', 'US06_mpg_fix'),
                         ('hev', 'FTP', 'FTP_mpg_fix'), ('hev', 'HWY', 'HWY_mpg_fix'), ('hev', 'US06', 'US06_mpg_fix')]:
    d = e[(e.architecture == arch) & e[ycol].notna() & e.F50.gt(0) & e.test_mass_kg.gt(0) & e.displacement_L.gt(0) & e.rated_hp.gt(0) & e.fuel.eq('gasoline')].copy()
    d['y'] = 1 / d[ycol]
    dv, ex = d[d.split == 'dev'], d[d.split == 'exam']
    Xc = ['test_mass_kg', 'F50', 'displacement_L', 'rated_hp']
    X = lambda q: np.c_[np.ones(len(q)), np.log(q[Xc].values.astype(float))]
    beta, *_ = np.linalg.lstsq(X(dv), np.log(dv.y.values), rcond=None)
    p = np.exp(X(ex) @ beta); pm = np.full(len(ex), dv.y.median())
    res[f'{arch}_{cyc_}'] = dict(n_dev=len(dv), n_exam=len(ex), regression_MdAPE=float(np.median(np.abs(p / ex.y.values - 1))),
                                 median_MdAPE=float(np.median(np.abs(pm / ex.y.values - 1))), beta=beta.tolist(), features=Xc)
    pd.DataFrame({'vehicle_id': ex.index, 'reg': p, 'y': ex.y.values}).to_parquet(f'{_P.WORK}/data/epa_reg_{arch}_{cyc_}.parquet')
    print(arch, cyc_, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in res[f'{arch}_{cyc_}'].items() if k not in ('beta', 'features')})
json.dump(res, open(_P.WORK + '/data/epa_baseline_fixed.json', 'w'), indent=1)
