"""Re-score the RELEASED v1 model (weights/, sigma_factors.json) on ZTBus mission totals under the by-month split.

V2_PLAN decisions log, 13 Nov 2026: the full-ZTBus split moved 292 missions from v1-exam to dev. v1 graded the mission
totals of all 1,407 v1-exam missions; this script repeats exactly v1's method (exam.py, 'ZTBus_missions': route proxy =
the traced v1 legs of the same bus route, run at the mission's mean temperature; baseline = v1 dev kWh/km x distance)
on the 1,115 missions that are exam under proc/hd/ztbus_full_splits.json, and on the old 1,407 as a reproduction check.
Nothing is chosen with it. python rescore_v1_ztbus.py  -> work/rescore_v1_ztbus.json (numbers go to NOTEBOOK.md)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.VEHICLE)
import numpy as np, pandas as pd
import evaluate as EV, features as F
from model_np import Ensemble
import realdata as rd

paths = sorted(os.path.join(_P.MODEL, 'weights', f) for f in os.listdir(os.path.join(_P.MODEL, 'weights')) if f.endswith('.npz'))
m = Ensemble(paths)
F.PHYS_K.update(m.meta.get('phys_k', {}))
K = json.load(open(os.path.join(_P.MODEL, 'sigma_factors.json')))
kz = K.get('HD_ZTBus_leg', 1.0)
split = json.load(open(os.path.join(_P.VEHICLE, 'data', 'proc', 'hd', 'ztbus_full_splits.json')))['missions']

hb = rd.baselines('summary')['HD_exam']
kpk = hb['ZTBus_legs_kWh']['kWh_per_km_dev'] * 3.6                       # v1 baseline: dev MJ/km
M = rd.hd_data()['ztbus_missions']
hd = EV.predict_real(m, 'hd')
legs = hd[hd.set == 'ZTBus_leg'].copy().join(M.set_index('name').busRoute, on='mission')
B_, T_, X_, E_, M_ = EV.real_set('hd')
li = np.where(M_.set.values == 'ZTBus_leg')[0]


def mission_rows(sel_missions):
    rows = []
    for r in sel_missions.itertuples():
        same = legs.busRoute.values == r.busRoute
        sel = li[same] if same.any() else li
        X = X_[sel].copy()
        X[:, 76:79] = r.T_amb_mean_C / 20.0
        X[:, 79] = float(r.T_amb_mean_C < 0); X[:, 80] = float(r.T_amb_mean_C > 25)
        p, s, c = m.predict(X, E_[sel])
        kpk_ = p.sum() / (B_['dist'][sel].sum() / 1000)
        rows.append(dict(name=r.name, route=r.busRoute, pred=kpk_ * r.drivenDistance / 1000, y=r.kWh * 3.6,
                         base=kpk * r.drivenDistance / 1000, sig=float(np.mean(s))))
    return pd.DataFrame(rows)


def score(mt):
    z = np.abs(np.log(mt.pred / mt.y)) / (mt.sig * kz)
    return dict(n=int(len(mt)), EnergyNet=float(np.median(np.abs(mt.pred / mt.y - 1))),
                baseline=float(np.median(np.abs(mt.base / mt.y - 1))), bias=float(np.median(mt.pred / mt.y - 1)),
                baseline_bias=float(np.median(mt.base / mt.y - 1)), cov68=float(np.mean(z < 1)), cov95=float(np.mean(z < 1.96)))


old = M[M.split == 'exam']
mt = mission_rows(M)
mt['split_v1'] = mt.name.map(M.set_index('name').split)
mt['split_by_month'] = mt.name.map(split)
R = {'model': 'EnergyNet v1.0 (released ensemble, weights/)', 'method': 'v1 exam.py ZTBus_missions (route proxy)',
     'old_v1_exam_1407': score(mt[mt.split_v1 == 'exam']),
     'by_month_exam_1115': score(mt[mt.split_by_month == 'exam']),
     'by_month_dev_294_info': score(mt[mt.split_by_month == 'dev'])}
mt.to_parquet(f'{_P.WORK}/data/rescore_v1_ztbus_missions.parquet')
json.dump(R, open(f'{_P.WORK}/rescore_v1_ztbus.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
