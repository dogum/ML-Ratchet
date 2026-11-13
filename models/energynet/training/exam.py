"""Final exam (real data never used for choosing). python exam.py <rung>
1. widen-only sigma factors per output group, fitted on DEV (68% coverage target)
2. VED exam trips vs label / VSP (MOVES-like) / class-median baselines; per-vehicle totals; breakdowns
3. EPA exam families (corrected cycle labels) vs dev-fitted log-linear regressions
4. Heavy duty: ZTBus legs, traced missions and all missions (route proxy), Mersin, ICCT
-> exam.json, data/exam_<set>.parquet"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.VEHICLE)
import numpy as np, pandas as pd
import evaluate as EV, dataset as DS, features as F
from model_np import ENet, Ensemble
import realdata as rd

EN = _P.WORK
rung = sys.argv[1]
_R = {r['name']: r for r in json.load(open(f'{EN}/rungs.json'))}
m = Ensemble([f'{EN}/ckpt/{x}.npz' for x in _R[rung]['members']]) if _R.get(rung, {}).get('members') else ENet(f'{EN}/ckpt/{rung}.npz')


def zscore(p, y, s):
    return np.abs(np.log(np.maximum(p, 1e-9) / np.maximum(y, 1e-9))) / s


# ---------- dev: sigma factors (widen only) ----------
dv = EV.predict_real(m, 'ved_dev'); dv = dv[dv.y > dv.group.map(EV.FLOOR)]
de = EV.epa_frame(m, 'dev'); dh = EV.predict_real(m, 'hd')
K = {}
for g, d in dv.groupby('group'):
    K[f'VED_{g}'] = float(max(1.0, np.quantile(zscore(d.pred, d.y, d.sig), 0.68)))
# EPA: per vehicle-test errors, sigma = mean sigma of its legs
r_dev, o_dev = EV.epa_metrics(de)
sg = de.groupby(['vehicle_id', 'test']).sig.mean()
o_dev = o_dev.join(sg, on=['vehicle_id', 'test'])
for a, d in o_dev.groupby('arch'):
    K[f'EPA_{a}'] = float(max(1.0, np.quantile(zscore(d.pred, d.y, d.sig), 0.68)))
for s_, d in dh[dh.split == 'dev'].groupby('set'):
    if s_ == 'Mersin':
        d = d.groupby('trip_id').agg(pred=('pred', 'mean'), y=('y', 'first'), sig=('sig', 'mean'), fuel=('fuel', 'first'))
        d['pred'] = d.pred * np.where(d.fuel == 'cng', EV.LHV / EV.CNG_LHV, 1.0)
    K[f'HD_{s_}'] = float(max(1.0, np.quantile(zscore(d.pred, d.y, d.sig), 0.68)))
print('sigma factors (dev):', K)


def cover(p, y, s, k):
    z = zscore(p, y, s * k)
    return dict(cov68=float(np.mean(z < 1.0)), cov95=float(np.mean(z < 1.96)))


R = {'rung': rung, 'sigma_factors_dev': K}
# ---------- VED exam ----------
ex = EV.predict_real(m, 'ved_exam'); ex = ex[ex.y > ex.group.map(EV.FLOOR)].copy()
ex['k'] = ex.group.map(lambda g: K[f'VED_{g}'])
ved = {}
for col, name in [('pred', 'EnergyNet'), ('b_label', 'label_mpg'), ('b_vsp', 'vsp_moves_like'), ('b_class', 'class_median'), ('E_phys', 'physics_estimate')]:
    ved[name] = EV.ved_metrics(ex.rename(columns={col: 'P'}).assign(**{col: ex[col]}), 'P')
ved['coverage'] = {g: cover(d.pred, d.y, d.sig, d.k.iloc[0]) for g, d in ex.groupby('group')}
ved['by_arch'] = {a: dict(n=int(len(d)), EnergyNet=float(np.median(np.abs(d.pred / d.y - 1))), vsp=float(np.median(np.abs(d.b_vsp / d.y - 1))),
                         label=float(np.median(np.abs(d.b_label / d.y - 1)))) for a, d in ex.groupby('arch')}
ex['Tb'] = pd.cut(ex['T'], [-40, -5, 5, 15, 25, 45], labels=['< -5 C', '-5..5 C', '5..15 C', '15..25 C', '> 25 C'])
ex['vb'] = pd.cut(ex.v_mean * 3.6, [0, 25, 40, 55, 200], labels=['< 25 km/h', '25-40 km/h', '40-55 km/h', '> 55 km/h'])
ex['gb'] = pd.cut(ex.climb / ex.dist_km, [-1, 5, 10, 20, 1000], labels=['< 5 m/km', '5-10 m/km', '10-20 m/km', '> 20 m/km'])
ex['lb'] = pd.cut(ex.dist_km, [0, 2, 5, 10, 1000], labels=['< 2 km', '2-5 km', '5-10 km', '> 10 km'])
br = {}
for b in ['Tb', 'vb', 'gb', 'lb']:
    br[b] = {}
    for (g, k), d in ex.groupby(['group', b], observed=True):
        br[b].setdefault(g, {})[str(k)] = dict(n=int(len(d)), EnergyNet=float(np.median(np.abs(d.pred / d.y - 1))),
                                              vsp=float(np.median(np.abs(d.b_vsp / d.y - 1))), bias=float(np.median(d.pred / d.y - 1)))
ved['breakdowns'] = br
R['VED'] = ved
ex.to_parquet(f'{EN}/data/exam_ved.parquet')
# ---------- EPA exam ----------
ee = EV.epa_frame(m, 'exam')
r_ex, o_ex = EV.epa_metrics(ee)
o_ex = o_ex.join(ee.groupby(['vehicle_id', 'test']).sig.mean(), on=['vehicle_id', 'test'])
fixb = json.load(open(f'{EN}/data/epa_baseline_fixed.json'))
summ = rd.baselines('summary')['EPA_exam']
epa = {}
for (a, t), d in o_ex.groupby(['arch', 'test']):
    key = f"{'hev' if a.startswith('hev') else a}_{t}"
    e = dict(n=int(len(d)), EnergyNet=float(np.median(np.abs(d.pred / d.y - 1))), bias=float(np.median(d.pred / d.y - 1)),
             **cover(d.pred, d.y, d.sig, K.get(f'EPA_{a}', 1.0)))
    if key in fixb:
        rg = pd.read_parquet(f'{EN}/data/epa_reg_{key}.parquet').set_index('vehicle_id')
        j = d.set_index('vehicle_id').join(rg.reg, how='inner')
        e['regression_same_vehicles'] = float(np.median(np.abs(j.reg / j.y - 1))); e['EnergyNet_same_vehicles'] = float(np.median(np.abs(j.pred / j.y - 1)))
        e['n_same'] = int(len(j)); e['regression_all'] = fixb[key]['regression_MdAPE']; e['dev_median_all'] = fixb[key]['median_MdAPE']
    bkey = {'bev_FTP': 'bev_CD_UDDS', 'bev_HWY': 'bev_CD_HWY'}.get(key)
    if bkey and bkey in summ:
        e['regression_all'] = summ[bkey]['regression_MdAPE']; e['dev_median_all'] = summ[bkey]['median_MdAPE']
    epa[key] = e
R['EPA'] = epa
o_ex.to_parquet(f'{EN}/data/exam_epa.parquet')
# ---------- heavy duty ----------
hd = EV.predict_real(m, 'hd')
H = {}
z = hd[(hd.set == 'ZTBus_leg') & (hd.split == 'exam')]
kz = K.get('HD_ZTBus_leg', 1.0)
hb = rd.baselines('summary')['HD_exam']
H['ZTBus_legs'] = dict(n=int(len(z)), EnergyNet=float(np.median(np.abs(z.pred / z.y - 1))), bias=float(np.median(z.pred / z.y - 1)),
                       baseline=hb['ZTBus_legs_kWh']['MdAPE'], **cover(z.pred, z.y, z.sig, kz))
zm = z.groupby('mission')[['pred', 'y']].sum()
kpk = hb['ZTBus_legs_kWh']['kWh_per_km_dev'] * 3.6
zb = z.assign(b=z.dist_km * kpk).groupby('mission')[['b', 'y']].sum()
H['ZTBus_traced_missions'] = dict(n=int(len(zm)), EnergyNet=float(np.median(np.abs(zm.pred / zm.y - 1))), baseline_same=float(np.median(np.abs(zb.b / zb.y - 1))))
# all missions: route proxy = traced legs of the same bus route (else all traced legs), run at the mission's mean temperature
M = rd.hd_data()['ztbus_missions']; legs = hd[hd.set == 'ZTBus_leg'].copy()
legs = legs.join(M.set_index('name').busRoute, on='mission')
B_, T_, X_, E_, M_ = EV.real_set('hd')
li = np.where(M_.set.values == 'ZTBus_leg')[0]
mt = []
for r in M[M.split == 'exam'].itertuples():
    sel = li[(legs.busRoute.values == r.busRoute)] if (legs.busRoute.values == r.busRoute).any() else li
    X = X_[sel].copy()
    X[:, 76:79] = r.T_amb_mean_C / 20.0                  # temperature mean / min / max features
    X[:, 79] = float(r.T_amb_mean_C < 0); X[:, 80] = float(r.T_amb_mean_C > 25)
    p, s, c = m.predict(X, E_[sel])
    kpk_ = p.sum() / (B_['dist'][sel].sum() / 1000)
    mt.append(dict(name=r.name, route=r.busRoute, pred=kpk_ * r.drivenDistance / 1000, y=r.kWh * 3.6, base=kpk * r.drivenDistance / 1000, sig=float(np.mean(s))))
mt = pd.DataFrame(mt)
H['ZTBus_missions'] = dict(n=int(len(mt)), EnergyNet=float(np.median(np.abs(mt.pred / mt.y - 1))), baseline=float(np.median(np.abs(mt.base / mt.y - 1))),
                           bias=float(np.median(mt.pred / mt.y - 1)), **cover(mt.pred, mt.y, mt.sig, kz))
ms = hd[(hd.set == 'Mersin') & (hd.split == 'exam')]
mdv = hd[(hd.set == 'Mersin') & (hd.split == 'dev')]
for fuel, d in ms.groupby('fuel'):
    g = d.groupby('trip_id').agg(p=('pred', 'mean'), y=('y', 'first'), sig=('sig', 'mean'))
    conv = EV.LHV / EV.CNG_LHV if fuel == 'cng' else 1.0
    base = mdv[mdv.fuel == fuel].groupby('trip_id').y.first().median()
    H[f'Mersin_{fuel}'] = dict(n=int(len(g)), EnergyNet=float(np.median(np.abs(g.p * conv / g.y - 1))), bias=float(np.median(g.p * conv / g.y - 1)),
                               baseline=float(np.median(np.abs(base / g.y - 1))), **cover(g.p * conv, g.y, g.sig, K.get('HD_Mersin', 1.0)))
t = hd[(hd.set == 'ICCT') & (hd.split == 'exam')]
tdv = hd[(hd.set == 'ICCT') & (hd.split == 'dev')].set_index('cycle')
tb = t.cycle.map(tdv.g_per_km) * t.dist_km * EV.LHV
H['ICCT'] = dict(n=int(len(t)), EnergyNet=float(np.median(np.abs(t.pred / t.y - 1))), baseline=float(np.median(np.abs(tb / t.y - 1))),
                 rows=[dict(truck=a, cycle=b, pred_L100=float(p / EV.LHV / dk / 832 * 100), meas_L100=float(l)) for a, b, p, dk, l in
                       zip(t.truck, t.cycle, t.pred, t.dist_km, t.L_per_100km)], **cover(t.pred, t.y, t.sig, K.get('HD_ICCT', 1.0)))
R['HD'] = H
hd.to_parquet(f'{EN}/data/exam_hd.parquet'); mt.to_parquet(f'{EN}/data/exam_ztbus_missions.parquet')
json.dump(R, open(f'{EN}/exam.json', 'w'), indent=1)
print(json.dumps({'VED': {k: {g: round(v['trip_MdAPE'], 3) for g, v in ved[k].items()} for k in ['EnergyNet', 'label_mpg', 'vsp_moves_like', 'class_median', 'physics_estimate']},
                  'VED_veh': {g: round(v['veh_MdAPE'], 3) for g, v in ved['EnergyNet'].items()},
                  'coverage': ved['coverage'], 'EPA': epa, 'HD': {k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in H.items()}}, indent=1))
