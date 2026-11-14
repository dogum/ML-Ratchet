"""Phase D final exam (real data never used for choosing). Run ONCE, after the release choice is committed.
python exam2.py <rung> [--sigma v1|car]

1. error bars fitted on DEV only (calib.py; the method chosen by rung r10): VED groups, EPA architectures, heavy duty
2. VED exam trips: EnergyNet v2 vs the v1 release (re-scored on the same trips) vs label / MOVES-like / class-median
3. EPA exam families: FTP, HWY, US06, 20 F cold FTP, SC03 vs the dev-fitted log-linear regression and the dev median
4. Full ZTBus exam (1,115 missions by month): legs and mission totals, v2 and v1 (both on the full traces) vs the dev
   kWh/km x distance baseline; v1's route-proxy number from rescore_v1_ztbus.json; Mersin and ICCT as in v1
-> work/exam2.json, work/data/exam2_<set>.parquet"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, argparse
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.VEHICLE)
import numpy as np, pandas as pd
import evaluate as EV, calib as CB
from model_np import Ensemble
from ratchet2 import RUNGS, load
import realdata as rd

ap = argparse.ArgumentParser(); ap.add_argument('rung'); ap.add_argument('--sigma', default='car')
a = ap.parse_args()
W = _P.WORK
R0 = {r['name']: r for r in RUNGS}[a.rung]
v2 = load(R0); bus_spec = bool(R0.get('bus_spec'))
v1 = Ensemble(sorted(os.path.join(_P.WORK, 'weights_v1', f) for f in os.listdir(os.path.join(_P.WORK, 'weights_v1')) if f.endswith('.npz'))) \
    if os.path.isdir(os.path.join(_P.WORK, 'weights_v1')) else None
K1 = json.load(open(os.path.join(_P.WORK, 'weights_v1', 'sigma_factors_v1.json'))) if v1 else {}


def frames(m, bs):
    return dict(ved_dev=EV.predict_real(m, 'ved_dev'), ved_exam=EV.predict_real(m, 'ved_exam'), epa_dev=EV.epa_frame(m, 'dev'),
                epa_exam=EV.epa_frame(m, 'exam'), hd=EV.predict_real(m, 'hd'), ztb=EV.predict_real(m, 'ztb', bs))


def sigma_fit(Fr, method):
    S = {}
    dv = Fr['ved_dev']; dv = dv[dv.y > dv.group.map(EV.FLOOR)]
    for g, d in dv.groupby('group'):
        S[f'VED_{g}'] = CB.fit(d, method)
    o = epa_vehicle(Fr['epa_dev'])
    for ar, d in o.groupby('arch'):
        S[f'EPA_{ar}'] = (CB.fit_k(d.assign(vehicle_id=d.vehicle_id)), 0.0)
    z = Fr['ztb']; zl = z[(z.split == 'dev') & (z.kind == 'leg') & z.label_ok & (z.y > 0)]
    S['HD_ZTBus'] = (CB.fit_k(zl.assign(vehicle_id=zl.mission)), 0.0)
    h = Fr['hd']
    ms = h[(h.set == 'Mersin') & (h.split == 'dev')].groupby('trip_id').agg(pred=('pred', 'mean'), y=('y', 'first'), sig=('sig', 'mean'), fuel=('fuel', 'first'))
    ms['pred'] = ms.pred * np.where(ms.fuel == 'cng', EV.LHV / EV.CNG_LHV, 1.0)
    S['HD_Mersin'] = (CB.fit_k(ms.reset_index().assign(vehicle_id=0)), 0.0)
    ic = h[(h.set == 'ICCT') & (h.split == 'dev')]
    S['HD_ICCT'] = (CB.fit_k(ic.assign(vehicle_id=0)), 0.0)
    return S


def epa_vehicle(M):
    r, o = EV.epa_metrics(M)
    return o.join(M.groupby(['vehicle_id', 'test']).sig.mean(), on=['vehicle_id', 'test'])


def cover(p, y, s, kt):
    k, tau = kt
    zz = np.abs(np.log(np.maximum(p, 1e-9) / np.maximum(y, 1e-9))) / np.sqrt((k * s) ** 2 + tau ** 2)
    return dict(cov68=float(np.mean(zz < 1.0)), cov95=float(np.mean(zz < 1.96)))


def mdape(p, y): return float(np.median(np.abs(p / y - 1)))
def bias(p, y): return float(np.median(p / y - 1))


def ved_block(ex, S):
    out = {}
    for g, d in ex.groupby('group'):
        v = d.groupby('vehicle_id')[['pred', 'y', 'b_label', 'b_vsp']].sum()
        out[g] = dict(n=int(len(d)), n_veh=int(len(v)), trip_MdAPE=mdape(d.pred, d.y), bias=bias(d.pred, d.y),
                      veh_MdAPE=mdape(v.pred, v.y), **cover(d.pred, d.y, d.sig, S.get(f'VED_{g}', (1.0, 0.0))))
    return out


def run_model(m, bs, S_method, label):
    Fr = frames(m, bs)
    S = sigma_fit(Fr, S_method)
    R = {'sigma_dev': {k: dict(k=v[0], tau=v[1]) for k, v in S.items()}}
    ex = Fr['ved_exam']; ex = ex[ex.y > ex.group.map(EV.FLOOR)].copy()
    R['VED'] = ved_block(ex, S)
    R['VED_by_arch'] = {ar: dict(n=int(len(d)), MdAPE=mdape(d.pred, d.y), bias=bias(d.pred, d.y)) for ar, d in ex.groupby('arch')}
    ex['Tb'] = pd.cut(ex['T'], [-40, -5, 5, 15, 25, 45], labels=['< -5 C', '-5..5 C', '5..15 C', '15..25 C', '> 25 C'])
    ex['vb'] = pd.cut(ex.v_mean * 3.6, [0, 25, 40, 55, 200], labels=['< 25 km/h', '25-40 km/h', '40-55 km/h', '> 55 km/h'])
    br = {}
    for b in ('Tb', 'vb'):
        br[b] = {}
        for (g, k), d in ex.groupby(['group', b], observed=True):
            br[b].setdefault(g, {})[str(k)] = dict(n=int(len(d)), MdAPE=mdape(d.pred, d.y), vsp=mdape(d.b_vsp, d.y), bias=bias(d.pred, d.y))
    R['VED_breakdowns'] = br
    o = epa_vehicle(Fr['epa_exam'])
    R['EPA'] = {}
    for (ar, t), d in o.groupby(['arch', 'test']):
        R['EPA'][f'{ar}_{t}'] = dict(n=int(len(d)), MdAPE=mdape(d.pred, d.y), bias=bias(d.pred, d.y), **cover(d.pred, d.y, d.sig, S.get(f'EPA_{ar}', (1.0, 0.0))))
    z = Fr['ztb']; ze = z[z.split == 'exam']
    zl = ze[(ze.kind == 'leg') & ze.label_ok & (ze.y > 0)]
    mi = EV.ztb_missions(ze)
    mi['sig_w'] = ze.assign(ps=ze.pred * ze.sig).groupby('mission').ps.sum() / mi.pred
    kz = S['HD_ZTBus']
    R['ZTBus_full'] = dict(legs=dict(n=int(len(zl)), MdAPE=mdape(zl.pred, zl.y), bias=bias(zl.pred, zl.y), **cover(zl.pred, zl.y, zl.sig, kz)),
                           missions=dict(n=int(len(mi)), MdAPE=mdape(mi.pred, mi.y), bias=bias(mi.pred, mi.y), **cover(mi.pred, mi.y, mi.sig_w, kz)))
    h = Fr['hd']
    for fuel, d in h[(h.set == 'Mersin') & (h.split == 'exam')].groupby('fuel'):
        g = d.groupby('trip_id').agg(p=('pred', 'mean'), y=('y', 'first'), sig=('sig', 'mean'))
        cv = EV.LHV / EV.CNG_LHV if fuel == 'cng' else 1.0
        R[f'Mersin_{fuel}'] = dict(n=int(len(g)), MdAPE=mdape(g.p * cv, g.y), bias=bias(g.p * cv, g.y), **cover(g.p * cv, g.y, g.sig, S['HD_Mersin']))
    t = h[(h.set == 'ICCT') & (h.split == 'exam')]
    R['ICCT'] = dict(n=int(len(t)), MdAPE=mdape(t.pred, t.y), **cover(t.pred, t.y, t.sig, S['HD_ICCT']))
    ex.to_parquet(f'{W}/data/exam2_ved_{label}.parquet'); o.to_parquet(f'{W}/data/exam2_epa_{label}.parquet')
    mi.to_parquet(f'{W}/data/exam2_ztb_missions_{label}.parquet')
    return R


OUT = {'rung': a.rung, 'sigma_method': a.sigma, 'v2': run_model(v2, bus_spec, a.sigma, 'v2')}
if v1 is not None:
    OUT['v1_release'] = run_model(v1, False, 'v1', 'v1')
# baselines
ex = pd.read_parquet(f'{W}/data/exam2_ved_v2.parquet')
OUT['baselines'] = {'VED': {g: dict(label=mdape(d.b_label, d.y), moves_like=mdape(d.b_vsp, d.y), class_median=mdape(d.b_class, d.y),
                                    label_veh=mdape(d.groupby('vehicle_id').b_label.sum(), d.groupby('vehicle_id').y.sum()),
                                    moves_like_veh=mdape(d.groupby('vehicle_id').b_vsp.sum(), d.groupby('vehicle_id').y.sum()))
                            for g, d in ex.groupby('group')}}
OUT['baselines']['VED_by_arch'] = {ar: dict(moves_like=mdape(d.b_vsp, d.y), label=mdape(d.b_label, d.y)) for ar, d in ex.groupby('arch')}
fixb = json.load(open(f'{W}/data/epa_baseline_fixed.json'))
summ = rd.baselines('summary')['EPA_exam']
o = pd.read_parquet(f'{W}/data/exam2_epa_v2.parquet')
eb = {}
for (ar, t), d in o.groupby(['arch', 'test']):
    key = f"{'hev' if ar.startswith('hev') else ar}_{t}"
    e = {}
    if key in fixb:
        rg = pd.read_parquet(f'{W}/data/epa_reg_{key}.parquet').set_index('vehicle_id')
        j = d.set_index('vehicle_id').join(rg.reg, how='inner')
        e = dict(regression_same=mdape(j.reg, j.y), EnergyNet_same=mdape(j.pred, j.y), n_same=int(len(j)),
                 regression_all=fixb[key]['regression_MdAPE'], dev_median_all=fixb[key]['median_MdAPE'])
    bk = {'bev_FTP': 'bev_CD_UDDS', 'bev_HWY': 'bev_CD_HWY'}.get(key)
    if bk and bk in summ:
        e = dict(regression_all=summ[bk]['regression_MdAPE'], dev_median_all=summ[bk]['median_MdAPE'])
    eb[f'{ar}_{t}'] = e
OUT['baselines']['EPA'] = eb
mi = pd.read_parquet(f'{W}/data/exam2_ztb_missions_v2.parquet')
kpk = EV.dev_kpk_ztb()
zz = EV.predict_real(v2, 'ztb', bus_spec); zl = zz[(zz.split == 'exam') & (zz.kind == 'leg') & zz.label_ok & (zz.y > 0)]
OUT['baselines']['ZTBus_full'] = dict(dev_MJ_per_km=kpk, missions=mdape(kpk * mi.km, mi.y), missions_bias=bias(kpk * mi.km, mi.y),
                                     legs=mdape(kpk * zl.dist_km, zl.y))
v1r = json.load(open(f'{W}/rescore_v1_ztbus.json'))
OUT['v1_ztbus_route_proxy_1115'] = v1r['by_month_exam_1115']
hb = rd.baselines('summary')['HD_exam']
h = EV.predict_real(v2, 'hd'); mdv = h[(h.set == 'Mersin') & (h.split == 'dev')]
for fuel in ('diesel', 'cng'):
    base = mdv[mdv.fuel == fuel].groupby('trip_id').y.first().median()
    g = h[(h.set == 'Mersin') & (h.split == 'exam') & (h.fuel == fuel)].groupby('trip_id').y.first()
    OUT['baselines'][f'Mersin_{fuel}'] = mdape(np.full(len(g), base), g.values)
t = h[(h.set == 'ICCT') & (h.split == 'exam')]; tdv = h[(h.set == 'ICCT') & (h.split == 'dev')].set_index('cycle')
OUT['baselines']['ICCT'] = mdape(t.cycle.map(tdv.g_per_km) * t.dist_km * EV.LHV, t.y)
json.dump(OUT, open(f'{W}/exam2.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != 'v2' and k != 'v1_release'}, indent=1)[:4000])
for lab in ('v2', 'v1_release'):
    if lab in OUT:
        r = OUT[lab]
        print(lab, 'VED', {g: (round(v['trip_MdAPE'], 3), round(v['bias'], 3), round(v['cov68'], 2)) for g, v in r['VED'].items()})
        print(lab, 'EPA', {k: (round(v['MdAPE'], 3), round(v['bias'], 3)) for k, v in r['EPA'].items()})
        print(lab, 'ZTB', r['ZTBus_full'], {k: r[k] for k in ('Mersin_diesel', 'Mersin_cng', 'ICCT')})
