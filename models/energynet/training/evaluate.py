"""Scoring: frozen simulated validation set (skill), dev gate (real dev units), exam (real, never used for choosing).
Every real set comes from realfeat.py records with their fixed tiers."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, functools
sys.path.insert(0, _P.SRC)
import numpy as np, pandas as pd
import dataset as DS, features as F
from model_np import ENet

D = _P.WORK + '/data'
LHV = 0.042; MI = 1609.344; GAL = 3.785411784
CNG_LHV = 0.0471
VAL_SEED = 777


@functools.lru_cache(maxsize=2)
def val_set(name='val'):
    B = DS.load_bank(name)
    T = DS.draw_tiers(B, np.random.default_rng(VAL_SEED))          # frozen tiers
    X, E = DS.assemble(B, T)
    return B, T, X, E


@functools.lru_cache(maxsize=4)
def dumb_rates(version, suf=''):
    """dumb baseline: median energy per km per (architecture, class) on the TRAIN bank."""
    B = DS.load_bank('train' + suf); T = DS.draw_tiers(B, np.random.default_rng(1))
    y = DS.target(B, T, version); k = B['arch'] * 10 + B['cls']
    ok = (y > 0.02) & (B['dist'] > 100); y, k, dist = y[ok], k[ok], B['dist'][ok]
    return {int(g): float(np.median(y[k == g] / dist[k == g])) for g in np.unique(k)}
    return {int(g): float(np.median(y[k == g] / B['dist'][k == g])) for g in np.unique(k)}


VMAP = {'v1': ('', 'v1'), 'v2': ('', 'v2'), 'v3': ('', 'v3'), 'v4': ('_v4', 'v3')}


def val_score(model, version, name='val'):
    suf, version = VMAP[version]
    B, T, X, E = val_set(name + suf)
    y = DS.target(B, T, version)
    ok = y > 0.02
    B = {k: v[ok] for k, v in B.items()}; T = {k: v[ok] for k, v in T.items()}; X, E, y = X[ok], E[ok], y[ok]
    p, s, c = model.predict(X, E)
    r = dumb_rates(version, suf); k = B['arch'] * 10 + B['cls']
    pd_ = np.array([r.get(int(g), np.nan) for g in k]) * B['dist']
    e_net = np.abs(np.log(p / y)); e_dumb = np.abs(np.log(pd_ / y)); e_phys = np.abs(np.log(E / y))
    out = dict(skill=float(1 - np.median(e_net) / np.median(e_dumb)), med_log_err=float(np.median(e_net)),
               MdAPE=float(np.median(np.abs(p / y - 1))), phys_MdAPE=float(np.median(np.abs(E / y - 1))),
               dumb_MdAPE=float(np.median(np.abs(pd_ / y - 1))), n=int(len(y)),
               z_cover68=float(np.mean(np.abs(np.log(p / y)) < s)))
    for a in range(7):
        m = B['arch'] == a
        if m.any(): out[f'MdAPE_{F.ARCHS[a]}'] = float(np.median(np.abs(p[m] / y[m] - 1)))
    for c_ in range(7):
        m = B['cls'] == c_
        if m.any(): out[f'MdAPE_{F.CLASSES[c_]}'] = float(np.median(np.abs(p[m] / y[m] - 1)))
    return out


@functools.lru_cache(maxsize=8)
def real_set(name):
    B = dict(np.load(f'{D}/real_{name}.npz'))
    T = {k: B['T_' + k] for k in DS.TIER_KEYS}
    M = pd.read_parquet(f'{D}/real_{name}_meta.parquet')
    X, E = DS.assemble(B, T)
    return B, T, X, E, M


def predict_real(model, name):
    B, T, X, E, M = real_set(name)
    p, s, c = model.predict(X, E)
    M = M.copy(); M['pred'] = p * B['vr']; M['sig'] = s; M['y'] = B['y']; M['E_phys'] = E * B['vr']
    for j, k in enumerate(['aero', 'rolling', 'brake', 'aux', 'losses']): M['c_' + k] = c[:, j]
    return M


FLOOR = {'ICE': 1 * LHV, 'HEV': 1 * LHV, 'EV': 0.01 * 3.6, 'PHEV': 0.05}


def ved_metrics(M, col='pred'):
    r = {}
    for g, d in M.groupby('group'):
        d = d[(d.y > FLOOR[g]) & d[col].notna()]
        ape = np.abs(d[col] / d.y - 1)
        v = d.groupby('vehicle_id')[[col, 'y']].sum()
        r[g] = dict(n=int(len(d)), n_veh=int(len(v)), trip_MdAPE=float(ape.median()), trip_MAPE=float(ape.mean()),
                    bias=float(np.median(d[col] / d.y - 1)), veh_MdAPE=float(np.median(np.abs(v[col] / v.y - 1))))
    return r


def epa_metrics(M, col='pred'):
    """per vehicle: FTP = 0.43 x cold UDDS + 0.57 x hot UDDS fuel per mile; target gal/mi (fuel) or wall kWh/mi (BEV)."""
    M = M.copy()
    miles = M['dist_m'] / MI if 'dist_m' in M else None
    r = {}
    out = []
    for (vid, test), d in M.groupby(['vehicle_id', 'test']):
        arch = d.arch.iloc[0]
        per = d[col].values / d['miles'].values
        if arch == 'bev':
            pr = per[0] / 3.6 / 0.88
            yv = d.y.iloc[0]
        else:
            if test == 'FTP' and len(d) == 2:
                w = np.where(d.cold.values, 0.43, 0.57); per = np.sum(w * per)
            else:
                per = per[0]
            pr = per / LHV / 745.0 / GAL           # gal/mi
            yv = 1.0 / d.y.iloc[0]
        out.append(dict(vehicle_id=vid, arch=arch, test=test, pred=pr, y=yv, family=d.family.iloc[0]))
    o = pd.DataFrame(out)
    for (a, t), d in o.groupby(['arch', 'test']):
        r[f'{a}_{t}'] = dict(n=int(len(d)), MdAPE=float(np.median(np.abs(d.pred / d.y - 1))), bias=float(np.median(d.pred / d.y - 1)))
    return r, o


def hd_metrics(M, split, col='pred'):
    r = {}
    z = M[(M.set == 'ZTBus_leg') & (M.split == split)]
    if len(z): r['ZTBus_legs'] = dict(n=int(len(z)), MdAPE=float(np.median(np.abs(z[col] / z.y - 1))), bias=float(np.median(z[col] / z.y - 1)))
    if len(z):
        zm = z.groupby('mission')[[col, 'y']].sum()
        r['ZTBus_traced_missions'] = dict(n=int(len(zm)), MdAPE=float(np.median(np.abs(zm[col] / zm.y - 1))))
    m = M[(M.set == 'Mersin') & (M.split == split)]
    for fuel, d in m.groupby('fuel'):
        g = d.groupby('trip_id').agg(p=(col, 'mean'), y=('y', 'first'))
        conv = (LHV / CNG_LHV) if fuel == 'cng' else 1.0        # predicted MJ -> grams at the fuel's LHV, label grams at 0.042
        r[f'Mersin_{fuel}'] = dict(n=int(len(g)), MdAPE=float(np.median(np.abs(g.p * conv / g.y - 1))), bias=float(np.median(g.p * conv / g.y - 1)))
    t = M[(M.set == 'ICCT') & (M.split == split)]
    if len(t): r['ICCT'] = dict(n=int(len(t)), MdAPE=float(np.median(np.abs(t[col] / t.y - 1))), bias=float(np.median(t[col] / t.y - 1)))
    return r


def epa_frame(model, split):
    M = predict_real(model, f'epa_{split}')
    B = real_set(f'epa_{split}')[0]
    M['miles'] = B['dist'] / MI
    return M


def dev_gate(model):
    res = {}
    res['VED'] = ved_metrics(predict_real(model, 'ved_dev'))
    res['EPA'] = epa_metrics(epa_frame(model, 'dev'))[0]
    res['HD'] = hd_metrics(predict_real(model, 'hd'), 'dev')
    vals = [res['VED'][g]['trip_MdAPE'] for g in ('ICE', 'HEV', 'EV', 'PHEV')]
    vals += [res['EPA'][k]['MdAPE'] for k in ('conv_FTP', 'conv_HWY', 'conv_US06', 'hev_powersplit_FTP', 'hev_powersplit_HWY', 'bev_FTP', 'bev_HWY') if k in res['EPA']]
    vals += [res['HD'][k]['MdAPE'] for k in ('ZTBus_legs', 'Mersin_diesel', 'Mersin_cng', 'ICCT') if k in res['HD']]
    res['D'] = float(np.mean(vals)); res['n_terms'] = len(vals)
    return res
