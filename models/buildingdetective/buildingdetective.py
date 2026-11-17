"""BuildingDetective inference (numpy + scipy only). Same code in the release.

    det = Detective()                                  # loads hourly / monthly / annual .npz from weights/ next to this file
    det = Detective('/path/to/weights_dir')            # or from another folder
    r = det.read_hourly(meters, T, GHI, loc, lat)       # meters: {'elec','fuel','dheat','dcool'} -> hourly kW
    r = det.read_monthly(mm, Tm, Hm, Cm, nd, lat)       # or det.read_monthly_from_hourly(meters, T, loc, lat)
    r = det.read_annual({'elec': kWh, 'fuel': kWh, 'dheat': kWh}, hdd18, cdd18)
r: dict with 'floor_area_m2', 'year_built', 'UA_W_K', 'UA_W_m2K', 'tau_h', 'Tb_heat_C', 'Tb_cool_C',
   'S_heat_W_K', 'S_cool_W_K', 'base_share', 'op_hours', 'weekend_ratio' -> (value, lo, hi) at +-1 sigma,
   and 'use', 'heat', 'cool' -> {class: probability}.
"""
import os

import numpy as np
from scipy.special import erf

import common as C
import features as F

__version__ = '1.0'
HERE = os.path.dirname(os.path.abspath(__file__))

OUT = [('floor_area_m2', 'logA', 'log'), ('year_built', 'year', 'lin'), ('UA_W_K', 'logUA', 'log'),
       ('UA_W_m2K', 'logUAm2', 'log'), ('tau_h', 'logtau', 'log'), ('Tb_heat_C', 'Tb_heat', 'lin'),
       ('Tb_cool_C', 'Tb_cool', 'lin'), ('S_heat_W_K', 'logS_heat', 'log'), ('S_cool_W_K', 'logS_cool', 'log'),
       ('base_share', 'base_share', 'lin'), ('op_hours', 'op_hours', 'lin'), ('weekend_ratio', 'weekend', 'lin')]


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


class Net:
    def __init__(self, fn):
        z = np.load(fn, allow_pickle=False)
        self.n = int(z['n_layers'])
        self.W = [z[f'W{i}'] for i in range(self.n)]
        self.b = [z[f'b{i}'] for i in range(self.n)]
        self.Wr, self.br, self.Wc, self.bc = z['Wr'], z['br'], z['Wc'], z['bc']
        self.xm, self.xs, self.ym, self.ys = z['xm'], z['xs'], z['ym'], z['ys']
        self.widen = z['widen'] if 'widen' in z else np.ones(len(self.ym))
        self.temp = z['temp'] if 'temp' in z else np.ones(3)

    def raw(self, X):
        h = (np.atleast_2d(X).astype(np.float64) - self.xm) / self.xs
        for W, b in zip(self.W, self.b):
            h = gelu(h @ W.T + b)
        r = h @ self.Wr.T + self.br
        nr = len(self.ym)
        mu = r[:, :nr] * self.ys + self.ym
        sg = np.exp(np.clip(r[:, nr:], -6, 3)) * self.ys * self.widen
        lg = h @ self.Wc.T + self.bc
        return mu, sg, lg


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def decode(mu, sg, lg, temp=(1, 1, 1)):
    res = []
    nu, nh = len(C.USES), len(C.HEAT)
    for i in range(len(mu)):
        r = {}
        for j, (name, key, kind) in enumerate(OUT):
            m, s = mu[i, j], sg[i, j]
            if kind == 'log':
                r[name] = (float(10 ** m), float(10 ** (m - s)), float(10 ** (m + s)))
            else:
                r[name] = (float(m), float(m - s), float(m + s))
        r['use'] = dict(zip(C.USES, _softmax(lg[i, :nu] / temp[0]).round(4).tolist()))
        r['heat'] = dict(zip(C.HEAT, _softmax(lg[i, nu:nu + nh] / temp[1]).round(4).tolist()))
        r['cool'] = dict(zip(C.COOL, _softmax(lg[i, nu + nh:] / temp[2]).round(4).tolist()))
        r['_mu'], r['_sigma'] = mu[i].tolist(), sg[i].tolist()
        res.append(r)
    return res


class Detective:
    def __init__(self, wdir=None):
        wdir = wdir or os.path.join(HERE, 'weights')
        self.nets = {m: Net(os.path.join(wdir, f'{m}.npz')) for m in ('hourly', 'monthly', 'annual')
                     if os.path.exists(os.path.join(wdir, f'{m}.npz'))}

    def _run(self, mode, X):
        n = self.nets[mode]
        mu, sg, lg = n.raw(X)
        return decode(mu, sg, lg, n.temp)

    def read_hourly(self, meters, T, GHI, loc, lat):
        return self._run('hourly', F.feat_hourly(meters, T, GHI, loc, lat))[0]

    def read_monthly(self, mm, Tm, Hm, Cm, nd, lat):
        return self._run('monthly', F.feat_monthly(mm, Tm, Hm, Cm, nd, lat))[0]

    def read_monthly_from_hourly(self, meters, T, loc, lat):
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(meters, T, loc)
        return self.read_monthly(mm, Tm, Hm, Cm, nd, lat)

    def read_annual(self, kwh, hdd, cdd):
        return self._run('annual', F.feat_annual(kwh, hdd, cdd))[0]
