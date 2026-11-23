"""LawFinder inference: evaluate the released one-line laws (numpy only).

    from lawfinder import LawFinder
    lf = LawFinder()                      # loads lawfinder_laws.json next to this file
    lf.names()                            # ['water', 'fire', ...]
    lf['water'].expr                      # the law as text
    mu, sig = lf['water'].predict(phi=0.8, fs=0.1, sync=-0.2, S=0.14)
    lf['water'].textbook(phi=0.8)         # the textbook rival with its dev-fitted constant

Each law is an expression tree with constants fitted by symbolic regression on that platform's simulator only.
sig is the 68 % error bar: the residual spread on the frozen simulated set, widened once on the real dev set.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _ev(t, X, c):
    k = t[0]
    if k == 'x':
        return X[t[1]]
    if k == 'c':
        return c[t[1]]
    a = _ev(t[1], X, c)
    if k == '+': return a + _ev(t[2], X, c)
    if k == '-': return a - _ev(t[2], X, c)
    if k == '*': return a * _ev(t[2], X, c)
    if k == '/': return a / _ev(t[2], X, c)
    if k == '^':
        b = _ev(t[2], X, c)
        return np.where(a > 0, np.abs(a) ** b, np.nan)
    if k == 'exp': return np.exp(np.clip(a, -50, 50))
    if k == 'log': return np.where(a > 0, np.log(np.abs(a) + 1e-300), np.nan)
    if k == 'sqrt': return np.where(a >= 0, np.sqrt(np.abs(a)), np.nan)
    if k == 'inv': return 1.0 / a
    if k == 'sq': return a * a
    raise ValueError(k)


def _tree(j):
    if j[0] in ('x', 'c'):
        return (j[0], int(j[1]))
    return (j[0],) + tuple(_tree(s) for s in j[1:])


class Law:
    def __init__(self, d):
        self.d = d
        self.vars = d['vars']
        self.expr = d['expr']
        self.tree = _tree(d['tree'])
        self.consts = np.array(d['consts'], float)
        self.sigma = float(d['sigma'])
        self.raw_sigma = float(d['sigma_sim'])

    def __call__(self, **kw):
        return self.predict(**kw)[0]

    def predict(self, raw_sigma=False, **kw):
        missing = [v for v in self.vars if v not in kw]
        if missing:
            raise KeyError(f'missing inputs {missing}; this law reads {self.vars}')
        X = [np.asarray(kw[v], float) for v in self.vars]
        with np.errstate(all='ignore'):
            y = _ev(self.tree, X, self.consts)
        y = np.broadcast_to(y, np.broadcast(*X).shape).astype(float)
        return y, (self.raw_sigma if raw_sigma else self.sigma)

    def textbook(self, **kw):
        t = self.d['textbook']
        return _TEXTBOOK[t['form']](t['const'], **kw)


def _fu(w, phi, **_):
    phi = np.asarray(phi, float)
    return (1 + phi ** w) ** (1 / w) - phi


def _k_times(k, D, **_):
    return k * np.asarray(D, float)


def _swing(k, L, E, **_):
    return k * np.asarray(L, float) ** 2 / np.asarray(E, float)


def _rothermel(k, ros_mmin, P, **_):
    return np.log10(k * np.asarray(ros_mmin, float) * np.asarray(P, float) * 1.44 + 0.0081)


def _signature(c, H, C, **_):
    return c[0] + c[1] * np.asarray(H, float) + c[2] * np.asarray(C, float)


def _road_load(eta, v, K, G, M, D, R, T, h, e, **_):
    """road-load energy at the wheels per km (MJ/km) over one efficiency per powertrain (eta: {'ICE', 'HEV', 'EV'})"""
    v, K, G, M, D, R, T, h, e = (np.asarray(x, float) for x in (v, K, G, M, D, R, T, h, e))
    m = M * 1e3
    rho = 101325.0 / (287.05 * (T + 273.15))
    roll = m * 9.81 * R / 1e3
    w = np.maximum(roll + 0.5 * rho * D * v ** 2 + m * K + m * 9.81 * G / 100, 0.05 * roll) / 1e3
    return w / np.where(e > 0.5, eta['EV'], np.where(h > 0.5, eta['HEV'], eta['ICE']))


_TEXTBOOK = {'budyko_fu': _fu, 'k_times_demand': _k_times, 'swing_ramp': _swing, 'rothermel_x_perimeter': _rothermel,
             'degree_day_signature': _signature, 'road_load': _road_load}


class LawFinder:
    def __init__(self, path=None):
        self.data = json.load(open(path or os.path.join(HERE, 'lawfinder_laws.json')))
        self.laws = {k: Law(v) for k, v in self.data['laws'].items()}

    def names(self):
        return list(self.laws)

    def __getitem__(self, k):
        return self.laws[k]
