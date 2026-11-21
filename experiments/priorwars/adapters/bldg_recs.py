"""BuildingDetective, RECS homes from annual totals (12 features; the released 'annual' network).

Targets = floor area (log10 m2), year built (sim: exact; RECS: midpoint of the survey's range, as the model's own
evaluate.eval_recs), heating group (electric = heat pump or resistance, fuel = fuel or district, none). Dev = the
1,848 RECS dev respondents; exam = the 7,401 RECS exam respondents. Sim pool = bank f2 annual shards 0-59 (R9 bank);
frozen sim set = shards 900-904.
"""
import functools
import numpy as np
from adapters import _bldg as B

NAME = 'bldg_recs'
TITLE = 'BuildingDetective (RECS, annual)'
MODE = 'annual'
TARGETS = [{'name': 'logA', 'kind': 'reg'}, {'name': 'year', 'kind': 'reg'}, {'name': 'heatgrp', 'kind': 'multi'}]
HEADLINE_ITEMS = ['floor area', 'heating fuel vs electric']
HEADLINE = 'mean of floor-area MdAPE (%, negated) and heating-group accuracy (%); vintage MAE reported, not in the headline'
GRP = {'heat_pump': 0, 'resistance': 0, 'fuel': 1, 'district': 1, 'none': 2}
HEAT2GRP = np.array([GRP[h] for h in B.HEAT], float)


def _sim(shards):
    X, Y, K = B.bank(MODE, shards)
    h = K[:, 1]
    g = np.where(h >= 0, HEAT2GRP[np.clip(h, 0, None)], np.nan)
    return X, {'logA': Y[:, 0], 'year': Y[:, 1], 'heatgrp': g}, None


@functools.lru_cache(maxsize=1)
def sim_pool():
    return _sim(range(0, 60))


@functools.lru_cache(maxsize=1)
def sim_val():
    return _sim(range(900, 905))


def _real(split):
    mats, labs = B.real(split, 'recs')
    X = mats[MODE].astype(np.float32)
    ok = np.isfinite(X).all(1)
    A = np.array([l['A'] for l in labs], float)
    lo = np.array([l['year_lo'] if l['year_lo'] is not None else np.nan for l in labs], float)
    hi = np.array([l['year_hi'] if l['year_hi'] is not None else np.nan for l in labs], float)
    ymid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(hi), hi - 10, lo + 5))
    g = np.array([GRP.get(l['heat'], np.nan) if l['heat'] else np.nan for l in labs], float)
    ids = np.array([l['id'] for l in labs])
    return X[ok], {'logA': np.log10(np.where(A > 0, A, np.nan))[ok], 'year': ymid[ok], 'heatgrp': g[ok]}, ids[ok]


def dev():
    return _real('dev')


def exam_X():
    X, Y, ids = _real('exam')
    return X, ids


def exam_truth():
    X, Y, ids = _real('exam')
    return Y, ids


def norm(X):
    raw, (xm, xs) = B.net(MODE)
    return ((np.asarray(X, float) - xm) / xs).astype(np.float32)


def P_predict(X):
    raw = B.net(MODE)[0]
    mu, lg = raw(X)
    nu = len(B.USES)
    h = lg[:, nu:nu + len(B.HEAT)].argmax(1)
    return {'logA': mu[:, 0], 'year': mu[:, 1], 'heatgrp': HEAT2GRP[h]}


def items(Y, P):
    return {'floor area': (10 ** Y['logA'], 10 ** np.asarray(P['logA'], float), ['mape']),
            'heating fuel vs electric': (Y['heatgrp'], np.asarray(P['heatgrp'], float), ['acc']),
            'vintage': (Y['year'], np.asarray(P['year'], float), ['mae'])}


def metrics(t):
    return {'logA': ['mape'], 'year': ['mae'], 'heatgrp': ['acc']}[t]


def C_dev_predict(Xd, Yd, Xe):
    from adapters import _cdev
    return _cdev.predict_all(TARGETS, Xd, Yd, Xe)


def C_dev_cv(X, Y):
    from adapters import _cdev
    import sys
    return _cdev.cv(sys.modules[__name__], X, Y)
