"""BuildingDetective, BDG2 buildings from monthly meter input (112 features; the released 'monthly' network).

Targets = floor area (log10 m2), year built, primary use (12 simulator classes; residential_* read as lodging, as in
the model's own evaluate.eval_bdg2). Dev = the 52 BDG2 dev buildings; exam = the 1,428 BDG2 exam buildings with a usable
2017 year. Sim pool = bank f2 shards 0-59 (the release R9 bank); frozen sim set = shards 900-904.
"""
import functools
import numpy as np
from adapters import _bldg as B

NAME = 'bldg_bdg2'
TITLE = 'BuildingDetective (BDG2, monthly)'
MODE = 'monthly'
TARGETS = [{'name': 'logA', 'kind': 'reg'}, {'name': 'year', 'kind': 'reg'}, {'name': 'use', 'kind': 'multi'}]
HEADLINE_ITEMS = ['floor area', 'primary use']
HEADLINE = 'mean of floor-area MdAPE (%, negated) and primary-use accuracy (%); year built reported, not in the headline'
LODGING = B.USES.index('lodging')
RES = [B.USES.index('residential_single'), B.USES.index('residential_multi')]


def _sim(shards):
    X, Y, K = B.bank(MODE, shards)
    use = K[:, 0].astype(float)
    use[use < 0] = np.nan
    use[np.isin(use, RES)] = LODGING      # TabPFN-v2 takes <= 10 classes; scoring reads residential_* as lodging anyway
    return X, {'logA': Y[:, 0], 'year': Y[:, 1], 'use': use}, None


@functools.lru_cache(maxsize=1)
def sim_pool():
    return _sim(range(0, 60))


@functools.lru_cache(maxsize=1)
def sim_val():
    return _sim(range(900, 905))


def _real(split):
    mats, labs = B.real(split, 'bdg2')
    X = mats[MODE].astype(np.float32)
    ok = np.isfinite(X).all(1)
    A = np.array([l['A'] for l in labs], float)
    yr = np.array([l['year'] if l['year'] is not None else np.nan for l in labs], float)
    use = np.array([B.USES.index(l['use']) for l in labs], float)
    ids = np.array([l['id'] for l in labs])
    return X[ok], {'logA': np.log10(np.where(A > 0, A, np.nan))[ok], 'year': yr[ok], 'use': use[ok]}, ids[ok]


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
    return {'logA': mu[:, 0], 'year': mu[:, 1], 'use': lg[:, :len(B.USES)].argmax(1).astype(float)}


def _lodge(u):
    u = np.asarray(u, float).copy()
    u[np.isin(u, RES)] = LODGING
    return u


def items(Y, P):
    return {'floor area': (10 ** Y['logA'], 10 ** np.asarray(P['logA'], float), ['mape']),
            'primary use': (Y['use'], _lodge(P['use']), ['acc']),
            'year built': (Y['year'], np.asarray(P['year'], float), ['mae'])}


def metrics(t):
    return {'logA': ['mape'], 'year': ['mae'], 'use': ['acc']}[t]


def C_dev_predict(Xd, Yd, Xe):
    from adapters import _cdev
    return _cdev.predict_all(TARGETS, Xd, Yd, Xe)


def C_dev_cv(X, Y):
    from adapters import _cdev
    import sys
    return _cdev.cv(sys.modules[__name__], X, Y)
