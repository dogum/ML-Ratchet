"""PlantDetective (grid platform) adapter. Read-only use of models/plantdetective/training and its workspace.

Rows = (unit, calendar-year window) read in the model's 'out' mode (hourly output only; the exam's second mode), built
exactly as training/exam.py builds its 'year' windows. The 'ctx' mode cannot be rebuilt here: its EIA-860 generator
table (PUDL out_eia__yearly_generators.parquet) is no longer on this machine. Features = the model's 171 inputs.
Targets = the model's own transformed targets for min stable load (logit), ramp up p99 (log), starts per year (log1p),
full-load heat rate (log), and the technology class (4). Truth = the unit's whole-record CAMPD truth (exam: 2018-2025;
dev: dev years 2019 and 2021, as the model's dev gate).
Dev = the 40 dev units' year windows; exam = the exam units' year windows (the exam's 'from one calendar year' rows),
a fixed subsample of at most 2 windows per unit (seed 20261118) for the CPU budget; P is scored on the same rows.
Sim pool = the release bank's training windows (set_v2.npz, V = False) of a full year (W = 365 days); frozen simulated
set = set_v2.npz validation windows (V = True), W = 365.

Paths (repository): the training code is models/plantdetective/training. set_v2.npz and the R4_ens networks
(data/models/R4_ens_s*.npz) are in the model's training workspace (its common.DATA). R4_ens is PlantDetective v1.0;
the repository now releases v1.1 (R6), and keeps v1.0's networks packed in plantdetective_weights_v1_0.npz.
The cached real windows go to pw_paths.WORK/plantdetective.
"""
import sys
sys.dont_write_bytecode = True
import pw_paths as _P
_P.use_training('plantdetective')
import functools
import os
import numpy as np

NAME = 'plantdetective'
TITLE = 'PlantDetective'
PD = _P.model_work('plantdetective')     # its data/ = the model's common.DATA
CACHE = f'{_P.WORK}/plantdetective'
UNIT = 'unit'
J = {'pmin_frac': 0, 'ramp_up': 1, 'starts_yr': 3, 'hr_full': 10}
TARGETS = [{'name': k, 'kind': 'reg'} for k in J] + [{'name': 'tech', 'kind': 'multi'}]
HEADLINE_ITEMS = ['min stable load', 'ramp up p99', 'starts per year', 'full-load heat rate', 'technology']
HEADLINE = ('mean over min stable load (median abs error, points of pmax), ramp-up p99, starts per year and full-load '
            'heat rate (median relative error, %; starts as |(p+1)/(t+1) - 1|, as the card) - all negated - and '
            'technology accuracy (%)')


def _sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


@functools.lru_cache(maxsize=1)
def _W():
    import common  # noqa
    from model import load_any
    return load_any('R4_ens')


def _build(split):
    import common  # noqa
    import pandas as pd
    from pdcore import truth, TECHS
    from trainset import tgt_vector, window_features
    from evaluate import real_unit, windows_for, DEV_YEARS
    import realdata
    U = realdata.units(); U = U[U.split == split]
    UI = realdata.units().set_index('unit_id')
    years = DEV_YEARS if split == 'dev' else tuple(range(2018, 2026))
    X, Y, C, uid_, yr_ = [], [], [], [], []
    for uid in U.unit_id:
        try:
            rec, t = real_unit(split, uid)
        except FileNotFoundError:
            continue
        tt = pd.to_datetime(t, unit='s')
        m = np.isin(tt.year, years)
        rec = {k: v[m] for k, v in rec.items()}; t = t[m]
        if len(t) < 24 * 30:
            continue
        tr = truth(rec['op'], rec['P'], rec['HI'], rec['meas'])
        if not np.isfinite(tr['pmax']):
            continue
        tech = realdata.truth_tech(UI.loc[uid])
        if tech not in TECHS:
            continue
        y = tgt_vector(tr)
        wins = windows_for(t, 'year')
        for s, L in wins:
            v, _ = window_features(rec, t, None, None, -6, None, np.random.default_rng(0), L // 24 + 1, start=s,
                                   gap=False, force=dict(price=False, ctx=False, net=False))
            X.append(v); Y.append(y); C.append(TECHS.index(tech)); uid_.append(uid)
            yr_.append(int(pd.to_datetime(t[s + L // 2], unit='s').year))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(f'{CACHE}/real_{split}_X.npz', X=np.array(X, np.float32), uid=np.array(uid_), year=np.array(yr_))
    np.savez(f'{CACHE}/real_{split}_Y.npz', Y=np.array(Y, np.float32), C=np.array(C), uid=np.array(uid_))


def _load(split, part):
    f = f'{CACHE}/real_{split}_{part}.npz'
    if not os.path.exists(f):
        _build(split)
    return np.load(f)


def _Ydict(Y, C):
    d = {k: Y[:, j].astype(float) for k, j in J.items()}
    d['tech'] = C.astype(float)
    return d


def dev():
    zx, zy = _load('dev', 'X'), _load('dev', 'Y')
    return zx['X'], _Ydict(zy['Y'], zy['C']), zx['uid']


def dev_groups():
    return _load('dev', 'X')['uid']


EXAM_PER_UNIT = 2          # CPU budget (PREREG Amendment A1): a FIXED subsample of <= 2 year windows per exam unit
EXAM_SEED = 20261118       # P is scored on the same rows


@functools.lru_cache(maxsize=1)
def _exam_idx():
    uid = _load('exam', 'X')['uid']
    rng = np.random.default_rng(EXAM_SEED)
    pick = []
    for u in sorted(set(uid.tolist())):
        pick.extend(rng.permutation(np.flatnonzero(uid == u))[:EXAM_PER_UNIT].tolist())
    return np.sort(np.array(pick))


def exam_X():
    z = _load('exam', 'X')
    i = _exam_idx()
    return z['X'][i], z['uid'][i]


def exam_truth():
    """Only called by score.py after PREREG is frozen."""
    z = _load('exam', 'Y')
    i = _exam_idx()
    assert (z['uid'][i] == _load('exam', 'X')['uid'][i]).all()
    return _Ydict(z['Y'][i], z['C'][i]), z['uid'][i]


def _setv2(val):
    z = np.load(f'{PD}/data/set_v2.npz')
    m = (z['V'] == val) & (z['W'] == 365)
    return z['X'][m].astype(np.float32), _Ydict(z['Y'][m], z['C'][m]), None


@functools.lru_cache(maxsize=1)
def sim_pool():
    return _setv2(False)


@functools.lru_cache(maxsize=1)
def sim_val():
    return _setv2(True)


def norm(X):
    import common  # noqa
    import glob
    W0 = dict(np.load(sorted(glob.glob(f'{PD}/data/models/R4_ens_s[0-9].npz'))[0]))
    mu, sd = W0['mu'], W0['sd']
    return ((np.asarray(X, float) - mu) / sd).astype(np.float32)


def P_predict(X):
    p, mm, ss = _W()(np.asarray(X, np.float32))
    out = {k: mm[:, j].astype(float) for k, j in J.items()}
    out['tech'] = np.asarray(p).argmax(1).astype(float)
    return out


def items(Y, P):
    g = lambda k: np.asarray(P[k], float)  # noqa
    pm_t, pm_p = _sig(Y['pmin_frac']) * 100, _sig(g('pmin_frac')) * 100
    st_t, st_p = np.expm1(Y['starts_yr']), np.maximum(np.expm1(g('starts_yr')), 0)
    return {'min stable load': (pm_t, pm_p, ['mae_med']),
            'ramp up p99': (np.exp(Y['ramp_up']), np.exp(g('ramp_up')), ['mape']),
            'starts per year': (st_t + 1, st_p + 1, ['mape']),
            'full-load heat rate': (np.exp(Y['hr_full']), np.exp(g('hr_full')), ['mape']),
            'technology': (Y['tech'], g('tech'), ['acc'])}


def baselines_exam():
    """Card baselines that need no oracle: simple statistics of the same year of output (classical)."""
    return {}


def metrics(t):
    return ['mape']
