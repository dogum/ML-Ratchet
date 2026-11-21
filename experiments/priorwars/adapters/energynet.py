"""EnergyNet (vehicle platform) adapter. Read-only use of models/energynet/training and its workspace.

Rows = trips. Features = the model's own assembled input vector (dataset.assemble with the record's tiers).
Target = the model's own: log(true energy / physics estimate E_phys).
Exam = VED exam trips (the headline rows; EPA/heavy-duty exam sets are not used here), a FIXED subsample of at most
8 trips per exam car (seed 20261118) to fit the CPU budget; P is scored on the same subsample.
Dev = the 16 VED dev cars' 1,498 trips (VED dev protocol). Bootstrap and dev CV are grouped by car.

Paths (repository): the training code is models/energynet/training (its _paths.py puts shared/vehicle on sys.path).
Banks (data/bank_*), real trips (data/real_ved_*.npz, *_meta.parquet, exam_ved.parquet) and checkpoints (ckpt/) are in
the model's training workspace. P is the rung r6_ens: EnergyNet v1.0. The repository now releases v2.1 (weights/r15_s*),
so rerunning P needs the v1.0 checkpoints in WORK/ckpt. The rung list is WORK/rungs.json if present, else the shipped
training/ratchet.json.
"""
import sys
sys.dont_write_bytecode = True
import pw_paths as _P
_P.use_training('energynet')
import functools
import numpy as np
import pandas as pd

NAME = 'energynet'
TITLE = 'EnergyNet'
EN = _P.model_work('energynet')
TARGETS = [{'name': 'lr', 'kind': 'reg'}]
GROUPS = ['ICE', 'HEV', 'PHEV', 'EV']
HEADLINE_ITEMS = ['ICE', 'HEV', 'PHEV']
HEADLINE = 'mean of the per-trip MdAPE (%) over the ICE, HEV and PHEV exam groups (negated: higher is better); EV (2 Leafs) reported, not in the headline'
CV_REPS = 1
EXAM_PER_CAR = 8
EXAM_SEED = 20261118
SIM_TRAIN_SEED = 1     # tiers drawn for the train pool (as evaluate.dumb_rates)
VAL_SEED = 777         # frozen tiers of the frozen simulated set (evaluate.VAL_SEED)


def _DS():
    import dataset as DS
    return DS


@functools.lru_cache(maxsize=1)
def _model():
    import json
    from model_np import ENet, Ensemble
    import os
    fn = f'{EN}/rungs.json'
    if os.path.exists(fn):
        rungs = json.load(open(fn))
    else:
        rungs = json.load(open(f"{_P.model_dir('energynet')}/training/ratchet.json"))['rungs']
    R = {r['name']: r for r in rungs}
    return Ensemble([f'{EN}/ckpt/{x}.npz' for x in R['r6_ens']['members']])


def _bank(name, seed):
    DS = _DS()
    B = DS.load_bank(name)
    T = DS.draw_tiers(B, np.random.default_rng(seed))
    X, E = DS.assemble(B, T)
    y = DS.target(B, T, 'v3')
    ok = (y > 0.02) & (E > 0)
    arch = np.array(['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series',
                     'bev'])[B['arch'].astype(int)]
    grp = np.where(arch == 'conv', 'ICE', np.where(arch == 'bev', 'EV', np.where(np.char.startswith(arch, 'phev'),
                                                                                 'PHEV', 'HEV')))
    # '_y', '_E', '_group' let the frozen simulated set be scored with the exam's own items (tuning only)
    return X[ok].astype(np.float32), {'lr': np.log(y[ok] / E[ok]), '_y': y[ok], '_E': E[ok], '_group': grp[ok]}, None


@functools.lru_cache(maxsize=1)
def sim_pool():
    return _bank('train_v4', SIM_TRAIN_SEED)


@functools.lru_cache(maxsize=1)
def sim_val():
    return _bank('val_v4', VAL_SEED)


@functools.lru_cache(maxsize=2)
def _real(name):
    DS = _DS()
    B = dict(np.load(f'{EN}/data/real_{name}.npz'))
    T = {k: B['T_' + k] for k in DS.TIER_KEYS}
    M = pd.read_parquet(f'{EN}/data/real_{name}_meta.parquet')
    X, E = DS.assemble(B, T)
    Es = E * B['vr']
    return X.astype(np.float32), B['y'].astype(float), Es, M


FLOOR = {'ICE': 1 * 0.042, 'HEV': 1 * 0.042, 'EV': 0.01 * 3.6, 'PHEV': 0.05}


def _Y(y, Es, M):
    return {'lr': np.log(y / Es), '_y': y, '_E': Es, '_group': M['group'].values.astype(str)}


@functools.lru_cache(maxsize=1)
def _dev():
    X, y, Es, M = _real('ved_dev')
    ok = y > M['group'].map(FLOOR).values
    return X[ok], y[ok], Es[ok], M[ok].reset_index(drop=True)


def dev():
    X, y, Es, M = _dev()
    return X, _Y(y, Es, M), M['trip_id'].values


def dev_groups():
    return _dev()[3]['vehicle_id'].values


@functools.lru_cache(maxsize=1)
def _exam_idx():
    X, y, Es, M = _real('ved_exam')
    ok = np.where(y > M['group'].map(FLOOR).values)[0]
    rng = np.random.default_rng(EXAM_SEED)
    pick = []
    Mo = M.iloc[ok]
    for vid, d in Mo.groupby('vehicle_id', sort=True):
        ii = ok[np.flatnonzero(M['vehicle_id'].values[ok] == vid)]
        pick.extend(rng.permutation(ii)[:EXAM_PER_CAR].tolist())
    return np.sort(np.array(pick))


def exam_X():
    X, y, Es, M = _real('ved_exam')
    i = _exam_idx()
    return X[i], M['trip_id'].values[i]


def exam_units():
    X, y, Es, M = _real('ved_exam')
    return M['vehicle_id'].values[_exam_idx()]


def exam_truth():
    """Only called by score.py after PREREG is frozen."""
    X, y, Es, M = _real('ved_exam')
    i = _exam_idx()
    Mi = M.iloc[i].reset_index(drop=True)
    return _Y(y[i], Es[i], Mi), Mi['trip_id'].values


def norm(X):
    m = _model().m[0]
    return np.clip((np.asarray(X, float) - m.mu) / m.sd, -8, 8).astype(np.float32)


def P_predict(X):
    # E_phys cancels in the log ratio: the network's own output is the correction m = log(pred / E_phys)
    outs = [mm.forward(X)[0] for mm in _model().m]
    return {'lr': np.mean(outs, axis=0)}


def items(Y, P):
    out = {}
    pred = Y['_E'] * np.exp(np.asarray(P['lr'], float))
    for g in GROUPS:
        sel = Y['_group'] == g
        y = np.where(sel, Y['_y'], np.nan)
        out[g] = (y, np.where(sel, pred, np.nan), ['mape'])
    out['all trips'] = (Y['_y'], pred, ['mape'])
    out['log ratio'] = (Y['lr'], np.asarray(P['lr'], float), ['rho', 'mdae'])
    return out


def baselines_exam():
    """The card's exam baselines on the same trips (exam_ved.parquet rows align with real_ved_exam)."""
    e = pd.read_parquet(f'{EN}/data/exam_ved.parquet')
    X, y, Es, M = _real('ved_exam')
    assert (e['trip_id'].values == M['trip_id'].values).all()
    i = _exam_idx()
    out = {}
    for col, nm in [('b_label', 'label fuel economy (classical)'), ('b_vsp', 'VSP MOVES-like (classical)'),
                    ('b_class', 'class median (dumb)'), ('E_phys', 'physics estimate alone')]:
        v = e[col].values[i].astype(float)
        out[nm] = {'lr': np.log(np.maximum(v, 1e-9) / Es[i])}
    return out


def metrics(t):
    return ['mape']
