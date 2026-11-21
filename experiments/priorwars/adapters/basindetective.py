"""BasinDetective (Watershed phase 2) adapter. Read-only use of the model's training workspace and released weights.

Rows = CAMELS-US basins. Features = the model's own ~150-feature vector (feats/real_*.npz, bank shards).
Targets = the six out-of-period change targets (real truth) + regulated (dor > 0.1), exactly the model's dev composite.
Dev: 52 dev basins, calibration 1999-2008 -> early window 1980-89 (the model's own dev protocol).
Exam: 619 basins, 1999-2008 -> 1989-99.

Paths (repository): bank/ and feats/ are in the model's training workspace (pw_paths.model_work), rebuilt by
models/basindetective/training (simbank.py, realfeats.py); the released network is models/basindetective/basindetective/
model.py with weights/basindetective_v1.npz. In the sandbox all of these sat under water/detective/ (release/ held the
package and weights).
"""
import glob
import json
import numpy as np
import pandas as pd
import pw_paths as _P

NAME = 'basindetective'
TITLE = 'BasinDetective'
D = _P.model_work('basindetective')
MD = _P.model_dir('basindetective')
DELTA_T = ['d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd']
TARGETS = [{'name': t, 'kind': 'reg'} for t in DELTA_T] + [{'name': 'reg', 'kind': 'clf'}]
UNIT = 'basin'
SIM_TRAIN = [('v3', range(0, 40)), ('v4', range(0, 40))]     # release R5 = banks v3 + v4, shards 0-39
SIM_VAL = [('v4', range(900, 904))]                            # frozen set of the release's physics (val-v4)
REG_THR = 0.1


def _bank(spec):
    X, L, names = [], [], None
    for phys, shards in spec:
        for s in shards:
            f = f'{D}/bank/{phys}/shard_{s:04d}.npz'
            z = np.load(f, allow_pickle=False)
            X.append(z['X']); L.append(pd.DataFrame.from_records(z['labels'])); names = [str(n) for n in z['names']]
    lab = pd.concat(L, ignore_index=True)
    Y = {t: lab[t].values.astype(float) for t in DELTA_T}
    Y['reg'] = lab['reg'].values.astype(float)
    return np.concatenate(X).astype(np.float32), Y, names


def sim_pool():
    return _bank(SIM_TRAIN)


def sim_val():
    return _bank(SIM_VAL)


def _real(split):
    z = np.load(f'{D}/feats/real_{split}.npz', allow_pickle=False)
    meta = pd.read_parquet(f'{D}/feats/real_{split}_meta.parquet')
    return z['X'].astype(np.float32), meta, [str(n) for n in z['names']]


def dev():
    X, meta, names = _real('dev')
    Y = {t: meta[f'true_{t}'].values.astype(float) for t in DELTA_T}
    Y['reg'] = (meta['dor'].values > REG_THR).astype(float)
    return X, Y, meta['basin'].values


def exam_X():
    X, meta, names = _real('exam')
    return X, meta['basin'].values


def exam_truth():
    """Only called by score.py after PREREG is frozen."""
    X, meta, names = _real('exam')
    Y = {t: meta[f'true_{t}'].values.astype(float) for t in DELTA_T}
    Y['reg'] = (meta['dor'].values > REG_THR).astype(float)
    return Y, meta['basin'].values


_M = None


def _model():
    global _M
    if _M is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('bd_model', f'{MD}/basindetective/model.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M = mod.Detective(f'{MD}/weights/basindetective_v1.npz')
        if hasattr(mod, 'load_sigma'):
            pass
    return _M


def norm(X):
    return _model().norm(X).astype(np.float32)


def P_predict(X):
    m = _model()
    mu, sd, p = m.forward(X)
    out = {t: mu[:, m.targets.index(t)] for t in DELTA_T}
    out['reg'] = p
    return out


def baselines_exam():
    """Classical + dumb baselines from the model's own exam table (same rows, same truths)."""
    X, meta, names = _real('exam')
    out = {'GR4J model-implied (classical)': {t: meta[f'gr4j_{t}'].values for t in DELTA_T},
           'no change (dumb)': {t: np.zeros(len(meta)) for t in DELTA_T}}
    out['GR4J model-implied (classical)']['reg'] = 1 - meta['gr4j_nse_cal'].values
    out['no change (dumb)']['reg'] = np.zeros(len(meta))
    return out


def metrics(t):
    """(name, kind) for each reported metric; 'primary' is first."""
    if t == 'reg':
        return ['auc']
    return ['rho', 'mdae']


HEADLINE = 'mean over the 6 change targets of Spearman rho and (2 AUC - 1) for regulated (= the model\'s dev composite)'
