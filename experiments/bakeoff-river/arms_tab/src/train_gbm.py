"""Train the GBM arm (LightGBM, one thread) on the sim-train table and export every tree to numpy arrays.

usage: .venv/bin/python train_gbm.py RUNG_DIR CONFIG_JSON
Models: the mean (L2 on qo / p_mean) and quantile models for 0.05, 0.25, 0.5, 0.75, 0.95. Early stopping on held-out
TRAIN runs (run index >= 11,400; never the frozen val set). Writes RUNG_DIR/gbm_trees.npz, RUNG_DIR/train.json.
"""
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats  # noqa: E402
import treeexport  # noqa: E402

out_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
os.makedirs(out_dir, exist_ok=True)
c0 = time.process_time()
D = np.load(cfg.get('data', '/home/claude/bakeoff/river/arms_tab/tab_gbm/work/tab_train.npz'))
X, y, run = D['X'], D['y'], D['run']
names = list(D['names'])
assert names == feats.NAMES[:len(names)]
keep = [i for i, n in enumerate(names) if n not in set(cfg.get('drop', []))]
X = X[:, keep] if len(keep) < X.shape[1] else X
fn = [names[i] for i in keep]
mono = [feats.MONOTONE[i] for i in keep] if cfg.get('monotone', True) else [0] * len(keep)
ntr = int((run < 11400).sum())
assert (run[:ntr] < 11400).all() and (run[ntr:] >= 11400).all()      # rows are ordered by run: slices, no copies
tr = slice(0, ntr)
va = slice(ntr, None)
log = open(os.path.join(out_dir, 'train.log'), 'a')


def fit(obj, alpha, params, rounds, es):
    p = {'objective': obj, 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 200,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
         'num_threads': 1, 'verbose': -1, 'seed': 3, 'monotone_constraints': mono,
         'monotone_constraints_method': 'intermediate', 'max_bin': 255}
    if obj == 'quantile':
        p['alpha'] = alpha
        del p['monotone_constraints'], p['monotone_constraints_method']   # LightGBM forbids them for quantile
    p.update(params)
    dtr = lgb.Dataset(X[tr], y[tr], feature_name=fn, free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    t0 = time.time()
    bst = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(es, verbose=False)])
    msg = f'{obj} {alpha} best_iter {bst.best_iteration} score {dict(bst.best_score["valid_0"])} {time.time() - t0:.0f}s'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    return bst


models = {}
models['mean'] = fit('regression', None, cfg.get('mean_params', {}), cfg.get('mean_rounds', 3000), 50)
for q, k in zip([0.05, 0.25, 0.5, 0.75, 0.95], ['q05', 'q25', 'q50', 'q75', 'q95']):
    models[k] = fit('quantile', q, {'learning_rate': 0.1, 'num_leaves': 31, **cfg.get('q_params', {})},
                    cfg.get('q_rounds', 400), 30)
arrs = {}
check = {}
rng = np.random.default_rng(0)
sel = rng.choice(np.arange(ntr, len(y)), 20000, replace=False)
for k, bst in models.items():
    a = treeexport.export(bst)
    for kk, v in a.items():
        arrs[f'{k}__{kk}'] = v
    p_lib = bst.predict(X[sel], num_iteration=bst.best_iteration)
    p_np = treeexport.predict(a, X[sel])
    check[k] = float(np.max(np.abs(p_lib - p_np)))
arrs['feature_index'] = np.array(keep, np.int32)
np.savez_compressed(os.path.join(out_dir, 'gbm_trees.npz'), **arrs)
info = {'cfg': cfg, 'features': fn, 'monotone': mono, 'numpy_vs_lightgbm_maxabs': check,
        'n_train_rows': ntr, 'n_es_rows': int(len(y) - ntr),
        'best_iter': {k: int(b.best_iteration) for k, b in models.items()},
        'cpu_s': time.process_time() - c0, 'lightgbm': lgb.__version__}
json.dump(info, open(os.path.join(out_dir, 'train.json'), 'w'), indent=1)
print(json.dumps(info, indent=1))
