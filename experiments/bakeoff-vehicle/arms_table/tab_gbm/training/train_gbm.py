"""GBM arm: LightGBM (one thread) on the table track's 151 inputs, exported to numpy arrays (treeexport.py).

usage: <priorwars venv>/bin/python train_gbm.py OUT_DIR CONFIG_JSON

Training table: the train bank (28,800 trips) under K independent tier draws (default_rng(1000 + k)), stacked:
every trip appears K times with different spec masking (EnergyNet redraws tiers every epoch; this is the tree analogue).
Rows with target y <= 0.02 MJ are dropped (EnergyNet's weight-0 rows). Target z = log(y / E_phys) (EnergyNet's
correction target). Early stopping on held-out TRAIN vehicles (vehicle index % 10 == 9, 10 %; never the frozen val set).
Models: 'mean' = L2 on z (optionally monotone-constrained), and quantile models for 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
Writes OUT_DIR/gbm_trees.npz, OUT_DIR/train.json (config, best iterations, numpy == LightGBM check, gains, cpu).
"""
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake  # noqa: E402
import treeexport  # noqa: E402

out_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
os.makedirs(out_dir, exist_ok=True)
c0 = time.process_time()
log = open(os.path.join(out_dir, 'train.log'), 'a')


def say(msg):
    print(msg, flush=True); log.write(msg + '\n'); log.flush()


B = bake.load_bank('train')
vix = bake.vehicle_index(B)
K = cfg.get('K', 3)
Xs, zs, hos = [], [], []
for k in range(K):
    T = bake.draw_tiers(B, np.random.default_rng(1000 + k))
    D = bake.sim_table(B, T)
    ok = D['y'] > 0.02
    Xs.append(D['X'][ok]); zs.append(np.log(D['y'][ok] / D['E_phys'][ok])); hos.append((vix % 10 == 9)[ok])
X = np.concatenate(Xs); z = np.concatenate(zs); ho = np.concatenate(hos)
del Xs, zs
tr = ~ho
say(f'table {X.shape} train {tr.sum()} es-holdout {ho.sum()} K {K}')
nfeat = X.shape[1]
mono = [0] * nfeat
for i, s in cfg.get('monotone', {}).items():
    mono[int(i)] = int(s)
Xtr, ztr, Xho, zho = X[tr], z[tr], X[ho], z[ho]
del X


def fit(obj, alpha, params, rounds, es):
    p = {'objective': obj, 'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 100,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0,
         'num_threads': 1, 'verbose': -1, 'seed': 3, 'max_bin': 255}
    if any(mono) and obj != 'quantile':
        p['monotone_constraints'] = mono
        p['monotone_constraints_method'] = cfg.get('mono_method', 'intermediate')
    if obj == 'quantile':
        p['alpha'] = alpha
    p.update(params)
    dtr = lgb.Dataset(Xtr, ztr, free_raw_data=False)
    dva = lgb.Dataset(Xho, zho, reference=dtr)
    t0 = time.process_time()
    bst = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(es, verbose=False)])
    say(f'{obj} {alpha} best_iter {bst.best_iteration} score {dict(bst.best_score["valid_0"])} cpu {time.process_time() - t0:.0f}s')
    return bst


models = {'mean': fit('regression', None, cfg.get('mean_params', {}), cfg.get('mean_rounds', 4000), 100)}
for q, k in zip([0.05, 0.25, 0.5, 0.75, 0.95], ['q05', 'q25', 'q50', 'q75', 'q95']):
    models[k] = fit('quantile', q, {'learning_rate': 0.1, 'num_leaves': 31, **cfg.get('q_params', {})},
                    cfg.get('q_rounds', 800), 40)
arrs, check = {}, {}
sel = np.random.default_rng(0).choice(len(Xho), min(10000, len(Xho)), replace=False)
for k, bst in models.items():
    a = treeexport.export(bst)
    for kk, v in a.items():
        arrs[f'{k}__{kk}'] = v
    check[k] = float(np.max(np.abs(bst.predict(Xho[sel], num_iteration=bst.best_iteration) - treeexport.predict(a, Xho[sel]))))
np.savez_compressed(os.path.join(out_dir, 'gbm_trees.npz'), **arrs)
pm = treeexport.predict({kk: arrs[f'mean__{kk}'] for kk in ('feat', 'thr', 'left', 'right', 'value', 'roots', 'depth')}, Xho)
gain = models['mean'].feature_importance('gain')
info = {'cfg': cfg, 'monotone': {i: s for i, s in enumerate(mono) if s}, 'numpy_vs_lightgbm_maxabs': check,
        'n_train_rows': int(len(ztr)), 'n_es_rows': int(len(zho)),
        'es_holdout_med_abs_log_err': float(np.median(np.abs(pm - zho))),
        'best_iter': {k: int(b.best_iteration) for k, b in models.items()},
        'gain_top40': [[int(i), float(gain[i])] for i in np.argsort(-gain)[:40]],
        'cpu_s': time.process_time() - c0, 'lightgbm': lgb.__version__}
json.dump(info, open(os.path.join(out_dir, 'train.json'), 'w'), indent=1)
say(json.dumps({k: info[k] for k in ('numpy_vs_lightgbm_maxabs', 'es_holdout_med_abs_log_err', 'best_iter', 'cpu_s')}))
