"""Sparse Gaussian process (Titsias 2009 SGPR / collapsed variational bound, inducing points) on a SCOPED feature set
of the sim-train table. ARD squared-exponential kernel. Hyperparameters and inducing inputs are optimised (Adam) on
`n_opt` random rows; the posterior is then accumulated over `n_post` rows at fixed hyperparameters (exact SGPR
algebra, chunked). Target: a transform of y = qo / p_mean ('sqrt' or 'log').
Exports numpy arrays for gp predict: feature index, standardisation, Z, lengthscales, signal/noise variance,
alpha = Sigma^-1 Kmn y / s_n^2, W = Kmm^-1 - Sigma^-1 (predictive variance), y offset, transform.

usage: python3 train_gp.py RUNG_DIR CONFIG_JSON
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats  # noqa: E402

torch.set_num_threads(1)
out_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = {'features': ['P_tp1', 'P_t', 'W_t', 'W_tm1', 'ema3', 'ema10', 'ema30', 'ema365', 'b50_frac', 'b300_frac',
                    'slow_out', 'log_swe', 's_log_aridity', 's_frac_snow'],
       'M': 500, 'n_opt': 20000, 'n_post': 200000, 'iters': 300, 'lr': 0.02, 'transform': 'sqrt', 'seed': 0}
cfg.update(json.load(open(cfg_path)))
os.makedirs(out_dir, exist_ok=True)
c0 = time.process_time()
D = np.load('/home/claude/bakeoff/river/arms_tab/tab_gbm/work/tab_train.npz')
names = list(D['names'])
fi = np.array([names.index(f) for f in cfg['features']])
tr = np.where(D['run'] < 11400)[0]
ho = np.where(D['run'] >= 11400)[0]
rng = np.random.default_rng(cfg['seed'])
X_all = D['X']
y_all = D['y'].astype(np.float64)
OFF = {'sqrt': 0.0, 'log': 0.02}[cfg['transform']]


def tf(y):
    y = np.maximum(y, 0.0)
    return np.sqrt(y) if cfg['transform'] == 'sqrt' else np.log(y + OFF)


sel_post = np.sort(rng.choice(tr, min(cfg['n_post'], len(tr)), replace=False))
Xp = X_all[sel_post][:, fi].astype(np.float64)
sh = rng.choice(ho, 20000, replace=False)                 # holdout rows for the final check (drawn after, as r0)
Xh_raw = X_all[sh][:, fi].astype(np.float64); yh_raw = y_all[sh]
del X_all, D                                              # keep memory low
xm, xs = Xp.mean(0), Xp.std(0) + 1e-9
Xp = (Xp - xm) / xs
zp = tf(y_all[sel_post])
y0 = float(zp.mean())
zp = zp - y0
sel_opt = rng.choice(len(Xp), cfg['n_opt'], replace=False)
Xo = torch.tensor(Xp[sel_opt]); zo = torch.tensor(zp[sel_opt])
d = Xp.shape[1]
M = cfg['M']
Z = torch.tensor(Xp[rng.choice(len(Xp), M, replace=False)], requires_grad=True)
log_l = torch.zeros(d, dtype=torch.float64, requires_grad=True)
log_s2 = torch.tensor(np.log(zp.var()), requires_grad=True)
log_n2 = torch.tensor(np.log(0.2 * zp.var()), requires_grad=True)


def kern(A, Bm, log_l, log_s2):
    A = A / torch.exp(log_l); Bm = Bm / torch.exp(log_l)
    d2 = (A * A).sum(1)[:, None] + (Bm * Bm).sum(1)[None] - 2 * A @ Bm.T
    return torch.exp(log_s2 - 0.5 * torch.clamp(d2, min=0.0))


def bound(X, y):
    N = X.shape[0]
    s2, n2 = torch.exp(log_s2), torch.exp(log_n2)
    Kmm = kern(Z, Z, log_l, log_s2) + 1e-6 * s2 * torch.eye(M, dtype=torch.float64)
    Kmn = kern(Z, X, log_l, log_s2)
    L = torch.linalg.cholesky(Kmm)
    A = torch.linalg.solve_triangular(L, Kmn, upper=False) / torch.sqrt(n2)
    Bm = torch.eye(M, dtype=torch.float64) + A @ A.T
    LB = torch.linalg.cholesky(Bm)
    c = torch.linalg.solve_triangular(LB, (A @ y)[:, None], upper=False)[:, 0] / torch.sqrt(n2)
    f = (-0.5 * N * np.log(2 * np.pi) - torch.log(torch.diagonal(LB)).sum() - 0.5 * N * torch.log(n2)
         - 0.5 * (y @ y) / n2 + 0.5 * (c @ c) - 0.5 * N * s2 / n2 + 0.5 * (A * A).sum())
    return f / N


opt = torch.optim.Adam([Z, log_l, log_s2, log_n2], lr=cfg['lr'])
log = open(os.path.join(out_dir, 'train.log'), 'a')
for it in range(cfg['iters']):
    opt.zero_grad()
    f = bound(Xo, zo)
    (-f).backward()
    opt.step()
    if it % 20 == 0 or it == cfg['iters'] - 1:
        msg = (f'it {it} bound/N {float(f):.4f} noise {float(torch.exp(log_n2)):.4f} signal {float(torch.exp(log_s2)):.4f} '
               f'ell {np.round(np.exp(log_l.detach().numpy()), 2).tolist()} cpu {time.process_time() - c0:.0f}s')
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
# posterior over n_post rows at fixed hyperparameters
with torch.no_grad():
    s2, n2 = torch.exp(log_s2), torch.exp(log_n2)
    Kmm = kern(Z, Z, log_l, log_s2) + 1e-6 * s2 * torch.eye(M, dtype=torch.float64)
    S = Kmm.clone(); bvec = torch.zeros(M, dtype=torch.float64)
    for s in range(0, len(Xp), 20000):
        Kmn = kern(Z, torch.tensor(Xp[s:s + 20000]), log_l, log_s2)
        S += Kmn @ Kmn.T / n2
        bvec += Kmn @ torch.tensor(zp[s:s + 20000]) / n2
    Sinv = torch.cholesky_inverse(torch.linalg.cholesky(S))
    Kinv = torch.cholesky_inverse(torch.linalg.cholesky(Kmm))
    alpha = Sinv @ bvec
    W = Kinv - Sinv
arr = {'feature_index': fi.astype(np.int32), 'xm': xm, 'xs': xs, 'Z': Z.detach().numpy(),
       'ell': np.exp(log_l.detach().numpy()), 's2': float(s2), 'n2': float(n2), 'alpha': alpha.numpy(),
       'W': W.numpy(), 'y0': y0, 'transform': cfg['transform'], 'off': OFF}
np.savez_compressed(os.path.join(out_dir, 'gp.npz'), **arr)
# quick holdout check (held-out train runs)
Xh = (Xh_raw - xm) / xs
with torch.no_grad():
    Ks = kern(torch.tensor(Xh), Z, log_l, log_s2).numpy()
mu = Ks @ arr['alpha'] + y0
zh = tf(yh_raw)
info = {'cfg': cfg, 'holdout_rmse_transformed': float(np.sqrt(np.mean((mu - zh) ** 2))),
        'holdout_r2_transformed': float(1 - np.mean((mu - zh) ** 2) / np.var(zh)),
        'noise_var': float(n2), 'signal_var': float(s2), 'ell': arr['ell'].tolist(), 'cpu_s': time.process_time() - c0}
json.dump(info, open(os.path.join(out_dir, 'train.json'), 'w'), indent=1)
print(json.dumps(info), flush=True)
