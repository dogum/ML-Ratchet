"""GP arm: sparse Gaussian process (SGPR: Titsias 2009 collapsed variational bound, M inducing points, ARD squared-
exponential kernel) on a scoped set of the table track's inputs. Adapted from the River Bake-off's tab_gp.

usage: python3 train_gp.py OUT_DIR CONFIG_JSON

Inputs: the top-k features of the GBM r0 gain ranking (simulated train bank only; one-hot arch / class columns
skipped there) + the 14 one-hot architecture / class columns, standardised. Target z = log(y / E_phys) (EnergyNet's
correction), centred. Optional cfg "archs": train on those architecture indices only (one GP per powertrain group). Training rows: the train bank under K tier draws (as the GBM arm), minus held-out train vehicles
(vehicle index % 10 == 9: the check set; never the frozen val set). Hyperparameters + inducing inputs: Adam on the
bound over n_opt random rows (fresh minibatch of n_opt every `resample` iterations if set); then the exact SGPR
posterior over all training rows at fixed hyperparameters (chunked). Exports gp.npz for the numpy predict.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake  # noqa: E402
import featsel  # noqa: E402

torch.set_num_threads(1)
out_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = {'k': 20, 'M': 500, 'n_opt': 20000, 'iters': 300, 'lr': 0.02, 'K': 3, 'seed': 0, 'resample': 0,
       'rank_from': '/home/claude/bakeoff/vehicle/arms_table/tab_gbm/work/rungs/r0/train.json'}
cfg.update(json.load(open(cfg_path)))
os.makedirs(out_dir, exist_ok=True)
c0 = time.process_time()
log = open(os.path.join(out_dir, 'train.log'), 'a')


def say(msg):
    print(msg, flush=True); log.write(msg + '\n'); log.flush()


fi = np.array(featsel.ranked(cfg['rank_from'], cfg['k']) + list(range(102, 116)), np.int32)
B = bake.load_bank('train')
vix = bake.vehicle_index(B)
Xs, zs, hos = [], [], []
for k in range(cfg['K']):
    T = bake.draw_tiers(B, np.random.default_rng(1000 + k))
    D = bake.sim_table(B, T)
    ok = D['y'] > 0.02
    if cfg.get('archs') is not None:
        ok &= np.isin(D['arch'], cfg['archs'])
    Xs.append(D['X'][ok][:, fi].astype(np.float64)); zs.append(np.log(D['y'][ok] / D['E_phys'][ok]))
    hos.append((vix % 10 == 9)[ok])
X = np.concatenate(Xs); z = np.concatenate(zs); ho = np.concatenate(hos)
del Xs, zs
Xp, zp = X[~ho], z[~ho]
Xh, zh = X[ho], z[ho]
del X
xm, xs = Xp.mean(0), Xp.std(0) + 1e-6
Xp = (Xp - xm) / xs; Xh = (Xh - xm) / xs
y0 = float(zp.mean()); zp = zp - y0
rng = np.random.default_rng(cfg['seed'])
d = Xp.shape[1]; M = cfg['M']
say(f'train rows {len(zp)} check rows {len(zh)} d {d} M {M} features {fi.tolist()}')
Z = torch.tensor(Xp[rng.choice(len(Xp), M, replace=False)], requires_grad=True)
log_l = torch.tensor(np.log(np.full(d, np.sqrt(d) * 0.7)), requires_grad=True)
log_s2 = torch.tensor(np.log(zp.var()), requires_grad=True)
log_n2 = torch.tensor(np.log(0.3 * zp.var()), requires_grad=True)


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
cfg['n_opt'] = min(cfg['n_opt'], len(Xp))
sel = rng.choice(len(Xp), cfg['n_opt'], replace=False)
Xo = torch.tensor(Xp[sel]); zo = torch.tensor(zp[sel])
for it in range(cfg['iters']):
    if cfg['resample'] and it and it % cfg['resample'] == 0:
        sel = rng.choice(len(Xp), cfg['n_opt'], replace=False)
        Xo = torch.tensor(Xp[sel]); zo = torch.tensor(zp[sel])
    opt.zero_grad()
    f = bound(Xo, zo)
    (-f).backward()
    opt.step()
    if it % 20 == 0 or it == cfg['iters'] - 1:
        say(f'it {it} bound/N {float(f):.4f} noise {float(torch.exp(log_n2)):.5f} signal {float(torch.exp(log_s2)):.4f} '
            f'cpu {time.process_time() - c0:.0f}s')
with torch.no_grad():
    s2, n2 = torch.exp(log_s2), torch.exp(log_n2)
    Kmm = kern(Z, Z, log_l, log_s2) + 1e-6 * s2 * torch.eye(M, dtype=torch.float64)
    S = Kmm.clone(); bvec = torch.zeros(M, dtype=torch.float64)
    for s in range(0, len(Xp), 10000):
        Kmn = kern(Z, torch.tensor(Xp[s:s + 10000]), log_l, log_s2)
        S += Kmn @ Kmn.T / n2
        bvec += Kmn @ torch.tensor(zp[s:s + 10000]) / n2
    Sinv = torch.cholesky_inverse(torch.linalg.cholesky(S))
    Kinv = torch.cholesky_inverse(torch.linalg.cholesky(Kmm))
    alpha = Sinv @ bvec
    W = Kinv - Sinv
arr = {'feature_index': fi, 'xm': xm, 'xs': xs, 'Z': Z.detach().numpy(), 'ell': np.exp(log_l.detach().numpy()),
       's2': float(s2), 'n2': float(n2), 'alpha': alpha.numpy(), 'W': W.numpy(), 'y0': y0}
np.savez_compressed(os.path.join(out_dir, 'gp.npz'), **arr)
# check on the held-out train vehicles (numpy, as predict.py does it)
A = Xh / arr['ell']; Bz = arr['Z'] / arr['ell']
mu = np.empty(len(Xh)); sd = np.empty(len(Xh))
for s in range(0, len(Xh), 5000):
    a = A[s:s + 5000]
    Kx = arr['s2'] * np.exp(-0.5 * np.maximum((a * a).sum(1)[:, None] + (Bz * Bz).sum(1)[None] - 2 * a @ Bz.T, 0))
    mu[s:s + 5000] = Kx @ arr['alpha'] + y0
    sd[s:s + 5000] = np.sqrt(np.maximum(arr['s2'] - np.einsum('ij,ij->i', Kx @ arr['W'], Kx), 0) + arr['n2'])
e = zh - mu
info = {'cfg': cfg, 'features': fi.tolist(), 'check_med_abs_log_err': float(np.median(np.abs(e))),
        'check_cov50': float(np.mean(np.abs(e) < 0.6745 * sd)), 'check_cov90': float(np.mean(np.abs(e) < 1.6449 * sd)),
        'noise_var': float(n2), 'signal_var': float(s2), 'ell': arr['ell'].tolist(), 'cpu_s': time.process_time() - c0}
json.dump(info, open(os.path.join(out_dir, 'train.json'), 'w'), indent=1)
say(json.dumps({k: info[k] for k in ('check_med_abs_log_err', 'check_cov50', 'check_cov90', 'noise_var', 'cpu_s')}))
