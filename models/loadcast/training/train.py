"""Train a LoadCast network (cold or warm) on simulated units only.

    OMP_NUM_THREADS=1 nice -n 5 python3 train.py <run> <mode> <bank> <noise> [epochs] [width] [depth]

Loss = weighted MSE on mu, weight 1 / ybar^wpow (ybar = the unit's target-year mean, floored). wpow = 2: each unit counts
like its CV^2 (warm). For an unknown lognormal magnitude the optimum of this loss is median * exp((1 - 2 wpow) s^2 / 2),
so wpow = 2 biases cold forecasts low by ~x0.5; wpow = 0.5 aims at the median (R4) + 0.1 x Gaussian NLL for sigma (mu detached). Fresh random target hours every epoch. Resumable per epoch.
"""
import sys, os, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, lcnet as N

torch.set_num_threads(1)
CK = f'{C.ROOT}/ckpt'


def jitter_static(X, rng, s):
    """Blur building identity in the metadata (so the network cannot memorise each simulated building): log10 area
    +-0.1 s, year built +-6 s yr, site climate (mean T +-0.5 s K, HDD/CDD +-8 s %), |lat| +-1 s deg; per row."""
    n = len(X)
    X[:, 10] += rng.normal(0, 0.2 * s, n)
    X[:, 11] += rng.normal(0, 0.2 * s, n) * (X[:, 12] == 0)
    X[:, 34] += rng.normal(0, 0.05 * s, n)
    X[:, 35] *= np.exp(rng.normal(0, 0.08 * s, n)); X[:, 36] *= np.exp(rng.normal(0, 0.08 * s, n))
    X[:, 37] += rng.normal(0, 0.02 * s, n)
    return X


def epoch_rows(units, mode, K, rng, jitter=0.0, wpow=2.0):
    Xs, Bs, Ys, Ws = [], [], [], []
    for u in units:
        t = u['tidx']; valid = t[np.isfinite(u['y'][t])]
        if len(valid) < 200: continue
        idx = rng.choice(valid, K)
        v = ['full', 'full', 'w8', 'w2'][rng.integers(4)] if mode == 'warm' else 'full'
        X, b, y, S = D.rows(u, idx, mode, v)
        ybar = float(np.nanmean(u['y'][valid])) / (S if mode == 'warm' else 1.0)
        fl = 0.2 if mode == 'cold' else 0.3      # weight 1/ybar^2: each unit contributes ~ its CV^2 (floored)
        Xs.append(X); Bs.append(b); Ys.append(y.astype(np.float32)); Ws.append(np.full(K, 1.0 / max(ybar, fl) ** wpow, np.float32))
    X = np.concatenate(Xs)
    if jitter > 0: X = jitter_static(X, rng, jitter)
    return X, np.concatenate(Bs), np.concatenate(Ys), np.concatenate(Ws)


def main(run, mode, bank, noise, epochs=10, width=192, depth=3, K=96, seed=0, jitter=0.0, dropout=0.0, wd=1e-4, wpow=2.0):
    os.makedirs(CK, exist_ok=True)
    path = f'{CK}/{run}_{mode}.pt'
    t0 = time.time()
    banks = bank.split('+')
    units = []
    for b in banks:                     # 'name@N' = first N buildings of a bank
        nm, _, nb = b.partition('@')
        units += D.load_bank(nm, noise, seed=seed, max_buildings=int(nb) if nb else None)
    for u in units: D.prepare(u, ('full', 'w8', 'w2') if mode == 'warm' else ('full',))
    print(f'{len(units)} units loaded+prepared in {time.time() - t0:.0f} s', flush=True)
    nin = D.N_COLD if mode == 'cold' else D.N_WARM
    net = N.MLP(nin, width, depth, dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)
    ep0 = 0; hist = []
    if os.path.exists(path + '.part'):
        ck = torch.load(path + '.part', weights_only=False)
        net.load_state_dict(ck['state']); opt.load_state_dict(ck['opt']); ep0 = ck['epoch'] + 1; hist = ck['hist']
        print('resumed at epoch', ep0, flush=True)
    rng = np.random.default_rng(seed)
    if ep0 == 0:
        X, B, Y, W = epoch_rows(units, mode, 32, rng)
        net.mu_x.copy_(torch.from_numpy(X.mean(0))); net.sd_x.copy_(torch.from_numpy(X.std(0) + 1e-3))
    total = epochs
    for ep in range(ep0, total):
        rng = np.random.default_rng([seed, ep])
        X, B, Y, W = epoch_rows(units, mode, K, rng, jitter, wpow)
        perm = rng.permutation(len(X)); n = len(X); bs = 1024
        lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * ep / total))
        for g in opt.param_groups: g['lr'] = lr
        net.train(); tl = 0.0; nb = 0
        Xt, Bt, Yt, Wt = map(torch.from_numpy, (X, B, Y, W))
        for i in range(0, n, bs):
            j = torch.from_numpy(perm[i:i + bs])
            mu, ls = net(Xt[j], Bt[j]); y = Yt[j]; w = Wt[j]
            mse = (w * (y - mu) ** 2).sum() / w.sum()
            nll = (((y - mu.detach()) ** 2) * torch.exp(-2 * ls) + 2 * ls).mean()
            loss = mse + 0.1 * nll
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tl += float(mse); nb += 1
        hist.append(dict(epoch=ep, mse=tl / nb, lr=lr, s=time.time() - t0))
        print(f'{run} {mode} epoch {ep} mse {tl / nb:.4f} rows {n} lr {lr:.5f} {time.time() - t0:.0f} s', flush=True)
        torch.save(dict(state=net.state_dict(), opt=opt.state_dict(), epoch=ep, hist=hist), path + '.part')
    net.eval()
    torch.save(dict(state=net.state_dict(), nin=nin, width=width, depth=depth, mode=mode, bank=bank, noise=noise,
                    epochs=epochs, hist=hist, dropout=dropout, jitter=jitter, wd=wd, K=K, wpow=wpow), path)
    print('saved', path, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    pos = [x for x in a[4:] if '=' not in x]; kw = dict(x.split('=') for x in a[4:] if '=' in x)
    main(a[0], a[1], a[2], a[3], *(int(x) for x in pos), **{k: float(v) for k, v in kw.items()})
