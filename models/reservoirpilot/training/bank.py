"""Build a training bank and a frozen simulated validation set for one generator version.
usage: python3 bank.py g1 [n_train] [n_val]
Out: data/bank_<gen>.npz (training), data/val_<gen>.pkl (frozen validation tasks + SOP / rule / DP scores).
Seeds: train 1000+, val 9000+ (disjoint). Resumable: skips files that exist."""
import os, sys, time, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from gen import make_task, task_stats                            # noqa: E402
from task import simulate, dp, cost, components                   # noqa: E402
from policy import fwd_mean                                       # noqa: E402

D_ = _paths.WORK + '/data'


def curves(tk):
    Dc, Ic = tk['Dcurve'], tk['Icurve']; d = np.arange(1, 367)
    return np.stack([Dc, fwd_mean(Dc, d, 30), fwd_mean(Dc, d, 90), fwd_mean(Ic, d, 30), fwd_mean(Ic, d, 90)])


def build_train(gen, n):
    out = f'{D_}/bank_{gen}.npz'
    if os.path.exists(out):
        return
    rng = np.random.default_rng(1000 + int(gen[1:]))
    L = 3653
    I = np.zeros((n, L), np.float32); doy = np.zeros((n, L), np.int16); C = np.zeros((n, 5, 366), np.float32)
    q = np.zeros(n); cap = np.zeros(n); S0 = np.zeros(n); Sc = np.zeros((n, 366), np.float32); info = []
    t0 = time.time()
    for i in range(n):
        tk = make_task(rng, gen, eval_len=L)
        m = len(tk['I'])
        I[i, :m] = tk['I']; I[i, m:] = tk['I'][-1]
        dd = tk['dates'].dayofyear.values; doy[i, :m] = dd; doy[i, m:] = dd[-1]
        C[i] = curves(tk); q[i] = tk['Qmax']; cap[i] = tk['cap']; S0[i] = tk['S0']; Sc[i] = tk['Scurve']
        info.append({**tk['info'], **task_stats(tk)})
        if i % 200 == 0:
            print('train', gen, i, round(time.time() - t0, 1), flush=True)
    np.savez(out, I=I, doy=doy, C=C, q=q, cap=cap, S0=S0, Sc=Sc)
    pickle.dump(info, open(f'{D_}/bank_{gen}_info.pkl', 'wb'))


def build_val(gen, n):
    out = f'{D_}/val_{gen}.pkl'
    if os.path.exists(out):
        return
    rng = np.random.default_rng(9000 + int(gen[1:]))
    tasks = []; t0 = time.time()
    for i in range(n):
        tk = make_task(rng, gen, eval_len=int(rng.integers(1000, 3654)))
        I, D, Q, cap, S0 = tk['I'], tk['D'], tk['Qmax'], tk['cap'], tk['S0']
        sc = {}
        for pol in ('sop', 'rule'):
            R, S = simulate(pol, I, D, Q, cap, S0, tk['Ss'])
            sc[pol] = components(R, S, D, Q, cap, S0)
        sc['dp'] = components(dp(I, D, Q, cap, S0), None, D, Q, cap, S0)
        tk['scores'] = sc; tk['stats'] = task_stats(tk); tasks.append(tk)
        if i % 25 == 0:
            print('val', gen, i, round(time.time() - t0, 1), {k: round(v['J'], 4) for k, v in sc.items()}, flush=True)
    pickle.dump(tasks, open(out, 'wb'))
    for pol in ('sop', 'rule', 'dp'):
        print(gen, pol, 'median J', np.median([t['scores'][pol]['J'] for t in tasks]))


if __name__ == '__main__':
    gen = sys.argv[1]; ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; nv = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    build_val(gen, nv)
    build_train(gen, ntr)
