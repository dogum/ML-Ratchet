"""Direct policy optimisation through the differentiable reservoir simulator (analytic policy gradient).

The rollout uses B's exact mass balance and B's exact daily cost; gradients of the cost flow back through the
storage dynamics to the policy weights (backpropagation through time over a T-day window). The objective per
window is B's J plus the DP baseline's terminal rule (1000 x storage deficit / capacity, summed then divided by T),
so the policy cannot win by emptying the reservoir. Batch loss = mean over windows of log(J_window + 0.02), which
weights every reservoir by its relative improvement (the exam statistic is the median, not the mean).

usage: python3 train.py NAME --gen g1 --fset base --iters 2500 --T 365 --batch 256 [--init weights.npz]
Resumable: checkpoints every 100 iterations in runs/NAME/ckpt.pt.
"""
import argparse, os, sys, time, json, pickle
import numpy as np, torch
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import torch_policy, rollout_torch, features, FEATS, export_weights, NumpyPolicy, rollout_np, EMA_TAUS  # noqa
from task import components                                                                                       # noqa

torch.set_num_threads(1)
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def load_bank(gen):
    z = np.load(f'{ROOT}/data/bank_{gen}.npz')
    return {k: z[k] for k in z.files}


def make_batch(bk, idx, t0, T, fset, rng, s_mode):
    ar = np.arange(T)
    I = bk['I'][idx[:, None], t0[:, None] + ar].astype(np.float64)
    doy = bk['doy'][idx[:, None], t0[:, None] + ar].astype(np.int64) - 1
    C = bk['C'][idx]                                     # (B, 5, 366)
    cal = {k: np.take_along_axis(C[:, j], doy, 1).astype(np.float64) for j, k in enumerate(['D', 'Dn30', 'Dn90', 'Ic30', 'Ic90'])}
    q = bk['q'][idx]; cap = bk['cap'][idx]
    u = rng.random(len(idx))
    Srand = rng.uniform(0.02, 1.0, len(idx)) * cap
    Ssc = bk['Sc'][idx, doy[:, 0]] * np.exp(rng.normal(0, 0.2, len(idx)))
    S0 = np.where(u < s_mode, Srand, np.clip(Ssc, 0, cap))
    ema0 = cal['Ic30'][:, 0] if fset == 'clim' else I[:, 0]
    b = {k: torch.tensor(v) for k, v in cal.items()}
    b.update({'I': torch.tensor(I), 'q': torch.tensor(q), 'cap': torch.tensor(cap), 'S0': torch.tensor(S0),
              'ema0': torch.tensor(ema0)})
    return b


def feat_stats(bk, fset, rng, n=512, T=200):
    """Feature mean/sd from SOP-like states (initial policy) on random windows."""
    idx = rng.integers(0, len(bk['q']), n); t0 = rng.integers(0, bk['I'].shape[1] - T, n)
    b = make_batch(bk, idx, t0, T, fset, rng, 0.5)
    S = b['S0']; emas = [b['ema0'] for _ in EMA_TAUS]; F = []
    K = 0.0864
    for t in range(T):
        It = b['I'][:, t]; D = b['D'][:, t]
        emas = [e + (It - e) / tau for e, tau in zip(emas, EMA_TAUS)]
        f = features(torch, S, It, emas, D, b['Dn30'][:, t], b['Dn90'][:, t], b['Ic30'][:, t], b['Ic90'][:, t],
                     b['q'], b['cap'], fset)
        F.append(torch.stack(f, -1))
        avail = S + It * K; r = torch.minimum(D, torch.relu(avail) / K); s1 = avail - r * K
        S = torch.relu(torch.minimum(s1, b['cap']))
    F = torch.cat(F).float()
    F = F[torch.isfinite(F).all(1)]
    return F.mean(0).double(), F.std(0).clamp_min(1e-3).double()


def val_score(w, fset, gen, sub=None, kind='val'):
    tasks = pickle.load(open(f'{ROOT}/data/{kind}_{gen}.pkl', 'rb'))
    if sub:
        tasks = tasks[:sub]
    pol = NumpyPolicy(w, fset)
    R, S = rollout_np(pol, tasks)
    J = np.array([components(r, s, t['D'], t['Qmax'], t['cap'], t['S0'])['J'] for r, s, t in zip(R, S, tasks)])
    Js = np.array([t['scores']['sop']['J'] for t in tasks])
    return 1 - np.median(J) / np.median(Js), float(np.median(J))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--gen', default='g1'); ap.add_argument('--fset', default='base')
    ap.add_argument('--iters', type=int, default=2500); ap.add_argument('--T', type=int, default=365)
    ap.add_argument('--batch', type=int, default=256); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--init', default=None); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--s_mode', type=float, default=0.5); ap.add_argument('--terminal', type=float, default=0.0); ap.add_argument('--tail', type=int, default=365); ap.add_argument('--eps', type=float, default=0.03); ap.add_argument('--head', default='rule')
    a = ap.parse_args()
    out = f'{ROOT}/runs/{a.name}'; os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f'{out}/args.json', 'w'), indent=1)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    bk = load_bank(a.gen); N, L = bk['I'].shape
    valid = np.where((bk['q'] > 0) & (bk['cap'] > 0))[0]          # a task with Qmax = 0 has no defined cost
    net = torch_policy(FEATS[a.fset], a.hidden, a.layers, a.head).double()
    ck = f'{out}/ckpt.pt'
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        net.load_state_dict(st['net']); mu, sd = st['mu'], st['sd']; it0 = st['it'] + 1
        opt = torch.optim.Adam(net.parameters(), lr=a.lr); opt.load_state_dict(st['opt'])
        rng = st['rng']; hist = st['hist']
        print('resumed at', it0, flush=True)
    else:
        mu, sd = feat_stats(bk, a.fset, rng)
        if a.init:
            w = np.load(a.init)
            lin = [m for m in net if hasattr(m, 'weight')]
            with torch.no_grad():
                for i, m in enumerate(lin):
                    m.weight.copy_(torch.tensor(w[f'W{i}'].T)); m.bias.copy_(torch.tensor(w[f'b{i}']))
                mu, sd = torch.tensor(w['mu']), torch.tensor(w['sd'])
        opt = torch.optim.Adam(net.parameters(), lr=a.lr); it0 = 0; hist = []
    sched = lambda it: 0.5 * (1 + np.cos(np.pi * min(it / a.iters, 1.0)))   # cosine decay
    t_start = time.time()
    for it in range(it0, a.iters):
        for g in opt.param_groups:
            g['lr'] = a.lr * (0.05 + 0.95 * sched(it))
        idx = valid[rng.integers(0, len(valid), a.batch)]; t0 = rng.integers(0, L - a.T - a.tail, a.batch)
        b = make_batch(bk, idx, t0, a.T + a.tail, a.fset, rng, a.s_mode)
        J, obj = rollout_torch(net, mu, sd, b, a.fset, a.terminal, a.tail, a.head)
        loss = torch.log(obj + a.eps).mean()
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        else:
            print('skipped non-finite gradient at', it, flush=True)
        hist.append([it, float(loss), float(J.median()), float(gn)]); obj = obj.detach()
        if it % 20 == 0:
            print(it, f'loss {float(loss):.4f} Jmed {float(J.median()):.4f} gn {float(gn):.3f} '
                      f'{(time.time() - t_start) / (it - it0 + 1):.2f}s/it', flush=True)
        if it % 100 == 99 or it == a.iters - 1:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'it': it, 'mu': mu, 'sd': sd, 'rng': rng,
                        'hist': hist}, ck + '.tmp')
            os.replace(ck + '.tmp', ck)
        if (it % 200 == 199 and it >= a.iters // 3) or it == a.iters - 1:
            w = export_weights(net, mu, sd); np.savez(f'{out}/weights_it{it + 1}.npz', **w, fset=a.fset, head=a.head)
            sj, jm = val_score({**w, 'head': a.head}, a.fset, a.gen, kind='sel')
            print(f'SEL it {it + 1}: S_J {sj:.4f} medJ {jm:.4f}', flush=True)
            sel = json.load(open(f'{out}/sel.json')) if os.path.exists(f'{out}/sel.json') else {}
            sel[str(it + 1)] = sj; json.dump(sel, open(f'{out}/sel.json', 'w'))
    # checkpoint choice on the selection set (not the ratchet's frozen set)
    sel = json.load(open(f'{out}/sel.json')); best = max(sel, key=sel.get)
    import shutil; shutil.copy(f'{out}/weights_it{best}.npz', f'{out}/weights.npz')
    print('selected it', best, sel[best])
    print('done', flush=True)


if __name__ == '__main__':
    main()
