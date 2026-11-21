"""Train the value network: features(scenario seen through the forecast, plan) -> saved fraction in the truth world.

  python3 train_value.py <rung> [--banks train_p1,onpol1_p1] [--hidden 64] [--epochs 400] [--recompute]

Rows come from the bank jsons (features stored at bank time; --recompute rebuilds them from the npz with the
current feature code). 10 % of scenarios (seeded) are a simulated selection set for early stopping. The mean head
is trained with MSE on the saved fraction (sigmoid output); the sigma head with a Gaussian NLL on the detached
residual. Exports models/<rung>.npz for fp.planner.ValueNet (numpy) and checks numpy == torch.
"""
import os
import sys
import json
import argparse

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = _paths.WORK


def load_rows(banks, recompute=False, seeds=None):
    from fp.planner import Context
    X, y, g = [], [], []
    for b in banks:
        d = os.path.join(ROOT, 'data', b)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json') or fn.endswith('.pf.json'):
                continue
            if seeds is not None and not any(a <= int(fn[1:7]) < b for a, b in seeds):
                continue
            rec = json.load(open(os.path.join(d, fn)))
            if 'skip' in rec:
                continue
            ctx = None
            if recompute:
                z = np.load(os.path.join(d, fn[:-5] + '.npz'))
                ctx = Context.from_store(z, rec)
            for r in rec['rows']:
                f = ctx.features(r['plan'])[0] if ctx is not None else np.array(r['f'])
                X.append(f)
                y.append(r['saved'])
                g.append(rec['name'])
    return np.array(X, np.float64), np.array(y, np.float64), np.array(g)


class Net(torch.nn.Module):
    def __init__(self, nin, h):
        super().__init__()
        self.l0 = torch.nn.Linear(nin, h)
        self.l1 = torch.nn.Linear(h, h)
        self.l2 = torch.nn.Linear(h, 2)

    def forward(self, x):
        h = torch.tanh(self.l0(x))
        h = torch.tanh(self.l1(h))
        return self.l2(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rung')
    ap.add_argument('--banks', default='train_p1')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--seeds', default=None, help='e.g. 0-150,300-450 (training scenario seed ranges)')
    a = ap.parse_args()
    seeds = [tuple(int(v) for v in r.split('-')) for r in a.seeds.split(',')] if a.seeds else None
    X, y, g = load_rows(a.banks.split(','), a.recompute, seeds)
    ug = np.unique(g)
    rs = np.random.default_rng(12345)
    sel = set(rs.choice(ug, max(1, len(ug) // 10), replace=False))
    is_sel = np.array([k in sel for k in g])
    mu, sd = X[~is_sel].mean(0), X[~is_sel].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    tr, va = np.flatnonzero(~is_sel), np.flatnonzero(is_sel)
    torch.manual_seed(a.seed)
    net = Net(X.shape[1], a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state, bs = 1e9, None, 256
    log = []
    for ep in range(a.epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(tr)
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]
            o = net(Xt[b])
            m = torch.sigmoid(o[:, 0])
            res = (yt[b] - m).detach()
            ls = torch.clamp(o[:, 1], -6, 2)
            loss = ((yt[b] - m) ** 2).mean() + 0.01 * (ls + 0.5 * res ** 2 * torch.exp(-2 * ls)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                o = net(Xt[va])
                mse = float(((torch.sigmoid(o[:, 0]) - yt[va]) ** 2).mean())
                otr = net(Xt[tr[:5000]])
                mtr = float(((torch.sigmoid(otr[:, 0]) - yt[tr[:5000]]) ** 2).mean())
            log.append((ep, mtr, mse))
            if mse < best:
                best = mse
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if ep % 50 == 0:
                print(f'ep {ep} train mse {mtr:.4f} sel mse {mse:.4f} best {best:.4f}', flush=True)
    net.load_state_dict(best_state)
    # sigma scale on the selection set (so that z-scores have unit rms there)
    with torch.no_grad():
        o = net(Xt[va]).numpy()
    m = 1 / (1 + np.exp(-o[:, 0]))
    s = np.exp(np.clip(o[:, 1], -6, 2))
    zrms = float(np.sqrt(np.mean(((y[va] - m) / s) ** 2)))
    W = [net.l0.weight.detach().numpy().T, net.l1.weight.detach().numpy().T, net.l2.weight.detach().numpy().T]
    B = [net.l0.bias.detach().numpy(), net.l1.bias.detach().numpy(), net.l2.bias.detach().numpy()]
    os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)
    fn = os.path.join(ROOT, 'models', a.rung + '.npz')
    np.savez(fn, W0=W[0], W1=W[1], W2=W[2], b0=B[0], b1=B[1], b2=B[2], mu=mu, sd=sd, sig_scale=1.0)
    from fp.planner import ValueNet
    vn = ValueNet(fn)
    mn, sn = vn(X[va])
    err = float(np.abs(mn - m).max())
    # proxy-only skill on the selection set, for reference
    from scipy.stats import spearmanr
    info = dict(rung=a.rung, banks=a.banks, seeds=a.seeds, n_rows=len(y), n_scen=len(ug), n_sel_rows=len(va), sel_mse=best,
                sel_rmse=best ** 0.5, sel_spearman=float(spearmanr(mn, y[va])[0]),
                proxy_spearman=float(spearmanr(X[va, -2], y[va])[0]), zrms_sel=zrms, numpy_vs_torch=err,
                hidden=a.hidden, epochs=a.epochs, n_params=int(sum(p.numel() for p in net.parameters())), log=log)
    json.dump(info, open(os.path.join(ROOT, 'models', a.rung + '.json'), 'w'), indent=1)
    print({k: v for k, v in info.items() if k != 'log'})


if __name__ == '__main__':
    main()
