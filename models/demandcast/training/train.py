"""Train the three DemandCast nets on a simulated bank.
python3 train.py <ckpt_name> <bank>[,<bank2>...] [--hidden 256,256] [--epochs 40] [--lr 2e-3] [--only d,w,s] [--init ckpt]
"""
import sys, os, time, argparse, json
import numpy as np, torch
from common import *
import model as M

torch.set_num_threads(1)


def masks(B, kind):
    if kind == 'd':
        y = B['yd']; m = np.isfinite(y)
    elif kind == 'w':
        y = B['yw']; m = np.isfinite(y)
        if 'hasw' in B: m &= B['hasw'][:, None]
    else:
        y = B['ys']; m = np.isfinite(y) & (B['night'] < 0.5)
        if 'hass' in B: m &= B['hass'][:, None]
    return np.nan_to_num(y).astype(np.float32), m.astype(np.float32)


XK = {'d': 'xd', 'w': 'xw', 's': 'xs'}
PK = {'w': 'pw', 's': 'ps'}


BASE_DIM = {'xd': 329, 'xw': 164, 'xs': 156}


def load_banks(names, base=False):
    parts = [dict(np.load(f'{DATA}/banks/{n}.npz')) for n in names]
    if base or len({p['xd'].shape[1] for p in parts}) > 1:      # mix of feature versions: keep the base features
        for p in parts:
            for k, d in BASE_DIM.items(): p[k] = p[k][:, :d]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if all(k in p for p in parts)}


def train_one(kind, B, hidden, epochs, lr, seed=0, init=None, wd=1e-4, bs=512):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = np.nan_to_num(B[XK[kind]].astype(np.float32))
    Y, Mk = masks(B, kind)
    P = B[PK[kind]].astype(np.float32) if kind != 'd' else np.zeros_like(Y)
    keep = Mk.sum(1) > 0
    X, Y, Mk, P = X[keep], Y[keep], Mk[keep], P[keep]
    net = M.Net(X.shape[1], hidden)
    if init is not None:
        net.load_state_dict(init)
    else:
        net.xm[:] = torch.tensor(X.mean(0)); net.xs[:] = torch.tensor(X.std(0) + (1e-3 if kind == 'd' else 2e-2))   # floor: near-constant sim features (night CF) must not amplify real-data quirks
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n = len(X); steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xt, Yt, Mt, Pt = map(torch.tensor, (X, Y, Mk, P))
    t0 = time.time(); step = 0
    for ep in range(epochs):
        perm = torch.tensor(rng.permutation(n)); tot = 0
        net.train()
        for b in range(n // bs):
            ix = perm[b * bs:(b + 1) * bs]
            mu, sig = M.head(kind, net, Xt[ix], Pt[ix])
            loss = M.nll(mu, sig, Yt[ix], Mt[ix])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += loss.item(); step += 1
        if ep % 5 == 4 or ep == epochs - 1:
            print(f'  {kind} epoch {ep + 1}/{epochs} nll {tot / (n // bs):.4f} {time.time() - t0:.0f}s', flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('banks')
    ap.add_argument('--hidden', default='256,256'); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--only', default='d,w,s')
    ap.add_argument('--init', default=None); ap.add_argument('--note', default='')
    a = ap.parse_args()
    B = load_banks(a.banks.split(','))
    hidden = tuple(int(h) for h in a.hidden.split(','))
    out = {'cfg': vars(a)}
    base = torch.load(f'{ROOT}/ckpt/{a.init}.pt', weights_only=False) if a.init else None
    for kind in ('d', 'w', 's'):
        if kind in a.only.split(','):
            print('train', kind, flush=True)
            net = train_one(kind, B, hidden, a.epochs, a.lr)
            out[kind] = net.state_dict(); out[kind + '_hidden'] = hidden
        elif base is not None:
            out[kind] = base[kind]; out[kind + '_hidden'] = base[kind + '_hidden']
    torch.save(out, f'{ROOT}/ckpt/{a.name}.pt')
    print('saved', a.name)


if __name__ == '__main__':
    main()
