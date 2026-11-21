"""Train a BasinDetective MLP on a simulated bank (resumable; one thread).

python3 train.py --run r1 --phys v1 --train 0-59 [--hidden 256 --layers 2 --epochs 80 --lr 1e-3 --dropout 0.1]
Writes models/<run>/{ckpt.pt, model.npz, result.json}.
"""
import common  # noqa: F401
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import D
from net import REG_TARGETS, CLS_TARGET, load_bank, Detective
import evaluate as E

torch.set_num_threads(1)


def rng_list(s):
    out = []
    for p in s.split(','):
        if '-' in p:
            a, b = map(int, p.split('-'))
            out += list(range(a, b + 1))
        elif p:
            out.append(int(p))
    return out


class MLP(nn.Module):
    def __init__(self, nin, nt, hidden, layers, dropout):
        super().__init__()
        self.hs = nn.ModuleList()
        d = nin
        for _ in range(layers):
            self.hs.append(nn.Linear(d, hidden))
            d = hidden
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, 2 * nt + 1)

    def forward(self, x):
        for h in self.hs:
            x = self.drop(nn.functional.gelu(h(x)))
        return self.out(x)


def prep(X, lab, stats=None):
    if stats is None:
        med = np.nanmedian(X, 0)
        q1, q3 = np.nanpercentile(X, [25, 75], 0)
        iqr = np.where(q3 - q1 > 1e-6, q3 - q1, np.nanstd(X, 0) + 1e-6)
        Y = lab[REG_TARGETS].values.astype(float)
        lo, hi = np.nanpercentile(Y, [1, 99], 0)
        Yc = np.clip(Y, lo, hi)
        stats = {'x_med': med, 'x_iqr': iqr, 'y_mu': np.nanmean(Yc, 0), 'y_sd': np.nanstd(Yc, 0) + 1e-9}
    Z = np.clip((X - stats['x_med']) / stats['x_iqr'], -6, 6)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    Y = (lab[REG_TARGETS].values.astype(float) - stats['y_mu']) / stats['y_sd']
    Y = np.clip(Y, -6, 6)
    C = lab[CLS_TARGET].values.astype(float)
    return Z.astype(np.float32), Y.astype(np.float32), C.astype(np.float32), stats


def export(net, stats, names, path, cfg):
    W = {'feat_names': np.array(names), 'targets': np.array(REG_TARGETS), 'n_layers': np.array(len(net.hs))}
    for i, h in enumerate(net.hs):
        W[f'w{i}'] = h.weight.detach().double().numpy()
        W[f'b{i}'] = h.bias.detach().double().numpy()
    W['wo'] = net.out.weight.detach().double().numpy()
    W['bo'] = net.out.bias.detach().double().numpy()
    for k, v in stats.items():
        W[k] = np.asarray(v, np.float64)
    W['cfg'] = np.array(json.dumps(cfg))
    np.savez(path, **W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--phys', default='v1')
    ap.add_argument('--train', default='0-59')
    ap.add_argument('--extra', default='', help='extra bank as phys:shards (mixing)')
    ap.add_argument('--val', default='900-903')
    ap.add_argument('--valphys', default=None)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--drop_feats', default='', help='comma list of feature-name prefixes to remove')
    a = ap.parse_args()
    cfg = vars(a)
    out = f'{D}/models/{a.run}'
    os.makedirs(out, exist_ok=True)
    X, lab, names = load_bank(a.phys, rng_list(a.train))
    if a.extra:
        ph, sh = a.extra.split(':')
        X2, lab2, _ = load_bank(ph, rng_list(sh))
        import pandas as pd
        X = np.concatenate([X, X2])
        lab = pd.concat([lab, lab2], ignore_index=True)
    keep = np.ones(len(names), bool)
    if a.drop_feats:
        for p in a.drop_feats.split(','):
            keep &= ~np.array([n.startswith(p) for n in names])
    # dropped features are kept in the vector but frozen at the median (so the feature code stays one)
    Xv, labv, _ = load_bank(a.valphys or a.phys, rng_list(a.val))
    Z, Y, C, st = prep(X, lab)
    if not keep.all():
        Z[:, ~keep] = 0.0
        st['x_iqr'] = np.where(keep, st['x_iqr'], 1e30)
    torch.manual_seed(a.seed)
    rs = np.random.default_rng(a.seed)
    nt = len(REG_TARGETS)
    net = MLP(Z.shape[1], nt, a.hidden, a.layers, a.dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per = int(np.ceil(len(Z) / 256))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per)
    ck = f'{out}/ckpt.pt'
    ep0 = 0
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        ep0 = s['epoch'] + 1
        rs = s['rs']
    Zt, Yt, Ct = map(torch.from_numpy, (Z, Y, C))
    Mt = torch.isfinite(Yt)
    Yt = torch.where(Mt, Yt, torch.zeros_like(Yt))
    t0 = time.time()
    for ep in range(ep0, a.epochs):
        net.train()
        perm = rs.permutation(len(Z))
        tot = 0.0
        for i in range(steps_per):
            b = torch.from_numpy(perm[i * 256:(i + 1) * 256])
            o = net(Zt[b])
            mu, ls = o[:, :nt], o[:, nt:2 * nt].clamp(-7, 5)
            m = Mt[b].float()
            nll = (0.5 * ((Yt[b] - mu) * torch.exp(-ls)) ** 2 + ls) * m
            loss = nll.sum(0) / m.sum(0).clamp(min=1)
            bce = nn.functional.binary_cross_entropy_with_logits(o[:, 2 * nt], Ct[b])
            L = loss.sum() + bce
            opt.zero_grad(); L.backward(); opt.step(); sched.step()
            tot += float(L)
        if ep % 5 == 4 or ep == a.epochs - 1:
            torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep,
                        'rs': rs}, ck)
            print(f'epoch {ep} loss {tot / steps_per:.4f} {time.time() - t0:.0f}s', flush=True)
    net.eval()
    export(net, st, names, f'{out}/model.npz', cfg)
    m = Detective(f'{out}/model.npz')
    # numpy == torch check
    with torch.no_grad():
        ot = net(Zt[:64]).numpy()
    torch_diff = float(np.abs(m.raw(X[:64]) - ot).max())
    sim = E.sim_scores(m, Xv, labv)
    dev, _ = E.dev_gate(f'{out}/model.npz')
    res = {'cfg': cfg, 'n_train': int(len(Z)), 'val': sim, 'val_skill': sim['_skill'], 'dev': dev,
           'dev_composite': dev['_composite'], 'n_params': int(sum(p.numel() for p in net.parameters())),
           'numpy_vs_torch_max_abs': torch_diff}
    json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps({'val_skill': res['val_skill'], 'dev_composite': res['dev_composite'],
                      'val': {k: v for k, v in sim.items()}, 'dev': dev}, indent=None, default=float)[:3000])


if __name__ == '__main__':
    main()
