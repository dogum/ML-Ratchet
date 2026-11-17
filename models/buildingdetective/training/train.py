"""Train one Detective network (hourly | monthly | annual) on simulated features only.

MLP with heteroscedastic regression heads (mu, log sigma per target, Gaussian NLL, NaN labels masked) and three
softmax heads (use, heating class, cooling class). One thread. Resumable: checkpoint every epoch.
usage: python3 train.py <run_name> <mode> <feat_tag> [hidden=256] [epochs=60] [layers=2] [drop=0.1]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import common as C

torch.set_num_threads(1)
NR, NU, NH, NC = len(C.REG), len(C.USES), len(C.HEAT), len(C.COOL)
MAX_SHARD = [899]


def load_feats(tag, mode, val):
    fs = sorted(glob.glob(os.path.join(C.ROOT, 'feats', tag, f'{mode}_*.npz')))
    fs = [f for f in fs if not f.endswith('.tmp.npz') and ((int(f[-8:-4]) >= 900) == val)]
    if not val:
        fs = [f for f in fs if int(f[-8:-4]) <= MAX_SHARD[0]]
    Xs, Ys, Ks, Is = [], [], [], []
    for f in fs:
        z = np.load(f)
        if len(z['X']) == 0:
            continue
        Xs.append(z['X']); Ys.append(z['Y']); Ks.append(z['K']); Is.append(z['idx'])
    return np.concatenate(Xs), C.clean_labels(np.concatenate(Ys)), np.concatenate(Ks), np.concatenate(Is)


class Net(nn.Module):
    def __init__(self, nin, hidden=256, layers=2, drop=0.1):
        super().__init__()
        L = []
        d = nin
        for _ in range(layers):
            L += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop)]
            d = hidden
        self.body = nn.Sequential(*L)
        self.reg = nn.Linear(d, 2 * NR)
        self.cls = nn.Linear(d, NU + NH + NC)

    def forward(self, x):
        h = self.body(x)
        r = self.reg(h)
        return r[:, :NR], r[:, NR:].clamp(-6, 3), self.cls(h)


def loss_fn(mu, ls, logit, y, k, wcls=1.0):
    m = torch.isfinite(y)
    yz = torch.where(m, y, torch.zeros_like(y))
    nll = 0.5 * ((yz - mu) / ls.exp()) ** 2 + ls
    lr = (nll * m).sum() / m.sum().clamp(min=1)
    ce = nn.functional.cross_entropy
    lc = (ce(logit[:, :NU], k[:, 0]) + ce(logit[:, NU:NU + NH], k[:, 1]) + ce(logit[:, NU + NH:], k[:, 2]))
    return lr * NR / 3 + wcls * lc, lr, lc


def prep(tag, mode):
    Xt, Yt, Kt, It = load_feats(tag, mode, False)
    Xv, Yv, Kv, Iv = load_feats(tag, mode, True)
    xm, xs = Xt.mean(0), Xt.std(0)
    xs = np.where(xs < 1e-4, 1.0, xs)
    ym, ys = np.nanmean(Yt, 0), np.nanstd(Yt, 0)
    ys = np.where(ys < 1e-6, 1.0, ys)
    return (Xt, Yt, Kt, It), (Xv, Yv, Kv, Iv), dict(xm=xm, xs=xs, ym=ym, ys=ys)


def predict(net, X, norm):
    net.eval()
    with torch.no_grad():
        x = torch.tensor((X - norm['xm']) / norm['xs'], dtype=torch.float32)
        out = [net(x[i:i + 4096]) for i in range(0, len(x), 4096)]
        mu = torch.cat([o[0] for o in out]).numpy() * norm['ys'] + norm['ym']
        sg = np.exp(torch.cat([o[1] for o in out]).numpy()) * norm['ys']
        lg = torch.cat([o[2] for o in out]).numpy()
    return mu, sg, lg


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def skills(mu, sg, lg, Y, K):
    """Per-task skill in [0,1]: regression R2 (clipped at 0), classification balanced accuracy rescaled to chance."""
    S = {}
    for j, n in enumerate(C.REG):
        m = np.isfinite(Y[:, j])
        if m.sum() < 20:
            continue
        r2 = 1 - np.mean((mu[m, j] - Y[m, j]) ** 2) / np.var(Y[m, j])
        S[n] = float(max(r2, 0))
    for (a, b, name, labs) in ((0, NU, 'use', C.USES), (NU, NU + NH, 'heat', C.HEAT), (NU + NH, NU + NH + NC, 'cool', C.COOL)):
        p = lg[:, a:b].argmax(1)
        t = K[:, ['use', 'heat', 'cool'].index(name)]
        rec = [np.mean(p[t == c] == c) for c in range(b - a) if (t == c).sum() >= 10]
        bacc = float(np.mean(rec))
        S[name] = float(max((bacc - 1 / len(rec)) / (1 - 1 / len(rec)), 0))
    S['_overall'] = float(np.mean([v for k, v in S.items() if not k.startswith('_')]))
    return S


def main(run, mode, tag, hidden=256, epochs=60, layers=2, drop=0.1, lr=2e-3, wd=1e-4, seed=0):
    od = os.path.join(C.ROOT, 'models', run)
    os.makedirs(od, exist_ok=True)
    (Xt, Yt, Kt, It), (Xv, Yv, Kv, Iv), norm = prep(tag, mode)
    print(run, mode, 'train', Xt.shape, 'val', Xv.shape, flush=True)
    torch.manual_seed(seed)
    net = Net(Xt.shape[1], hidden, layers, drop)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(Xt) + 255) // 256))
    ck = os.path.join(od, 'ckpt.pt')
    ep0 = 0
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched']); ep0 = st['ep'] + 1
    xt = torch.tensor((Xt - norm['xm']) / norm['xs'], dtype=torch.float32)
    yt = torch.tensor((Yt - norm['ym']) / norm['ys'], dtype=torch.float32)
    kt = torch.tensor(Kt.astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    for ep in range(ep0, epochs):
        net.train()
        t0 = time.time()
        perm = torch.randperm(len(xt), generator=g)
        tl = []
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            mu, ls, lg = net(xt[b])
            loss, lr_, lc_ = loss_fn(mu, ls, lg, yt[b], kt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl.append(float(loss))
        if ep % 5 == 4 or ep == epochs - 1:
            mu, sg, lg = predict(net, Xv, norm)
            S = skills(mu, sg, lg, Yv, Kv)
            print(ep, 'loss %.3f' % np.mean(tl), 'val skill %.4f' % S['_overall'], '%.0fs' % (time.time() - t0), flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ep=ep), ck)
    mu, sg, lg = predict(net, Xv, norm)
    S = skills(mu, sg, lg, Yv, Kv)
    cov = {n: float(np.mean(np.abs(mu[np.isfinite(Yv[:, j]), j] - Yv[np.isfinite(Yv[:, j]), j]) <= sg[np.isfinite(Yv[:, j]), j]))
           for j, n in enumerate(C.REG)}
    cfg = dict(run=run, mode=mode, tag=tag, max_shard=MAX_SHARD[0], hidden=hidden, epochs=epochs, layers=layers, drop=drop, lr=lr, wd=wd,
               seed=seed, n_train=int(len(Xt)), n_val=int(len(Xv)), nin=int(Xt.shape[1]),
               n_params=int(sum(p.numel() for p in net.parameters())))
    torch.save(dict(net=net.state_dict(), norm=norm, cfg=cfg), os.path.join(od, 'final.pt'))
    json.dump(dict(cfg=cfg, val_skill=S, val_cov1sigma=cov), open(os.path.join(od, 'result.json'), 'w'), indent=1)
    print(json.dumps(S, indent=0))
    return S


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('mode'); ap.add_argument('tag')
    ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--layers', type=int, default=2); ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_shard', type=int, default=899)
    a = ap.parse_args()
    MAX_SHARD[0] = a.max_shard
    main(a.run, a.mode, a.tag, a.hidden, a.epochs, a.layers, a.drop, a.lr, a.wd, a.seed)
