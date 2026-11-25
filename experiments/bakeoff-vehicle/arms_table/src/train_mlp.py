"""MLP control: EnergyNet v2.1's network and training recipe (vendor/energynet/training/train.py), retrained on the
bake-off's shared train bank (28,800 trips; v2.1 saw 3 banks = 86,400). Resumable per epoch, one thread.

usage: python3 train_mlp.py OUT_DIR --seed S [--epochs 60] [--hidden 256] [--lr 2e-3] [--wc 0.2] [--batch 256]

As train.py with `--version e --mode corr --era` (v2.1's r15 members): target log(y / E_phys) with the tier's target
version (bake.target), Gaussian NLL with a learned log-sigma head + 0.2 x MSE of the 5 log component shares, AdamW
(wd 1e-4), OneCycle, grad clip 5, tiers redrawn every epoch, input scaling from one tier draw (rng 12345).
Writes OUT_DIR/s<seed>.pt (checkpoint) and OUT_DIR/s<seed>.npz (numpy weights in EnergyNet's export format) and
OUT_DIR/s<seed>.json (cpu seconds, final train error).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake  # noqa: E402
import features as F  # noqa: E402

torch.set_num_threads(1)


class Net(nn.Module):
    def __init__(self, nin, h=256, nc=5):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(nin, h), nn.GELU(approximate='tanh'), nn.Linear(h, h), nn.GELU(approximate='tanh'),
                                  nn.Linear(h, h), nn.GELU(approximate='tanh'), nn.Linear(h, h // 2), nn.GELU(approximate='tanh'))
        self.head = nn.Linear(h // 2, 2 + nc)

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, 0], o[:, 1].clamp(-5, 2), o[:, 2:]


def comp_targets(B, E):
    c = B['comp'][:, :5].astype(float)
    return np.log(np.clip(c, 1e-3 * E[:, None], None) / E[:, None])


def export(net, mu, sd, path, meta):
    sdict = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(path, x_mu=mu.astype(np.float32), x_sd=sd.astype(np.float32), meta=json.dumps(meta),
             **{k.replace('.', '_'): v for k, v in sdict.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wc', type=float, default=0.2); ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f's{a.seed}'
    ck = os.path.join(a.out, tag + '.pt')
    B = bake.load_bank('train')
    n = len(B['arch'])
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    T0 = bake.draw_tiers(B, np.random.default_rng(12345))
    X0 = bake.sim_table(B, T0)['X']
    mu = X0.mean(0); sd = X0.std(0) + 1e-3
    net = Net(X0.shape[1], a.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    bs = a.batch
    steps_per = (n + bs - 1) // bs; total = a.epochs * steps_per
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    ep0 = 0; cpu_prev = 0.0
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        ep0 = s['epoch'] + 1; mu, sd = s['mu'], s['sd']; rng = s['rng']; cpu_prev = s.get('cpu', 0.0)
        torch.set_rng_state(s['torch_rng'])
        print('resume', ep0, flush=True)
    c0 = time.process_time()
    mae = float('nan')
    for ep in range(ep0, a.epochs):
        t0 = time.time()
        T = bake.draw_tiers(B, rng)
        D = bake.sim_table(B, T)
        X, E, yv = D['X'], D['E_phys'], D['y']
        okm = yv > 0.02
        yy = np.log(np.maximum(yv, 0.02)) - np.log(E)
        C = comp_targets(B, E)
        Xt = torch.tensor(np.clip((X - mu) / sd, -8, 8), dtype=torch.float32)
        yt = torch.tensor(yy, dtype=torch.float32); Ct = torch.tensor(C, dtype=torch.float32)
        Wt = torch.tensor(okm.astype(np.float32))
        perm = torch.randperm(n); tot = 0.0
        net.train()
        for k in range(steps_per):
            idx = perm[k * bs:(k + 1) * bs]
            m, ls, c = net(Xt[idx])
            w = Wt[idx]
            nll = ((ls + 0.5 * ((yt[idx] - m) / ls.exp()) ** 2) * w).sum() / w.sum()
            loss = nll + a.wc * ((c - Ct[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(nll)
        if ep % 10 == 9 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                m, ls, c = net(Xt)
                mae = float(np.median(np.abs(m.numpy() - yy)[okm]))
        print(f'ep {ep} nll {tot / steps_per:.4f} train med|err| {mae:.4f} {time.time() - t0:.1f}s', flush=True)
        cpu = cpu_prev + time.process_time() - c0
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, mu=mu, sd=sd,
                        rng=rng, cpu=cpu, torch_rng=torch.get_rng_state()), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
    cpu = cpu_prev + time.process_time() - c0
    export(net, mu, sd, os.path.join(a.out, tag + '.npz'),
           dict(rung=os.path.basename(a.out.rstrip('/')), version='e', mode='corr', hidden=a.hidden, nin=int(X0.shape[1]),
                epochs=a.epochs, seed=a.seed, bank='bakeoff train (28,800)', phys_k=F.PHYS_K, era=True, lr=a.lr,
                masked_inputs=[]))
    json.dump({'cpu_s': cpu, 'train_med_abs_err': mae, 'epochs': a.epochs, 'seed': a.seed},
              open(os.path.join(a.out, tag + '.json'), 'w'))
    print('exported', tag, 'cpu_s', round(cpu, 1), flush=True)


if __name__ == '__main__':
    main()
