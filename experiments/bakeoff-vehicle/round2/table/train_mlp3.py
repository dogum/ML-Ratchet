"""Round 2: the MLP control (v1.0 tab_mlp r3 recipe = EnergyNet v2.1's network and loss) on several banks, GPU.

python train_mlp3.py <name> <seed> <bank> [<bank> ...] [--epochs 30] [--device cuda]
-> round2/work/mlp/<name>/s<seed>.npz (EnergyNet's numpy export format, read by tab_mlp's model_np.py) + s<seed>.json

Identical to arms_table/src/train_mlp.py (imported for Net / export / comp_targets, not edited) except: each epoch draws
tiers for every listed bank and concatenates them (the 3 banks are one data set of 86,400 trips), and it runs on the
GPU. Input scaling from one tier draw of train101 (rng 12345), as v1.0. Batch 256, AdamW 2e-3, OneCycle (pct 0.1),
Gaussian NLL + 0.2 x component MSE, grad clip 5.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
P.setup()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import bake  # noqa: E402
import features as F  # noqa: E402
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location('v1_train_mlp', os.path.join(P.BAKEOFF, 'arms_table', 'src', 'train_mlp.py'))
TM = importlib.util.module_from_spec(_s); _s.loader.exec_module(TM)   # its harness sys.path line is a no-op here


def load_bank(name):
    B = dict(np.load(os.path.join(P.bank_dir(name), 'rows.npz')))
    B['_split'] = 'train'; B['_bank'] = name
    return B


def main():
    a = sys.argv[1:]
    epochs = int(a[a.index('--epochs') + 1]) if '--epochs' in a else 30
    dev = torch.device(a[a.index('--device') + 1] if '--device' in a else 'cuda')
    pos = [x for i, x in enumerate(a) if not x.startswith('--') and (i == 0 or not a[i - 1].startswith('--'))]
    name, seed, banks = pos[0], int(pos[1]), pos[2:]
    out = os.path.join(P.WORK, 'mlp', name); os.makedirs(out, exist_ok=True)
    tag = f's{seed}'
    if os.path.exists(os.path.join(out, tag + '.json')):
        print('done already'); return
    Bs = [load_bank(b) for b in banks]
    B101 = Bs[0]                                   # train101 (or train101_v4): scaling as v1.0
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    X0 = bake.sim_table(B101, bake.draw_tiers(B101, np.random.default_rng(12345)))['X']
    mu = X0.mean(0); sd = X0.std(0) + 1e-3
    net = TM.Net(X0.shape[1], 256).to(dev)
    lr, bs, wc = 2e-3, 256, 0.2
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = sum(len(B['arch']) for B in Bs)
    steps_per = (n + bs - 1) // bs; total = epochs * steps_per
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    t0 = time.time(); c0 = time.process_time(); mae = float('nan')
    for ep in range(epochs):
        Xs, Es, Ys, Cs = [], [], [], []
        for B in Bs:
            D = bake.sim_table(B, bake.draw_tiers(B, rng))
            Xs.append(D['X']); Es.append(D['E_phys']); Ys.append(D['y']); Cs.append(TM.comp_targets(B, D['E_phys']))
        X = np.concatenate(Xs); E = np.concatenate(Es); yv = np.concatenate(Ys); Cc = np.concatenate(Cs)
        okm = yv > 0.02
        yy = np.log(np.maximum(yv, 0.02)) - np.log(E)
        Xt = torch.tensor(np.clip((X - mu) / sd, -8, 8), dtype=torch.float32, device=dev)
        yt = torch.tensor(yy, dtype=torch.float32, device=dev); Ct = torch.tensor(Cc, dtype=torch.float32, device=dev)
        Wt = torch.tensor(okm.astype(np.float32), device=dev)
        perm = torch.randperm(n).to(dev); tot = 0.0
        net.train()
        for k in range(steps_per):
            idx = perm[k * bs:(k + 1) * bs]
            m, ls, c = net(Xt[idx])
            w = Wt[idx]
            nll = ((ls + 0.5 * ((yt[idx] - m) / ls.exp()) ** 2) * w).sum() / w.sum()
            loss = nll + wc * ((c - Ct[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(nll.detach())
        if ep % 10 == 9 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                m, ls, c = net(Xt)
                mae = float(np.median(np.abs(m.cpu().numpy() - yy)[okm]))
        print(f'ep {ep} nll {tot / steps_per:.4f} train med|err| {mae:.4f} {time.time() - t0:.0f}s', flush=True)
    net = net.cpu()
    TM.export(net, mu, sd, os.path.join(out, tag + '.npz'),
              dict(rung=name, version='e', mode='corr', hidden=256, nin=int(X0.shape[1]), epochs=epochs, seed=seed,
                   bank='+'.join(banks), phys_k=F.PHYS_K, era=True, lr=lr, masked_inputs=[]))
    json.dump({'gpu_wall_s': round(time.time() - t0, 1), 'cpu_s': round(time.process_time() - c0, 1),
               'train_med_abs_err': mae, 'epochs': epochs, 'seed': seed, 'banks': banks},
              open(os.path.join(out, tag + '.json'), 'w'))
    print('exported', tag, flush=True)


if __name__ == '__main__':
    main()
