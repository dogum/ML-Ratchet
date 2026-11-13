"""Train the HybridPilot MLP on DP-labelled banks (behaviour cloning / DAgger aggregate).
usage: train.py OUT_NAME BANK_DIR [BANK_DIR ...] [--hidden 96] [--epochs 30] [--w_on 1.0] [--init ckpt.npz]
Outputs ckpt/<OUT>.npz (numpy weights incl. standardisation) and a json with validation numbers."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, glob, json, time, argparse, zlib
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(1)
ROOT = _P.WORK


def load_banks(dirs):
    X, S, ON, C, A = [], [], [], [], []
    off = 0
    for d in dirs:
        for i, f in enumerate(sorted(glob.glob(f'{ROOT}/data/{d}/*.npz'))):
            z = np.load(f)
            X.append(z['X']); S.append(z['s']); ON.append(z['on']); A.append(z['arch'])
            C.append(z['case'].astype(np.int64) + (zlib.crc32((d + os.path.basename(f).split('_')[0]).encode()) % 10_000_000) * 100000)
    return np.concatenate(X), np.concatenate(S), np.concatenate(ON), np.concatenate(C), np.concatenate(A)


class Net(nn.Module):
    def __init__(self, nin, h, mu, sd, depth=2):
        super().__init__()
        self.register_buffer('mu', torch.tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.tensor(sd, dtype=torch.float32))
        layers = []; n = nin
        for _ in range(depth):
            layers += [nn.Linear(n, h), nn.Tanh()]; n = h
        layers += [nn.Linear(n, 2)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f((x - self.mu) / self.sd)


def export(net, path):
    p = {'mu': net.mu.numpy().astype(np.float64), 'sd': net.sd.numpy().astype(np.float64)}
    lin = [m for m in net.f if isinstance(m, nn.Linear)]
    for i, m in enumerate(lin):
        p[f'W{i}'] = m.weight.detach().numpy().T.astype(np.float64); p[f'b{i}'] = m.bias.detach().numpy().astype(np.float64)
    np.savez(path, **p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('banks', nargs='+')
    ap.add_argument('--hidden', type=int, default=96); ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--w_on', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--drop_feats', default='')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    X, S, ON, C, A = load_banks(a.banks)
    ok = np.isfinite(S) & np.all(np.isfinite(X), axis=1)
    X, S, ON, C, A = X[ok], S[ok], ON[ok], C[ok], A[ok]
    if a.drop_feats:
        from hp_policy import FEAT_NAMES
        for nm in a.drop_feats.split(','):
            X[:, [i for i, f in enumerate(FEAT_NAMES) if f.startswith(nm)]] = 0.0
    y_s = np.log(np.clip(S, 0, 8) + 0.05).astype(np.float32)
    rng = np.random.default_rng(0)
    uc = np.unique(C); val_c = set(rng.choice(uc, max(1, len(uc) // 10), replace=False).tolist())
    va = np.array([c in val_c for c in C]); tr = ~va
    mu = X[tr].mean(0); sd = X[tr].std(0); sd = np.where(sd < 1e-3, 1.0, sd)
    print(f'samples {len(X)} train {tr.sum()} val {va.sum()} cases {len(uc)}; on-rate {ON.mean():.3f}', flush=True)
    net = Net(X.shape[1], a.hidden, mu, sd, a.depth)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xt = torch.tensor(X[tr]); Yt = torch.tensor(y_s[tr]); Ot = torch.tensor(ON[tr])
    Xv = torch.tensor(X[va]); Yv = torch.tensor(y_s[va]); Ov = torch.tensor(ON[va])
    hub = nn.SmoothL1Loss(beta=0.1); bce = nn.BCEWithLogitsLoss()
    bs = 1024; n = len(Xt)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(n); tl = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            out = net(Xt[b])
            loss = hub(out[:, 0], Yt[b]) + a.w_on * bce(out[:, 1], Ot[b])
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            ov = net(Xv)
            ls = hub(ov[:, 0], Yv).item(); lo = bce(ov[:, 1], Ov).item()
            acc = ((ov[:, 1] > 0).float() == Ov).float().mean().item()
            mae_s = (torch.exp(ov[:, 0]) - torch.exp(Yv)).abs().median().item()
        print(f'ep {ep} train {tl / n:.4f} val_s {ls:.4f} val_on {lo:.4f} acc_on {acc:.4f} med|ds| {mae_s:.3f} {time.time() - t0:.0f}s', flush=True)
    os.makedirs(f'{ROOT}/ckpt', exist_ok=True)
    p = export(net, f'{ROOT}/ckpt/{a.out}.npz')
    torch.save(net.state_dict(), f'{ROOT}/ckpt/{a.out}.pt')
    # numpy == torch check
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hp_policy import MLP
    m = MLP.load(f'{ROOT}/ckpt/{a.out}.npz')
    xs = X[va][:200]
    with torch.no_grad():
        yt = net(torch.tensor(xs)).numpy()
    yn = np.stack([m(x.astype(np.float64)) for x in xs])
    err = float(np.abs(yt - yn).max())
    info = dict(out=a.out, banks=a.banks, samples=int(len(X)), val_s_huber=ls, val_on_bce=lo, val_on_acc=acc, val_med_abs_ds=mae_s,
                numpy_vs_torch_maxabs=err, hidden=a.hidden, depth=a.depth, epochs=a.epochs, w_on=a.w_on,
                n_params=int(sum(p.numel() for p in net.parameters())))
    json.dump(info, open(f'{ROOT}/ckpt/{a.out}.json', 'w'), indent=1)
    print(info, flush=True)


if __name__ == '__main__':
    main()
