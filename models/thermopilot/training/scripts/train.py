"""Train the ThermoPilot MLP by imitation of the perfect-foresight DP teacher (cost-sensitive: soft targets from
the teacher's action values + expected regret).

usage: train.py OUT.npz CONFIG.json BANK_DIR [BANK_DIR ...]
config: hidden [256,256], tau 0.1, w_regret 1.0, epochs 20, lr 1e-3, batch 512, wd 1e-5, seed 0
"""
import os, sys, json, glob, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
torch.set_num_threads(1)


def load_banks(dirs):
    X, Q, S, E = [], [], [], []
    for d in dirs:
        for i, f in enumerate(sorted(glob.glob(os.path.join(d, 'ep_*.npz')))):
            z = np.load(f)
            X.append(z['X']); Q.append(z['Q']); S.append(z['S'])
            E.append(np.full(len(z['S']), hash(f) % (1 << 30)))
    return np.concatenate(X), np.concatenate(Q), np.concatenate(S), np.concatenate(E)


def main():
    out, cfgf, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    cfg = dict(hidden=[256, 256], tau=0.1, w_regret=1.0, epochs=20, lr=1e-3, batch=512, wd=1e-5, seed=0)
    cfg.update(json.load(open(cfgf)))
    t0 = time.time()
    X, Q, S, E = load_banks(dirs)
    X = X.astype(np.float32)
    ue = np.unique(E)
    r = np.random.default_rng(cfg['seed'])
    hold = set(r.choice(ue, size=max(1, len(ue) // 10), replace=False).tolist())
    mh = np.isin(E, list(hold))
    mu = X[~mh].mean(0); sd = X[~mh].std(0) + 1e-3
    Xn = (X - mu) / sd
    Qn = Q / S[:, None]
    print(f'{len(X)} samples, {len(ue)} episodes, {X.shape[1]} features; load {time.time() - t0:.0f}s', flush=True)
    torch.manual_seed(cfg['seed'])
    layers, d = [], X.shape[1]
    for h in cfg['hidden']:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers += [torch.nn.Linear(d, Q.shape[1])]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    Xt, Qt = torch.tensor(Xn[~mh]), torch.tensor(Qn[~mh])
    Xv, Qv = torch.tensor(Xn[mh]), torch.tensor(Qn[mh])
    n = len(Xt)
    steps = cfg['epochs'] * (n // cfg['batch'])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=steps)
    best = (1e9, None)

    def lossf(logit, q):
        tgt = torch.softmax(-q / cfg['tau'], 1)
        lp = torch.log_softmax(logit, 1)
        ce = -(tgt * lp).sum(1).mean()
        reg = (lp.exp() * q).sum(1).mean()
        return ce + cfg['w_regret'] * reg, ce, reg

    for ep in range(cfg['epochs']):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n - cfg['batch'] + 1, cfg['batch']):
            b = perm[i:i + cfg['batch']]
            loss, _, _ = lossf(net(Xt[b]), Qt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            lv = net(Xv)
            l, ce, reg = lossf(lv, Qv)
            a = lv.argmax(1)
            hard = Qv[torch.arange(len(a)), a].mean().item()
            agree = (a == Qv.argmin(1)).float().mean().item()
        print(f'ep {ep} loss {l.item():.4f} ce {ce.item():.4f} soft-regret {reg.item():.4f} hard-regret {hard:.4f} '
              f'agree {agree:.3f}  {time.time() - t0:.0f}s', flush=True)
        if hard < best[0]:
            best = (hard, {k: v.detach().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    W = dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        W[f'W{i}'] = m.weight.detach().numpy().T.astype(np.float32)
        W[f'b{i}'] = m.bias.detach().numpy().astype(np.float32)
    np.savez(out, **W)
    # numpy == torch check
    from_np = Xn[mh][:256]
    h = from_np
    for i in range(len(lin)):
        h = h @ W[f'W{i}'] + W[f'b{i}']
        if i < len(lin) - 1:
            h = np.maximum(h, 0)
    with torch.no_grad():
        ht = net(torch.tensor(from_np)).numpy()
    err = float(np.abs(h - ht).max())
    info = dict(cfg=cfg, n=len(X), episodes=len(ue), best_hard_regret=best[0], np_torch_maxerr=err,
                params=int(sum(p.numel() for p in net.parameters())), wall_s=time.time() - t0)
    json.dump(info, open(out.replace('.npz', '.json'), 'w'), indent=1)
    print(info, flush=True)


if __name__ == '__main__':
    main()
