"""Train one EnergyNet rung on the simulated bank (resumable).
python train.py <rung_name> --version v1|v2|v3 --mode corr|abs [--epochs 60] [--hidden 256] [--seed 0] [--bank train]

mode corr: the network predicts log(E / E_phys) (a correction to the generic physics estimate)
mode abs : the network predicts log(E) directly (physics features still among the inputs)
Outputs: ckpt/<rung>.pt (checkpoint), ckpt/<rung>.npz (numpy weights + input scaling)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, time, json, argparse
sys.path.insert(0, _P.SRC)
import numpy as np, torch, torch.nn as nn
import dataset as DS, features as F

torch.set_num_threads(1)
CK = _P.WORK + '/ckpt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MY_COLS = [F.NR + F.NF + 47, F.NR + F.NF + 48]      # model-year input (Phase D rung r7)


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
    c = B['comp'][:, :5].astype(float)            # aero, rolling, brake, aux, powertrain losses (MJ)
    return np.log(np.clip(c, 1e-3 * E[:, None], None) / E[:, None])


def norm_stats(X0, robust=False, q=(0.5, 99.5)):
    """Input standardisation. The historical rungs (robust=False) take the plain mean and std of the drawn bank.
    A few moving downhill trips with zero net positive wheel work make features.phys_vec's work shares (aero,
    rolling, grade_dn, inertia over Ep) explode to 1e5..1e6, which stretched x_sd by 2e4-2.5e5x and left those four
    inputs near-constant on real trips (audit 2026-11-25, docs/audits/energynet-v2.1.md). robust=True takes the
    statistics of each column clipped to its q-percentiles; the network's existing z-clip at +-8 then saturates
    those trips instead of letting them set the scale."""
    X0 = np.asarray(X0, float)
    if robust:
        lo, hi = np.percentile(X0, q[0], axis=0), np.percentile(X0, q[1], axis=0)
        X0 = np.clip(X0, lo, hi)
    return X0.mean(0), X0.std(0) + 1e-3


def export(net, mu, sd, path, meta):
    sdict = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    np.savez(path, x_mu=mu.astype(np.float32), x_sd=sd.astype(np.float32), meta=json.dumps(meta), **{k.replace('.', '_'): v for k, v in sdict.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rung'); ap.add_argument('--version', default='v1'); ap.add_argument('--mode', default='corr')
    ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--bank', default='train'); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wc', type=float, default=0.2)
    ap.add_argument('--era', action='store_true', help='read the model-year input (rung r7); otherwise it is held at zero')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--robust-norm', action='store_true', help='percentile-clipped input statistics (see norm_stats)')
    ap.add_argument('--mask-comp', action='store_true', help='weight the component loss by the same y > 0.02 mask as the NLL')
    ap.add_argument('--lab', default='lab', help="'lab_epa': window-sticker labels from the EPA procedure emulation (rung r11)")
    a = ap.parse_args()
    B = DS.load_bank(a.bank)
    if a.lab != 'lab': B['lab'] = B[a.lab]
    n = len(B['arch']); print('bank', n, flush=True)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    def tiers(r):
        T = DS.draw_tiers(B, r)
        if not a.era: T['my'] = np.zeros(n, bool)
        return T
    T0 = tiers(np.random.default_rng(12345))
    X0, E0 = DS.assemble(B, T0)
    mu, sd = norm_stats(X0, robust=a.robust_norm)
    if not a.era: mu[MY_COLS] = 0.0; sd[MY_COLS] = 1.0
    net = Net(X0.shape[1], a.hidden).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    bs = a.batch
    steps_per = (n + bs - 1) // bs; total = a.epochs * steps_per
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    ck = f'{CK}/{a.rung}.pt'; ep0 = 0
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False, map_location=DEV); net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        ep0 = s['epoch'] + 1; mu, sd = s['mu'], s['sd']; rng = s['rng']; print('resume', ep0, flush=True)
    for ep in range(ep0, a.epochs):
        t0 = time.time()
        T = tiers(rng)
        X, E = DS.assemble(B, T)
        yv = DS.target(B, T, a.version)
        okm = yv > 0.02
        y = np.log(np.maximum(yv, 0.02))
        yy = y - np.log(E) if a.mode == 'corr' else y
        C = comp_targets(B, E)
        Xt = torch.tensor(np.clip((X - mu) / sd, -8, 8), dtype=torch.float32, device=DEV)
        yt = torch.tensor(yy, dtype=torch.float32, device=DEV); Ct = torch.tensor(C, dtype=torch.float32, device=DEV)
        Wt = torch.tensor(okm.astype(np.float32), device=DEV)
        perm = torch.randperm(n, device=DEV); tot = 0.0
        net.train()
        for k in range(steps_per):
            idx = perm[k * bs:(k + 1) * bs]
            m, ls, c = net(Xt[idx])
            w = Wt[idx]
            nll = ((ls + 0.5 * ((yt[idx] - m) / ls.exp()) ** 2) * w).sum() / w.sum()
            if a.mask_comp:   # rows with y <= 0.02 are already out of the NLL; keep them out of the component loss too
                comp = (((c - Ct[idx]) ** 2).mean(1) * w).sum() / w.sum()
            else:
                comp = ((c - Ct[idx]) ** 2).mean()
            loss = nll + a.wc * comp
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(nll)
        net.eval()
        with torch.no_grad():
            m, ls, c = net(Xt)
            mae = float(np.median(np.abs(m.cpu().numpy() - yy)[okm]))
        print(f'ep {ep} nll {tot / steps_per:.4f} train med|err| {mae:.4f} {time.time() - t0:.1f}s', flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, mu=mu, sd=sd, rng=rng), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
    export(net, mu, sd, f'{CK}/{a.rung}.npz', dict(rung=a.rung, version=a.version, mode=a.mode, hidden=a.hidden, nin=int(X0.shape[1]),
                                                    epochs=a.epochs, seed=a.seed, bank=a.bank, phys_k=F.PHYS_K, era=bool(a.era), lab=a.lab, lr=a.lr,
                                                    masked_inputs=[] if a.era else MY_COLS,
                                                    robust_norm=bool(a.robust_norm), mask_comp=bool(a.mask_comp)))
    print('exported', flush=True)


if __name__ == '__main__':
    main()
