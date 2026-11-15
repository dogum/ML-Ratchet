"""Train a GridPulse rung: python3 train.py <rung> <bank[,bank2]> [mode] [tau] [epochs]
Re-augments every epoch (phase, noise segment, truncation N). Checkpoints every epoch (resumable)."""
from common import *
import torch, torch.nn as nn, time
torch.set_num_threads(1)
import bank as BK, dataset as DS, net as NET


def main(rung, banks, mode='mean', tau=0.0, epochs=40, hid=NET.HID, lr=2e-3, bs=256, seed=0, ctx=False, p_unknown=0.25):
    dev, L = zip(*[BK.load(b) for b in banks.split(',')]); dev = np.concatenate(dev); L = pd.concat(L, ignore_index=True)
    keep = DS.selected(dev, L); dev, L = dev[keep], L[keep].reset_index(drop=True)
    print(rung, 'train events', len(L), L.system.value_counts().to_dict(), flush=True)
    noise = DS.noise_split('train'); rng = np.random.default_rng(seed); torch.manual_seed(seed)
    gb = (L.system.values == 'GB').astype(np.float32); f0 = L.f0.values
    # normalisation from one draw
    dem = L.demand_MW.values

    def feats(meas, N):
        d = np.where(rng.random(len(dem)) < p_unknown, np.nan, dem * rng.normal(1, 0.03, len(dem))) if ctx else None
        return DS.features(meas['w'], N, gb, f0, d, ctx)
    meas, N, y = DS.make(dev, L, rng, noise, mode, tau); X = feats(meas, N)
    xm, xs = X.mean(0), X.std(0) + 1e-3; ym, ys = y.mean(0), y.std(0) + 1e-6
    net = NET.Net(nin=X.shape[1], hid=hid); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ck = f'{CKPT}/{rung}.pt'; ep0 = 0
    if os.path.exists(ck + '.tmp'):
        s = torch.load(ck + '.tmp', weights_only=False); net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); ep0 = s['ep'] + 1
        rng = np.random.default_rng(seed + 1000 * ep0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(L) // bs + 1), last_epoch=ep0 * (len(L) // bs + 1) - 1 if ep0 else -1)
    for ep in range(ep0, epochs):
        t = time.time()
        meas, N, y = DS.make(dev, L, rng, noise, mode, tau)
        X = torch.tensor((feats(meas, N) - xm) / xs); Y = torch.tensor((y - ym) / ys)
        perm = torch.randperm(len(X)); tot = 0
        net.train()
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            mu, ls = net(X[idx])
            loss = (ls + 0.5 * ((Y[idx] - mu) / ls.exp()) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(idx)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), ep=ep), ck + '.tmp')
        print(f'ep {ep} nll {tot / len(X):.4f} {time.time() - t:.0f}s', flush=True)
    torch.save(dict(net=net.state_dict(), xm=xm, xs=xs, ym=ym, ys=ys, hid=hid, mode=mode, tau=tau, banks=banks, ctx=ctx, nin=X.shape[1],
                    n_train=len(L)), ck)
    os.remove(ck + '.tmp')


if __name__ == '__main__':
    a = sys.argv
    kw = json.loads(a[6]) if len(a) > 6 else {}
    if 'hid' in kw:
        kw['hid'] = tuple(kw['hid'])
    main(a[1], a[2], a[3] if len(a) > 3 else 'mean', float(a[4]) if len(a) > 4 else 0.0, int(a[5]) if len(a) > 5 else 40, **kw)
