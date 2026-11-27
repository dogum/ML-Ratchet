"""Train a FlowCast rung. usage: python3 train.py <config name>  (configs/<name>.json)
config: {name, phys, H, steps, batch, lr, nextday, init_from, dropout, val_every}
Resumable: ckpt/<name>.resume.pt every 250 steps; final ckpt/<name>.pt (best-on-sim-val is NOT used: last step)."""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch
torch.set_num_threads(1)
from model import FlowNet, loss_fn, QS
import bankdata as bd


def sim_eval(net, B, nextday, nmax=None, bs=50, nowcast=False, lead=1):
    """Frozen-set score: per sample, two fixed windows (offsets 0 and 731), the last 365 days of each scored.
    Returns dict with median NSE (mean vs gauge-like target), median KGE, 90 % interval coverage."""
    net.eval()
    n = len(B['qo']) if nmax is None else min(nmax, len(B['qo']))
    nses, kges, cov90, cov50 = [], [], [], []
    with torch.no_grad():
        for a in range(0, n, bs):
            idx = list(range(a, min(a + bs, n)))
            preds, qs, ys = [], [], []
            for o in (0, 731):
                X, Y, S, _ = bd.batch(B, idx, [o] * len(idx), nextday, nowcast, None, [lead] * len(idx))
                m, q = net(torch.from_numpy(X), torch.from_numpy(S))
                preds.append(m[:, bd.WARM:].numpy()); qs.append(q[:, bd.WARM:].numpy()); ys.append(Y[:, bd.WARM:])
            P = np.concatenate(preds, 1); Qq = np.concatenate(qs, 1); Yy = np.concatenate(ys, 1)
            for j in range(len(idx)):
                y, p = Yy[j], P[j]
                v = np.var(y)
                nses.append(1 - np.mean((p - y) ** 2) / v if v > 0 else np.nan)
                r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 and v > 0 else 0.0
                al = np.std(p) / max(np.std(y), 1e-9); be = np.mean(p) / max(np.mean(y), 1e-9)
                kges.append(1 - math.sqrt((r - 1) ** 2 + (al - 1) ** 2 + (be - 1) ** 2))
                cov90.append(np.mean((y >= Qq[j, :, 0]) & (y <= Qq[j, :, 4])))
                cov50.append(np.mean((y >= Qq[j, :, 1]) & (y <= Qq[j, :, 3])))
    net.train()
    return {'nse_median': float(np.nanmedian(nses)), 'kge_median': float(np.nanmedian(kges)),
            'cov90': float(np.mean(cov90)), 'cov50': float(np.mean(cov50)), 'n': n,
            'frac_nse_lt0': float(np.nanmean(np.array(nses) < 0)), 'nse_all': [float(x) for x in nses]}


def build_net(cfg, Btr=None):
    net = FlowNet(bd.n_feat(cfg.get('nextday', False), cfg.get('nowcast', False)), len(STATIC_NAMES), H=cfg.get('H', 64),
                  dropout=cfg.get('dropout', 0.1))
    if Btr is not None:
        net.st_mu.copy_(torch.from_numpy(Btr['st'].mean(0)))
        net.st_sd.copy_(torch.from_numpy(Btr['st'].std(0) + 1e-3))
    return net


def main(name):
    cfg = json.load(open(f'{CONFIGS}/{name}.json'))
    nextday = cfg.get('nextday', False)
    Btr = bd.add_sigma(bd.load_bank(cfg['phys'], 'train'), cfg.get('weight', 'target_std'))
    Bva = bd.add_sigma(bd.load_bank(cfg['phys'], 'val'), cfg.get('weight', 'target_std'))
    print(f'train {len(Btr["qo"])} val {len(Bva["qo"])}', flush=True)
    # Seed torch before the network is built: until 27 Nov 2026 only NumPy's batch sampler was seeded, so the same
    # config gave different initial weights and dropout masks (audit). Historical rungs were trained unseeded.
    torch.manual_seed(cfg.get('seed', 0))
    net = build_net(cfg, Btr)
    if cfg.get('init_from'):
        sd = torch.load(f'{ROOT}/ckpt/{cfg["init_from"]}.pt', weights_only=False)['state']
        if sd['lstm.weight_ih_l0'].shape != net.lstm.weight_ih_l0.shape:   # new input channels: zero-init them
            w = torch.zeros_like(net.lstm.weight_ih_l0)
            old = sd['lstm.weight_ih_l0']; nd_old = old.shape[1] - len(STATIC_NAMES)
            w[:, :nd_old] = old[:, :nd_old]; w[:, -len(STATIC_NAMES):] = old[:, nd_old:]
            sd['lstm.weight_ih_l0'] = w
        sd = {k: v for k, v in sd.items() if not k.startswith('st_')}
        net.load_state_dict(sd, strict=False)
        print('init from', cfg['init_from'], flush=True)
    steps, bs = cfg.get('steps', 3000), cfg.get('batch', 32)
    lr0 = cfg.get('lr', 2e-3)
    opt = torch.optim.Adam(net.parameters(), lr=lr0)
    rp = f'{ROOT}/ckpt/{name}.resume.pt'
    step, hist = 0, []
    rng = np.random.default_rng(cfg.get('seed', 0))
    if os.path.exists(rp):
        r = torch.load(rp, weights_only=False)
        net.load_state_dict(r['state']); opt.load_state_dict(r['opt']); step = r['step']; hist = r['hist']
        rng = np.random.default_rng([cfg.get('seed', 0), step])
        if 'np_rng' in r:                                   # resume files written since 27 Nov 2026
            rng.bit_generator.state = r['np_rng']; torch.set_rng_state(r['torch_rng'])
        print('resumed at', step, flush=True)
    N = len(Btr['qo']); maxo = Btr['F'].shape[1] - bd.WIN - 1
    t0 = time.time(); run = []
    net.train()
    while step < steps:
        lr = 1e-4 + 0.5 * (lr0 - 1e-4) * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g['lr'] = lr
        idx = rng.integers(0, N, bs); offs = rng.integers(0, maxo + 1, bs)
        X, Y, S, SIG = bd.batch(Btr, idx, offs, nextday, cfg.get('nowcast', False), rng)
        m, q = net(torch.from_numpy(X), torch.from_numpy(S))
        Yt = torch.from_numpy(Y[:, bd.WARM:]);
        loss, l1, l2 = loss_fn(m[:, bd.WARM:], q[:, bd.WARM:], Yt, torch.from_numpy(SIG))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); step += 1
        run.append([float(loss), float(l1), float(l2)])
        if step % 50 == 0:
            a = np.mean(run, 0); run = []
            print(f'step {step} loss {a[0]:.4f} nse* {a[1]:.4f} pin {a[2]:.4f} lr {lr:.2e} {time.time()-t0:.0f}s', flush=True)
        if step % cfg.get('val_every', 500) == 0 or step == steps:
            ev = sim_eval(net, Bva, nextday, nmax=300, nowcast=cfg.get('nowcast', False))
            hist.append({'step': step, **{k: v for k, v in ev.items() if k != 'nse_all'}})
            print('VAL', hist[-1], flush=True)
        if step % 250 == 0 or step == steps:
            torch.save({'state': net.state_dict(), 'opt': opt.state_dict(), 'step': step, 'hist': hist,
                        'np_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state()},
                       rp + '.tmp'); os.replace(rp + '.tmp', rp)
    torch.save({'state': net.state_dict(), 'cfg': cfg, 'hist': hist}, f'{ROOT}/ckpt/{name}.pt')
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
