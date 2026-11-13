"""Train CellSense NMC811 on simulated banks only. Gaussian heads with beta-NLL (beta 0.5), EMA weights, and augmentation that
mimics the cycler and the way the curves are used:
  - capacity gain error (0.2%) and noise; a voltage offset of a few mV (the curve slides along the voltage axis)
  - the reference curve withheld 30% of the time
  - p_trunc of the readings become partial discharges from full that stop after a random depth (1.0 Ah to the end)
--target rest: the capacity head learns the charge still to come below where the curve stops (q_total minus the last
recorded capacity), a correction to a model-free reading, instead of the capacity itself
Resumable: a checkpoint is written every 500 steps and picked up again if the run restarts.

    python nmc_train.py --name n1 --banks bank_textbook_*.npz [--steps 8000]
"""
import _paths as _P
import os, sys, json, time, copy, argparse, numpy as np, torch
sys.path.insert(0, _P.SRC); import nmc_model as NM
CK = os.path.join(_P.WORK, 'nmc', 'ckpt'); os.makedirs(CK, exist_ok=True)
DV = 0.0066265       # grid spacing (V)


def load_bank(paths):
    Z = [np.load(p) for p in paths]; lab = [str(x) for x in Z[0]['labels']]; j = [lab.index(k) for k in NM.LABELS]
    return {'Q0': np.concatenate([z['Q0'] for z in Z]), 'Q1': np.concatenate([z['Q1'] for z in Z]), 'y': np.concatenate([z['y'][:, j] for z in Z])}


def vshift(Q, dv):
    """the curve as a cycler with a voltage offset dv (V) would log it: Q at grid voltage V is the true Q at V + dv"""
    f = dv / DV; k = np.floor(f).astype(int); w = (f - k)[:, None]; L = Q.shape[1]; idx = np.arange(L)[None]
    a = np.clip(idx - k[:, None], 0, L - 1); b = np.clip(idx - k[:, None] - 1, 0, L - 1)        # grid descends: V + dv sits dv/DV points earlier
    return np.take_along_axis(Q, a, 1) * (1 - w) + np.take_along_axis(Q, b, 1) * w


def augment(rng, Q1, Q0, p_ref=0.7):
    B = len(Q1); g = 1 + rng.normal(0, 0.002, (B, 1)); Q1 = Q1 * g + rng.normal(0, 2e-4, Q1.shape); Q0 = Q0 * (g + rng.normal(0, 5e-4, (B, 1))) + rng.normal(0, 2e-4, Q0.shape)
    Q1 = vshift(Q1, rng.normal(0, 0.003, B)); Q0 = vshift(Q0, rng.normal(0, 0.003, B))
    drop = rng.random(B) > p_ref; Q0 = np.where(drop[:, None], np.nan, Q0); return Q1, Q0


def truncate(Q, D):
    """a partial discharge from full that stops after delivering D Ah: grid points beyond it are missing"""
    return np.where(Q > D[:, None], np.nan, Q)


def trunc_aug(rng, Q1, p, lo=1.0):
    B = len(Q1); m = rng.random(B) < p; end = np.nanmax(Q1, 1); D = rng.uniform(lo, end); return truncate(Q1, np.where(m, D, np.inf))


def bnll(mu, ls, y, beta=0.5):
    var = torch.exp(2 * ls); l = 0.5 * ((y - mu) ** 2 / var + 2 * ls); return (l * var.detach() ** beta).mean()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--widths', default='32,48,64,96,128'); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--p_trunc', type=float, default=0.7); ap.add_argument('--smooth', type=float, default=0)
    ap.add_argument('--target', default='total', help="'total': capacity itself; 'rest': the charge still to come below where the curve stops (q_total - last recorded Q)")
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load_bank([os.path.join(_P.WORK, 'nmc', b) if not os.path.isabs(b) else b for b in a.banks]); N = len(D['y']); y = D['y'].astype(np.float64)
    mu_y, sd_y = y.mean(0), y.std(0) + 1e-6; nl = len(NM.LABELS)
    if a.target == 'rest':        # scale of the remaining charge, from one fixed draw of truncations
        r0 = np.random.default_rng(1); j = r0.integers(0, N, 20000); rest = y[j, 0] - NM.q_last(trunc_aug(r0, D['Q1'][j].astype(np.float64), a.p_trunc))
        mu_y[0], sd_y[0] = rest.mean(), rest.std() + 1e-6
    net = NM.Net(tuple(int(w) for w in a.widths.split(','))); ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    path = os.path.join(CK, a.name + '.pt'); res = path + '.resume'; start = 1
    if os.path.exists(res):
        r = torch.load(res, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); start = r['step'] + 1
        rng = np.random.default_rng(a.seed + start); print('resumed at', start, flush=True)
    print('bank', N, 'params', sum(p.numel() for p in net.parameters()), flush=True); t0 = time.time()
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * min(1, step / a.steps)))
        for g in opt.param_groups: g['lr'] = lr
        i = rng.integers(0, N, a.bs); Q1, Q0 = augment(rng, D['Q1'][i].astype(np.float64), D['Q0'][i].astype(np.float64))
        Q1 = trunc_aug(rng, Q1, a.p_trunc); yi = y[i].copy()
        if a.target == 'rest': yi[:, 0] = yi[:, 0] - NM.q_last(Q1)
        X = torch.from_numpy(NM.features(Q1, Q0, smooth=a.smooth)); Y = torch.from_numpy(((yi - mu_y) / sd_y).astype(np.float32))
        out = net(X); loss = bnll(out[:, :nl], out[:, nl:].clamp(-7, 3), Y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(float(loss.detach()), 4), 'lr': round(lr, 6), 't': round(time.time() - t0)}), flush=True)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'step': step}, res)
    torch.save({'state': ema.state_dict(), 'mu_y': mu_y.tolist(), 'sd_y': sd_y.tolist(), 'args': vars(a), 'labels': NM.LABELS, 'widths': a.widths, 'smooth': a.smooth, 'target': a.target}, path)
    if os.path.exists(res): os.remove(res)
    print('saved', path, flush=True)
