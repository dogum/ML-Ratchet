"""Train TransitNet on simulated banks: cross-entropy on the class, beta-NLL (beta 0.5) on the four planet parameters
(planets only), EMA weights. Augmentation: views mirrored in time half the time (a transit looks the same backwards)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, time, copy, argparse, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M
CK = _P.WORK + '/ckpt/'

def load(paths):
    Z = [np.load(p) for p in paths]; keys = ['local', 'secondary', 'glob', 'oddeven', 's', 'cls', 'y', 'snr'] + (['cent', 'sc'] if all('cent' in z.files for z in Z) else [])
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}

def batch(D, i, flip=None, cent=False):
    keys = ('local', 'secondary', 'glob', 'oddeven') + (('cent',) if cent else ())
    X = [torch.from_numpy(np.nan_to_num(D[k][i].astype(np.float32))) for k in keys]
    if flip is not None and flip.any():
        fl = torch.from_numpy(flip)
        X = [torch.where(fl[:, None, None], torch.flip(x, [-1]), x) for x in X]
    S = torch.from_numpy(np.nan_to_num(D['s'][i]).astype(np.float32))
    return X[:4] + [S] + ([X[4], torch.from_numpy(np.nan_to_num(D['sc'][i]).astype(np.float32))] if cent else [])

def bnll(mu, ls, y, m, beta=0.5):
    var = torch.exp(2 * ls); l = 0.5 * ((y - mu) ** 2 / var + 2 * ls) * var.detach() ** beta; return (l * m).sum() / m.sum().clamp(min=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--init'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=float, default=1.0); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--w_cls', type=float, default=1.0); ap.add_argument('--residual', type=int, default=0); ap.add_argument('--cent', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load(a.banks); N = len(D['cls']); y = D['y'].astype(np.float64); planet = D['cls'] == 0
    if a.residual: y = y - M.base(D['s'])
    mu_y, sd_y = np.nanmean(y[planet], 0), np.nanstd(y[planet], 0)
    net = M.TransitNet(a.width, bool(a.cent))
    if a.init: net.load_state_dict(torch.load(a.init, weights_only=False)['state'])
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time(); ce = torch.nn.CrossEntropyLoss()
    print('bank', N, 'class counts', np.bincount(D['cls']), 'params', sum(p.numel() for p in net.parameters()), flush=True)
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        i = rng.integers(0, N, a.bs); X = batch(D, i, rng.random(a.bs) < 0.5, bool(a.cent)); net.train(); out = net(*X)
        c = torch.from_numpy(D['cls'][i].astype(np.int64)); Y = torch.from_numpy(np.nan_to_num((y[i] - mu_y) / sd_y).astype(np.float32)); m = torch.from_numpy((D['cls'][i] == 0).astype(np.float32))[:, None].expand(-1, 4)
        loss = a.w_cls * ce(out[:, :4], c) + bnll(out[:, 4:8], out[:, 8:12].clamp(-6, 3), Y, m)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 'lr': round(lr, 6), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'mu_y': mu_y.tolist(), 'sd_y': sd_y.tolist(), 'args': vars(a), 'width': a.width, 'residual': a.residual, 'cent': a.cent}, CK + a.name + '.pt'); print('saved', a.name)
