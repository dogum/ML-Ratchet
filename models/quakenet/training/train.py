"""Train QuakeNet on simulated banks: beta-NLL (beta 0.5) on Mw, distance and log10 depth, von Mises NLL on back-azimuth,
EMA weights."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, time, copy, argparse, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M
CK = _P.WORK + '/ckpt/'

def load(paths):
    Z = [np.load(p) for p in paths]; return {k: np.concatenate([z[k] for z in Z]) for k in ('x', 'ls', 'y', 'peak')}

def targets(y):
    return np.stack([y[:, 0], y[:, 1], np.log10(np.maximum(y[:, 2], 1.0))], 1)

def scal(ls, peak):
    return np.stack([(np.asarray(ls) + 6) / 2, np.log10(np.maximum(np.asarray(peak, float), 1.0))], 1).astype(np.float32)

def losses(o, Y, sb, cb, beta=0.5):
    n = len(M.OUT); mu, ls = o[:, :n], o[:, n:2 * n].clamp(-6, 3); var = torch.exp(2 * ls)
    l = (0.5 * ((Y - mu) ** 2 / var + 2 * ls) * var.detach() ** beta).mean()
    v = o[:, 2 * n:2 * n + 2]; kap = torch.sqrt((v ** 2).sum(1) + 1e-6); ang = torch.atan2(v[:, 0], v[:, 1]); th = torch.atan2(sb, cb)
    vm = (-kap * torch.cos(th - ang) + torch.log(torch.special.i0e(kap)) + kap).mean()
    return l + 0.5 * vm

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=1.5e-3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--init')
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load(a.banks); N = len(D['y']); T = targets(D['y']); mu_y, sd_y = T.mean(0), T.std(0); S = scal(D['ls'], D['peak'])
    net = M.QuakeNet()
    if a.init: net.load_state_dict(torch.load(a.init, weights_only=False)['state'])
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time()
    print('bank', N, 'params', sum(p.numel() for p in net.parameters()), flush=True); RS = CK + a.name + '.resume'; start = 1
    if os.path.exists(RS):     # the sandbox can stop a long job: pick up from the last 500-step checkpoint
        r = torch.load(RS, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); rng.bit_generator.state = r['rng']; start = r['step'] + 1; print('resumed at', start, flush=True)
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        i = rng.integers(0, N, a.bs); x = torch.from_numpy(D['x'][i].astype(np.float32)); s = torch.from_numpy(S[i])
        Y = torch.from_numpy(((T[i] - mu_y) / sd_y).astype(np.float32)); sb, cb = torch.from_numpy(D['y'][i, 3]), torch.from_numpy(D['y'][i, 4])
        net.train(); loss = losses(net(x, s), Y, sb, cb); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'rng': rng.bit_generator.state, 'step': step}, RS)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'mu_y': mu_y.tolist(), 'sd_y': sd_y.tolist(), 'args': vars(a)}, CK + a.name + '.pt'); print('saved', a.name)
    if os.path.exists(RS): os.remove(RS)
