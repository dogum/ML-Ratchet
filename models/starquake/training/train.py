"""Train StarQuake on simulated banks: beta-NLL (beta 0.5) on ln numax and ln dnu, EMA weights, cosine schedule,
resumable checkpoints every 500 steps."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, time, copy, argparse, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M
CK = _P.WORK + '/ckpt/'; BK = _P.WORK + '/bank/'

def load(names):
    import glob; names = sorted(sum([glob.glob(BK + p) for p in names], [])); print('banks', [os.path.basename(p) for p in names], flush=True)
    Z = [np.load(p) for p in names]; out = {k: np.concatenate([z[k] for z in Z]) for k in ('x1', 'x2', 's', 'y')}
    out['det'] = np.concatenate([z['det'] if 'det' in z.files else np.ones(len(z['y']), np.float32) for z in Z]); return out

def nll(o, Y, det=None, beta=0.5):
    """beta-NLL on the oscillating examples; plus, if the net has a detection output, binary cross-entropy on all."""
    n = len(M.OUT); mu, ls = o[:, :n], o[:, n:2 * n].clamp(-7, 3); var = torch.exp(2 * ls)
    l = (0.5 * ((Y - mu) ** 2 / var + 2 * ls) * var.detach() ** beta).sum(1)
    if det is None or o.shape[1] == 2 * n: return l.mean() if det is None else (l * det).sum() / det.sum().clamp(min=1)
    return (l * det).sum() / det.sum().clamp(min=1) + torch.nn.functional.binary_cross_entropy_with_logits(o[:, 2 * n], det)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--init'); ap.add_argument('--arch', default='{}')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load(a.banks); N = len(D['y']); pos = D['det'] > 0.5; mu_y, sd_y = D['y'][pos].mean(0), D['y'][pos].std(0); arch = json.loads(a.arch)
    net = M.StarQuake(**arch)
    if a.init:      # warm start: copy every tensor whose shape matches (a new detection output starts fresh)
        st = torch.load(a.init, weights_only=False)['state']; own = net.state_dict()
        own.update({k: v for k, v in st.items() if k in own and own[k].shape == v.shape}); net.load_state_dict(own)
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time()
    print('bank', N, 'params', sum(p.numel() for p in net.parameters()), flush=True); RS = CK + a.name + '.resume'; start = 1
    if os.path.exists(RS):
        r = torch.load(RS, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); rng.bit_generator.state = r['rng']; start = r['step'] + 1; print('resumed at', start, flush=True)
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        i = np.sort(rng.integers(0, N, a.bs)); x1 = torch.from_numpy(D['x1'][i].astype(np.float32)); x2 = torch.from_numpy(D['x2'][i].astype(np.float32)); s = torch.from_numpy(D['s'][i])
        Y = torch.from_numpy(((D['y'][i] - mu_y) / sd_y).astype(np.float32))
        dt = torch.from_numpy(D['det'][i].astype(np.float32)); net.train(); loss = nll(net(x1, x2, s), Y, dt); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'rng': rng.bit_generator.state, 'step': step}, RS)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'mu_y': mu_y.tolist(), 'sd_y': sd_y.tolist(), 'arch': arch, 'args': vars(a)}, CK + a.name + '.pt'); print('saved', a.name)
    if os.path.exists(RS): os.remove(RS)
