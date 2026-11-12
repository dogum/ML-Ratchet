"""Train BeatNet on simulated banks only: cross-entropy (label smoothing 0.05) on N/S/V/F plus Gaussian NLL on log QRS
width; EMA weights; resumable checkpoints every 500 steps."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, time, copy, glob, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, _P.SRC); import model as M
CK = _P.WORK + '/ckpt/'; os.makedirs(CK, exist_ok=True)
QMU, QSD = np.log(100.0), 0.35         # QRS width target normalisation (log ms)

def load(pattern):
    Z = [np.load(p) for p in sorted(glob.glob(pattern))]; return {k: np.concatenate([z[k] for z in Z]) for k in ('w', 'tpl', 'rr', 'y', 'qrs', 'info')}

def inputs(D, i, cin):
    x = D['w'][i][:, None] if cin == 1 else np.stack([D['w'][i], D['tpl'][i]], 1); return x.astype(np.float32), np.nan_to_num(D['rr'][i]).astype(np.float32)

def loss_fn(o, y, q, af=None):
    ce = F.cross_entropy(o[:, :4], y, label_smoothing=0.05); mu, ls = o[:, 4], o[:, 5].clamp(-4, 2)
    t = (torch.log(q) - QMU) / QSD; nll = (0.5 * (t - mu) ** 2 * torch.exp(-2 * ls) + ls).mean(); L = ce + 0.3 * nll
    if af is not None and o.shape[1] > 6: L = L + 0.3 * F.binary_cross_entropy_with_logits(o[:, 6], af)   # base rhythm = atrial fibrillation
    return L, ce

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--bank', required=True); ap.add_argument('--cin', type=int, default=1)
    ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--init'); ap.add_argument('--nout', type=int, default=6)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load(a.bank); N = len(D['y']); net = M.BeatNet(cin=a.cin, nout=a.nout)
    if a.init:   # warm start from the incumbent; tensors whose shape changed (extra input channel, extra output) keep the overlap, the rest starts at zero / fresh init
        src = torch.load(a.init, weights_only=False)['state']; dst = net.state_dict()
        for k, v in src.items():
            d = dst[k]
            if d.shape == v.shape: dst[k] = v.clone(); continue
            if d.dim() == 3 and d.shape[0] == v.shape[0] and d.shape[1] > v.shape[1]: n = torch.zeros_like(d); n[:, :v.shape[1]] = v; dst[k] = n   # new input channel: zero weights
            elif d.dim() >= 1 and d.shape[0] > v.shape[0]: n = d.clone(); n[:v.shape[0]] = v; dst[k] = n                                           # new output row: fresh init
        net.load_state_dict(dst); print('warm start from', a.init, flush=True)
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time()
    print('bank', N, np.bincount(D['y'], minlength=4).tolist(), 'params', sum(p.numel() for p in net.parameters()), flush=True); RS = CK + a.name + '.resume'; start = 1
    if os.path.exists(RS):
        r = torch.load(RS, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); rng.bit_generator.state = r['rng']; start = r['step'] + 1; print('resumed at', start, flush=True)
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        i = rng.integers(0, N, a.bs); x, r = inputs(D, i, a.cin)
        x = x * rng.uniform(0.8, 1.25, (a.bs, 1, 1)).astype(np.float32); sh = int(rng.integers(-3, 4)); x = np.roll(x, sh, -1)
        o = net.train()(torch.from_numpy(x), torch.from_numpy(r)); loss, ce = loss_fn(o, torch.from_numpy(D['y'][i].astype(np.int64)), torch.from_numpy(D['qrs'][i]), torch.from_numpy((D['info'][i, 1] == 1).astype(np.float32)))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'rng': rng.bit_generator.state, 'step': step}, RS)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 'ce': round(ce.item(), 4), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'cin': a.cin, 'nout': a.nout, 'args': vars(a)}, CK + a.name + '.pt'); print('saved', a.name)
    if os.path.exists(RS): os.remove(RS)
