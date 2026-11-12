"""Train BearingNet on simulated banks only. Loss: cross-entropy (label smoothing 0.02) on the class + Gaussian beta-NLL on
speed (all) and size (faulty only). EMA weights, cosine LR, resumable every 500 steps."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, time, copy, argparse, numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, _P.SRC); import model as M, bn_simbank as SB
CK = _P.WORK + '/ckpt/'

def nll(mu, ls, y, beta=0.5):
    ls = ls.clamp(-5, 3); var = torch.exp(2 * ls); return 0.5 * ((y - mu) ** 2 / var + 2 * ls) * var.detach() ** beta

def loss_fn(o, c, sp, sz, cw=None):
    ce = Fn.cross_entropy(o[:, :4], c, weight=cw, label_smoothing=0.02)
    ls = nll(o[:, 4], o[:, 5], sp).mean(); f = (c > 0).float()
    lz = (nll(o[:, 6], o[:, 7], sz) * f).sum() / f.sum().clamp(min=1)
    return ce + 0.3 * ls + 0.2 * lz, ce

def targets(D):
    return D['cls'].astype(np.int64), (D['lsp'] / M.SP_SCALE).astype(np.float32), ((D['lsize'] - M.SZ_MU) / M.SZ_SD).astype(np.float32)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True)
    ap.add_argument('--steps', type=int, default=12000); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--init'); ap.add_argument('--chdrop', type=float, default=0.0); ap.add_argument('--arch', default='base')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    Ds = [SB.load(b) for b in a.banks]; D = {k: np.concatenate([d[k] for d in Ds]) for k in Ds[0]}; N = len(D['cls']); C, SP, SZ = targets(D)
    net = M.ARCH[a.arch]()
    if a.init: net.load_state_dict(torch.load(a.init, weights_only=False)['state'])
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time(); os.makedirs(CK, exist_ok=True)
    print('bank', N, 'params', sum(p.numel() for p in net.parameters()), 'classes', np.bincount(C), flush=True); RS = CK + a.name + '.resume'; start = 1
    if os.path.exists(RS):
        r = torch.load(RS, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); rng.bit_generator.state = r['rng']; start = r['step'] + 1; print('resumed at', start, flush=True)
    run = []
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        i = rng.integers(0, N, a.bs); x = torch.from_numpy(D['x'][i].astype(np.float32)); s = torch.from_numpy(D['s'][i])
        if a.chdrop:   # randomly blank whole demodulation bands so the net cannot lean on one band of one machine
            m = torch.from_numpy((rng.random((a.bs, 5)) > a.chdrop).astype(np.float32)); m[:, -1] = 1
            x[:, :20] *= m.repeat_interleave(4, 1)[:, :, None]
        net.train(); o = net(x, s); loss, ce = loss_fn(o, torch.from_numpy(C[i]), torch.from_numpy(SP[i]), torch.from_numpy(SZ[i]))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); run.append(ce.item())
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'rng': rng.bit_generator.state, 'step': step}, RS)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 'ce': round(float(np.mean(run[-250:])), 4), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'args': vars(a)}, CK + a.name + '.pt'); print('saved', a.name)
    if os.path.exists(RS): os.remove(RS)
