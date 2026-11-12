"""Train SwellNet on simulated banks: beta-NLL (beta 0.5) on log10 D, t0 and log10 storm Hs, von Mises NLL on direction
(only where the background window has directional data), BCE detection loss (a share --pnull of each batch is replaced by
real null windows; injected swells count as positives when their ridge contrast >= 0.3 dex, weaker ones are left out), EMA weights, resumable every 500 steps."""
import os, sys, json, time, copy, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import model as M
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CK = os.path.join(ROOT, 'ckpt/'); BANK = os.path.join(ROOT, 'bank/')

def load(names):
    X = [np.load(BANK + n + '_x.npy', mmap_mode='r') for n in names]; Y = [np.load(BANK + n + '_y.npy') for n in names]
    return X, Y

def losses(o, Y, sd, cd, wd, isnull, det_t, det_w, beta=0.5):
    n = len(M.OUT); mu, ls = o[:, :n], o[:, n:2 * n].clamp(-6, 3); var = torch.exp(2 * ls); reg = 1 - isnull
    l = ((0.5 * ((Y - mu) ** 2 / var + 2 * ls) * var.detach() ** beta).mean(1) * reg).sum() / reg.sum().clamp(min=1)
    wd = wd * reg
    ld = (torch.nn.functional.binary_cross_entropy_with_logits(o[:, 2 * n + 2], det_t, reduction='none') * det_w).sum() / det_w.sum().clamp(min=1)
    v = o[:, 2 * n:2 * n + 2]; kap = torch.sqrt((v ** 2).sum(1) + 1e-6); ang = torch.atan2(v[:, 0], v[:, 1]); th = torch.atan2(sd, cd)
    vm = ((-kap * torch.cos(th - ang) + torch.log(torch.special.i0e(kap)) + kap) * wd).sum() / wd.sum().clamp(min=1)
    return l + 0.5 * vm + 0.5 * ld

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--init'); ap.add_argument('--null', default='null_train1'); ap.add_argument('--pnull', type=float, default=0.25)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); os.makedirs(CK, exist_ok=True)
    XS, YS = load(a.banks); sizes = np.array([len(y) for y in YS]); N = sizes.sum(); Yall = np.concatenate(YS); T, DIR = M.targets(Yall)
    mu_y, sd_y = T.mean(0), T.std(0)
    net = M.SwellNet()
    if a.init: sd0 = torch.load(a.init, weights_only=False); net.load_state_dict(sd0['state']); mu_y, sd_y = np.array(sd0['mu_y']), np.array(sd0['sd_y'])
    net.mu_y[:] = torch.tensor(mu_y); net.sd_y[:] = torch.tensor(sd_y)
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time()
    print('bank', N, 'params', sum(p.numel() for p in net.parameters()), flush=True); RS = CK + a.name + '.resume'; start = 1
    if os.path.exists(RS):
        r = torch.load(RS, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); rng.bit_generator.state = r['rng']; start = r['step'] + 1; print('resumed at', start, flush=True)
    off = np.concatenate([[0], np.cumsum(sizes)]); XN = np.load(BANK + a.null + '_x.npy', mmap_mode='r'); print('null windows', len(XN), flush=True)
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 300) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        gi = np.sort(rng.integers(0, N, a.bs)); bk = np.searchsorted(off, gi, side='right') - 1
        xb = np.stack([XS[k][i - off[k]] for k, i in zip(bk, gi)])
        nl = rng.random(a.bs) < a.pnull; ni = rng.integers(0, len(XN), nl.sum())
        if nl.any(): xb[nl] = XN[np.sort(ni)]
        con = Yall[gi, 4]; det_t = torch.from_numpy((~nl & (con >= 0.3)).astype(np.float32)); det_w = torch.from_numpy((nl | (con >= 0.3)).astype(np.float32))
        isnull = torch.from_numpy(nl.astype(np.float32))
        x = torch.from_numpy(M.features(xb)); Y = torch.from_numpy(((T[gi] - mu_y) / sd_y).astype(np.float32))
        th = np.radians(DIR[gi]); sd, cd = torch.from_numpy(np.sin(th).astype(np.float32)), torch.from_numpy(np.cos(th).astype(np.float32))
        wd = torch.from_numpy((np.isfinite(xb[:, 1].astype(np.float32)).mean((1, 2)) > 0.5).astype(np.float32))
        net.train(); loss = losses(net(x), Y, sd, cd, wd, isnull, det_t, det_w); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'rng': rng.bit_generator.state, 'step': step}, RS)
        if step % 100 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'mu_y': mu_y.tolist(), 'sd_y': sd_y.tolist(), 'args': vars(a)}, CK + a.name + '.pt'); print('saved', a.name)
    if os.path.exists(RS): os.remove(RS)
