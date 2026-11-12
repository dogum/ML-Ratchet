"""Train CellSense on the simulated bank. Gaussian heads with beta-NLL (beta = 0.5), EMA weights, augmentation that mimics
the cycler: capacity gain error and noise, thermocouple offset and noise, and the reference curve withheld 25% of the time."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, time, copy, argparse, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M
CK = _P.WORK + '/ckpt/'

def load_bank(paths):
    """health banks (no cycle number) and lifetime banks can be mixed: labels a bank lacks are NaN, cycle/charging-time NaN"""
    Z = [np.load(p) for p in paths]; labs = max(([str(x) for x in z['labels']] for z in Z), key=len); out = {k: [] for k in ('Q0', 'T0c', 'Q1', 'T1c', 'y', 's')}
    for z in Z:
        n = len(z['y']); y = np.full((n, len(labs)), np.nan, np.float32); y[:, :z['y'].shape[1]] = z['y']
        for k in ('Q0', 'T0c', 'Q1', 'T1c'): out[k].append(z[k])
        out['y'].append(y); out['s'].append(z['s'] if 's' in z.files else np.full((n, 2), np.nan, np.float32))
    D = {k: np.concatenate(v) for k, v in out.items()}; D['labels'] = labs; D['has_s'] = bool(np.isfinite(D['s']).any()); return D

def scalars(s):   # cycle number (log), charging time 0-80% (min), and whether they are known
    s = np.asarray(s, np.float64); ok = np.isfinite(s).all(1)
    return np.stack([np.where(ok, np.log10(np.maximum(np.nan_to_num(s[:, 0], nan=1), 1)) / 3.5, 0), np.where(ok, np.nan_to_num(s[:, 1]) / 15.0, 0), ok.astype(float)], 1).astype(np.float32)

def augment(rng, Q1, T1, Q0, T0, p_ref=0.75):
    B = len(Q1); g = 1 + rng.normal(0, 0.003, (B, 1)); Q1 = Q1 * g + rng.normal(0, 0.0003, Q1.shape); Q0 = Q0 * (g + rng.normal(0, 0.001, (B, 1))) + rng.normal(0, 0.0003, Q0.shape)
    off = rng.normal(0, 0.8, (B, 1)); T1 = T1 + off + rng.normal(0, 0.08, T1.shape); T0 = T0 + off + rng.normal(0, 0.3, (B, 1)) + rng.normal(0, 0.08, T0.shape)
    drop = rng.random(B) > p_ref; Q0 = np.where(drop[:, None], np.nan, Q0); T0 = np.where(drop[:, None], np.nan, T0)
    return Q1, T1, Q0, T0

def truncate(Q, T, D):
    """a partial discharge from full that stops after delivering D Ah: grid points beyond it are missing (NaN)"""
    cut = Q > D[:, None]; return np.where(cut, np.nan, Q), np.where(cut, np.nan, T)

def trunc_aug(rng, Q1, T1, Q0, T0, p=0.5, lo=0.25):
    """half the readings become partial discharges of random depth (25-100% of what the curve delivers); for half of those
    the reference curve is cut at the same depth too (a cell that only ever sees partial discharges)"""
    B = len(Q1); m = rng.random(B) < p; D = np.nanmax(Q1, 1) * rng.uniform(lo, 1.0, B); D = np.where(m, D, np.inf)
    Q1, T1 = truncate(Q1, T1, D); both = m & (rng.random(B) < 0.5); D0 = np.where(both, D, np.inf); Q0, T0 = truncate(Q0, T0, D0)
    return Q1, T1, Q0, T0

def bnll(mu, ls, y, beta=0.5):
    m = torch.isfinite(y); y = torch.where(m, y, torch.zeros_like(y)); var = torch.exp(2 * ls); l = 0.5 * ((y - mu) ** 2 / var + 2 * ls)
    return (l * var.detach() ** beta * m).sum() / m.sum().clamp(min=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--banks', nargs='+', required=True); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--init', default=None); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--widths', default='48,64,96,128,160'); ap.add_argument('--warm', type=int, default=200)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--smooth', type=float, default=0); ap.add_argument('--p_trunc', type=float, default=0)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load_bank(a.banks); N = len(D['y']); y = D['y'].astype(np.float64)
    mu_y, sd_y = np.nanmean(y, 0), np.nanstd(y, 0) + 1e-6
    LAB = D['labels']; extra = 3 if D['has_s'] else 0
    net = M.CellSense(tuple(int(w) for w in a.widths.split(',')), n_out=len(LAB), extra=extra)
    if a.init:
        ck = torch.load(a.init, weights_only=False); own = net.state_dict()
        for k, v in ck['state'].items():
            if k in own and own[k].shape == v.shape: own[k] = v
            elif k in own and k.startswith('head.0.weight'): own[k][:, :v.shape[1]] = v           # new scalar inputs start at zero weight
            elif k in own and k.startswith('head.2'):                                                  # more outputs: keep mu rows and log-sigma rows aligned
                no, nn_ = v.shape[0] // 2, own[k].shape[0] // 2; own[k][:no] = v[:no]; own[k][nn_:nn_ + no] = v[no:]
        net.load_state_dict(own)
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); t0 = time.time()
    print('bank', N, 'params', sum(p.numel() for p in net.parameters()), flush=True)
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / a.warm) * 0.5 * (1 + np.cos(np.pi * min(1, step / a.steps))); [g.__setitem__('lr', lr) for g in opt.param_groups]
        i = rng.integers(0, N, a.bs); Q1, T1, Q0, T0 = augment(rng, D['Q1'][i], D['T1c'][i], D['Q0'][i], D['T0c'][i])
        if a.p_trunc: Q1, T1, Q0, T0 = trunc_aug(rng, Q1, T1, Q0, T0, a.p_trunc)
        X = torch.from_numpy(M.features(Q1, T1, Q0, T0, smooth=a.smooth)); Y = torch.from_numpy(((y[i] - mu_y) / sd_y).astype(np.float32))
        out = net(X, torch.from_numpy(scalars(D['s'][i])) if extra else None); n = Y.shape[1]; loss = bnll(out[:, :n], out[:, n:].clamp(-7, 3), Y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.998).add_(pn, alpha=0.002)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if step % 200 == 0: print(json.dumps({'step': step, 'loss': float(loss), 'lr': lr, 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'mu_y': mu_y.tolist(), 'sd_y': sd_y.tolist(), 'args': vars(a), 'labels': LAB, 'widths': a.widths, 'extra': extra, 'smooth': a.smooth}, CK + a.name + '.pt'); print('saved', a.name)
