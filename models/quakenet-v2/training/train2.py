"""Train QuakeNet v2's graph network on simulated multi-station samples (prep.py). Resumable (checkpoint every 500 steps).
Each batch re-draws the P travel-time residual of every station (3-D Earth; N(0, --res) s) and the record-start
rounding (U(-0.5, 0.5) s), so pick times never repeat; picks = ak135 P time + residual + the simulated picker's error.
Static node/edge features are computed once per sample (featurize); only the pick-time columns change per batch."""
import os, sys, json, time, copy, argparse, numpy as np, torch
from _paths import H; sys.path.insert(0, H + 'src')
import gnn as G, tt as TT

def load(path):
    z = np.load(path); return {k: z[k] for k in z.files}

def cc_feats(pz):
    """P-wave cross-correlation between station pairs on the stored vertical windows (pick-60 s .. pick+180 s; we use
    pick-20 .. pick+100 s): the lag (s, /10) that best aligns j to i around their picks (+-20 s), and the peak |correlation|
    (either polarity: radiation patterns flip it)"""
    K = len(pz); W = pz[:, 40:160].astype(np.float32); W = W - W.mean(1, keepdims=True); W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    out = np.zeros((K, K, 2), np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            c = np.correlate(np.pad(W[i], 20), W[j], 'valid'); k = int(np.argmax(np.abs(c))); lag = k - 20
            out[i, j] = (lag / 10, abs(c[k])); out[j, i] = (-lag / 10, abs(c[k]))
    return out

def featurize(D, real=False, cache=None):
    """per sample: static node (S,10,16) and edge (S,10,10,8) features, mask, base pick term (S,10): pick - ak135 P time at
    the anchor (simulated picks without the residual), and the static targets (S,5; the t0 column filled per batch)"""
    if cache and os.path.exists(cache): return load(cache)
    S = len(D['K']); XN = np.zeros((S, 10, 16), np.float32); XE = np.zeros((S, 10, 10, 8), np.float16); M = np.zeros((S, 10), np.float32)
    BASE = np.zeros((S, 10), np.float32); Y = np.zeros((S, 5), np.float32)
    for s in range(S):
        K = int(D['K'][s]); idx = D['idx'][s, :K]; a = D['anchor'][s]; A = dict(lat=a[0], lon=a[1], logz=a[2], mw=a[3], t0=0.0)
        sl, so = D['n_slat'][idx], D['n_slon'][idx]; dA = TT.dist(A['lat'], A['lon'], sl, so); ttA = TT.tt('P', dA, 10 ** A['logz'])
        tp = D['n_tpick'][idx] if real else D['n_tP'][idx] + D['n_pickerr'][idx]
        R = {k[4:]: D[k][idx] for k in D if k.startswith('n_r_')}
        xn, xe = G.node_edge_features(sl, so, R, ttA, D['n_ls'][idx], D['n_peak'][idx], A, True, cc_feats(D['n_pz'][idx]))
        XN[s, :K] = xn; XE[s, :K, :K] = xe; M[s, :K] = 1; BASE[s, :K] = tp - ttA
        t = D['truth'][s]; Y[s] = G.targets(A, t[0], t[1], t[2], t[3], 0.0)
    F = dict(xn=XN, xe=XE, mask=M, base=BASE, y=Y)
    if cache: np.savez(cache, **F)
    return F

def make_batch(F, D, rows, res, use_time=True, use_cc=False):
    """res: (B, 10) residuals in s (zeros for real). Fills the pick-residual columns: node 13 r_i = (base_i + res_i -
    mean_j(base_j + res_j)) / 10 (clipped +-10), edge 0 = r_j - r_i; target t0 offset = -mean(base + res) / 10."""
    m = F['mask'][rows]; b = F['base'][rows] + res; t0 = (b * m).sum(1) / m.sum(1); r = np.clip((b - t0[:, None]) / 10, -10, 10) * m
    xn = F['xn'][rows].copy(); xe = F['xe'][rows].astype(np.float32); y = F['y'][rows].copy(); y[:, 4] = -t0 / 10
    if use_time: xn[:, :, 13] = r; xe[:, :, :, 0] = (r[:, None, :] - r[:, :, None]) * m[:, :, None] * m[:, None, :]
    else: xn[:, :, 13] = 0; xn[:, :, 14] = 0; xe[:, :, :, 0] = 0
    if not use_cc: xe = xe[..., :6]
    em = D['n_emb'][np.maximum(D['idx'][rows], 0)].astype(np.float32) * m[..., None]
    return [torch.from_numpy(np.ascontiguousarray(v)) for v in (xn, xe, m, em, y)], t0

def picks(D, idx, res):
    """simulated absolute pick times (s after origin): ak135 P + residual + the simulated picker's error"""
    return D['n_tP'][idx] + res + D['n_pickerr'][idx]

def draw_res(rng, B, sig):
    return rng.normal(0, sig, (B, 10)) + rng.uniform(-0.5, 0.5, (B, 10))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--data', default=H + 'bank/tr1_samples.npz')
    ap.add_argument('--steps', type=int, default=4000); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--res', type=float, default=1.5); ap.add_argument('--no_time', action='store_true'); ap.add_argument('--emb', action='store_true'); ap.add_argument('--cc', action='store_true')
    ap.add_argument('--h', type=int, default=64); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--init')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load(a.data); F = featurize(D, cache=a.data.replace('.npz', '_feat.npz')); N = len(D['K'])
    net = G.GNN(fe=8 if a.cc else 6, h=a.h, use_emb=a.emb)
    if a.init: net.load_state_dict(torch.load(H + f'ckpt/{a.init}.pt', weights_only=False)['state'])
    ema = copy.deepcopy(net); opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4); start = 1; RS = H + f'ckpt/{a.name}.resume'; t0 = time.time()
    print(json.dumps(dict(samples=N, params=sum(p.numel() for p in net.parameters()), **vars(a))), flush=True)
    if os.path.exists(RS):
        r = torch.load(RS, weights_only=False); net.load_state_dict(r['net']); ema.load_state_dict(r['ema']); opt.load_state_dict(r['opt']); rng.bit_generator.state = r['rng']; start = r['step'] + 1; print('resumed at', start, flush=True)
    for step in range(start, a.steps + 1):
        lr = a.lr * min(1, step / 200) * 0.5 * (1 + np.cos(np.pi * step / a.steps)); [g.__setitem__('lr', lr) for g in opt.param_groups]
        rows = rng.integers(0, N, a.bs); (xn, xe, m, em, y), _ = make_batch(F, D, rows, draw_res(rng, a.bs, a.res), not a.no_time, a.cc)
        net.train(); loss = G.loss(net(xn, xe, m, em if a.emb else None), y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.995).add_(pn, alpha=0.005)
        if step % 500 == 0: torch.save({'net': net.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'rng': rng.bit_generator.state, 'step': step}, RS)
        if step % 250 == 0: print(json.dumps({'step': step, 'loss': round(loss.item(), 4), 't': round(time.time() - t0)}), flush=True)
    torch.save({'state': ema.state_dict(), 'args': vars(a)}, H + f'ckpt/{a.name}.pt'); print('saved', a.name, flush=True)
    if os.path.exists(RS): os.remove(RS)
