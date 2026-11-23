"""Scoring helpers: frozen simulated set skill, dev-event errors, posterior-predictive amplitude forecasts."""
import os, json, numpy as np, torch
from _paths import BANK, DATA
from model import TsunamiNet
import tsim, realdata
torch.set_num_threads(1)

def load_net(rung):
    st = torch.load(os.path.join(BANK, 'ckpt', rung + '.pt'))['net']
    extra = st['ctx.0.weight'].shape[1] - 257
    net = TsunamiNet(extra=extra); net.load_state_dict(st); net.eval(); return net

@torch.no_grad()
def posterior(net, X, S, M, n=400, seed=0):
    g = torch.Generator().manual_seed(seed); out = []
    for i in range(len(X)):
        z = torch.randn(n, 4, generator=g)
        x = torch.from_numpy(np.asarray(X[i:i + 1], np.float32)); s = torch.from_numpy(np.asarray(S[i:i + 1], np.float32)); m = torch.from_numpy(np.asarray(M[i:i + 1]))
        e = None
        if net.extra:
            import features as F
            e = torch.from_numpy(F.nnls_features(np.asarray(X[i], np.float32), np.asarray(M[i]))[None])
        out.append(net.sample(x, s, m, n, z, e).numpy())
    return np.stack(out)

def frozen_score(net, bank='val_p0', n=200, limit=None):
    z = np.load(os.path.join(BANK, f'bank_{bank}.npz')); X, S, M, TH = z['X'], z['S'], z['M'], z['TH']
    if limit: X, S, M, TH = X[:limit], S[:limit], M[:limit], TH[:limit]
    P = posterior(net, X, S, M, n)
    med = np.median(P, 1); lo, hi = np.percentile(P, 16, 1), np.percentile(P, 84, 1)
    out = {}
    sk = []
    for k, name in [(0, 'mw'), (1, 'sc')]:
        mae = np.abs(med[:, k] - TH[:, k]).mean(); ref = np.abs(np.median(TH[:, k]) - TH[:, k]).mean()
        out[name + '_mae'] = float(mae); out[name + '_skill'] = float(1 - mae / ref); sk.append(1 - mae / ref)
        out[name + '_cov68'] = float(((TH[:, k] >= lo[:, k]) & (TH[:, k] <= hi[:, k])).mean())
    out['score'] = float(np.mean(sk)); out['n'] = int(len(X))
    out['mw_mae_by_size'] = {f'{a}-{b}': float(np.abs(med[(TH[:, 0] >= a) & (TH[:, 0] < b), 0] - TH[(TH[:, 0] >= a) & (TH[:, 0] < b), 0]).mean()) for a, b in [(7.3, 8.0), (8.0, 8.6), (8.6, 9.3)]}
    return out

def amp_forecast(gf, geo, prep, P, delay_frac=0.0, n_use=100, gain=1.0):
    """Posterior-predictive max |eta| at each target from n_use posterior samples. Returns list of (median, p16, p84) in m."""
    res = []
    S = P[np.linspace(0, len(P) - 1, n_use).astype(int)]
    slips = [tsim.theta_slip(geo, prep['frame'], th) for th in S]
    for t in prep['targets']:
        d = t['ta'] * delay_frac
        a = gain * np.array([realdata.predict_amp(gf, t['site'], u, s, t['a'], t['b'], t['mask'], d)[0] for u, s in slips])
        res.append((float(np.median(a)), float(np.percentile(a, 16)), float(np.percentile(a, 84)), a))
    return res
