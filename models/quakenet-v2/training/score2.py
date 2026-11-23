"""Score QuakeNet v2 and the baselines on a set of graph samples (simulated frozen set, real dev, or real exam).
Errors: epicentre (km, great circle), depth (|log10 ratio| and km), Mw, origin time (s).
Skill per output = 1 - mean error / mean error of the dumb baseline (station centroid; dev-median depth 77 km and Mw 5.85;
origin time from the centroid). Objective = mean over epicentre, log depth, Mw and origin time of max(0, skill)."""
import sys, json, numpy as np, torch
from _paths import H; sys.path.insert(0, H + 'src')
import gnn as G, tt as TT, locate2 as LC, train2 as T2
Z_MED, MW_MED = 77.0, 5.85; CHI2_68 = 2.2789

def load_net(name):
    ck = torch.load(H + f'ckpt/{name}.pt', weights_only=False); a = ck['args']; net = G.GNN(fe=8 if a.get('cc') else 6, h=a.get('h', 64), use_emb=a.get('emb', False))
    net.load_state_dict(ck['state']); net.eval(); return net, a

_FEAT = {}
def feats(D, real, key=None):
    k = key or id(D)
    if k not in _FEAT: _FEAT[k] = T2.featurize(D, real)
    return _FEAT[k]

def predict(net, a, D, rows, res_sig=None, fixed_res=None, real=False, bs=256, key=None):
    """GNN predictions (dicts, raw error bars) for sample rows; simulated picks use fixed_res * res_sig"""
    F = feats(D, real, key); out = []
    for i in range(0, len(rows), bs):
        r = np.asarray(rows[i:i + bs]); res = np.zeros((len(r), 10))
        if not real:
            for j, s in enumerate(r): K = int(D['K'][s]); res[j, :K] = fixed_res[D['idx'][s, :K]] * res_sig
        (xn, xe, m, em, y), t0 = T2.make_batch(F, D, r, res, not a.get('no_time'), a.get('cc'))
        with torch.no_grad(): o = net(xn, xe, m, em if a.get('emb') else None).numpy()
        for oo, s, tt0 in zip(o, r, t0):
            an = D['anchor'][s]; out.append(G.decode(oo, dict(lat=an[0], lon=an[1], logz=an[2], mw=an[3], t0=float(tt0))))
    return out

def pick_times(D, rows, res_sig=None, fixed_res=None, real=False):
    out = []
    for s in rows:
        K = int(D['K'][s]); idx = D['idx'][s, :K]
        out.append(D['n_tpick'][idx] if real else T2.picks(D, idx, fixed_res[idx] * res_sig))
    return out

def baselines(D, rows, res_sig=None, fixed_res=None, real=False, tt_loc=True):
    """v1.2 combined (anchor), centroid, travel-time locator"""
    TP = pick_times(D, rows, res_sig, fixed_res, real); V, C, L = [], [], []
    for s, tp in zip(rows, TP):
        K = int(D['K'][s]); idx = D['idx'][s, :K]; sl, so = D['n_slat'][idx], D['n_slon'][idx]
        R = {k[4:]: D[k][idx] for k in D if k.startswith('n_r_')}; V.append(LC.combine_v12(sl, so, R, tp)); C.append(LC.centroid(sl, so, tp, Z_MED, MW_MED))
        if tt_loc:
            if real: ts = [None if not np.isfinite(v) else v for v in D['n_tspick'][idx]]
            else:
                se = D['n_serr'][idx]; ts = [None if not np.isfinite(e) else D['n_tS'][i] + fixed_res[i] * res_sig * 1.8 + e for i, e in zip(idx, se)]
            L.append(LC.tt_locate(sl, so, tp, ts))
    return V, C, L

def errors(P, D, rows):
    T = D['truth'][rows]; e = {}
    lat = np.array([p['lat'] for p in P]); lon = np.array([p['lon'] for p in P]); e['epi_km'] = TT.dist(lat, lon, T[:, 0], T[:, 1]) * TT.KM_PER_DEG
    dz = np.array([p.get('depth', np.nan) for p in P], float); e['logz'] = np.abs(np.log10(np.maximum(dz, 1)) - T[:, 2]); e['depth_km'] = np.abs(dz - 10 ** T[:, 2])
    e['mw'] = np.abs(np.array([p.get('mw', np.nan) for p in P], float) - T[:, 3]); e['t0'] = np.abs(np.array([p.get('t0', np.nan) for p in P], float) - T[:, 4])
    if 'cov' in P[0]:
        n, ee = LC.local_offset(lat, lon, T[:, 0], T[:, 1]); m2 = []
        for p, a, b in zip(P, n, ee):
            r = np.array([a, b]); m2.append(r @ np.linalg.solve(p['cov'] + 1e-6 * np.eye(2), r))
        e['epi_m2'] = np.array(m2)
        if 'logz_sig' in P[0]: e['z_ratio'] = e['logz'] / np.array([p['logz_sig'] for p in P]); e['mw_ratio'] = e['mw'] / np.array([p['mw_sig'] for p in P]); e['t0_ratio'] = e['t0'] / np.array([p['t0_sig'] for p in P])
    return e

def summary(e, ref=None):
    s = dict(n=int(len(e['epi_km'])), epi_med_km=float(np.median(e['epi_km'])), epi_mean_km=float(np.mean(e['epi_km'])), depth_med_km=float(np.nanmedian(e['depth_km'])),
             depth_f2=float(np.nanmean(e['logz'] < np.log10(2))), mw_mae=float(np.nanmean(e['mw'])), t0_med_s=float(np.nanmedian(e['t0'])), t0_mean_s=float(np.nanmean(e['t0'])))
    if 'epi_m2' in e: s['cov_epi'] = float(np.mean(e['epi_m2'] < CHI2_68))
    for k in ('z', 'mw', 't0'):
        if k + '_ratio' in e: s['cov_' + k] = float(np.mean(e[k + '_ratio'] < 1))
    if ref is not None:
        sk = {k: float(1 - np.nanmean(e[k]) / np.nanmean(ref[k])) for k in ('epi_km', 'logz', 'mw', 't0')}; s['skill'] = sk; s['objective'] = float(np.mean([max(0, v) for v in sk.values()]))
    return s

def widen_factors(e):
    """widen-only factors fitted on dev so that 68% of truths fall inside the bars"""
    f = {'epi': max(1.0, float(np.sqrt(np.quantile(e['epi_m2'], 0.68) / CHI2_68)))}
    for k in ('z', 'mw', 't0'): f[k] = max(1.0, float(np.quantile(e[k + '_ratio'], 0.68)))
    return f

def apply_factors(P, f):
    for p in P:
        p['cov'] = p['cov'] * f['epi'] ** 2; p['logz_sig'] *= f['z']; p['mw_sig'] *= f['mw']; p['t0_sig'] *= f['t0']
    return P
