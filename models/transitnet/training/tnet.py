"""Load TransitNet and run it (both time directions averaged); scoring on the frozen simulated set, the real dev set and the
real exam."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M
D = _P.WORK + '/data/'; B = _P.WORK + '/bank/'

def load_net(path):
    ck = torch.load(path, weights_only=False); net = M.TransitNet(ck.get('width', 1.0), bool(ck.get('cent', 0))); net.load_state_dict(ck['state']); net.eval()
    net.mu_y, net.sd_y = np.array(ck['mu_y']), np.array(ck['sd_y']); net.residual = ck.get('residual', 0); return net

def predict(net, Z, bs=512):
    """Z: dict with local, secondary, glob, oddeven, s -> class probabilities and planet parameters with 1-sigma errors"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(Z['s']), bs):
            X = [torch.from_numpy(np.asarray(Z[k][i:i + bs], np.float32)) for k in ('local', 'secondary', 'glob', 'oddeven')]; S = torch.from_numpy(np.nan_to_num(np.asarray(Z['s'][i:i + bs], np.float32)))
            ex = [torch.from_numpy(np.nan_to_num(np.asarray(Z['cent'][i:i + bs], np.float32))), torch.from_numpy(np.asarray(Z['sc'][i:i + bs], np.float32))] if net.use_cent else []
            o1 = net(*X, S, *ex); o2 = net(*[torch.flip(x, [-1]) for x in X], S, *([torch.flip(ex[0], [-1]), ex[1]] if ex else [])); outs.append(torch.stack([o1, o2]).numpy())
    o = np.concatenate(outs, 1); pr = np.exp(o[:, :, :4] - o[:, :, :4].max(-1, keepdims=True)); pr = (pr / pr.sum(-1, keepdims=True)).mean(0)
    mu = o[:, :, 4:8].mean(0) * net.sd_y + net.mu_y + (M.base(Z['s']) if getattr(net, 'residual', 0) else 0); sg = np.sqrt((np.exp(2 * np.clip(o[:, :, 8:12], -6, 3)) * net.sd_y ** 2).mean(0) + o[:, :, 4:8].var(0) * net.sd_y ** 2)
    R = {'p_' + c: pr[:, j] for j, c in enumerate(M.CLASSES)}
    for j, p in enumerate(M.PARAMS): R[p] = mu[:, j]; R[p + '_sig'] = sg[:, j]
    return R

def auc(score, pos):
    s = np.asarray(score); pos = np.asarray(pos, bool); r = np.argsort(np.argsort(s)) + 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

VAL = B + 'val2.npz'     # frozen simulated set, versioned with the physics: val.npz (v1), val2.npz (v2: limb darkening from the dev planets, centroids)

def val(net, path=None):
    path = path or VAL
    Z = np.load(path); P = predict(net, Z); cls = Z['cls']; y = Z['y']; ok = (cls == 0) & (Z['snr'] >= 7); out = {'auc': auc(P['p_planet'], cls == 0)}
    for j, p in enumerate(M.PARAMS):
        e = np.abs(P[p][ok] - y[ok, j]); base = np.abs(y[ok, j] - np.median(y[ok, j])).mean(); out[p] = {'mae': float(e.mean()), 'skill': float(1 - e.mean() / base), 'in_1sigma': float(np.mean(e < P[p + '_sig'][ok]))}
    out['objective'] = float(0.5 * (2 * out['auc'] - 1) + 0.5 * np.mean([max(0, out[p]['skill']) for p in M.PARAMS])); return out

def real_arrays(idx):
    Z = np.load(D + 'real.npz'); return {k: Z[k][idx] for k in ('local', 'secondary', 'glob', 'oddeven', 's', 'cent', 'sc') if k in Z.files}

def param_errors(meta, P):
    k, b, aR = (np.array([m.get(x, np.nan) for m in meta], float) for x in ('k', 'b', 'aR'))
    return {'k_frac': np.abs(10 ** P['log_k'] / k - 1), 'b_abs': np.abs(P['b'] - b), 'aR_frac': np.abs(10 ** P['log_aR'] / aR - 1)}

def dev(net):
    S = json.load(open(D + 'split.json')); Mt = json.load(open(D + 'real_meta.json')); idx = S['dev']; meta = [Mt[i] for i in idx]; P = predict(net, real_arrays(idx))
    isp = np.array([m['kind'] == 'planet' for m in meta]); pp = np.clip(P['p_planet'], 1e-4, 1 - 1e-4)
    E = param_errors([m for m in meta if m['kind'] == 'planet'], {k: v[isp] for k, v in P.items()})
    return {'n': len(idx), 'logloss': float(-np.mean(np.where(isp, np.log(pp), np.log(1 - pp)))), 'acc': float(np.mean((pp > 0.5) == isp)), 'auc': auc(pp, isp),
            'k_frac_med': float(np.nanmedian(E['k_frac'])), 'b_abs_med': float(np.nanmedian(E['b_abs'])), 'aR_frac_med': float(np.nanmedian(E['aR_frac']))}

def exam(net, min_snr=7):
    S = json.load(open(D + 'split.json')); Mt = json.load(open(D + 'real_meta.json')); idx = [i for i in S['exam'] if Mt[i]['info']['snr'] >= min_snr]; meta = [Mt[i] for i in idx]; P = predict(net, real_arrays(idx))
    kind = np.array([m['kind'] for m in meta]); isp = kind == 'planet'; isf = kind == 'FP'; m2 = isp | isf; pp = P['p_planet']
    out = {'n_planets': int(isp.sum()), 'n_fp': int(isf.sum()), 'n_fa': int((kind == 'FA').sum()), 'auc': auc(pp[m2], isp[m2]),
           'planets_called_planet': float(np.mean(pp[isp] > 0.5)), 'fp_called_planet': float(np.mean(pp[isf] > 0.5)), 'fa_called_planet': float(np.mean(pp[kind == 'FA'] > 0.5)) if (kind == 'FA').any() else None}
    pm = [m for m in meta if m['kind'] == 'planet']; Pp = {k: v[isp] for k, v in P.items()}; E = param_errors(pm, Pp)
    snr = np.array([m['info']['snr'] for m in pm]); depth = np.array([m['info']['depth'] for m in pm]); k_naive = np.sqrt(np.clip(depth, 0, None)); kt = np.array([m['k'] for m in pm])
    def summ(mask):
        z = {}
        for key, e in E.items():
            ee = e[mask & np.isfinite(e)]; z[key] = {'n': int(len(ee)), 'median': float(np.median(ee)) if len(ee) else None, 'p68': float(np.percentile(ee, 68)) if len(ee) else None}
        nb = np.abs(k_naive / kt - 1)[mask & np.isfinite(kt)]; z['k_frac_naive'] = {'median': float(np.median(nb)) if len(nb) else None}
        # coverage: published value inside CellSense-style 1-sigma (network sigma and published error combined)
        for key, p, tr, te in (('k', 'log_k', 'k', 'k_err'), ('b', 'b', 'b', 'b_err'), ('aR', 'log_aR', 'aR', 'aR_err')):
            truth = np.array([m.get(tr, np.nan) for m in pm], float); terr = np.array([m.get(te, np.nan) for m in pm], float)
            if p.startswith('log'):
                d = np.abs(Pp[p] - np.log10(truth)); s = np.sqrt(Pp[p + '_sig'] ** 2 + (np.nan_to_num(terr / truth) / np.log(10)) ** 2)
            else:
                d = np.abs(Pp[p] - truth); s = np.sqrt(Pp[p + '_sig'] ** 2 + np.nan_to_num(terr) ** 2)
            ok = mask & np.isfinite(d); z[key + '_in_1sigma'] = float(np.mean(d[ok] < s[ok])) if ok.any() else None
        return z
    out['params_all'] = summ(np.ones(len(pm), bool)); out['params_by_snr'] = {f'{lo}-{hi}': summ((snr >= lo) & (snr < hi)) for lo, hi in ((7, 20), (20, 100), (100, 1e9))}
    return out, idx, P
