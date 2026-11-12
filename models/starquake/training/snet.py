"""Load StarQuake, predict, and score: the frozen simulated set, the real dev stars, the real exam."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M
D = _P.WORK + '/data/'; B = _P.WORK + '/bank/'
VAL = B + 'val2.npz'      # frozen simulated set, versioned with the physics (val1: published relations, v1; val2: dev-calibrated v2)
VALNEG = B + 'valneg.npz' # frozen simulated NON-oscillating set (v2 physics, noise_val stars): for the detection score
HIT = (0.05, 0.02)        # a reading 'hits' if numax is within 5% and dnu within 2%

def load_net(path):
    ck = torch.load(path, weights_only=False); net = M.StarQuake(**ck.get('arch', {})); net.load_state_dict(ck['state']); net.eval()
    net.mu_y, net.sd_y = np.array(ck['mu_y']), np.array(ck['sd_y']); return net

def predict(net, x1, x2, s, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(x1), bs):
            out.append(net(torch.from_numpy(np.asarray(x1[i:i + bs], np.float32)), torch.from_numpy(np.asarray(x2[i:i + bs], np.float32)), torch.from_numpy(np.asarray(s[i:i + bs], np.float32))).numpy())
    o = np.concatenate(out); mu = o[:, :2] * net.sd_y + net.mu_y; sg = np.exp(np.clip(o[:, 2:4], -7, 3)) * net.sd_y
    pd = 1 / (1 + np.exp(-np.clip(o[:, 4], -30, 30))) if o.shape[1] > 4 else np.ones(len(o))     # a net without a detection output always says 'yes'
    return dict(ln_numax=mu[:, 0], ln_dnu=mu[:, 1], sig_numax=sg[:, 0], sig_dnu=sg[:, 1], p_det=pd)

def scores(P, ln_nm, ln_dn, fac=(1.0, 1.0)):
    e1 = P['ln_numax'] - ln_nm; e2 = P['ln_dnu'] - ln_dn
    return dict(n=int(len(e1)), numax_hit=float(np.mean(np.abs(e1) < HIT[0])), dnu_hit=float(np.mean(np.abs(e2) < HIT[1])),
                numax_med=float(np.median(np.abs(e1))), dnu_med=float(np.median(np.abs(e2))), numax_bias=float(np.median(e1)), dnu_bias=float(np.median(e2)),
                numax_out=float(np.mean(np.abs(e1) > 0.15)), dnu_out=float(np.mean(np.abs(e2) > 0.05)),
                numax_cov=float(np.mean(np.abs(e1) < fac[0] * P['sig_numax'])), dnu_cov=float(np.mean(np.abs(e2) < fac[1] * P['sig_dnu'])))

def val(net, path=None, neg=None):
    """Objective = mean of three rates on the frozen simulated sets: numax within 5%, dnu within 2% (oscillating stars),
    and the detection balanced accuracy (oscillating stars called 'yes', non-oscillating called 'no', threshold 0.5)."""
    z = np.load(path or VAL); P = predict(net, z['x1'], z['x2'], z['s']); s = scores(P, z['y'][:, 0], z['y'][:, 1])
    full = z['s'][:, 0] > np.log10(1000) - 2.5; s['full'] = scores({k: v[full] for k, v in P.items()}, z['y'][full, 0], z['y'][full, 1])
    zn = np.load(neg or VALNEG); Pn = predict(net, zn['x1'], zn['x2'], zn['s'])
    s['tpr'] = float(np.mean(P['p_det'] > 0.5)); s['fpr'] = float(np.mean(Pn['p_det'] > 0.5)); s['det_bal'] = 0.5 * (s['tpr'] + 1 - s['fpr'])
    s['objective'] = (s['numax_hit'] + s['dnu_hit'] + s['det_bal']) / 3; return s

def real(net, which, L=None, fac=(1.0, 1.0)):
    """Real stars: splits dev/exam (giants, Yu et al. 2018 truth) or dev_neg/exam_neg (quiet dwarfs: detection only)."""
    neg = which.endswith('_neg'); z = np.load(D + ('realneg.npz' if neg else 'real.npz')); meta = json.load(open(D + ('realneg_meta.json' if neg else 'real_meta.json')))
    idx = np.array([i for i, m in enumerate(meta) if m['split'] == which and (L is None or m['L'] == L)]); mt = [meta[i] for i in idx]
    P = predict(net, z['x1'][idx], z['x2'][idx], z['s'][idx])
    if neg: return dict(n=len(idx), fpr=float(np.mean(P['p_det'] > 0.5))), P, mt
    nm = np.log([m['numax'] for m in mt]); dn = np.log([m['dnu'] for m in mt]); sc = scores(P, nm, dn, fac); sc['tpr'] = float(np.mean(P['p_det'] > 0.5))
    return sc, P, mt
