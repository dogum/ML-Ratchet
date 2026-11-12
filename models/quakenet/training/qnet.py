"""Load QuakeNet, predict, and score: frozen simulated set, real dev records, real exam."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import model as M, train as TR
D = _P.WORK + '/data/'; B = _P.WORK + '/bank/'

def load_net(path):
    ck = torch.load(path, weights_only=False); net = M.QuakeNet(); net.load_state_dict(ck['state']); net.eval(); net.mu_y, net.sd_y = np.array(ck['mu_y']), np.array(ck['sd_y']); return net

def predict(net, x, ls, peak, bs=256):
    S = TR.scal(ls, peak); out = []
    with torch.no_grad():
        for i in range(0, len(x), bs): out.append(net(torch.from_numpy(np.asarray(x[i:i + bs], np.float32)), torch.from_numpy(S[i:i + bs])).numpy())
    o = np.concatenate(out); n = len(M.OUT); mu = o[:, :n] * net.sd_y + net.mu_y; sg = np.exp(np.clip(o[:, n:2 * n], -6, 3)) * net.sd_y
    v = o[:, 2 * n:]; return dict(mw=mu[:, 0], mw_sig=sg[:, 0], dist=mu[:, 1], dist_sig=sg[:, 1], depth=10 ** mu[:, 2], logz_sig=sg[:, 2], logz=mu[:, 2],
                                baz=np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360, kappa=np.sqrt((v ** 2).sum(1)))

def angdiff(a, b): return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)

def scores(P, mw, dist, depth, baz):
    lz = np.log10(np.maximum(depth, 1)); out = {'n': int(len(mw)), 'mw_mae': float(np.mean(np.abs(P['mw'] - mw))), 'mw_bias': float(np.mean(P['mw'] - mw)), 'mw_in_1sigma': float(np.mean(np.abs(P['mw'] - mw) < P['mw_sig'])),
           'dist_mae': float(np.mean(np.abs(P['dist'] - dist))), 'dist_in_1sigma': float(np.mean(np.abs(P['dist'] - dist) < P['dist_sig'])),
           'depth_mae_km': float(np.mean(np.abs(P['depth'] - depth))), 'depth_med_km': float(np.median(np.abs(P['depth'] - depth))), 'logz_in_1sigma': float(np.mean(np.abs(P['logz'] - lz) < P['logz_sig'])),
           'baz_med_deg': float(np.median(angdiff(P['baz'], baz))), 'baz_within_30': float(np.mean(angdiff(P['baz'], baz) < 30))}
    for nm, m in (('shallow', depth <= 70), ('deep', depth > 70)):
        if m.any(): out[nm] = {'n': int(m.sum()), 'depth_med_km': float(np.median(np.abs(P['depth'][m] - depth[m]))), 'mw_mae': float(np.mean(np.abs(P['mw'][m] - mw[m])))}
    sk = lambda e, t: 1 - np.mean(e) / np.mean(np.abs(t - np.median(t)))
    out['skill'] = {'mw': float(sk(np.abs(P['mw'] - mw), mw)), 'dist': float(sk(np.abs(P['dist'] - dist), dist)), 'logz': float(sk(np.abs(P['logz'] - lz), lz)), 'baz': float(1 - np.mean(angdiff(P['baz'], baz)) / 90.0)}
    return out

VAL = B + 'val8.npz'     # frozen simulated set, versioned with the physics (val.npz: v1; val2.npz: surface-wave delay; val3.npz: + scattering coda; val4.npz: + the world's depth and magnitude mix, 1990-2011 catalog; val5.npz: + ocean and station layering; val7.npz: v4 + short-period surface waves depleted as measured on dev (layering dropped after rung 5 was rejected); val8.npz: v4 + per-component surface-wave gains and short-period Rayleigh delay)
def val(net):
    z = np.load(VAL); P = predict(net, z['x'], z['ls'], z['peak']); y = z['y']; s = scores(P, y[:, 0], y[:, 1], y[:, 2], np.degrees(np.arctan2(y[:, 3], y[:, 4])) % 360)
    s['objective'] = float(np.mean([max(0, v) for v in s['skill'].values()])); return s

def real(net, which):
    z = np.load(D + 'real.npz'); J = json.load(open(D + 'real_meta.json')); idx = np.array([i for i, s in enumerate(J['split']) if s == which]); meta = [J['meta'][i] for i in idx]
    P = predict(net, z['x'][idx], z['ls'][idx], z['peak'][idx]); g = lambda k: np.array([m[k] for m in meta], float)
    return scores(P, g('mw'), g('dist'), g('depth'), g('baz')), idx, P
