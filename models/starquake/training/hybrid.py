"""StarQuake v1.1 candidate (15 Nov 2026): the classical SYD-style pipeline as a backbone, the network as a guard.

On whole Kepler light curves the classical pipeline (classical.py) reads numax and dnu better than the network on the
18 dev stars (numax 0.78 % vs 1.01 %, dnu 0.63 % vs 0.78 %); on short pieces it fails often and the network wins. The
candidate moves the network's reading towards the classical one by a weight that depends on the observing length,
and only where the two agree to within k network error bars (otherwise the classical reading is taken to have failed):

    ln x = ln x_net + w(T) * d,    d = ln x_cl - ln x_net if |d| <= k sigma_net else 0

w(T) (five length bins) and k are fitted on SIMULATED stars only (noise_train pool), per output. The error bar is the
network's, scaled per length bin by the ratio of hybrid to network median errors on the same simulations.

    python hybrid.py sims fit   noise_train 3000 11     # simulated giants: truth, network and classical readings
    python hybrid.py sims val3  noise_val   5000 12     # the new frozen set (v2 physics, held-out noise stars)
    python hybrid.py negs val3neg noise_val 2000 13     # frozen non-oscillating set (detection score)
    python hybrid.py fit fit                            # -> work/hybrid/params.json
    python hybrid.py score val3 [params.json]           # objective of the release (r5) and of the hybrid
"""
import _paths as _P  # portable paths (see _paths.py)
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import sq, starsim as SS, simbank as SB, classical as C, calib as CB  # noqa: E402

H = _P.WORK + '/hybrid/'
EDGES = [0, 40, 70, 150, 600, 1e9]           # observing-length bins (days)
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
K_GRID = [1.0, 2.0, 3.0, 5.0, 1e9]
HIT = {'numax': 0.05, 'dnu': 0.02}


def release_net():
    """the released network, or the numpy weights named by SQ_NET (a candidate rung exported by export.py)"""
    import starquake
    return starquake.Net(os.environ.get('SQ_NET') or os.path.join(_P.MODEL, 'starquake_weights.npz'))


def reading(net, y):
    """Light curve -> network outputs, classical numax/dnu and observing length."""
    F1, F2, s, (f, P), T = sq.prepare(y); o = net.forward(F1, F2, s)
    try:
        c = C.measure(f, P, F1, F2); cn, cd = c['numax'], c['dnu']
    except Exception:
        cn, cd = np.nan, np.nan
    return [o['ln_numax'], o['ln_dnu'], o['sig_numax'], o['sig_dnu'], o['p_det'], np.log(cn) if cn > 0 else np.nan, np.log(cd) if cd > 0 else np.nan, T]


def _sims_chunk(args):
    which, n, seed, neg = args
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    rng = np.random.default_rng(seed); pool = SB.load_pool(which); net = release_net(); calib = CB.make('v2'); R, Y = [], []
    for i in range(n):
        k, yn, kp = pool[rng.integers(len(pool))]; seg = SB.segment(yn, rng, 0.4).astype(np.float64); m = len(seg)
        if neg:
            y = seg.copy(); truth = [np.nan, np.nan]
        else:
            s = SS.draw_star(rng, (5.0, 275.0), calib=calib); y = seg + SS.synth(s, m, rng); truth = [np.log(s['numax']), np.log(s['dnu'])]
        if rng.random() < 0.25: y = y + rng.normal(0, float(np.exp(rng.uniform(np.log(20), np.log(400)))), m)
        R.append(reading(net, y)); Y.append(truth)
    return np.array(R), np.array(Y)


def sims(name, which, n, seed, neg=False, workers=3):
    workers = int(os.environ.get('SQ_WORKERS', workers))
    from multiprocessing import Pool
    os.makedirs(H, exist_ok=True); t = time.time(); per = n // workers
    with Pool(workers) as p:
        out = p.map(_sims_chunk, [(which, per, seed * 100 + j, neg) for j in range(workers)])
    R = np.concatenate([o[0] for o in out]); Y = np.concatenate([o[1] for o in out])
    np.savez(H + name + '.npz', r=R, y=Y, cols=json.dumps(['net_ln_numax', 'net_ln_dnu', 'sig_numax', 'sig_dnu', 'p_det', 'cl_ln_numax', 'cl_ln_dnu', 'T']))
    print(name, len(R), '%.0fs' % (time.time() - t))


def combine(r, params):
    """r: (N, 8) readings -> dict ln_numax, ln_dnu, sig_numax, sig_dnu (the release's raw sigmas scaled per length bin)."""
    out = {}; b = np.clip(np.digitize(r[:, 7], EDGES) - 1, 0, len(EDGES) - 2)
    for j, name in enumerate(('numax', 'dnu')):
        p = params[name]; mu, sg, cl = r[:, j], r[:, 2 + j], r[:, 5 + j]
        d = np.where(np.isfinite(cl), cl - mu, 0.0); d = np.where(np.abs(d) <= p['k'] * sg, d, 0.0)
        out['ln_' + name] = mu + np.array(p['w'])[b] * d; out['sig_' + name] = sg * np.array(p['s'])[b]
    return out


def loss(e):
    return float(np.mean(np.minimum(np.abs(e), 0.10)))      # robust L1: a reading off by more than 10% counts as 10%


def fit(name):
    z = np.load(H + name + '.npz'); r, y = z['r'], z['y']; b = np.clip(np.digitize(r[:, 7], EDGES) - 1, 0, len(EDGES) - 2); P = {}
    for j, o in enumerate(('numax', 'dnu')):
        best = None
        for k in K_GRID:
            ws = []
            for i in range(len(EDGES) - 1):
                m = b == i; mu, sg, cl = r[m, j], r[m, 2 + j], r[m, 5 + j]
                d = np.where(np.isfinite(cl), cl - mu, 0.0); d = np.where(np.abs(d) <= k * sg, d, 0.0)
                ws.append(min(W_GRID, key=lambda w: loss(mu + w * d - y[m, j])))
            tot = loss(combine(r, {o: {'k': k, 'w': ws, 's': [1] * 5}, ('dnu' if o == 'numax' else 'numax'): {'k': 0, 'w': [0] * 5, 's': [1] * 5}})['ln_' + o] - y[:, j])
            if best is None or tot < best[0]: best = (tot, k, ws)
        _, k, ws = best; e = combine(r, {o: {'k': k, 'w': ws, 's': [1] * 5}, ('dnu' if o == 'numax' else 'numax'): {'k': 0, 'w': [0] * 5, 's': [1] * 5}})['ln_' + o] - y[:, j]
        s = [float(np.median(np.abs(e[b == i])) / max(np.median(np.abs(r[b == i, j] - y[b == i, j])), 1e-9)) for i in range(len(EDGES) - 1)]
        P[o] = {'k': float(k), 'w': [float(w) for w in ws], 's': s}
        print(o, 'k', k, 'w', ws, 's', np.round(s, 2))
    P['edges_days'] = EDGES[:-1] + ['inf']; P['fitted_on'] = name
    json.dump(P, open(H + os.environ.get('SQ_PARAMS', 'params.json'), 'w'), indent=1); return P


def scores(est, y, T):
    out = {}
    for sub, m in (('all', np.ones(len(T), bool)), ('full', T > 1000), ('long', T >= 300), ('short', T < 300)):
        for o in ('numax', 'dnu'):
            e = est['ln_' + o][m] - y[m, 0 if o == 'numax' else 1]
            out[f'{sub}_{o}_hit'] = float(np.mean(np.abs(e) < HIT[o])); out[f'{sub}_{o}_med'] = float(np.median(np.abs(e)))
            out[f'{sub}_{o}_bias'] = float(np.median(e)); out[f'{sub}_{o}_cov'] = float(np.mean(np.abs(e) < est['sig_' + o][m]))
    return out


def score(name, params=None, neg='val3neg'):
    z = np.load(H + name + '.npz'); r, y = z['r'], z['y']; zn = np.load(H + neg + '.npz')
    det = 0.5 * (np.mean(r[:, 4] > 0.5) + 1 - np.mean(zn['r'][:, 4] > 0.5)); res = {}
    rel = {'ln_numax': r[:, 0], 'ln_dnu': r[:, 1], 'sig_numax': r[:, 2], 'sig_dnu': r[:, 3]}
    for tag, est in (('release', rel), ('hybrid', combine(r, params) if params else None)):
        if est is None: continue
        s = scores(est, y, r[:, 7]); s['det_bal'] = float(det); s['objective'] = (s['all_numax_hit'] + s['all_dnu_hit'] + det) / 3; res[tag] = s
        print(f"{tag:8s} objective {s['objective']:.4f} | numax hit {s['all_numax_hit']:.3f} med {s['all_numax_med']:.4f} | dnu hit {s['all_dnu_hit']:.3f} med {s['all_dnu_med']:.4f} | det {det:.3f}")
        for sub in ('full', 'long', 'short'):
            print(f"         {sub:5s} numax med {s[sub + '_numax_med']:.4f} bias {s[sub + '_numax_bias']:+.4f} cov {s[sub + '_numax_cov']:.2f} | dnu med {s[sub + '_dnu_med']:.4f} bias {s[sub + '_dnu_bias']:+.4f} cov {s[sub + '_dnu_cov']:.2f}")
    return res


def real(split, out_name):
    """Network + classical readings of real stars of one split, at the five lengths (realbank.pieces)."""
    import realbank as RB
    os.makedirs(H, exist_ok=True); sp = json.load(open(_P.WORK + '/data/split.json')); Y = json.load(open(_P.WORK + '/data/yu2018.json')); net = release_net(); R, T, M = [], [], []
    for k in sp[split]:
        p = f'{_P.WORK}/data/lc/{k}.npz'
        if not os.path.exists(p): continue
        for L, seg in RB.pieces(sq.read_npz(p, trim=os.environ.get('SQ_TRIM') == '1'), k):
            if seg is None: continue
            R.append(reading(net, seg)); T.append([np.log(Y[str(k)]['numax']), np.log(Y[str(k)]['dnu'])]); M.append(dict(kic=k, L=L, e_numax=Y[str(k)]['e_numax'] / Y[str(k)]['numax'], e_dnu=Y[str(k)]['e_dnu'] / Y[str(k)]['dnu']))
    np.savez(H + out_name + '.npz', r=np.array(R), y=np.array(T), meta=json.dumps(M)); print(out_name, len(R))


def real_report(name, params):
    z = np.load(H + name + '.npz'); r, y = z['r'], z['y']; M = json.loads(str(z['meta'])); L = np.array([m['L'] for m in M]); hy = combine(r, params); res = {}
    for l in (0, 365, 90, 54, 27):
        m = L == l; row = {}
        for j, o in enumerate(('numax', 'dnu')):
            row[o] = {'net': float(np.median(np.abs(r[m, j] - y[m, j]))), 'classical': float(np.median(np.abs(np.nan_to_num(r[m, 5 + j] - y[m, j], nan=9.0)))),
                      'hybrid': float(np.median(np.abs(hy['ln_' + o][m] - y[m, j]))), 'net_bias': float(np.median(r[m, j] - y[m, j])), 'hybrid_bias': float(np.median(hy['ln_' + o][m] - y[m, j]))}
        res[str(l)] = row
        print(f"L={l:>4} n={m.sum():3d} numax net {row['numax']['net']:.4f} cl {row['numax']['classical']:.4f} hyb {row['numax']['hybrid']:.4f} | dnu net {row['dnu']['net']:.4f} cl {row['dnu']['classical']:.4f} hyb {row['dnu']['hybrid']:.4f}")
    return res


def exam(params_path, widen_path, name='exam', out_name='exam_result.json'):
    """The exam, once (after the release choice): the released r5 network, the hybrid, the classical pipeline (same
    readings) and the dumb dev-median guess, per length; coverage of each one's dev-widened 68% bars (published error
    added in quadrature, as exam.py); mass and radius vs Yu and Gaia DR3 for the whole light curves."""
    params = json.load(open(params_path)); W = json.load(open(widen_path)); z = np.load(H + name + '.npz'); r, y = z['r'], z['y']; M = json.loads(str(z['meta']))
    L = np.array([m['L'] for m in M]); hy = combine(r, params); pub = np.array([[m['e_numax'], m['e_dnu']] for m in M])
    dv = np.load(H + 'dev.npz'); dmed = np.median(np.exp(dv['y'][[m['L'] == 0 for m in json.loads(str(dv['meta']))]]), 0)
    est = {'release': (r[:, 0], r[:, 1], r[:, 2] * W['release'][0], r[:, 3] * W['release'][1]),
           'hybrid': (hy['ln_numax'], hy['ln_dnu'], hy['sig_numax'] * W['hybrid'][0], hy['sig_dnu'] * W['hybrid'][1]),
           'classical': (r[:, 5], r[:, 6], None, None), 'dumb': (np.full(len(r), np.log(dmed[0])), np.full(len(r), np.log(dmed[1])), None, None)}
    out = {'lengths': {}}
    for l in (0, 365, 90, 54, 27):
        m = L == l; row = {'n': int(m.sum())}
        for name, (a, b, sa, sb) in est.items():
            e1 = np.nan_to_num(a[m] - y[m, 0], nan=9.0); e2 = np.nan_to_num(b[m] - y[m, 1], nan=9.0)
            row[name] = {'numax_med': float(np.median(np.abs(e1))), 'dnu_med': float(np.median(np.abs(e2))), 'numax_hit': float(np.mean(np.abs(e1) < 0.05)), 'dnu_hit': float(np.mean(np.abs(e2) < 0.02)),
                         'numax_bias': float(np.median(e1)), 'dnu_bias': float(np.median(e2))}
            if sa is not None:
                row[name]['numax_cov68'] = float(np.mean(np.abs(e1) < np.hypot(sa[m], pub[m, 0]))); row[name]['dnu_cov68'] = float(np.mean(np.abs(e2) < np.hypot(sb[m], pub[m, 1])))
        out['lengths'][str(l)] = row
        print(f"L={l:>4} n={m.sum()} numax  rel {row['release']['numax_med']:.4f} hyb {row['hybrid']['numax_med']:.4f} cl {row['classical']['numax_med']:.4f} dumb {row['dumb']['numax_med']:.3f} | cov rel {row['release']['numax_cov68']:.2f} hyb {row['hybrid']['numax_cov68']:.2f}")
        print(f"              dnu    rel {row['release']['dnu_med']:.4f} hyb {row['hybrid']['dnu_med']:.4f} cl {row['classical']['dnu_med']:.4f} dumb {row['dumb']['dnu_med']:.3f} | cov rel {row['release']['dnu_cov68']:.2f} hyb {row['hybrid']['dnu_cov68']:.2f}")
    # mass and radius, whole light curves, Yu's Teff (exam.py's formulas)
    gp = _P.WORK + '/data/gaia.json'; G = json.load(open(gp)) if os.path.exists(gp) else {}       # gaia.py; empty if the ESA archive could not be reached
    Y = json.load(open(_P.WORK + '/data/yu2018.json')); m = L == 0; kic = [M[i]['kic'] for i in np.flatnonzero(m)]
    t = np.array([Y[str(k)]['teff'] for k in kic]) / 5777.0; gR = np.array([G.get(str(k), {}).get('R_flame') or np.nan for k in kic], float); g = np.isfinite(gR)
    yR = np.array([Y[str(k)]['R'] for k in kic]); yM = np.array([Y[str(k)]['M'] for k in kic]); ok = np.isfinite(yM); out['mass_radius'] = {'gaia_n': int(g.sum())}
    sc = lambda v: float(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))))
    for name in ('release', 'hybrid', 'classical'):
        a, b = np.exp(est[name][0][m]) / 3090.0, np.exp(est[name][1][m]) / 135.1; R = a * b ** -2 * t ** 0.5; Ms = a ** 3 * b ** -4 * t ** 1.5
        out['mass_radius'][name] = {'R_vs_gaia_ratio': float(np.nanmedian(R[g] / gR[g])) if g.any() else None, 'R_vs_gaia_scatter': sc(R[g] / gR[g]) if g.any() else None,
                                    'M_vs_yu_med': float(np.nanmedian(np.abs(Ms[ok] / yM[ok] - 1))), 'R_vs_yu_med': float(np.nanmedian(np.abs(R[ok] / yR[ok] - 1)))}
    out['mass_radius']['yu'] = {'R_vs_gaia_ratio': float(np.median(yR[g] / gR[g])) if g.any() else None, 'R_vs_gaia_scatter': sc(yR[g] / gR[g]) if g.any() else None}
    print(json.dumps(out['mass_radius'], indent=1)); json.dump(out, open(H + out_name, 'w'), indent=1); return out


def widen(name, params=None):
    """Widen-only 68% factors per output on the dev stars (calib_sigma.py's rule), for the release and the hybrid."""
    z = np.load(H + name + '.npz'); r, y = z['r'], z['y']; M = json.loads(str(z['meta'])); pub = np.array([[m['e_numax'], m['e_dnu']] for m in M]); out = {}
    for tag, est in (('release', {'ln_numax': r[:, 0], 'ln_dnu': r[:, 1], 'sig_numax': r[:, 2], 'sig_dnu': r[:, 3]}), ('hybrid', combine(r, params) if params else None)):
        if est is None: continue
        fs = []
        for j, o in enumerate(('numax', 'dnu')):
            e = np.abs(est['ln_' + o] - y[:, j]); f = 1.0
            while np.mean(e < np.hypot(f * est['sig_' + o], pub[:, j])) < 0.683 and f < 20: f *= 1.02
            fs.append(float(f))
        out[tag] = fs
    print(out); json.dump(out, open(H + 'widen.json', 'w')); return out


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'sims': sims(a[1], a[2], int(a[3]), int(a[4]))
    elif a[0] == 'negs': sims(a[1], a[2], int(a[3]), int(a[4]), neg=True)
    elif a[0] == 'fit': fit(a[1])
    elif a[0] == 'score': score(a[1], json.load(open(a[2])) if len(a) > 2 else None)
    elif a[0] == 'real': real(a[1], a[2])
    elif a[0] == 'real_report': real_report(a[1], json.load(open(a[2] if len(a) > 2 else H + 'params.json')))
    elif a[0] == 'widen': widen(a[1], json.load(open(H + 'params.json')))
    elif a[0] == 'exam': exam(H + 'params.json', H + 'widen.json')
    elif a[0] == 'exam_trim': exam(H + 'params.json', H + 'widen.json', 'exam_trim', 'exam_trim_result.json')   # 27 Nov 2026 supplementary re-score
