"""Real Kepler red giants (dev and exam), read once: features of the whole ~4-year light curve and of shorter pieces
(27 d = one TESS sector, 54 d, 90 d = one Kepler quarter, 365 d), each piece at a start fixed by the star's KIC number.
Also runs the classical pipeline on every reading. Truth: Yu et al. 2018 (from the whole light curve)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); import sq, classical as C
D = _P.WORK + '/data/'; LENS = [27, 54, 90, 365, 0]

def pieces(y, kic):
    rng = np.random.default_rng(kic); out = []
    for L in LENS:
        if L == 0: out.append((0, y)); continue
        n = int(L * 86400 / sq.DT); seg = None
        for _ in range(50):
            s = rng.integers(0, len(y) - n + 1); c = y[s:s + n]
            if np.isfinite(c).mean() > 0.8: seg = c; break
        out.append((L, seg))
    return out

if __name__ == '__main__':
    sp = json.load(open(D + 'split.json')); Y = json.load(open(D + 'yu2018.json')); R = {k: [] for k in ('x1', 'x2', 's')}; meta = []; t = time.time()
    for which in ('dev', 'exam'):
        for k in sp[which]:
            p = f'{D}lc/{k}.npz'
            if not os.path.exists(p): continue
            y = sq.read_npz(p)
            for L, seg in pieces(y, k):
                if seg is None: continue
                F1, F2, s, (f, P), T = sq.prepare(seg); c = C.measure(f, P, F1, F2)
                R['x1'].append(F1.astype(np.float16)); R['x2'].append(F2.astype(np.float16)); R['s'].append(s)
                meta.append(dict(kic=k, split=which, L=L, T=T, cl_numax=c['numax'], cl_dnu=c['dnu'], **{a: Y[str(k)][a] for a in ('numax', 'e_numax', 'dnu', 'e_dnu', 'kp', 'teff', 'e_teff', 'M', 'e_M', 'R', 'e_R', 'phase', 'amp', 'width', 'gran', 'dlen')}))
            if len(meta) % 100 < 5: print(len(meta), '%.0fs' % (time.time() - t), flush=True)
    np.savez(D + 'real.npz', **{a: np.stack(v) for a, v in R.items()}); json.dump(meta, open(D + 'real_meta.json', 'w')); print('readings', len(meta))
