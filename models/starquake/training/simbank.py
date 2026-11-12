"""Build a bank of simulated red giants injected into real Kepler light curves of quiet dwarfs.
Each example: a random star (starsim.draw_star), a random noise star from the pool, a random observing length
(40%: the whole ~4-year light curve; otherwise log-uniform from 27 days to the whole), sometimes extra white noise
(25%: 20-400 ppm per cadence, as for a fainter star). Stored float16: F1 (1024), F2 (48x110), scalars, labels."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, argparse, numpy as np
sys.path.insert(0, _P.SRC); import sq, starsim as SS
D = _P.WORK + '/data/'; B = _P.WORK + '/bank/'

def load_pool(which):
    sp = json.load(open(D + 'split.json')); out = []; scr = json.load(open(D + 'noise_screen.json'))
    for k in sp[which]:
        if not scr[str(k)]['keep']: continue
        p = f'{D}lc/{k}.npz'
        if os.path.exists(p):
            y = sq.read_npz(p); z = np.load(p); out.append((k, y.astype(np.float32), json.loads(str(z['hdr']))['KEPMAG']))
    return out

def segment(y, rng, p_full=0.4, tmin=27.0):
    n = len(y); full = n * sq.DT / 86400
    if rng.random() < p_full: return y
    T = np.exp(rng.uniform(np.log(tmin), np.log(full))); L = max(int(T * 86400 / sq.DT), 64)
    for _ in range(20):
        s = rng.integers(0, n - L + 1); seg = y[s:s + L]
        if np.isfinite(seg).mean() > 0.5: return seg
    return y

def one(rng, pool, calib=None, p_full=0.4, p_neg=0.0):
    """p_neg: fraction of examples WITHOUT oscillations in the band (detection label 0): 60% the real quiet star alone,
    40% a very luminous giant (numax 0.5-2 uHz) whose hump lies below the band and only its granulation shows."""
    k, yn, kp = pool[rng.integers(len(pool))]; seg = segment(yn, rng, p_full).astype(np.float64); n = len(seg)
    neg = rng.random() < p_neg; kind = 'osc'
    if neg and rng.random() < 0.6:
        kind = 'noise'; s = dict(numax=np.nan, dnu=np.nan); y = seg.copy()
    else:
        if neg: kind = 'lowgiant'
        s = SS.draw_star(rng, (0.5, 2.0) if neg else (5.0, 275.0), calib=calib); x = SS.synth(s, n, rng); y = seg + x
    s['det'] = int(not neg); s['kind'] = kind
    wn = 0.0
    if rng.random() < 0.25: wn = float(np.exp(rng.uniform(np.log(20), np.log(400)))); y = y + rng.normal(0, wn, n)
    F1, F2, sc, _, T = sq.prepare(y)
    s.update(noise_kic=int(k), kp=float(kp), T=float(T), white=wn)
    return F1, F2, sc, np.nan_to_num(np.array([np.log(s['numax']), np.log(s['dnu'])], np.float32)), s

def build(out, n, seed, which, calib=None, p_full=0.4, p_neg=0.0):
    rng = np.random.default_rng(seed); pool = load_pool(which); print('pool', len(pool), flush=True)
    X1 = np.zeros((n, sq.NB), np.float16); X2 = np.zeros((n, sq.NC, sq.NL), np.float16); S = np.zeros((n, 2), np.float32); Yl = np.zeros((n, 2), np.float32); DET = np.ones(n, np.float32); P = []
    t = time.time()
    for i in range(n):
        F1, F2, sc, yl, s = one(rng, pool, calib, p_full, p_neg); X1[i], X2[i], S[i], Yl[i], DET[i] = F1, F2, sc, yl, s['det']; P.append(s)
        if (i + 1) % 1000 == 0: print(i + 1, n, '%.0fs' % (time.time() - t), flush=True)
    np.savez(out, x1=X1, x2=X2, s=S, y=Yl, det=DET, params=json.dumps(P)); print('saved', out, '%.0fs' % (time.time() - t))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=20000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pool', default='noise_train'); ap.add_argument('--calib', default=''); ap.add_argument('--p_full', type=float, default=0.4); ap.add_argument('--p_neg', type=float, default=0.0)
    a = ap.parse_args(); calib = None
    if a.calib:
        import calib as CB; calib = CB.make(a.calib)
    if a.n <= 5000: build(B + a.out, a.n, a.seed, a.pool, calib, a.p_full, a.p_neg)
    else:      # resumable: 5,000-example chunks, each with its own seed; existing chunks are kept
        for c in range(a.n // 5000):
            out = B + a.out.replace('.npz', f'_{c:02d}.npz')
            if not os.path.exists(out): build(out, 5000, a.seed * 1000 + c, a.pool, calib, a.p_full, a.p_neg)
