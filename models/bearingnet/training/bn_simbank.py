"""Build a simulated bank: physics defects in REAL healthy backgrounds (CWRU noise file, Paderborn K001-K003), features
computed exactly as at inference. Chunked and resumable. Usage: simbank.py NAME N SEED [VERSION]"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); import bearsim as BS, features as FE, realdata as RD
OUT = _P.WORK + '/banks/'; CH = 2500

def backgrounds():
    S = RD.splits(); B = []
    for i in S['cwru_noise']:
        c, ch, fr = RD.cwru_rec(i)
        for k in ('DE', 'FE'): B.append(dict(x=ch[k], f=fr, src='CWRU%d_%s' % (i, k), grp='cwru'))
    for b in S['pb_noise']:
        z = np.load(RD.D + 'pb/%s.npz' % b); meta = json.loads(str(z['meta']))
        for x, m in zip(z['x'], meta): B.append(dict(x=x.astype(float), f=m['rpm'] / 60, src=b + '/' + m['cond'], grp='pb'))
    return B

def make(rng, B, version='v1', **kw):
    grp = 'cwru' if rng.random() < 0.5 else 'pb'; pool = [b for b in B if b['grp'] == grp]; b = pool[rng.integers(len(pool))]
    x, tr = BS.simulate(rng, b['x'], b['f'], version=version, **kw)
    f_nom = tr['f_r'] * float(np.exp(rng.uniform(np.log(0.95), np.log(1.05))))
    F, sc, band = FE.window_features(x, tr['geom'], f_nom)
    return F, sc, dict(cls=BS.CLS.index(tr['cls']), lsp=float(np.log2(tr['f_r'] / f_nom)), lsize=float(np.log10(max(tr['size_mm'], 1e-3))),
                       snr=tr['snr_db'] if tr['snr_db'] is not None else np.nan, src=b['src'], distract=tr['distract'], aperiodic=tr['aperiodic'])

def build(name, n, seed, version='v1', rev=False):
    os.makedirs(OUT, exist_ok=True); B = backgrounds(); t0 = time.time()
    for c0 in (range(0, n, CH)[::-1] if rev else range(0, n, CH)):     # rev: a second worker fills from the end
        fn = OUT + '%s_%05d.npz' % (name, c0)
        if os.path.exists(fn): continue
        rng = np.random.default_rng([seed, c0]); m = min(CH, n - c0)
        X = np.zeros((m, FE.NCH, len(FE.V)), np.float16); S = np.zeros((m, FE.NSC), np.float32); Y = []
        for j in range(m):
            F, sc, y = make(rng, B, version); X[j] = F; S[j] = sc; Y.append(y)
        tmp = fn[:-4] + '.tmp%d.npz' % os.getpid(); np.savez(tmp, x=X, s=S, cls=np.array([y['cls'] for y in Y], np.int8), lsp=np.array([y['lsp'] for y in Y], np.float32),
                 lsize=np.array([y['lsize'] for y in Y], np.float32), snr=np.array([y['snr'] for y in Y], np.float32), src=np.array([y['src'] for y in Y]),
                 flags=np.array([[y['distract'], y['aperiodic']] for y in Y], bool), version=version); os.replace(tmp, fn)
        print(fn, round(time.time() - t0), flush=True)

def load(name):
    fs = sorted(f for f in os.listdir(OUT) if f.startswith(name + '_') and f.endswith('.npz') and '.tmp' not in f); Z = [np.load(OUT + f) for f in fs]
    return {k: np.concatenate([z[k] for z in Z]) for k in ('x', 's', 'cls', 'lsp', 'lsize', 'snr', 'flags')}

if __name__ == '__main__':
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else 'v1', len(sys.argv) > 5 and sys.argv[5] == 'rev')
