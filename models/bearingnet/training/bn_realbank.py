"""Features of every real dev / exam reading (windows), plus the classical baseline's per-window scores.
Writes data/real_<part>.npz (window features) and data/real_<part>.json (one record per reading)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, time, numpy as np
sys.path.insert(0, _P.SRC); import features as FE, realdata as RD, classical as CL

def build(part):
    S = RD.splits(); X, SC, W, meta = [], [], [], []; t0 = time.time()
    gen = ((lambda i=i: RD.cwru_readings([i])) for i in S['cwru_' + part]); gen2 = ((lambda b=b: RD.pb_readings([b])) for b in S['pb_' + part])
    def stream():                      # one file at a time (RAM)
        for g in list(gen) + list(gen2): yield from g()
    for k, r in enumerate(stream()):
        x = r.pop('x'); r['short'] = len(x) < FE.NWIN; x = FE.pad_short(x); st = FE.windows(x); r['n_win'] = len(st); r['win0'] = len(W); cs, csn = [], []
        f_cl = CL.speed(x[st[0]:st[0] + FE.NWIN], r['f_nom']); r['f_classical'] = float(f_cl)
        for s in st:
            w = x[s:s + FE.NWIN]; F, sc, band = FE.window_features(w, r['geom'], r['f_nom']); X.append(F.astype(np.float16)); SC.append(sc); W.append(k)
            cs.append(CL.scores(w, r['geom'], f_cl)); csn.append(CL.scores(w, r['geom'], r['f_nom']))
        r['classical'] = np.mean(cs, 0).tolist(); r['classical_fnom'] = np.mean(csn, 0).tolist(); meta.append(r)
        if k % 50 == 0: print(part, k, round(time.time() - t0), flush=True)
    np.savez(RD.D + 'real_%s.npz' % part, x=np.stack(X), s=np.stack(SC), w=np.array(W)); json.dump(meta, open(RD.D + 'real_%s.json' % part, 'w'))

if __name__ == '__main__':
    for p in sys.argv[1:]: build(p)
