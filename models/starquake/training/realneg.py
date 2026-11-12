"""Real NON-oscillating stars for the detection test: dev_neg = the 60 noise_val dwarfs (screened or not; never exam),
exam_neg = 120 fresh unscreened dwarfs (negpick.py). Same five pieces per star as realbank.py; classical readings too.
Writes data/realneg.npz + realneg_meta.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, numpy as np
sys.path.insert(0, _P.SRC); import sq, classical as C, realbank as RB
D = _P.WORK + '/data/'

if __name__ == '__main__':
    sp = json.load(open(D + 'split.json')); R = {k: [] for k in ('x1', 'x2', 's')}; meta = []; t = time.time()
    for which, src in (('dev_neg', 'noise_val'), ('exam_neg', 'exam_neg')):
        for k in sp[src]:
            p = f'{D}lc/{k}.npz'
            if not os.path.exists(p): continue
            y = sq.read_npz(p); kp = json.loads(str(np.load(p)['hdr']))['KEPMAG']
            for L, seg in RB.pieces(y, k):
                if seg is None: continue
                F1, F2, s, (f, P), T = sq.prepare(seg); c = C.measure(f, P, F1, F2)
                R['x1'].append(F1.astype(np.float16)); R['x2'].append(F2.astype(np.float16)); R['s'].append(s)
                meta.append(dict(kic=k, split=which, L=L, T=T, kp=kp, cl_numax=c['numax'], cl_dnu=c['dnu']))
        print(which, len(meta), '%.0fs' % (time.time() - t), flush=True)
    np.savez(D + 'realneg.npz', **{a: np.stack(v) for a, v in R.items()}); json.dump(meta, open(D + 'realneg_meta.json', 'w')); print('readings', len(meta))
