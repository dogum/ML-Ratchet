"""QRS-width exam data from the QT database (Laguna et al. 1997): manual QRS onset/offset (q1c, first cardiologist) on
30-100 beats per record. Beats found by qrsdet on the first lead; every detected beat goes through the front end, the
manually annotated ones carry the true width. Split: a dev set of 8 records by fixed seed; the rest is the exam."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, numpy as np, wfdb
sys.path.insert(0, _P.SRC); import prep, qrsdet as Q
D = _P.WORK + '/data/qtdb/'

def qt_split(seed=11):
    recs = sorted(f[:-4] for f in os.listdir(D) if f.endswith('.q1c')); rng = np.random.default_rng(seed); dev = sorted(rng.choice(recs, 8, replace=False).tolist())
    return recs, dev

if __name__ == '__main__':
    recs, dev = qt_split(); R = {k: [] for k in ('w', 'tpl', 'rr', 'qrs', 'rec')}; meta = []
    for i, rec in enumerate(recs):
        r = wfdb.rdrecord(D + rec); a = wfdb.rdann(D + rec, 'q1c'); x = r.p_signal[:, 0]; fs = r.fs; det = Q.detect(x, fs)
        sym, smp = np.array(a.symbol), np.array(a.sample); truth = []
        for k in np.nonzero(sym == 'N')[0]:
            if k > 0 and k + 1 < len(sym) and sym[k - 1] == '(' and sym[k + 1] == ')':
                j = np.argmin(np.abs(det - smp[k])) if len(det) else -1
                if j >= 0 and abs(det[j] - smp[k]) < 0.06 * fs: truth.append((j, (smp[k + 1] - smp[k - 1]) / fs * 1000))
        if not truth: continue
        keep = np.array([t[0] for t in truth]); f = prep.features(x, fs, det, keep)
        for k in ('w', 'tpl', 'rr'): R[k].append(f[k])
        R['qrs'].append(np.array([t[1] for t in truth], np.float32)); R['rec'].append(np.full(len(keep), i))
        meta.append(dict(rec=rec, idx=i, lead=r.sig_name[0], fs=fs, n=len(keep), split='dev' if rec in dev else 'exam'))
        print(rec, r.sig_name[0], len(keep), np.median(R['qrs'][-1]), flush=True)
    out = {k: np.concatenate(v) for k, v in R.items()}; out['w'] = out['w'].astype(np.float16); out['tpl'] = out['tpl'].astype(np.float16); out['y'] = np.zeros(len(out['qrs']), np.int32)
    np.savez(_P.WORK + '/data/qt_beats.npz', **out); json.dump(meta, open(_P.WORK + '/data/qt_meta.json', 'w'), indent=0)
