"""Round 2 (15 Nov 2026): can the 8 QT-database dev records show the exam's QRS-width loss (lead MLII: 18 ms vs classical
delineation 15 ms)? Reads the dev beats with the released numpy module (beatnet.py, its dev-fitted linear map) and the
classical width (baselines.qrs_width, with the dev-fitted map stored in exam.json). Dev records only; the exam beats are
loaded with the same file but never scored here. Writes work/qt_devprobe_r2.json.

    python qtbank.py && python qt_devprobe.py
"""
import _paths as _P
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.MODEL)
import beatnet as BM, baselines as BL

if __name__ == '__main__':
    z = np.load(_P.WORK + '/data/qt_beats.npz'); meta = json.load(open(_P.WORK + '/data/qt_meta.json'))
    sp = {m['idx']: m['split'] for m in meta}; lead = {m['idx']: m['lead'] for m in meta}; name = {m['idx']: m['rec'] for m in meta}
    dev = np.array([sp[r] == 'dev' for r in z['rec']]); F = {k: z[k][dev].astype(np.float32) for k in ('w', 'tpl', 'rr')}
    q = BM.BeatNet().predict_features(F)['qrs_ms']; E = json.load(open(_P.SRC + '/exam.json'))['qrs']
    cl = E['classical_a'] + E['classical_b'] * BL.qrs_width(F['w']); t = z['qrs'][dev]; rec = z['rec'][dev]
    out = dict(n=int(dev.sum()), leads=sorted({lead[r] for r in rec}), mlii_records=int(sum(lead[r] == 'MLII' for r in set(rec))),
               net_mae=float(np.abs(q - t).mean()), classical_mae=float(np.abs(cl - t).mean()), median_mae=float(np.abs(np.median(t) - t).mean()), records={})
    for r in sorted(set(rec)):
        m = rec == r; out['records'][name[r]] = dict(lead=lead[r], n=int(m.sum()), truth_median=float(np.median(t[m])), net_mae=float(np.abs(q[m] - t[m]).mean()),
                                                     classical_mae=float(np.abs(cl[m] - t[m]).mean()), net_bias=float((q[m] - t[m]).mean()), classical_bias=float((cl[m] - t[m]).mean()))
    print(json.dumps(out, indent=1)); json.dump(out, open(_P.WORK + '/qt_devprobe_r2.json', 'w'), indent=1)
