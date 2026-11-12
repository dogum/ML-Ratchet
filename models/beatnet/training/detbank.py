"""Real beats located by our own QRS detector (qrsdet.py) instead of the reference annotations: MIT-BIH dev + DS2 records.
Every detection goes through the same front end. Each detection is matched to the nearest unused reference beat within
150 ms: matched -> that beat's AAMI class; unmatched -> class 5 ('no beat', a false detection). Reference beats with no
detection are kept as a count per class ('missed'). Writes data/mitdb_det_beats.npz."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, wfdb
sys.path.insert(0, _P.SRC); import common as C, prep, qrsdet as Q
D = C.D

def one(rec):
    r = wfdb.rdrecord(D + f'mitdb/{rec}'); a = wfdb.rdann(D + f'mitdb/{rec}', 'atr'); ch = r.sig_name.index('MLII'); x = r.p_signal[:, ch]
    pos, cls, sym = C.beats(np.asarray(a.sample), np.asarray(a.symbol)); det = Q.detect(x, r.fs)
    j = np.searchsorted(det, pos); lab = np.full(len(det), 5); used = np.zeros(len(det), bool); hit = np.zeros(len(pos), bool)
    for i, p in enumerate(pos):
        best = None
        for k in (j[i] - 1, j[i]):
            if 0 <= k < len(det) and not used[k] and abs(det[k] - p) <= 0.15 * r.fs and (best is None or abs(det[k] - p) < abs(det[best] - p)): best = k
        if best is not None: used[best] = hit[i] = True; lab[best] = cls[i]
    f = prep.features(x, r.fs, det); missed = np.bincount(cls[~hit], minlength=5)
    return f, lab, det, missed, np.bincount(cls, minlength=5)

if __name__ == '__main__':
    dev = C.dev_records(); R = {k: [] for k in ('w', 'tpl', 'rr', 'y', 'rec', 'pos')}; meta = []
    for rec in dev + C.DS2:
        f, lab, det, missed, ref = one(rec)
        for k in ('w', 'tpl', 'rr'): R[k].append(f[k])
        R['y'].append(lab); R['rec'].append(np.full(len(lab), rec)); R['pos'].append(det)
        meta.append(dict(rec=rec, split='dev' if rec in dev else 'ds2', n_det=len(det), ref_counts=ref.tolist(), missed=missed.tolist(), false_det=int((lab == 5).sum())))
        print(rec, len(det), 'ref', ref.tolist(), 'missed', missed.tolist(), 'false', int((lab == 5).sum()), flush=True)
    np.savez(D + 'mitdb_det_beats.npz', **{k: np.concatenate(v).astype(np.float16 if k in ('w', 'tpl') else (np.float32 if k == 'rr' else np.int32)) for k, v in R.items()})
    json.dump(meta, open(D + 'mitdb_det_beats_meta.json', 'w'), indent=0)
