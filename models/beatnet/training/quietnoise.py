"""Quiet real background from the dev records only: each beat minus the median of its 20 neighbours on each side
(aligned on the annotation, -0.35..+0.65 s), which leaves the recording's own noise floor, small baseline drift and
beat-to-beat jitter. 10 s pieces are kept with their local QRS size so they can be scaled to any simulated record.
Writes data/quiet_dev.npz (360 Hz, relative to the record's median QRS peak-to-peak)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np, wfdb, scipy.signal as ss
sys.path.insert(0, _P.SRC); import common as C
pieces = []
for rec in C.dev_records():
    r = wfdb.rdrecord(f'{C.D}mitdb/{rec}'); a = wfdb.rdann(f'{C.D}mitdb/{rec}', 'atr'); x = r.p_signal[:, r.sig_name.index('MLII')].astype(float)
    pos, cls, _ = C.beats(np.asarray(a.sample), np.asarray(a.symbol)); lo, hi = int(0.35 * 360), int(0.65 * 360)
    x = x - ss.medfilt(x, 721)          # remove the slowest drift (1 s median)
    P = pos; mid = np.concatenate([[0], (P[1:] + P[:-1]) // 2, [len(x)]]); res = x.copy()
    for i, p in enumerate(P):
        off = np.arange(mid[i], mid[i + 1]) - p; nb = [j for j in range(max(0, i - 20), min(len(P), i + 21)) if j != i]
        idx = np.clip(P[nb][:, None] + off[None, :], 0, len(x) - 1); res[mid[i]:mid[i + 1]] -= np.median(x[idx], 0)
    W = np.stack([x[p - lo:p + hi] for p in P[(P > lo) & (P < len(x) - hi)]])
    qamp = float(np.median(np.ptp(W[:, lo - 25:lo + 25], 1)))
    for s in range(0, len(x) - 3600, 3600):
        seg = res[s:s + 3600]; pieces.append((seg - seg.mean()) / qamp)
    print(rec, 'qamp', round(qamp, 3), 'pieces', len(pieces))
Pz = np.array(pieces, np.float32); rms = Pz.std(1); keep = rms < np.percentile(rms, 50)     # the quieter half
np.savez(_P.WORK + '/data/quiet_dev.npz', x=Pz[keep]); print('kept', keep.sum(), 'median rms (x QRS)', float(np.median(rms[keep])))
