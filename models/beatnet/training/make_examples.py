"""Demo excerpts for the release: one DS2 record per stratum (V-rich, S-rich, with F, normal), drawn by a fixed seed;
60 s shown with 30 s of context either side."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, wfdb
sys.path.insert(0, _P.SRC); import common as C
rng = np.random.default_rng(5); cnt = {}
for r in C.DS2:
    a = wfdb.rdann(f'{C.D}mitdb/{r}', 'atr'); _, c, _ = C.beats(np.asarray(a.sample), np.asarray(a.symbol)); cnt[r] = np.bincount(c, minlength=5)
strata = [('V-rich', [r for r in C.DS2 if cnt[r][2] >= 100], 2), ('S-rich', [r for r in C.DS2 if cnt[r][1] >= 50], 1), ('fusion beats', [r for r in C.DS2 if cnt[r][3] >= 10], 3), ('normal', [r for r in C.DS2 if cnt[r][1:4].sum() < 5], 0)]
used = []
for why, recs, cl in strata:
    rec = int(rng.choice([r for r in recs if r not in used])); used.append(rec)
    R = wfdb.rdrecord(f'{C.D}mitdb/{rec}'); a = wfdb.rdann(f'{C.D}mitdb/{rec}', 'atr'); x = R.p_signal[:, R.sig_name.index('MLII')]; pos, c, _ = C.beats(np.asarray(a.sample), np.asarray(a.symbol))
    starts = np.arange(30 * 360, len(x) - 90 * 360, 360 * 5); ok = [s for s in starts if (not cl) or np.any((c == cl) & (pos >= s) & (pos < s + 60 * 360))]; s0 = int(rng.choice(ok))
    lo, hi = s0 - 30 * 360, s0 + 90 * 360; m = (pos >= lo) & (pos < hi); ref = np.array(['NSVFQ'[k] for k in c[m]])
    np.savez(f'{_P.WORK}/release/examples/{rec}.npz', ecg=x[lo:hi].astype(np.float32), beats=pos[m] - lo, ref=ref, show=np.array([30 * 360, 90 * 360]),
             meta=json.dumps(dict(record=rec, fs=360, start_s=s0 / 360, why=why, lead='MLII')))
    print(why, rec, s0 / 360, np.bincount(c[m], minlength=5))
