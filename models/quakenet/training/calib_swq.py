"""Surface waves in the real Earth lose more of their short periods than in ak135: real/synthetic RMS ratio after S, by
period band, on shallow dev quakes with USGS moment tensors (noise energy before P subtracted; bands below 2x noise skipped).
Fit an extra attenuation exp(-pi f t*) for everything after S."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, simbank as SB, quakesim as QS, pipeline as PL, crust as CR
DAMP = '--check' in sys.argv
from scipy.signal import butter, sosfiltfilt
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); P0 = PL.PRE
BANDS = [(6, 9), (9, 12), (12, 18), (18, 25), (25, 35), (35, 50), (50, 80), (80, 140)]
rows = []
for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
    if s != 'dev' or m['depth'] > 70: continue
    M, _ = CP.usgs_mt(m['id']); X = CP.synth_for(m, M); X = CR.damp_short_sw(None, X, P0 + SB.s_minus_p(min(QS.DEPTHS, key=lambda v: abs(v - m['depth'])), min(max(int(round(m['dist'])), 25), 95)), 1.0) if DAMP else X; R = z['x'][i].astype(float) * 10 ** z['ls'][i]; L = min(R.shape[1], X.shape[1])
    zz = min(QS.DEPTHS, key=lambda v: abs(v - m['depth'])); tS = P0 + int(SB.s_minus_p(zz, min(max(int(round(m['dist'])), 25), 95)))
    r = []
    for a, b in BANDS:
        sos = butter(2, [1 / b, 1 / a], 'band', fs=1.0, output='sos'); rr = sosfiltfilt(sos, R[0, :L]); xx = sosfiltfilt(sos, X[0, :L])
        nz = np.mean(rr[5:P0 - 5] ** 2); er = np.mean(rr[tS:] ** 2) - nz; ex = np.mean(xx[tS:] ** 2)
        r.append(0.5 * np.log10(er / ex) if er > 3 * nz else np.nan)
    rows.append(dict(i=i, mw=m['mw'], dist=m['dist'], ratio=r))
A = np.array([r['ratio'] for r in rows]); f = np.array([2 / (a + b) for a, b in BANDS]); med = np.nanmedian(A, 0); n = np.sum(np.isfinite(A), 0)
for (a, b), mm, nn in zip(BANDS, med, n): print(f'{a}-{b} s  log10 real/synth {mm:+.2f}  n {nn}')
ok = n >= 5; tstar = float(-np.sum(med[ok] * np.log(10) * np.pi * f[ok]) / np.sum((np.pi * f[ok]) ** 2))    # least squares through the origin
print('extra t* after S: %.1f s' % tstar)
if not DAMP: json.dump(dict(bands=BANDS, median=med.tolist(), n=n.tolist(), tstar=tstar, records=rows), open(D + 'swq_calibration.json', 'w'), indent=1)
