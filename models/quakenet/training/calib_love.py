"""Surface waves on all three components, real vs ak135 synthetic of the same quake (USGS moment tensor), shallow dev
quakes. Horizontals rotated to radial/transverse with the true back-azimuth.
  energy  log10 real/synthetic RMS after S, by period band and component (Z, R Rayleigh; T Love); noise before P subtracted
  timing  group arrival (envelope peak) lag real - synthetic, by band, on Z and T"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, simbank as SB, quakesim as QS, pipeline as PL, crust as CR
from scipy.signal import butter, sosfiltfilt, hilbert
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); P0 = PL.PRE
BANDS = [(9, 12), (12, 18), (18, 25), (25, 35), (35, 50), (50, 80)]
DAMP = '--damped' in sys.argv; PATH = '--path' in sys.argv
def rt(X, baz):
    b = np.radians(baz); N, E = X[1], X[2]; R = -N * np.cos(b) - E * np.sin(b); T = N * np.sin(b) - E * np.cos(b); return np.stack([X[0], R, T])
rows = []
for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
    if s != 'dev' or m['depth'] > 70: continue
    M, _ = CP.usgs_mt(m['id']); X = CP.synth_for(m, M); Rr = z['x'][i].astype(float) * 10 ** z['ls'][i]; L = min(Rr.shape[1], X.shape[1])
    zz = min(QS.DEPTHS, key=lambda v: abs(v - m['depth'])); tS = P0 + int(SB.s_minus_p(zz, min(max(int(round(m['dist'])), 25), 95)))
    if DAMP: X = CR.damp_short_sw(None, X, tS, 1.0)
    if PATH: X = CR.surface_zrt(None, X, tS, m['dist'], np.radians(m['baz']), 1.0, 1.0)
    A, B = rt(Rr[:, :L], m['baz']), rt(X[:, :L], m['baz']); en = np.full((3, len(BANDS)), np.nan); lag = np.full((3, len(BANDS)), np.nan)
    for j, (a, b) in enumerate(BANDS):
        sos = butter(2, [1 / b, 1 / a], 'band', fs=1.0, output='sos'); ra = sosfiltfilt(sos, A, axis=1); sb = sosfiltfilt(sos, B, axis=1)
        for c in range(3):
            nz = np.mean(ra[c, 5:P0 - 5] ** 2); er = np.mean(ra[c, tS:] ** 2) - nz; ex = np.mean(sb[c, tS:] ** 2)
            if er > 3 * nz:
                en[c, j] = 0.5 * np.log10(er / ex)
                e1 = np.abs(hilbert(ra[c])); e2 = np.abs(hilbert(sb[c])); e1[:tS] = 0; e2[:tS] = 0; lag[c, j] = float(np.argmax(e1) - np.argmax(e2))
    rows.append(dict(i=i, mw=m['mw'], dist=m['dist'], depth=m['depth'], lat=m['lat'], lon=m['lon'], station=m['station'], en=en.tolist(), lag=lag.tolist()))
EN = np.array([r['en'] for r in rows]); LG = np.array([r['lag'] for r in rows]); dist = np.array([r['dist'] for r in rows])
print('n', len(rows), '| log10 real/synth after S (median, n)')
for j, (a, b) in enumerate(BANDS):
    print(f'{a:>3}-{b:<3}s  Z {np.nanmedian(EN[:, 0, j]):+.2f} ({np.sum(np.isfinite(EN[:, 0, j]))})  R {np.nanmedian(EN[:, 1, j]):+.2f} ({np.sum(np.isfinite(EN[:, 1, j]))})  T {np.nanmedian(EN[:, 2, j]):+.2f} ({np.sum(np.isfinite(EN[:, 2, j]))})'
          f'   | lag Z {np.nanmedian(LG[:, 0, j]):+5.0f} s (iqr {np.nanpercentile(LG[:, 0, j], 25):+.0f}..{np.nanpercentile(LG[:, 0, j], 75):+.0f})  T {np.nanmedian(LG[:, 2, j]):+5.0f} s')
TZ = EN[:, 2, :] - EN[:, 0, :]; print('Love minus Rayleigh (T - Z) real/synth, by band:', np.round(np.nanmedian(TZ, 0), 2))
# dispersion: the short-period lag relative to the long-period lag, per record, per 1000 km
d = LG[:, 0, 1] - LG[:, 0, 4]; print('Z lag 12-18 s minus 35-50 s: median %+.0f s, iqr %s, per 1000 km %+.1f s' % (np.nanmedian(d), np.nanpercentile(d, [25, 75]).round(0), np.nanmedian(d / (dist * 0.11119))))
if not (DAMP or PATH): json.dump(dict(bands=BANDS, records=rows), open(D + 'love_calibration.json', 'w'), indent=1)
