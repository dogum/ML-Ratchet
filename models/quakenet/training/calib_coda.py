"""P coda on the dev events: real seismograms keep ringing after P (scattering in a 3-D Earth); 1-D synthetics are quiet
between the main phases. Fit the scattering coda (amplitude a, decay tau) so synthetic records with the event's USGS moment
tensor match the real energy 30-100 s and 100-200 s after P relative to the first 30 s (vertical, 0.05-0.3 Hz), on dev
records of M6.0+ events. Noise energy before P is subtracted; records where the coda sinks into noise are skipped."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL, scatter as SC
from scipy.signal import butter, sosfiltfilt
D = _P.WORK + '/data/'; SOS = butter(2, [0.05, 0.3], 'band', fs=1.0, output='sos'); P = PL.PRE
J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz')
e = lambda x, a, b: float(np.mean(x[P + a:P + b] ** 2))
recs = []
for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
    if s != 'dev' or m['mw'] < 6.0: continue
    M, _ = CP.usgs_mt(m['id'])
    if M is None: continue
    R = sosfiltfilt(SOS, z['x'][i].astype(float)[0] * 10 ** z['ls'][i]); nz = e(R, -55, -5); p = e(R, 0, 30) - nz
    w1, w2 = e(R, 30, 100) - nz, e(R, 100, 200) - nz
    if p <= 0 or w1 < 1.5 * nz or w2 < 1.5 * nz: print('skip', i, 'coda in noise'); continue
    recs.append(dict(i=i, mw=m['mw'], depth=m['depth'], dist=m['dist'], X=CP.synth_for(m, M), r1=np.log10(w1 / p), r2=np.log10(w2 / p)))
print('records', len(recs))
def ratios(X, rng, a, tau):
    Y = sosfiltfilt(SOS, SC.scatter(rng, X, a, tau)[0]) if a > 0 else sosfiltfilt(SOS, X[0]); p = e(Y, 0, 30)
    return np.log10(e(Y, 30, 100) / p), np.log10(e(Y, 100, 200) / p)
best = None; grid = []
for tau in (30, 60, 100, 160, 250):
    for a in (0.0, 0.08, 0.15, 0.25, 0.4, 0.6, 0.9):
        rng = np.random.default_rng(0); res = []
        for r in recs:
            s1, s2 = np.mean([ratios(r['X'], rng, a, tau) for _ in range(3)], 0); res.append((r['r1'] - s1, r['r2'] - s2))
        res = np.array(res); cost = float(np.abs(np.median(res, 0)).sum() + 0.5 * np.median(np.abs(res)))
        grid.append(dict(a=a, tau=tau, med1=float(np.median(res[:, 0])), med2=float(np.median(res[:, 1])), cost=cost))
        if best is None or cost < best['cost']: best = grid[-1]; best_res = res
        print(grid[-1], flush=True)
# per-record scatter of the best fit gives the spread of a across events (log10 residual / 2 in amplitude)
sd = float(np.std(best_res.mean(1)) / 2)
json.dump(dict(best=best, log10_a_sd=sd, n=len(recs), grid=grid, real=[dict(i=r['i'], mw=r['mw'], depth=r['depth'], dist=r['dist'], r1=r['r1'], r2=r['r2']) for r in recs]), open(D + 'coda_calibration.json', 'w'), indent=1)
print('best', best, 'log10 a sd', round(sd, 2))
