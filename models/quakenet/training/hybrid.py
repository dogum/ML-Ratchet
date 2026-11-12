"""Where in the record does the 'too deep' reading come from? Shallow dev quakes: swap the part after S between the real
record and its simulated twin, and read both hybrids."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL, simbank as SB, qnet as Q, quakesim as QS, twin as TW
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz')
net = Q.load_net(sys.argv[1]); NP = SB.noise_pool(); rng = np.random.default_rng(0); res = {k: [] for k in ('true', 'real', 'twin', 'real_surface_Z_only', 'real_surface_horiz_only', 'real_surface_long_periods', 'real_surface_short_periods', 'real_surface_realigned', 'lag')}
from scipy.signal import butter, sosfiltfilt
LO = butter(4, 1 / 25, 'low', fs=1.0, output='sos')
for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
    if s != 'dev' or m['depth'] > 70: continue
    M, _ = CP.usgs_mt(m['id']); W, ls, p, pk = TW.twin(m, M, rng, NP, layers=False, damp=True)
    T = W.astype(float) * 10 ** ls; R = z['x'][i].astype(float) * 10 ** z['ls'][i]
    zz = min(QS.DEPTHS, key=lambda v: abs(v - m['depth'])); tS = PL.PRE + int(SB.s_minus_p(zz, min(max(int(round(m['dist'])), 25), 95))) - 20
    w = np.clip((np.arange(PL.L) - tS) / 30, 0, 1)
    Rl, Tl = sosfiltfilt(LO, R, axis=1), sosfiltfilt(LO, T, axis=1); mix = lambda S: T * (1 - w) + S * w
    Hz = T.copy(); Hz[0] = mix(R)[0]; Hh = T.copy(); Hh[1:] = mix(R)[1:]
    from scipy.signal import hilbert
    eR = np.abs(hilbert(sosfiltfilt(LO, R[0]))) * w; eT = np.abs(hilbert(sosfiltfilt(LO, T[0]))) * w
    lags = np.arange(-300, 301, 2); cc = [np.dot(np.roll(eR, -k), eT) for k in lags]; lag = int(lags[int(np.argmax(cc))]); res['lag'].append(lag)
    Rs = np.roll(R, -lag, axis=1); Ha = mix(Rs)
    for nm, X in (('real_surface_realigned', Ha), ('real', R), ('twin', T), ('real_surface_Z_only', Hz), ('real_surface_horiz_only', Hh), ('real_surface_long_periods', mix(Rl + (T - Tl))), ('real_surface_short_periods', mix(Tl + (R - Rl)))):
        Wn, l = PL.normalize(X.astype(np.float32)); P = Q.predict(net, Wn[None], np.array([l]), np.array([pk])); res[nm].append(float(P['depth'][0]))
    res['true'].append(m['depth'])
t = np.array(res['true'])
print('lag of real surface waves behind twin: median %d s, 16-84%% %s' % (np.median(res['lag']), np.percentile(res['lag'], [16, 84])))
for k in res:
    if k not in ('true', 'lag'): print(f"{k:28s} pred/true median {np.median(np.array(res[k]) / t):.2f}")
