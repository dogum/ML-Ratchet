"""Twin test on the dev stars. For each dev star, simulate 4 twins with its published numax, dnu, Teff, mass and
evolutionary phase (everything else from the simulator's relations), injected into the 4 quiet noise stars nearest in
Kepler magnitude, with the same observing length. Compare the real and twin log power spectra in three bands
(below the hump, the hump, above the hump) and, given a network, the readings of real and twins."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import sq, starsim as SS, simbank as SB
D = _P.WORK + '/data/'

def twins(kic, pool, calib=None, k=4, seed=0):
    Y = json.load(open(D + 'yu2018.json'))[str(kic)]; rng = np.random.default_rng(seed + kic); yr = sq.read_npz(f'{D}lc/{kic}.npz'); n = len(yr)
    kp = np.array([p[2] for p in pool]); out = []
    for j in np.argsort(np.abs(kp - Y['kp']))[:k]:
        yn = pool[j][1].astype(np.float64)
        if len(yn) < n: yn = np.concatenate([yn, np.full(n - len(yn), np.nan)])
        yn = yn[:n].copy(); yn[~np.isfinite(yr)] = np.nan          # the real star's own gaps
        s = SS.draw_star(rng, (Y['numax'], Y['numax'] * 1.000001), calib=calib); cl = Y['phase'] == 2; M = Y['M']; teff = Y['teff']
        s.update(numax=Y['numax'], dnu=Y['dnu'], teff=teff, M=M, clump=cl, amp=float(SS.yu_amp(Y['numax'], Y['dnu'], teff, cl)), fwhm=0.66 * Y['numax'] ** 0.88,
                 a_gran=3710 * Y['numax'] ** -0.613 * M ** -0.26, b1=0.317 * Y['numax'] ** 0.970, b2=0.948 * Y['numax'] ** 0.992, v1=SS.VIS[1],
                 eps_off=-0.04 if cl else 0.0, dpi=290.0 if cl else 75.0, q=0.3 if cl else 0.14, gamma0=0.12 * (teff / 4750) ** 3)
        if calib: calib(s, None)
        out.append(sq.prepare(yn + SS.synth(s, n, rng)))
    return Y, sq.prepare(yr), out

def bands(F1, numax):
    f = sq.FC; m = lambda lo, hi: (f > max(lo, 3.2)) & (f < min(hi, 250))
    return np.array([F1[m(0.3 * numax, 0.6 * numax)].mean(), F1[m(numax * np.exp(-0.12), numax * np.exp(0.12))].mean(), F1[m(1.6 * numax, 2.5 * numax)].mean() if (m(1.6 * numax, 2.5 * numax)).sum() > 3 else np.nan])

if __name__ == '__main__':
    sp = json.load(open(D + 'split.json')); pool = SB.load_pool('noise_train'); R = []
    for kic in sp['dev']:
        Y, real, tw = twins(kic, pool); br = bands(real[0], Y['numax']); bt = np.mean([bands(t[0], Y['numax']) for t in tw], 0)
        R.append(br - bt); print(kic, Y['phase'], '%.1f' % Y['numax'], 'log10 real/twin: below %.2f hump %.2f above %.2f' % tuple(br - bt), flush=True)
    R = np.array(R); print('median', np.nanmedian(R, 0), 'by numax<40', np.nanmedian(R[[json.load(open(D + 'yu2018.json'))[str(k)]['numax'] < 40 for k in sp['dev']]], 0))
