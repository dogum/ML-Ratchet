"""real vs synthetic on a dev record: plain ak135 synthetic, and with the two dev-measured corrections (slow surface waves, scattering coda)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL, scatter as SC, simbank as SB
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz')
def corrected(m, X, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(X.shape[1]); zz = min(SB.QS.DEPTHS, key=lambda v: abs(v - m['depth'])); d = min(max(int(round(m['dist'])), 25), 95)
    tS = 60 + SB.s_minus_p(zz, d); src = np.where(t < tS, t, tS + (t - tS) / 1.065); X = np.stack([np.interp(src, t, x, left=0, right=0) for x in X])
    import crust as CR; return CR.surface_zrt(rng, SC.scatter(rng, X, 0.2, 120), tS, m['dist'], np.radians(m['baz']), 1.0, 1.0)
if __name__ == '__main__':
    ids = [int(a) for a in sys.argv[1:]] or [84, 1086, 1427, 2831]; fig, ax = plt.subplots(len(ids), 1, figsize=(12, 2.6 * len(ids)))
    for a, i in zip(np.atleast_1d(ax), ids):
        m = J['meta'][i]; M, _ = CP.usgs_mt(m['id']); X = CP.synth_for(m, M); R = z['x'][i].astype(float) * 10 ** z['ls'][i]; L = min(R.shape[1], X.shape[1]); t = np.arange(L) - PL.PRE
        C = corrected(m, X); s = np.abs(R[0, :L]).max()
        a.plot(t, R[0, :L] / s + 2.4, 'k', lw=.6); a.plot(t, X[0, :L] / s, 'C0', lw=.6); a.plot(t, C[0, :L] / s - 2.4, 'C3', lw=.6)
        a.set_title(f"{m['place']}  M{m['mw']} {m['depth']} km  {m['dist']:.0f} deg  {m['station']}   black real / blue ak135 / red corrected", fontsize=8); a.set_yticks([])
    plt.tight_layout(); plt.savefig(_P.WORK + '/scratch/compare.png', dpi=80)
