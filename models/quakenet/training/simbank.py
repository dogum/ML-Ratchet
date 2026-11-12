"""QuakeNet training data: Syngine ak135f Green's functions combined into a random earthquake (double couple plus a small
non-double-couple part, Mw 5.4-8.0, GCMT-scaled source duration) seen at a random distance (25-95 degrees) and back-azimuth,
added to a real quiet-time noise window from a GSN station, then read by the same pipeline as the real records.
Augmentations for what a 1-D Earth model leaves out: travel-time stretch +-1.5% around P, amplitude scatter (site effects,
x/ 1.25), sensor misorientation (+-5 degrees)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, glob, json, time, numpy as np
sys.path.insert(0, _P.SRC); import quakesim as QS, pipeline as PL, scatter as SC, crust as CR
from scipy.signal import sosfiltfilt
D = _P.WORK + '/data/'

SW = int(os.environ.get('QSW', 6)); LAYERS = int(os.environ.get('QLAYERS', 1)); STFV = int(os.environ.get('QSTF', 1)); NOISEV = int(os.environ.get('QNOISE', 0))   # v6: short-period surface waves depleted as on the dev quakes (crust.damp_short_sw)
#   # v4: + ocean water-column reverberation at the source and crust/sediment at the station (crust.py);
# v3: + scattering coda above 0.04 Hz with amplitude and decay set on the dev events (calib_coda.py, scatter.py);
# v1: no surface-wave delay; v2: real surface waves arrive later than ak135's, warp after S measured on the dev events (calib_sw.py)
_SP = None
def s_minus_p(z, d):
    global _SP
    if _SP is None:
        f = _P.WORK + '/data/sp_table.npy'
        if os.path.exists(f): _SP = np.load(f)
        else:
            from obspy.taup import TauPyModel
            tp = TauPyModel('ak135'); _SP = np.array([[tp.get_travel_times(zz, dd, ['S'])[0].time - tp.get_travel_times(zz, dd, ['P'])[0].time for dd in QS.DISTS] for zz in QS.DEPTHS]); np.save(f, _SP)
    return _SP[QS.DEPTHS.index(z), QS.DISTS.index(d)]

def noise_pool():
    X = []
    for p in sorted(glob.glob(D + 'wf/no_*.npz')):
        x = np.load(p)['x']
        if np.isfinite(x).all() and x.std() > 0 and -8.5 < np.log10(x.std(1).max()) < -5.3: X.append(x)    # drop dead channels and glitchy or storm-swamped windows
    return X

def random_mt(rng, m0):
    strike, dip, rake = rng.uniform(0, 360), np.degrees(np.arccos(rng.uniform(0, 1))), rng.uniform(-180, 180)
    M = QS.double_couple(strike, dip, rake, m0)
    clvd = rng.normal(0, 0.1) * m0; M = M + clvd * np.array([2, -1, -1, 0, 0, 0]) / np.sqrt(3); return M

PRIOR = int(os.environ.get('QPRIOR', 0))   # 1: 70% of events drawn from the real world's depth and magnitude mix (USGS catalog 1990-2011, before every exam event), 30% uniform
_CAT = None
def catalog_prior():
    global _CAT
    if _CAT is None:
        import csv; R = list(csv.DictReader(open(D + 'catalog_1990_2011.csv'))); _CAT = np.array([[float(r['depth']), float(r['mag'])] for r in R if r['depth'] and r['mag']])
    return _CAT

def sample(rng, NP):
    if PRIOR and rng.random() < 0.7:
        zc, mc = catalog_prior()[rng.integers(len(catalog_prior()))]; z = int(min(QS.DEPTHS, key=lambda v: abs(v - zc))); mw = float(np.clip(mc + rng.uniform(-0.05, 0.05), 5.4, 8.0))
    else:
        shallow = rng.random() < 0.6; zs = [z for z in QS.DEPTHS if (z <= 70) == shallow]; z = int(rng.choice(zs)); mw = rng.uniform(5.4, 8.0)
    d = int(rng.choice(QS.DISTS))
    m0 = QS.mw_to_m0(mw); M = random_mt(rng, m0); az, baz = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    G = QS.gf(z, d); Z, R, T = QS.zrt(G, M, az); hd = QS.half_duration(m0) * (np.exp(rng.normal(0, np.log(1.6))) * (0.75 if z > 300 else 1.0) if SW >= 5 and STFV else 1.0); k = QS.stf(hd);   # v5: real ruptures scatter around the size scaling (x/ 1.6), deep ones run shorter
    Z, R, T = (np.convolve(x, k)[:len(x)] for x in (Z, R, T))
    mis = np.radians(rng.normal(0, 3)); X = QS.zne(Z, R, T, baz + mis)                                  # sensor misorientation
    X = sosfiltfilt(PL.SOS_BAND, np.pad(X, ((0, 0), (200, 200))), axis=1)[:, 200:-200][:, ::2]        # band, 1 sps; starts at P-60
    st = np.exp(rng.uniform(np.log(0.985), np.log(1.015))); t = np.arange(X.shape[1]); X = np.stack([np.interp((t - 60) / st + 60, t, x, left=0, right=0) for x in X])
    if SW >= 2:
        w = float(np.clip(np.exp(rng.normal(np.log(1.065), 0.15)), 0.92, 1.5)); tS = 60 + s_minus_p(z, d) * st; src = np.where(t < tS, t, tS + (t - tS) / w)
        X = np.stack([np.interp(src, t, x, left=0, right=0) for x in X])
    if SW >= 3:
        a = 0.2 * 10 ** rng.normal(0, 0.3); tau = float(np.exp(rng.uniform(np.log(60), np.log(250)))); X = SC.scatter(rng, X, a, tau)
    if SW >= 4 and LAYERS: X, _ = CR.apply(rng, X, baz + mis, z)
    if SW >= 7: X = CR.surface_zrt(rng, X, 60 + s_minus_p(z, d) * st, d, baz + mis)          # v7: per-component surface-wave gains + short-period Rayleigh delay
    elif SW >= 6: X = CR.damp_short_sw(rng, X, 60 + s_minus_p(z, d) * st)
    X = X * np.exp(rng.normal(0, np.log(1.25)))
    nz = NP[rng.integers(len(NP))]; full = nz[:, :3000].astype(np.float64).copy() * (10 ** rng.normal(0.15, 0.15) if NOISEV else 1.0);   # v7: the quiet-time pool runs ~1.4x quieter than the noise before P on dev records
    p_true = 300 + int(rng.integers(-8, 9))   # real noise, P near 300 s
    n = min(X.shape[1], 3000 - (p_true - 60)); full[:, p_true - 60:p_true - 60 + n] += X[:, :n]
    W, ls, p, peak = PL.read(full, 300)
    return W, np.float32(ls), np.array([mw, d, z, np.sin(baz), np.cos(baz)], np.float32), np.float32(peak), int(p - p_true)

def make(seed, n, out):
    rng = np.random.default_rng(seed); NP = noise_pool(); print('noise windows', len(NP), flush=True); R = {k: [] for k in ('x', 'ls', 'y', 'peak', 'pickerr')}; t = time.time()
    for i in range(n):
        W, ls, y, peak, pe = sample(rng, NP)
        R['x'].append(W.astype(np.float16)); R['ls'].append(ls); R['y'].append(y); R['peak'].append(peak); R['pickerr'].append(pe)
        if (i + 1) % 2000 == 0: print(seed, i + 1, '%.0fs' % (time.time() - t), flush=True)
    np.savez(out, **{k: np.array(v) for k, v in R.items()})

if __name__ == '__main__':
    make(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
