"""Twin test on the dev records: simulate each dev quake (USGS moment tensor, USGS depth, the station's true distance and
direction) through the full simulator chain, in real noise at the record's own noise level, and let QuakeNet read both.
If the twin reads right and the real record doesn't, the simulator is missing physics; toggles show which."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import compare as CP, pipeline as PL, simbank as SB, scatter as SC, crust as CR, qnet as Q, quakesim as QS
from obspy.geodetics import gps2dist_azimuth
D = _P.WORK + '/data/'; J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); S = json.load(open(D + 'stations.json'))
def twin(m, M, rng, NP, hd_factor=1.0, layers=True, damp=False, noise=1.0, smear=False, love=False, disp=False, s_disp=1.0, zrt=False):
    zz = min(QS.DEPTHS, key=lambda v: abs(v - m['depth'])); d = min(max(int(round(m['dist'])), 25), 95)
    _, az, baz = gps2dist_azimuth(m['lat'], m['lon'], S[m['station']]['lat'], S[m['station']]['lon'])
    G = QS.gf(zz, d); Z, R, T = QS.zrt(G, M, np.radians(az)); m0 = np.sqrt(0.5 * np.sum(M[:3] ** 2) + np.sum(M[3:] ** 2)); k = QS.stf(QS.half_duration(m0) * hd_factor)
    Z, R, T = (np.convolve(x, k)[:len(x)] for x in (Z, R, T)); X = QS.zne(Z, R, T, np.radians(baz))
    from scipy.signal import sosfiltfilt
    X = sosfiltfilt(PL.SOS_BAND, np.pad(X, ((0, 0), (200, 200))), axis=1)[:, 200:-200][:, ::2]; t = np.arange(X.shape[1])
    tS = 60 + SB.s_minus_p(zz, d); src = np.where(t < tS, t, tS + (t - tS) / 1.065); X = np.stack([np.interp(src, t, x, left=0, right=0) for x in X])
    X = SC.scatter(rng, X, 0.2, 120)
    if layers: X, _ = CR.apply(rng, X, np.radians(baz), m['depth'])
    if damp: X = CR.damp_short_sw(rng, X, tS, 1.0)
    if zrt: X = CR.surface_zrt(rng, X, tS, m['dist'], np.radians(baz), 1.0, s_disp, disp)
    if smear: X = CR.smear_short_sw(rng, X, tS, smear, 50)
    if (love or disp) and not zrt: X = CR.path_surface(rng, X, tS, m['dist'], np.radians(baz), love, disp, 1.0, s_disp)
    nz = NP[rng.integers(len(NP))][:, :3000].astype(float); full = nz.copy() * noise; n = min(X.shape[1], 3000 - 240); full[:, 240:240 + n] += X[:, :n]
    return PL.read(full, 300)
if __name__ == '__main__':
    net = Q.load_net(sys.argv[1]); NP = SB.noise_pool(); rng = np.random.default_rng(0); rows = []
    for i, (m, s) in enumerate(zip(J['meta'], J['split'])):
        if s != 'dev': continue
        M, _ = CP.usgs_mt(m['id'])
        if M is None: continue
        rows.append((i, m, M))
    variants = {'v6': dict(layers=False, damp=True), 'zrt_gains': dict(layers=False, zrt=True), 'zrt_gains+disp': dict(layers=False, zrt=True, disp=True)}
    real = Q.predict(net, z['x'][[r[0] for r in rows]], z['ls'][[r[0] for r in rows]], z['peak'][[r[0] for r in rows]])
    true = np.array([r[1]['depth'] for r in rows]); sh = true <= 70
    print('n', len(rows), 'real: shallow pred/true median %.2f, deep %.2f' % (np.median(real['depth'][sh] / true[sh]), np.median(real['depth'][~sh] / true[~sh])))
    out = {}
    for nm, kw in variants.items():
        W = [twin(m, M, rng, NP, **kw) for _, m, M in rows]; P = Q.predict(net, np.stack([w[0] for w in W]), np.array([w[1] for w in W]), np.array([w[3] for w in W]))
        print(nm, 'twin: shallow pred/true median %.2f, deep %.2f | mw bias shallow %+.2f' % (np.median(P['depth'][sh] / true[sh]), np.median(P['depth'][~sh] / true[~sh]), np.median(P['mw'][sh] - np.array([r[1]['mw'] for r in rows])[sh])), flush=True)
        out[nm] = P['depth'].tolist()
    json.dump(dict(true=true.tolist(), real=real['depth'].tolist(), **out), open(D + 'twin_dev.json', 'w'))
