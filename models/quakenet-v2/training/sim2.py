"""QuakeNet v2 training data: one simulated earthquake seen at 2-10 real GSN stations at once.

Per event (shared by every station): epicentre, depth, Mw, mechanism (double couple + small CLVD), source duration
(GCMT scaling x/ 1.6), and, for offshore sources, the water-column reverberation (pwP and multiples).
Per station: QuakeNet v1.2's physics v7 chain exactly (Syngine ak135f Green's functions; misorientation; travel-time
stretch; surface waves 6.5% slower; scattering coda; Moho Ps and sediment ringing; per-component surface-wave gains and
short-period Rayleigh delay; site amplitude scatter), laid into a real quiet-time noise window from the same station
when one exists. New for v2:
  - exact distance: the nearest Green's function (1-degree grid) is stretched about P so S-P matches ak135 at the exact
    distance;
  - absolute timing: P arrives at origin + ak135 P time(exact distance, depth) + a station/path residual (3-D Earth);
    the record is cut as the real ones are (300 s before the ak135 prediction), so the residual moves P in the record;
  - an S pick (horizontal STA/LTA trigger refined by AIC) for the classical travel-time locator.
Each record is read by QuakeNet v1.2's pipeline (STA/LTA P pick within 30 s of sample 300) and its frozen network
(rung 7, q7_zrt.pt); we store the network's 256-number summary and its 8 outputs per station (node features), plus the
pick errors and truths. Chunks of 500 events, resumable."""
import os, sys, json, glob, time, csv, numpy as np, torch
from _paths import H; sys.path.insert(0, H + 'src'); sys.path.insert(0, H + 'src/v12')
import quakesim as QS, pipeline as PL, scatter as SC, crust as CR, tt as TT, model as MD
from scipy.signal import sosfiltfilt, butter
from _paths import Q
ST = json.load(open(Q + 'stations.json')); SN = sorted(ST); SLAT = np.array([ST[s]['lat'] for s in SN]); SLON = np.array([ST[s]['lon'] for s in SN])
RES_SIG = float(os.environ.get('Q2RES', 1.5))      # P residual spread used when simulating (s); re-drawn at training time
SOS_S = butter(2, [0.02, 0.2], 'band', fs=1.0, output='sos')

def noise_pool():
    X, who = [], []
    for p in sorted(glob.glob(Q + 'wf/no_*.npz')):
        x = np.load(p)['x']
        if np.isfinite(x).all() and x.std() > 0 and -8.5 < np.log10(x.std(1).max()) < -5.3:
            X.append(x[:, :3000].astype(np.float32)); who.append(p.split('_')[-1][:-4])
    by = {}
    for i, s in enumerate(who): by.setdefault(s, []).append(i)
    return X, by

_CAT = None
def catalog():
    global _CAT
    if _CAT is None:
        R = list(csv.DictReader(open(Q + 'catalog_1990_2011.csv')))
        _CAT = np.array([[float(r['latitude']), float(r['longitude']), float(r['depth']), float(r['mag'])] for r in R if r['depth'] and r['mag']])
    return _CAT

def random_mt(rng, m0):
    strike, dip, rake = rng.uniform(0, 360), np.degrees(np.arccos(rng.uniform(0, 1))), rng.uniform(-180, 180)
    M = QS.double_couple(strike, dip, rake, m0); clvd = rng.normal(0, 0.1) * m0
    return M + clvd * np.array([2, -1, -1, 0, 0, 0]) / np.sqrt(3)

def sample_source(rng):
    C = catalog()
    if rng.random() < 0.7:          # the world's mix (1990-2011, before every exam event), as QuakeNet v1.2 rung 4
        la, lo, zc, mc = C[rng.integers(len(C))]; z = int(min(QS.DEPTHS, key=lambda v: abs(v - zc))); mw = float(np.clip(mc + rng.uniform(-0.05, 0.05), 5.4, 8.0))
    else:
        la, lo = C[rng.integers(len(C)), :2]; shallow = rng.random() < 0.6; z = int(rng.choice([v for v in QS.DEPTHS if (v <= 70) == shallow])); mw = float(rng.uniform(5.4, 8.0))
    if rng.random() < 0.8: la, lo = TT.destination(la, lo, rng.uniform(0, 360), abs(rng.normal(0, 1.5)))
    else: la, lo = np.degrees(np.arcsin(rng.uniform(-1, 1))), rng.uniform(-180, 180)
    return float(la), float(lo), z, mw

def s_pick(X, p):
    """classical S pick: first horizontal STA/LTA (10 s / 120 s) trigger above 2.5 from P+150 s to P+800 s, refined by the
    AIC minimum (Maeda 1985) within 40 s of it. Returns the sample index or -1."""
    hz = sosfiltfilt(SOS_S, X[1:3], axis=1); e = (hz ** 2).sum(0); a2 = e; ns, nl = 10, 120
    from scipy.signal import lfilter
    sta = lfilter([1 / ns], [1, -(1 - 1 / ns)], a2); lta = lfilter([1 / nl], [1, -(1 - 1 / nl)], a2); c = sta / np.maximum(lta, 1e-300)
    lo, hi = p + 150, min(p + 800, X.shape[1] - 1)
    if hi <= lo + 10: return -1
    above = np.where(c[lo:hi] > 2.5)[0]
    if not len(above): return -1
    k = lo + above[0]; a, b = max(k - 40, 0), min(k + 40, X.shape[1]); return int(a + aic_pick(hz[:, a:b]))

def aic_pick(seg):
    """AIC(q) = q log var(x[:q]) + (n-q-1) log var(x[q:]), summed over components; index of its minimum"""
    seg = np.atleast_2d(seg); n = seg.shape[1]; q = np.arange(2, n - 2); aic = np.zeros(len(q))
    for x in seg:
        c1, c2 = np.cumsum(x), np.cumsum(x * x); t1, t2 = c1[-1], c2[-1]
        v1 = c2[q - 1] / q - (c1[q - 1] / q) ** 2; m = n - q; v2 = (t2 - c2[q - 1]) / m - ((t1 - c1[q - 1]) / m) ** 2
        aic += q * np.log(np.maximum(v1, 1e-300)) + (n - q - 1) * np.log(np.maximum(v2, 1e-300))
    return int(q[np.argmin(aic)])

def ocean_params(rng, z):
    if rng.random() < (0.8 if z <= 70 else 0.7):
        Hw = float(np.exp(rng.uniform(np.log(0.3), np.log(6.0)))); return dict(tau=2 * Hw / 1.5 * 0.97, a=rng.uniform(0.15, 0.5), r=rng.uniform(0.4, 0.8))
    return None

def layers(rng, X, baz, ocean):
    """crust.apply with the source-side water layer fixed per event (QuakeNet drew it per record)"""
    Xh = sosfiltfilt(CR.HP, X, axis=1); Y = X.copy()
    if ocean is not None: Y += np.stack([CR.train(x, ocean['tau'], ocean['a'], ocean['r']) for x in Xh])
    tps = rng.uniform(2.5, 6.5); aps = rng.uniform(0.05, 0.3); Rp = aps * CR.shift(Xh[0], tps); Y[1] += -Rp * np.cos(baz); Y[2] += -Rp * np.sin(baz)
    if rng.random() < 0.5:
        ts = rng.uniform(0.5, 3.0); rs = rng.uniform(0.2, 0.5); Y += np.stack([sum(((-rs) ** k) * CR.shift(x, k * ts) for k in range(1, 5)) for x in Xh])
    return Y

def station_record(rng, src, M, hd, ocean, d, az, baz, res, nz):
    z = src[2]; dg = int(min(max(round(d), 25), 95)); G = QS.gf(z, dg); Z, R, T = QS.zrt(G, M, np.radians(az)); k = QS.stf(hd)
    Z, R, T = (np.convolve(x, k)[:len(x)] for x in (Z, R, T)); b = np.radians(baz) + np.radians(rng.normal(0, 3))
    X = QS.zne(Z, R, T, b); X = sosfiltfilt(PL.SOS_BAND, np.pad(X, ((0, 0), (200, 200))), axis=1)[:, 200:-200][:, ::2]
    sp = lambda dd: float(TT.tt('S', dd, z) - TT.tt('P', dd, z))
    st0 = np.exp(rng.uniform(np.log(0.985), np.log(1.015))); st = st0 * sp(d) / sp(dg)
    pf = 300 + res; pi = int(np.floor(pf)); frac = pf - pi; t = np.arange(X.shape[1])
    X = np.stack([np.interp((t - 60 - frac) / st + 60, t, x, left=0, right=0) for x in X]); tS = 60 + frac + sp(d) * st0
    w = float(np.clip(np.exp(rng.normal(np.log(1.065), 0.15)), 0.92, 1.5)); src_t = np.where(t < tS, t, tS + (t - tS) / w); X = np.stack([np.interp(src_t, t, x, left=0, right=0) for x in X])
    a = 0.2 * 10 ** rng.normal(0, 0.3); tau = float(np.exp(rng.uniform(np.log(60), np.log(250)))); X = SC.scatter(rng, X, a, tau)
    X = layers(rng, X, b, ocean); X = CR.surface_zrt(rng, X, tS, d, b); X = X * np.exp(rng.normal(0, np.log(1.25)))
    full = nz.astype(np.float64).copy(); n = min(X.shape[1], 3000 - (pi - 60)); full[:, pi - 60:pi - 60 + n] += X[:, :n]
    W, ls, p, peak = PL.read(full, 300); sp_i = s_pick(full, p)
    return W, ls, p, peak, pf, sp_i, pi - 60 + tS

class Encoder:
    """QuakeNet v1.2 (rung 7) frozen: window -> 256-number summary (last hidden layer) and its 8 raw outputs"""
    def __init__(self, path=H + 'ckpt/q7_zrt.pt'):
        ck = torch.load(path, weights_only=False); self.net = MD.QuakeNet(); self.net.load_state_dict(ck['state']); self.net.eval()
        self.mu, self.sd = np.array(ck['mu_y']), np.array(ck['sd_y'])
    def __call__(self, W, ls, peak, bs=64):
        S = np.stack([(np.asarray(ls) + 6) / 2, np.log10(np.maximum(np.asarray(peak, float), 1.0))], 1).astype(np.float32); E, O = [], []
        with torch.no_grad():
            for i in range(0, len(W), bs):
                x = torch.from_numpy(np.asarray(W[i:i + bs], np.float32)); h = self.net.body(MD.features(x)); z = torch.cat([h.mean(-1), h.amax(-1), torch.from_numpy(S[i:i + bs])], 1)
                hd = self.net.head; e = hd[3](hd[2](hd[1](hd[0](z))))
                e = hd[4](e); o = hd[5](e); E.append(e.numpy()); O.append(o.numpy())
        return np.concatenate(E), np.concatenate(O)

def decode(O, mu, sd):
    """v1.2 raw outputs -> mw, dist, logz with raw sigmas, baz (deg), kappa"""
    m = O[:, :3] * sd + mu; s = np.exp(np.clip(O[:, 3:6], -6, 3)) * sd; v = O[:, 6:8]
    return dict(mw=m[:, 0], mw_sig=s[:, 0], dist=m[:, 1], dist_sig=s[:, 1], logz=m[:, 2], logz_sig=s[:, 2], baz=np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360, kappa=np.sqrt((v ** 2).sum(1)))

def make_chunk(seed, n_ev, out, enc, NP, by, kmin=2, kmax=10):
    rng = np.random.default_rng(seed); Wb, LS, PK, PZ = [], [], [], []; node = {k: [] for k in ('ev', 'sta', 'd', 'az', 'baz', 'res', 'pickerr', 'tP', 'serr', 'tS', 'pick_rec', 'p_rec')}; evs = []
    for e in range(n_ev):
        la, lo, z, mw = sample_source(rng); D = TT.dist(la, lo, SLAT, SLON); cand = np.where((D >= 25) & (D <= 95))[0]
        if len(cand) < kmin: continue
        K = int(min(rng.integers(kmin, kmax + 1), len(cand))); sel = rng.choice(cand, K, replace=False)
        m0 = QS.mw_to_m0(mw); M = random_mt(rng, m0); hd = QS.half_duration(m0) * np.exp(rng.normal(0, np.log(1.6))) * (0.75 if z > 300 else 1.0); oc = ocean_params(rng, z)
        evs.append([la, lo, z, mw])
        for s in sel:
            d = float(D[s]); az = float(TT.azimuth(la, lo, SLAT[s], SLON[s])); baz = float(TT.azimuth(SLAT[s], SLON[s], la, lo)); res = float(rng.normal(0, RES_SIG))
            pool = by.get(SN[s]); nz = NP[rng.choice(pool)] if pool and rng.random() < 0.8 else NP[rng.integers(len(NP))]
            W, ls, p, peak, pf, spk, tSrec = station_record(rng, (la, lo, z, mw), M, hd, oc, d, az, baz, res, nz)
            Wb.append(W); LS.append(ls); PK.append(peak); PZ.append(W[0, :240].astype(np.float16)); tP = float(TT.tt('P', d, z)); tS = float(TT.tt('S', d, z))
            for k, v in (('ev', len(evs) - 1), ('sta', s), ('d', d), ('az', az), ('baz', baz), ('res', res), ('pickerr', p - pf), ('tP', tP),
                         ('serr', (spk - tSrec) if spk >= 0 else np.nan), ('tS', tS), ('pick_rec', p), ('p_rec', pf)): node[k].append(v)
    E, O = enc(np.stack(Wb), LS, PK)
    np.savez(out + '.tmp.npz', emb=E.astype(np.float16), pz=np.stack(PZ), out8=O.astype(np.float32), ls=np.array(LS, np.float32), peak=np.array(PK, np.float32),
             **{k: np.array(v) for k, v in node.items()}, events=np.array(evs, np.float32))
    os.replace(out + '.tmp.npz', out + '.npz')

if __name__ == '__main__':
    torch.set_num_threads(1); tag, s0, nch, n_ev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(H + 'bank', exist_ok=True); enc = Encoder(); NP, by = noise_pool(); print('noise', len(NP), 'stations with own noise', len(by), flush=True); t = time.time()
    for c in range(nch):
        out = f'{H}bank/{tag}_{s0 + c:04d}'
        if os.path.exists(out + '.npz'): continue
        make_chunk(s0 + c, n_ev, out, enc, NP, by); print(tag, s0 + c, '%.0fs' % (time.time() - t), flush=True)
