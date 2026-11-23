"""TsunamiNet inference: numpy (+ scipy) only.
From the first DART bottom-pressure records after a subduction earthquake, a normalizing flow gives the posterior of
the tsunami source (Mw, along-strike centroid, deep-row moment fraction, rupture length) on NOAA SIFT unit sources,
and the posterior is pushed through linear shallow-water Green's functions to forecast the wave amplitude at other DARTs.

    net = TsunamiNet()                       # weights next to this file
    gf  = GreensFunctions()                  # compact 8' Green's-function bank shipped with the release
    ev  = gf.event(lon, lat, records)        # records: {site: (eta_m[1441], mask[1441])} minute series from origin
    post = net.posterior(ev)                 # dict with samples, medians, 68%/90% ranges (dev-widened)
    fc  = gf.forecast(ev, post, site, eta, mask)   # amplitude forecast at a held-out DART
"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NCOL, HALF, WIN0, WIN1 = 9, 4, -20, 70
NW = WIN1 - WIN0; K = 3; TA_MAX = 290; MIN_COV = 0.5; MIN_SCALE = 0.05
MU_RIG = 4.0e10; UNIT_AREA = 100e3 * 50e3

def _gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

def _conv1d(x, w, b, stride, pad):
    """x (B, C, L), w (O, C, k)."""
    B, C, L = x.shape; O, _, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    Lo = (L + 2 * pad - k) // stride + 1
    idx = np.arange(Lo)[:, None] * stride + np.arange(k)[None]
    cols = xp[:, :, idx]                         # B, C, Lo, k
    return np.einsum('bclk,ock->bol', cols, w, optimize=True) + b[None, :, None]

def nnls_features(X, M):
    """SIFT-style non-negative least squares on the input windows: slip (m) on the 18 basis unit sources. Returns 23
    features: Mw_nnls - 8.3, centroid column, deep fraction, relative misfit, log10(total slip + 0.1), 18 slip shares."""
    from scipy.optimize import nnls
    X = np.asarray(X, np.float64); out = np.zeros(23, np.float32); A, b = [], []
    for k in range(X.shape[0]):
        if not M[k]: continue
        m = X[k, 1] > 0; A.append(X[k, 2:][:, m].T); b.append(X[k, 0, m] * 10.0)
    if not A: return out
    A = np.concatenate(A); b = np.concatenate(b); s, res = nnls(A, b, maxiter=500); tot = s.sum()
    if tot <= 0:
        out[0] = -1.5; out[3] = 1.0; out[4] = -1.0; return out
    cols = np.tile(np.arange(9) - 4, 2)
    out[0] = (np.log10(MU_RIG * UNIT_AREA * tot) - 9.1) / 1.5 - 8.3; out[1] = (cols * s).sum() / tot; out[2] = s[9:].sum() / tot
    out[3] = res / (np.linalg.norm(b) + 1e-9); out[4] = np.log10(tot + 0.1); out[5:] = s / tot
    return out

class TsunamiNet:
    def __init__(self, path=os.path.join(HERE, 'tsunaminet_weights.npz')):
        z = np.load(path, allow_pickle=False); self.meta = json.loads(str(z['meta']))
        self.conv = [(z[f'c{i}w'], z[f'c{i}b'], s, p) for i, (s, p) in enumerate(self.meta['conv'])]
        self.st = [(z['st0w'], z['st0b']), (z['st1w'], z['st1b'])]; self.cx = [(z['ctx0w'], z['ctx0b']), (z['ctx1w'], z['ctx1b'])]
        self.flow = [(z[f'f{i}m'], [(z[f'f{i}w{j}'], z[f'f{i}b{j}']) for j in range(3)]) for i in range(self.meta['n_flow'])]
        self.mu = np.array(self.meta['mu'], np.float32); self.sd = np.array(self.meta['sd'], np.float32)
        self.cal = self.meta.get('calibration', {})

    def context(self, X, S, M):
        X = np.asarray(X, np.float32); S = np.asarray(S, np.float32); M = np.asarray(M, bool)
        h = X
        for w, b, s, p in self.conv: h = _gelu(_conv1d(h, w, b, s, p))
        f = np.concatenate([h.mean(-1), h.max(-1), S], 1)
        e = _gelu(f @ self.st[0][0].T + self.st[0][1]) @ self.st[1][0].T + self.st[1][1]
        m = M[:, None].astype(np.float32)
        mean = (e * m).sum(0) / max(m.sum(), 1); mx = (e - 1e4 * (1 - m)).max(0)
        c = np.concatenate([mean, mx, [m.sum() / K]])
        if self.cx[0][0].shape[1] > len(c):          # physics anchor: the NNLS solution on the same windows
            c = np.concatenate([c, nnls_features(X, M)])
        c = c[None]
        return _gelu(c @ self.cx[0][0].T + self.cx[0][1]) @ self.cx[1][0].T + self.cx[1][1]

    def sample(self, X, S, M, n=2000, seed=0, z=None, raw=False):
        """X (K, 20, 90), S (K, 3), M (K,) -> (n, 4) samples of (Mw, centroid along strike [100 km], deep fraction, log10 L)."""
        c = self.context(X, S, M)
        z = np.random.default_rng(seed).standard_normal((n, 4)).astype(np.float32) if z is None else np.asarray(z, np.float32)
        C = np.repeat(c, len(z), 0)
        for m, net in reversed(self.flow):
            h = np.concatenate([z * m, C], 1)
            h = _gelu(h @ net[0][0].T + net[0][1]); h = _gelu(h @ net[1][0].T + net[1][1]); o = h @ net[2][0].T + net[2][1]
            s, t = o[:, :4], o[:, 4:]
            s = 2.5 * np.tanh(s / 2.5) * (1 - m); t = t * (1 - m)
            z = z * m + (1 - m) * ((z - t) * np.exp(-s))
        return z * self.sd + self.mu

    def posterior(self, ev, n=2000, seed=0):
        P = self.sample(ev['X'], ev['S'], ev['M'], n, seed)
        med = np.median(P, 0); wf = self.cal.get('widen', {'0': 1.0, '1': 1.0})
        def rng(k, q):
            lo, hi = np.percentile(P[:, k], [50 - q / 2, 50 + q / 2])
            return [float(med[k] - wf[str(k)] * (med[k] - lo)), float(med[k] + wf[str(k)] * (hi - med[k]))]
        return dict(samples=P, mw=float(med[0]), mw68=rng(0, 68), mw90=rng(0, 90), centroid_km=float(100 * med[1]),
                    centroid_km68=[100 * v for v in rng(1, 68)], deep_fraction=float(np.clip(med[2], 0, 1)), length_km=float(10 ** med[3]))


# ---------------------------------------------------------------- geometry (NOAA SIFT unit sources)
def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def mw_to_m0(mw): return 10 ** (1.5 * mw + 9.1)

class GreensFunctions:
    """Compact 8' linear shallow-water Green's functions (mm of sea level per m of slip on each SIFT unit source),
    int8-quantised windows [arrival-60, arrival+300] min per (DART site, unit source), plus the unit-source table."""
    def __init__(self, path=os.path.join(HERE, 'gf_compact.npz')):
        z = np.load(path, allow_pickle=False)
        self.sites = [str(s) for s in z['sites']]; self.site_ll = z['site_ll']; self.sidx = {s: i for i, s in enumerate(self.sites)}
        self.q = z['q']; self.scale = z['scale']; self.arr = z['arr']; self.W0 = int(z['w0'])
        self.units = json.loads(str(z['units'])); self.centers = z['centers']; self.chains = json.loads(str(z['chains']))
        self.col_ll = np.array([[c['lon'], c['lat']] for ch in self.chains for c in ch])
        self.colist = [(ci, j) for ci, ch in enumerate(self.chains) for j in range(len(ch))]

    def series(self, site, u, n=1441):
        i = self.sidx[site]; out = np.zeros(n)
        a = int(self.arr[i, u])
        if a < 0: return out
        v = self.q[i, u].astype(np.float64) * self.scale[i, u] / 127.0
        lo = a + self.W0; idx = np.arange(lo, lo + len(v)); ok = (idx >= 0) & (idx < n)
        out[idx[ok]] = v[ok]; return out

    def arrival(self, site, units):
        a = [int(self.arr[self.sidx[site], u]) for u in units if u >= 0 and self.arr[self.sidx[site], u] >= 0]
        return min(a) if a else None

    def frame(self, lon, lat):
        d = hav(lat, lon % 360, self.col_ll[:, 1], self.col_ll[:, 0]); n = int(np.argmin(d)); ci, j = self.colist[n]; ch = self.chains[ci]
        units = -np.ones(2 * NCOL, int)
        for m in range(NCOL):
            jj = j - HALF + m
            if 0 <= jj < len(ch): units[m] = ch[jj]['b']; units[NCOL + m] = ch[jj]['a']
        return dict(chain=ci, j=j, units=units, s0=ch[j]['s'], dist_km=float(d[n]))

    def wave(self, site, units, slips, delay=0.0):
        w = sum(s * self.series(site, u) for u, s in zip(units, slips)) / 1000.0
        if delay: w = np.interp(np.arange(len(w)) - delay, np.arange(len(w)), w, left=0.0)
        return w

    def basis(self, site, units, ta):
        B = np.zeros((2 * NCOL, NW), np.float32)
        for k, u in enumerate(units):
            if u >= 0:
                g = self.series(site, u); i0 = ta + WIN0; seg = g[max(i0, 0): i0 + NW]; B[k, max(0, -i0): max(0, -i0) + len(seg)] = seg
        return B

    def station_block(self, site, units, ta, eta, mask):
        B = self.basis(site, units, ta); scale = float(np.abs(B).max())
        i0 = ta + WIN0; o = np.zeros(NW, np.float32); m = np.zeros(NW, np.float32)
        lo, hi = max(i0, 0), min(i0 + NW, len(eta)); o[lo - i0: hi - i0] = eta[lo:hi]; m[lo - i0: hi - i0] = mask[lo:hi]; o *= m
        if scale <= 0: return None, None, float(m.mean()), scale
        x = np.concatenate([(o * 1000.0 / scale / 10.0)[None], m[None], B / scale], 0)
        s = np.array([ta / 180.0, np.log10(scale) / 2.0, np.log10(np.abs(o).max() * 1000.0 / scale + 1e-3) / 2.0], np.float32)
        return x, s, float(m.mean()), scale

    def event(self, lon, lat, records):
        """records: {site: (eta [m, minutes from origin], mask)}. Picks up to 3 input DARTs by predicted arrival."""
        fr = self.frame(lon, lat); cen = [u for u in (fr['units'][HALF], fr['units'][NCOL + HALF]) if u >= 0]
        cands = []
        for site, (eta, mask) in records.items():
            if site not in self.sidx: continue
            ta = self.arrival(site, cen)
            if ta is None or ta > TA_MAX: continue
            x, s, cov, scale = self.station_block(site, fr['units'], ta, np.asarray(eta, np.float32), np.asarray(mask, bool))
            if x is not None and cov >= MIN_COV and scale >= MIN_SCALE: cands.append((ta, x, s, site))
        cands.sort(key=lambda c: c[0]); cands = cands[:K]
        X = np.zeros((K, 2 + 2 * NCOL, NW), np.float32); S = np.zeros((K, 3), np.float32); M = np.zeros(K, bool)
        for k, c in enumerate(cands): X[k], S[k], M[k] = c[1], c[2], True
        return dict(frame=fr, X=X, S=S, M=M, inputs=[c[3] for c in cands], input_ta=[c[0] for c in cands], lon=lon, lat=lat)

    def theta_slip(self, fr, theta):
        mw, sc, fd, lL = float(theta[0]), float(theta[1]) * 100, float(np.clip(theta[2], 0, 1)), float(theta[3])
        L = 10 ** lL; ch = self.chains[fr['chain']]; s = np.array([q['s'] for q in ch]) - fr['s0']
        xs = np.linspace(sc - L / 2, sc + L / 2, 48); w = np.clip(np.sin(np.pi * (xs - xs[0]) / L), 0, None) ** 0.7 + 1e-3
        colw = np.array([w[(xs >= sk - 50) & (xs < sk + 50)].sum() for sk in s])
        if colw.sum() <= 0: colw[np.argmin(np.abs(s - sc))] = 1.0
        colw /= colw.sum(); units, mom = [], []
        for jj in np.nonzero(colw > 1e-4)[0]:
            units += [ch[jj]['b'], ch[jj]['a']]; mom += [colw[jj] * (1 - fd), colw[jj] * fd]
        mom = np.array(mom); mom /= mom.sum()
        return np.array(units), mom * mw_to_m0(mw) / (MU_RIG * UNIT_AREA)

    def centroid_latlon(self, fr, sc, fdeep):
        ch = self.chains[fr['chain']]; s = np.array([c['s'] for c in ch]); x = fr['s0'] + 100.0 * sc
        def interp(P):
            return np.degrees(np.interp(x, s, np.unwrap(np.radians(P[:, 0])))) % 360, np.interp(x, s, P[:, 1])
        lb, tb = interp(self.centers[[c['b'] for c in ch]]); la, ta = interp(self.centers[[c['a'] for c in ch]])
        return float((1 - fdeep) * lb + fdeep * la), float((1 - fdeep) * tb + fdeep * ta)

    def forecast(self, ev, post, site, a, b, mask=None, n_use=100, delay_frac=None, widen=None, model_sd=None):
        """Max |eta| (m) at `site` in minutes [a, b] (observed minutes `mask` if given): median and 68% range."""
        cal = getattr(self, 'cal', {})
        delay_frac = cal.get('delay_frac', 0.0) if delay_frac is None else delay_frac
        gain = cal.get('amp_gain', 1.0)
        P = post['samples']; S = P[np.linspace(0, len(P) - 1, n_use).astype(int)]
        ta = self.arrival(site, [u for u in (ev['frame']['units'][HALF], ev['frame']['units'][NCOL + HALF]) if u >= 0]) or 0
        amps = []
        for th in S:
            u, s = self.theta_slip(ev['frame'], th); w = self.wave(site, u, s, ta * delay_frac)[a:b + 1]
            amps.append(gain * np.abs(w[mask] if mask is not None else w).max())
        lg = np.log10(np.maximum(amps, 1e-6)); m = np.median(lg)
        sd = np.sqrt(((widen or 1.0) * lg.std()) ** 2 + (model_sd or 0.0) ** 2)
        return dict(median=float(10 ** m), lo68=float(10 ** (m - sd)), hi68=float(10 ** (m + sd)))
