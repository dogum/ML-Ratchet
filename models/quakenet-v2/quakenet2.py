"""QuakeNet v2: several seismometers, one earthquake. From 41 minutes of three-component ground motion at 2-10 GSN
stations, where was the earthquake (epicentre), how deep, how big (Mw), and when (origin time)? Each with an error bar.

How it works:
  1. Each station is read by QuakeNet v1.2 (quakenet.py, unchanged): distance, back-azimuth, Mw, depth, and the
     network's 256-number summary of the record; plus the P pick time.
  2. The physics anchor: v1.2's readings combined classically (least squares on the sphere from distance and
     back-azimuth; inverse-variance Mw and log depth; origin time from the P picks and ak135 travel times).
  3. A graph network (stations = nodes, station pairs = edges) predicts the correction to the anchor, and error bars.
Learned only from simulated earthquakes laid into real station noise; examined on real USGS-located earthquakes.

numpy and scipy only. See demo.py and MODEL_CARD.md."""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pipeline as PL
from quakenet import QuakeNet, _gelu
KM = np.pi * 6371.0 / 180
CHI2_68 = 2.2789      # chi-square (2 dof) at 68 %: err_ellipse is the 68 % contour of cov_km2 (as training/score2.py scores it
                      # and the lab page draws it); fixed 22 Nov 2026, the release returned the 1-sigma (39 %) axes

# ---------------------------------------------------------------- geometry and ak135 travel times
_TT = np.load(os.path.join(HERE, 'tt_ak135.npz')); _DEP = _TT['depths']; _P, _S = _TT['P'], _TT['S']

def tt(phase, dist, depth):
    T = _P if phase == 'P' else _S; d = np.clip(np.asarray(dist, float), 0, 100); z = np.clip(np.asarray(depth, float), 2, 700)
    fj = d / 0.25; j = np.clip(np.floor(fj).astype(int), 0, T.shape[1] - 2); wj = fj - j
    i = np.clip(np.searchsorted(_DEP, z) - 1, 0, len(_DEP) - 2); wi = (z - _DEP[i]) / (_DEP[i + 1] - _DEP[i])
    return (T[i, j] * (1 - wj) + T[i, j + 1] * wj) * (1 - wi) + (T[i + 1, j] * (1 - wj) + T[i + 1, j + 1] * wj) * wi

def slowness(dist, depth): return (tt('P', np.asarray(dist) + 0.25, depth) - tt('P', np.asarray(dist) - 0.25, depth)) / 0.5

def gc_dist(lat1, lon1, lat2, lon2):
    a, b = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    c = np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(dl); s = np.hypot(np.cos(b) * np.sin(dl), np.cos(a) * np.sin(b) - np.sin(a) * np.cos(b) * np.cos(dl))
    return np.degrees(np.arctan2(s, c))

def azimuth(lat1, lon1, lat2, lon2):
    a, b = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(b), np.cos(a) * np.sin(b) - np.sin(a) * np.cos(b) * np.cos(dl))) % 360

def destination(lat, lon, az, d):
    a, l, t, dd = np.radians(lat), np.radians(lon), np.radians(az), np.radians(d)
    b = np.arcsin(np.sin(a) * np.cos(dd) + np.cos(a) * np.sin(dd) * np.cos(t)); l2 = l + np.arctan2(np.sin(t) * np.sin(dd) * np.cos(a), np.cos(dd) - np.sin(a) * np.sin(b))
    return np.degrees(b), (np.degrees(l2) + 180) % 360 - 180

def angdiff(a, b): return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180

def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon); return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

# ---------------------------------------------------------------- the classical combination (the anchor)
V12_FACTOR = dict(mw=1.0, dist=1.6404, logz=2.4165, baz=1.1398)

def combine_v12(slat, slon, R, tpick):
    """QuakeNet v1.2 per station, combined classically. R: per-station arrays (raw error bars)."""
    f = V12_FACTOR; sd = R['dist_sig'] * f['dist']; sb = np.maximum(np.degrees(1 / np.sqrt(np.maximum(R['kappa'], 1e-3))) * f['baz'], 0.5)
    c0 = [destination(a, b, z, d) for a, b, z, d in zip(slat, slon, R['baz'], R['dist'])]
    res = lambda p: np.concatenate([(gc_dist(p[0], p[1], slat, slon) - R['dist']) / sd, angdiff(azimuth(slat, slon, p[0], p[1]), R['baz']) / sb])
    u = _unit(np.array([c[0] for c in c0]), np.array([c[1] for c in c0])); w = 1 / (sd * sb); m = (u * w[:, None]).sum(0); m /= np.linalg.norm(m)
    starts = c0 + [(np.degrees(np.arcsin(m[2])), np.degrees(np.arctan2(m[1], m[0])))]; cost = [np.sum(res(np.array(s, float)) ** 2) for s in starts]; best = None
    for i in np.argsort(cost)[:2]:
        r = least_squares(res, np.array(starts[i], float), method='lm', max_nfev=200)
        if best is None or r.cost < best.cost: best = r
    lat, lon = float(np.clip(best.x[0], -90, 90)), float((best.x[1] + 180) % 360 - 180)
    cov = np.linalg.pinv(best.jac.T @ best.jac); s2 = max(1.0, 2 * best.cost / max(len(best.fun) - 2, 1)) if len(slat) > 2 else 1.0; cl = np.cos(np.radians(lat))
    C = np.array([[cov[0, 0], cov[0, 1] * cl], [cov[0, 1] * cl, cov[1, 1] * cl * cl]]) * KM * KM * s2
    wm = 1 / (R['mw_sig'] * f['mw']) ** 2; wz = 1 / (R['logz_sig'] * f['logz']) ** 2; lz = float((R['logz'] * wz).sum() / wz.sum())
    D = gc_dist(lat, lon, slat, slon); r = np.asarray(tpick) - tt('P', D, 10 ** lz)
    return dict(lat=lat, lon=lon, cov=C, mw=float((R['mw'] * wm).sum() / wm.sum()), mw_sig=float(1 / np.sqrt(wm.sum())), logz=lz, logz_sig=float(1 / np.sqrt(wz.sum())),
                depth=10 ** lz, t0=float(np.mean(r)))

# ---------------------------------------------------------------- the graph network
def cc_feats(pz):
    K = len(pz); W = np.asarray(pz, np.float32)[:, 40:160]; W = W - W.mean(1, keepdims=True); W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9; out = np.zeros((K, K, 2), np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            c = np.correlate(np.pad(W[i], 20), W[j], 'valid'); k = int(np.argmax(np.abs(c))); lag = k - 20; out[i, j] = (lag / 10, abs(c[k])); out[j, i] = (-lag / 10, abs(c[k]))
    return out

def features(slat, slon, R, tpick, ls, peak, pz, A, use_time=True, use_cc=False):
    """node (K, 16) and edge (K, K, 6 or 8) features in the anchor's frame (see training/gnn.py)"""
    dA = gc_dist(A['lat'], A['lon'], slat, slon); azA = np.radians(azimuth(A['lat'], A['lon'], slat, slon)); toA = azimuth(slat, slon, A['lat'], A['lon']); zA = 10 ** A['logz']
    sb = np.degrees(1 / np.sqrt(np.maximum(R['kappa'], 1e-3))); base = np.asarray(tpick) - tt('P', dA, zA); t0 = float(np.mean(base))
    r = np.clip((base - t0) / 10, -10, 10) if use_time else np.zeros(len(slat)); p = slowness(dA, zA) / 10 if use_time else np.zeros(len(slat))
    xn = np.stack([(R['dist'] - dA) / 10, np.log(R['dist_sig']), angdiff(R['baz'], toA) / 30, np.log(sb), R['mw'] - A['mw'], np.log(R['mw_sig']), R['logz'] - A['logz'],
                   np.log(R['logz_sig']), (np.asarray(ls) + 6) / 2, np.log10(np.maximum(peak, 1.0)), dA / 90, np.sin(azA), np.cos(azA), r, p, np.full(len(slat), len(slat) / 10)], 1)
    sep = gc_dist(slat[:, None], slon[:, None], slat[None], slon[None]) / 90; daz = azA[None] - azA[:, None]
    xe = np.stack([r[None] - r[:, None], sep, np.sin(daz), np.cos(daz), (dA[None] - dA[:, None]) / 30, np.zeros_like(sep)], -1)
    if use_cc: xe = np.concatenate([xe, cc_feats(pz)], -1)
    return np.nan_to_num(xn).astype(np.float32), np.nan_to_num(xe).astype(np.float32), t0

class GraphNet:
    def __init__(self, W, meta):
        self.W, self.m = W, meta
    def _mlp(self, name, x, n):
        for j in range(n):
            x = x @ self.W[f'{name}_{j}_w'].T + self.W[f'{name}_{j}_b']
            if j < n - 1: x = _gelu(x)
        return x
    def __call__(self, xn, xe, emb=None):
        m = self.m; K = len(xn)
        if m['use_emb']: xn = np.concatenate([xn, np.tanh(emb @ self.W['proj_w'].T + self.W['proj_b'])], -1)
        h = self._mlp('enc', xn, 2); pm = 1 - np.eye(K); cnt = max(K - 1, 1)
        for r in range(m['rounds']):
            hi = np.broadcast_to(h[:, None], (K, K, h.shape[1])); hj = np.broadcast_to(h[None], (K, K, h.shape[1]))
            msg = self._mlp(f'msg{r}', np.concatenate([hi, hj, xe], -1), 2) * pm[..., None]
            h = h + self._mlp(f'upd{r}', np.concatenate([h, msg.sum(1) / cnt], -1), 2)
        return self._mlp('out', np.concatenate([h.mean(0), h.max(0)]), 3)

class QuakeNet2:
    def __init__(self, path=os.path.join(HERE, 'quakenet2_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); W = {k: z[k].astype(np.float32) for k in z.files if k != 'meta'}
        self.gnn = GraphNet(W, self.meta); self.station = QuakeNet()

    def read_station(self, zne, p_guess):
        """one station: v1.2's window, readings, 256-number summary and the P pick (sample)"""
        W, ls, p, peak = PL.read(np.asarray(zne, float), int(p_guess)); net = self.station
        s = np.array([[(ls + 6) / 2, np.log10(max(peak, 1.0))]], np.float32)
        x = W[None].astype(np.float32); emb = self._embedding(x, s); j = net.meta['n_head'] - 1; o = emb @ net.W[f'h{j}_w'].T + net.W[f'h{j}_b']
        mu = o[0, :3] * net.sd + net.mu; sg = np.exp(np.clip(o[0, 3:6], -6, 3)) * net.sd; v = o[0, 6:8]
        R = dict(mw=mu[0], mw_sig=sg[0], dist=mu[1], dist_sig=sg[1], logz=mu[2], logz_sig=sg[2], baz=np.degrees(np.arctan2(v[0], v[1])) % 360, kappa=np.sqrt((v ** 2).sum()))
        return dict(R=R, emb=emb[0], pz=W[0, :240].astype(np.float32), ls=ls, peak=peak, pick=p)

    def _embedding(self, x, s):
        """v1.2's last hidden layer (256 numbers), the same forward pass as QuakeNet.forward without its final layer"""
        net = self.station; W = net.W
        env = np.log10(np.stack([np.convolve(np.pad(np.abs(c), 4), np.ones(9) / 9, 'valid') for c in np.abs(x).reshape(-1, x.shape[-1])]).reshape(x.shape) + 1e-3)
        from quakenet import _conv
        h = np.concatenate([x, env], 1).astype(np.float32)
        for i in range(net.meta['n_blocks']):
            y = _gelu(_conv(h, W[f'b{i}_c1_w'], W[f'b{i}_c1_b'], 2)); y = _conv(y, W[f'b{i}_c2_w'], W[f'b{i}_c2_b']); h = _gelu(y + _conv(h[:, :, ::2], W[f'b{i}_sk_w'], W[f'b{i}_sk_b']))
        z = np.concatenate([h.mean(-1), h.max(-1), s], 1)
        for j in range(net.meta['n_head'] - 1): z = _gelu(z @ W[f'h{j}_w'].T + W[f'h{j}_b'])
        return z

    def locate(self, stations, calibrated=True):
        """stations: list (2-10) of dicts with zne (3, n) velocity at 1 sps (as QuakeNet v1.2 wants it), lat, lon,
        t_start (time of the first sample, s, any common clock such as a UNIX timestamp) and p_guess (sample where P
        is expected, within ~30 s). Returns a dict: lat, lon, err_ellipse (semi-axes km and orientation), epi_sig_km,
        depth (km) with logz_sig, mw with mw_sig, t0 (origin time on the same clock) with t0_sig, plus the anchor and
        the per-station readings."""
        rd = [self.read_station(s['zne'], s['p_guess']) for s in stations]
        return self.locate_read(np.array([s['lat'] for s in stations], float), np.array([s['lon'] for s in stations], float),
                                np.array([s['t_start'] + r['pick'] for s, r in zip(stations, rd)]), rd, calibrated)

    def locate_read(self, slat, slon, tpick, rd, calibrated=True):
        K = len(rd); assert 2 <= K <= 10, 'QuakeNet v2 reads 2 to 10 stations'
        R = {k: np.array([r['R'][k] for r in rd], float) for k in rd[0]['R']}; ref = float(np.min(tpick)); tp = np.asarray(tpick, float) - ref
        A = combine_v12(slat, slon, R, tp); m = self.meta
        xn, xe, t0A = features(slat, slon, R, tp, [r['ls'] for r in rd], [r['peak'] for r in rd], [r['pz'] for r in rd], A, m['use_time'], m['use_cc'])
        o = self.gnn(xn, xe, np.stack([r['emb'] for r in rd]) if m['use_emb'] else None); f = m['sigma_factor'] if calibrated else dict(epi=1, z=1, mw=1, t0=1)
        a, b, off = np.exp(np.clip(o[5], -5, 4)), np.exp(np.clip(o[6], -5, 4)), o[7]; L = np.array([[a, 0], [off, b]]) * 100 * f['epi']; C = L @ L.T
        n, e = o[0] * 100, o[1] * 100; lat, lon = destination(A['lat'], A['lon'], np.degrees(np.arctan2(e, n)) % 360, np.hypot(n, e) / KM)
        ev, evec = np.linalg.eigh(C); s = np.exp(np.clip(o[8:11], -6, 3))
        return dict(lat=float(lat), lon=float(lon), cov_km2=C, epi_sig_km=float(np.sqrt(np.trace(C) / 2)),
                    err_ellipse=dict(major_km=float(np.sqrt(ev[1] * CHI2_68)), minor_km=float(np.sqrt(ev[0] * CHI2_68)), major_azimuth=float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)),
                    depth=float(10 ** (A['logz'] + o[2])), logz=float(A['logz'] + o[2]), logz_sig=float(s[0] * f['z']), mw=float(A['mw'] + o[3]), mw_sig=float(s[1] * f['mw']),
                    t0=float(ref + t0A + 10 * o[4]), t0_sig=float(10 * s[2] * f['t0']),
                    anchor=dict(lat=A['lat'], lon=A['lon'], depth=A['depth'], mw=A['mw'], t0=ref + A['t0']), stations=[dict(R={k: float(v) for k, v in r['R'].items()}, pick=r['pick']) for r in rd])
