"""QuakeNet v2's graph network. Stations are nodes, station pairs are edges.

Everything is expressed in the frame of a physics anchor A: QuakeNet v1.2's per-station readings combined classically
(locate2.combine_v12: least-squares epicentre from distance + back-azimuth, inverse-variance Mw and log depth, origin
time from the P picks). The network predicts the correction to A, with error bars:
  epicentre  (north, east) offset in units of 100 km, bivariate Gaussian (Cholesky factor: 2 log-diagonals + 1 off-diagonal)
  log10 depth, Mw, origin time (units of 10 s): offset to A, Gaussian each.
Node features (per station, rotation-invariant: no absolute coordinates):
  v1.2's readings relative to A (distance - distance to A, back-azimuth - direction to A, Mw - Mw_A, log depth - log
  depth_A) with its log error bars; amplitude scale and STA/LTA strength; distance to A and the direction of the station
  seen from A (sin, cos); the P pick residual at A (pick - t0_A - ak135 P time) and the ak135 slowness there.
  Optionally v1.2's 256-number summary (projected to 32).
Edge features: residual difference, station separation, difference in direction from A, difference in distance from A,
and optionally a P-wave cross-correlation lag and its coefficient."""
import numpy as np, torch, torch.nn as nn
import tt as TT
NMAX = 10; KM = TT.KM_PER_DEG

def node_edge_features(slat, slon, R, tpick, ls, peak, A, use_time=True, cc=None):
    """one event; arrays of length K. A: anchor dict (lat, lon, logz, mw, t0). Returns xn (K, 16), xe (K, K, 6)"""
    dA = TT.dist(A['lat'], A['lon'], slat, slon); azA = np.radians(TT.azimuth(A['lat'], A['lon'], slat, slon)); toA = TT.azimuth(slat, slon, A['lat'], A['lon'])
    zA = 10 ** A['logz']; sb = np.degrees(1 / np.sqrt(np.maximum(R['kappa'], 1e-3)))
    r = (np.asarray(tpick) - A['t0'] - TT.tt('P', dA, zA)) / 10 if use_time else np.zeros(len(slat)); r = np.clip(r, -10, 10)
    p = TT.slowness(dA, zA) / 10 if use_time else np.zeros(len(slat))
    xn = np.stack([(R['dist'] - dA) / 10, np.log(R['dist_sig']), TT.angdiff(R['baz'], toA) / 30, np.log(sb), R['mw'] - A['mw'], np.log(R['mw_sig']),
                   R['logz'] - A['logz'], np.log(R['logz_sig']), (np.asarray(ls) + 6) / 2, np.log10(np.maximum(peak, 1.0)), dA / 90, np.sin(azA), np.cos(azA),
                   r, p, np.full(len(slat), len(slat) / 10)], 1)
    sep = TT.dist(slat[:, None], slon[:, None], slat[None], slon[None]) / 90; daz = azA[None] - azA[:, None]
    xe = np.stack([r[None] - r[:, None], sep, np.sin(daz), np.cos(daz), (dA[None] - dA[:, None]) / 30, np.zeros_like(sep)], -1)
    if cc is not None: xe = np.concatenate([xe, cc], -1)
    return np.nan_to_num(xn).astype(np.float32), np.nan_to_num(xe).astype(np.float32)

def targets(A, lat, lon, logz, mw, t0):
    d = TT.dist(A['lat'], A['lon'], lat, lon) * KM; a = np.radians(TT.azimuth(A['lat'], A['lon'], lat, lon))
    return np.array([d * np.cos(a) / 100, d * np.sin(a) / 100, logz - A['logz'], mw - A['mw'], (t0 - A['t0']) / 10], np.float32)

def mlp(i, h, o, n=2):
    L = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 2): L += [nn.Linear(h, h), nn.GELU()]
    return nn.Sequential(*L, nn.Linear(h, o))

class GNN(nn.Module):
    def __init__(self, fn=16, fe=6, h=96, rounds=3, use_emb=False):
        super().__init__(); self.use_emb = use_emb; self.h = h
        self.proj = nn.Linear(256, 32) if use_emb else None
        self.enc = mlp(fn + (32 if use_emb else 0), h, h)
        self.msg = nn.ModuleList([mlp(2 * h + fe, h, h) for _ in range(rounds)]); self.upd = nn.ModuleList([mlp(2 * h, h, h) for _ in range(rounds)])
        self.out = mlp(2 * h, 2 * h, 11, 3)
    def forward(self, xn, xe, mask, emb=None):
        if self.use_emb: xn = torch.cat([xn, torch.tanh(self.proj(emb))], -1)
        h = self.enc(xn) * mask[..., None]; N = h.shape[1]
        pm = (mask[:, :, None] * mask[:, None, :]) * (1 - torch.eye(N))[None]; cnt = pm.sum(2, keepdim=True).clamp(min=1)
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([h[:, :, None].expand(-1, -1, N, -1), h[:, None].expand(-1, N, -1, -1), xe], -1)) * pm[..., None]
            h = (h + upd(torch.cat([h, m.sum(2) / cnt], -1))) * mask[..., None]
        mean = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1); mx = (h - 1e4 * (1 - mask[..., None])).amax(1)
        return self.out(torch.cat([mean, mx], -1))

def unpack(o):
    """raw (B, 11) -> mean (B, 5) [n, e, logz, mw, t0] and scale params"""
    return o[:, :5], o[:, 5:7].clamp(-5, 4), o[:, 7], o[:, 8:11].clamp(-6, 3)

def loss(o, Y, beta=0.5):
    mu, ld, off, ls = unpack(o); r = Y - mu
    a, b = torch.exp(ld[:, 0]), torch.exp(ld[:, 1])                      # Sigma = L L^T, L = [[a, 0], [off, b]]
    u = r[:, 0] / a; v = (r[:, 1] - off * u) / b                           # L^-1 r
    nll_e = 0.5 * (u ** 2 + v ** 2) + ld[:, 0] + ld[:, 1]
    var = torch.exp(2 * ls); nll = 0.5 * (r[:, 2:] ** 2 / var + 2 * ls) * var.detach() ** beta
    w_e = (a * b).detach() ** beta
    return (nll_e * w_e).mean() + nll.sum(1).mean()

def decode(o, A):
    """numpy raw outputs (11,) + anchor -> dict with lat, lon, cov (km^2, north/east), logz, logz_sig, mw, mw_sig, t0, t0_sig"""
    o = np.asarray(o, float); a, b, off = np.exp(np.clip(o[5], -5, 4)), np.exp(np.clip(o[6], -5, 4)), o[7]
    L = np.array([[a, 0], [off, b]]) * 100; C = L @ L.T; n, e = o[0] * 100, o[1] * 100
    lat, lon = TT.destination(A['lat'], A['lon'], np.degrees(np.arctan2(e, n)) % 360, np.hypot(n, e) / KM); s = np.exp(np.clip(o[8:11], -6, 3))
    return dict(lat=float(lat), lon=float(lon), cov=C, logz=A['logz'] + o[2], logz_sig=s[0], depth=10 ** (A['logz'] + o[2]), mw=A['mw'] + o[3], mw_sig=s[1], t0=A['t0'] + 10 * o[4], t0_sig=10 * s[2])
