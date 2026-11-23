"""TsunamiNet simulator: random interface earthquakes on SIFT unit-source columns -> Okada slip on unit sources ->
DART waveforms by Green's-function superposition (gf_bank.py) -> laid into real quiet DART noise windows.
Also: the epicentre frame (9 columns x 2 rows of unit sources around the epicentre) the network reads, the source
parameters theta = (Mw, along-strike centroid [100 km], deep-row moment fraction, log10 length [km]), and
helpers to turn theta back into slip (for posterior-predictive forecasts)."""
import os, re, json, numpy as np
from scipy.signal import sosfilt
from _paths import DATA, BANK
from dart import SOS
MU = 4.0e10; UNIT_AREA = 100e3 * 50e3
NCOL = 9; HALF = NCOL // 2
WIN0, WIN1 = -20, 70          # input window around the predicted arrival (minutes)
NW = WIN1 - WIN0
PHYS = {'p0': dict(delay_frac=0.0, amp_sigma=0.0, shift_sigma=0.0, rupture_vel=None)}

def m0(mw): return 10 ** (1.5 * mw + 9.1)
def mw_of(m): return (np.log10(np.maximum(m, 1.0)) - 9.1) / 1.5

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def unit_center(u):
    """Centre of the unit rectangle (lon, lat): lower-edge centre moved W/2 cos(dip) toward the up-dip side."""
    d = 0.5 * u['width'] * np.cos(np.radians(u['dip']))
    az = np.radians(u['strike'] - 90.0)
    dlat = d * np.cos(az) / 111.195; dlon = d * np.sin(az) / (111.195 * np.cos(np.radians(u['lat'])))
    return (u['lon'] + dlon) % 360, u['lat'] + dlat

class Geometry:
    def __init__(self):
        self.U = json.load(open(os.path.join(DATA, 'unit_sources_pacific.json')))
        self.uidx = {u['name']: i for i, u in enumerate(self.U)}
        self.centers = np.array([unit_center(u) for u in self.U])
        cols = {}
        for i, u in enumerate(self.U):
            m = re.match(r'([a-z]{4})([a-z])(\d+)', u['name']); z, r, k = m.group(1), m.group(2), int(m.group(3))
            if r in 'ab': cols.setdefault((z, k), {})[r] = i
        # ordered column chains per zone (a and b both present), broken where neighbours are > 250 km apart
        self.chains = []
        for z in sorted(set(k[0] for k in cols)):
            ks = sorted(k for (zz, k) in cols if zz == z and len(cols[(zz, k)]) == 2)
            chain = []
            for k in ks:
                a, b = cols[(z, k)]['a'], cols[(z, k)]['b']
                c = 0.5 * (self.centers[a] + self.centers[b])
                if chain and hav(chain[-1]['lat'], chain[-1]['lon'], c[1], c[0]) > 250:
                    self.chains.append(chain); chain = []
                chain.append(dict(zone=z, k=k, a=a, b=b, lon=c[0], lat=c[1]))
            if chain: self.chains.append(chain)
        for ch in self.chains:
            s = [0.0]
            for p, q in zip(ch[:-1], ch[1:]): s.append(s[-1] + hav(p['lat'], p['lon'], q['lat'], q['lon']))
            for c, v in zip(ch, s): c['s'] = v
        self.colist = [(ci, j) for ci, ch in enumerate(self.chains) for j in range(len(ch))]
        self.col_ll = np.array([[self.chains[ci][j]['lon'], self.chains[ci][j]['lat']] for ci, j in self.colist])

    def frame(self, lon, lat):
        """Epicentre frame: nearest column; returns dict(chain, j, units[2*NCOL] (b row then a row; -1 if none), s0)."""
        d = hav(lat, lon, self.col_ll[:, 1], self.col_ll[:, 0]); n = int(np.argmin(d))
        ci, j = self.colist[n]; ch = self.chains[ci]
        units = -np.ones(2 * NCOL, int)
        for m in range(NCOL):
            jj = j - HALF + m
            if 0 <= jj < len(ch): units[m] = ch[jj]['b']; units[NCOL + m] = ch[jj]['a']
        return dict(chain=ci, j=j, units=units, s0=ch[j]['s'], dist_km=float(d[n]))

    def theta_to_latlon(self, fr, s_c_100km, f_deep):
        """Map (along-strike centroid, deep fraction) to lon/lat by interpolating column centres of rows b and a."""
        ch = self.chains[fr['chain']]; s = np.array([c['s'] for c in ch]); x = fr['s0'] + 100.0 * np.asarray(s_c_100km)
        cb = self.centers[[c['b'] for c in ch]]; ca = self.centers[[c['a'] for c in ch]]
        def interp(P):
            lon = np.interp(x, s, np.unwrap(np.radians(P[:, 0]))); return np.degrees(lon) % 360, np.interp(x, s, P[:, 1])
        lb, tb = interp(cb); la, ta = interp(ca)
        return (1 - f_deep) * lb + f_deep * la, (1 - f_deep) * tb + f_deep * ta

    def project_point(self, fr, lon, lat):
        """Along-strike coordinate (100 km units, relative to the frame column) of a point: nearest location on the chain."""
        ch = self.chains[fr['chain']]; s = np.array([c['s'] for c in ch])
        ss = np.linspace(s[0], s[-1], max(2, int((s[-1] - s[0]) / 5) + 1))
        lo = np.degrees(np.interp(ss, s, np.unwrap(np.radians([c['lon'] for c in ch])))); la = np.interp(ss, s, [c['lat'] for c in ch])
        k = int(np.argmin(hav(lat, lon, la, lo % 360)))
        return (ss[k] - fr['s0']) / 100.0

class GFBank:
    def __init__(self, sites=None, gfdir=os.path.join(BANK, 'gf8'), highpass=True):
        self.dir = gfdir; self.G = {}; self.meta = {}
        names = sites or sorted(f[:-4] for f in os.listdir(gfdir) if f.endswith('.npz') and not f.endswith('.tmp.npz'))
        for n in names:
            z = np.load(os.path.join(gfdir, n + '.npz'))
            G = z['G'].astype(np.float32)
            if highpass: G = sosfilt(SOS, G, axis=1).astype(np.float32)   # same causal 3-h high-pass as the DART records
            self.G[n] = G.astype(np.float16); self.meta[n] = dict(lat=float(z['lat']), lon=float(z['lon']), depth=float(z['depth']))
    def wave(self, site, slip_units, slips, delays=None):
        """eta (m) at a site, 1441 minutes, from slips (m) on unit indices; optional per-unit delays (minutes, >= 0)."""
        G = self.G[site]; out = np.zeros(G.shape[1], np.float64)
        for k, (u, s) in enumerate(zip(slip_units, slips)):
            g = G[u].astype(np.float64) * s
            if delays is not None and delays[k] > 0:
                d = delays[k]; i = int(d); f = d - i
                g = np.concatenate([np.zeros(i + 1), g])[: G.shape[1] + 1]
                g = (1 - f) * g[1:] + f * g[:-1]
            out += g
        return out / 1000.0
    def arrival(self, site, units):
        """Predicted first arrival (minutes): the earliest, over the given unit sources, of the first time a unit's
        response exceeds 2% of its own maximum (per-unit arrivals are what the release ships)."""
        best = None
        for u in units:
            if u < 0: continue
            g = np.abs(self.G[site][u].astype(np.float32)); mx = g.max()
            if mx <= 1e-3: continue
            a = int(np.argmax(g > 0.02 * mx))
            best = a if best is None else min(best, a)
        return best

def random_source(geo, rng, mw_range=(7.3, 9.3)):
    """A random interface earthquake. Returns dict with epicentre, unit slips, theta."""
    ci, j = geo.colist[rng.integers(len(geo.colist))]; ch = geo.chains[ci]
    c = ch[j]
    # epicentre: column centre + jitter (along +-50 km, across +-40 km)
    lon, lat = c['lon'] + rng.uniform(-0.5, 0.5) / np.cos(np.radians(c['lat'])) * 0.6, c['lat'] + rng.uniform(-0.45, 0.45)
    fr = geo.frame(lon, lat)
    ch = geo.chains[fr['chain']]; s = np.array([q['s'] for q in ch]) - fr['s0']
    mw = rng.uniform(*mw_range)
    L = 10 ** (-2.477 + 0.585 * mw + 0.18 * rng.standard_normal())       # Strasser et al. (2010) interface
    L = float(np.clip(L, 40, 1400))
    u = rng.uniform(0, 1); x0 = -u * L; x1 = x0 + L                       # epicentre inside the rupture
    xs = np.linspace(x0, x1, 64)
    # heterogeneous slip along strike: tapered bell x correlated log-normal noise
    taper = np.clip(np.sin(np.pi * (xs - x0) / L), 0, None) ** rng.uniform(0.3, 1.5) + 1e-3
    corr = max(L / 4, 50.0); kk = np.arange(1, 9)
    noise = (rng.standard_normal(8)[:, None] * np.sin(np.pi * kk[:, None] * (xs - x0)[None] / L) / kk[:, None] ** 1.2).sum(0)
    w = taper * np.exp(0.6 * noise / (np.std(noise) + 1e-9))
    # bin onto columns (each column spans s +- 50 km)
    colw = np.array([w[(xs >= sk - 50) & (xs < sk + 50)].sum() for sk in s])
    if colw.sum() <= 0: colw[np.argmin(np.abs(s))] = 1.0
    colw = colw / colw.sum()
    fdeep = float(np.clip(rng.beta(1.3, 1.3), 0, 1))
    fd = np.clip(fdeep + 0.15 * rng.standard_normal(len(s)), 0, 1)
    units, mom = [], []
    for jj in np.nonzero(colw > 1e-4)[0]:
        units += [ch[jj]['b'], ch[jj]['a']]; mom += [colw[jj] * (1 - fd[jj]), colw[jj] * fd[jj]]
    mom = np.array(mom); mom /= mom.sum(); slips = mom * m0(mw) / (MU * UNIT_AREA)
    sc = float((np.repeat(s[colw > 1e-4], 2) * mom).sum())
    fdeep_eff = float(mom[1::2].sum())
    return dict(lon=float(lon % 360), lat=float(lat), frame=fr, units=np.array(units), slips=slips, mw=float(mw), L=L,
                theta=np.array([mw, sc / 100.0, fdeep_eff, np.log10(L)], np.float32),
                col_s=np.repeat(s[colw > 1e-4], 2))

def theta_slip(geo, fr, theta):
    """Deterministic slip from theta (for posterior-predictive forecasts): uniform-bell rupture of length L centred at
    s_c, deep-row fraction f, moment from Mw. Returns (units, slips)."""
    mw, sc, fd, lL = float(theta[0]), float(theta[1]) * 100, float(np.clip(theta[2], 0, 1)), float(theta[3])
    L = 10 ** lL; ch = geo.chains[fr['chain']]; s = np.array([q['s'] for q in ch]) - fr['s0']
    xs = np.linspace(sc - L / 2, sc + L / 2, 48); w = np.clip(np.sin(np.pi * (xs - xs[0]) / L), 0, None) ** 0.7 + 1e-3
    colw = np.array([w[(xs >= sk - 50) & (xs < sk + 50)].sum() for sk in s])
    if colw.sum() <= 0: colw[np.argmin(np.abs(s - sc))] = 1.0
    colw /= colw.sum(); units, mom = [], []
    for jj in np.nonzero(colw > 1e-4)[0]:
        units += [ch[jj]['b'], ch[jj]['a']]; mom += [colw[jj] * (1 - fd), colw[jj] * fd]
    mom = np.array(mom); mom /= mom.sum()
    return np.array(units), mom * m0(mw) / (MU * UNIT_AREA)
