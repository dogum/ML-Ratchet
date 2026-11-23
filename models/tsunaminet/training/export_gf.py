"""Compact Green's functions for the release: for each DART site and unit source, the 8' response in the window
[arrival-60, arrival+300] min, int8-quantised with one scale per (site, unit); arrival = first time |G| > 2% of its
max (the same rule as training). Plus the unit-source table and the column chains."""
import os, json, numpy as np
from _paths import BANK, ROOT
import tsim
W0, W1 = -60, 300
geo = tsim.Geometry(); gf = tsim.GFBank()
sites = sorted(gf.G); nu = len(geo.U); L = W1 - W0 + 1
q = np.zeros((len(sites), nu, L), np.int8); sc = np.zeros((len(sites), nu), np.float32); arr = -np.ones((len(sites), nu), np.int16)
for i, s in enumerate(sites):
    G = gf.G[s].astype(np.float32)
    for u in range(nu):
        g = G[u]; mx = np.abs(g).max()
        if mx <= 1e-3: continue
        a = int(np.argmax(np.abs(g) > 0.02 * mx)); arr[i, u] = a
        idx = np.arange(a + W0, a + W0 + L); ok = (idx >= 0) & (idx < len(g)); seg = np.zeros(L, np.float32); seg[ok] = g[idx[ok]]
        m = np.abs(seg).max() + 1e-12; sc[i, u] = m; q[i, u] = np.round(seg / m * 127).astype(np.int8)
chains = [[dict(zone=c['zone'], k=c['k'], a=int(c['a']), b=int(c['b']), lon=float(c['lon']), lat=float(c['lat']), s=float(c['s'])) for c in ch] for ch in geo.chains]
site_ll = np.array([[gf.meta[s]['lon'], gf.meta[s]['lat']] for s in sites])
out = os.path.join(ROOT, 'release', 'gf_compact.npz')
np.savez_compressed(out, sites=np.array(sites), site_ll=site_ll, q=q, scale=sc, arr=arr, w0=W0, units=json.dumps(geo.U), centers=geo.centers, chains=json.dumps(chains))
# check reconstruction error on the input-window scale
import sys; sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model folder: tsunaminet.py; import tsunaminet
C = tsunaminet.GreensFunctions(out); rng = np.random.default_rng(0); errs = []
for _ in range(300):
    s = sites[rng.integers(len(sites))]; u = int(rng.integers(nu)); a = arr[sites.index(s), u]
    if a < 0: continue
    g = gf.G[s][u].astype(np.float32); r = C.series(s, u); w = slice(max(a - 20, 0), a + 180)
    errs.append(np.abs(g[w] - r[w]).max() / (np.abs(g).max() + 1e-9))
print('sites', len(sites), 'size MB', os.path.getsize(out) / 1e6, 'max rel err in window', max(errs), 'median', np.median(errs))
