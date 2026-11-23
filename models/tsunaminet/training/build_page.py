"""Assemble the showpiece page: web/page.html + tnet.js + app.js + DATA (hero field, weights, exam events, tables)."""
import os, json, base64, pickle, numpy as np
from _paths import ROOT, BANK, DATA as DDIR
import tsim
W = os.path.join(ROOT, 'web')
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
D = json.load(open(os.path.join(W, 'page_text.json')))      # text, tables (written by summarize.py)
# hero
h = np.load(os.path.join(BANK, 'hero_tohoku.npz'))
arr = h['arr']; amax = h['amax']; land = h['h'] <= 0
A = np.where(np.isfinite(arr), np.clip(arr, 0, 65000), 65534).astype(np.uint16); A[land] = 65535
la = np.log10(np.maximum(amax, 1e-3)); amp = np.clip((la + 2.0) / 2.5 * 255, 0, 255).astype(np.uint8)   # 1 cm .. 3 m
lat, lon = h['lat'], h['lon']; d = float(lon[1] - lon[0])
sites = json.load(open(os.path.join(DDIR, 'dart_sites.json')))
gfs = set(f[:-4] for f in os.listdir(os.path.join(BANK, 'gf8')) if f.endswith('.npz'))
ev = {e['id']: e for e in json.load(open(os.path.join(DDIR, 'events.json')))}
D['hero'] = dict(nx=int(len(lon)), ny=int(len(lat)), lon0=float(lon[0] - d / 2), lat0=float(lat[0] - d / 2), d=d, arr=b64(A), amp=b64(amp), epi=[ev[5413]['lon'], ev[5413]['lat']])
D['darts'] = [[s['lon'], s['lat']] for s in sites if s['site'] in gfs]
# weights
z = np.load(os.path.join(ROOT, 'release', 'tsunaminet_weights.npz'))
meta = json.loads(str(z['meta'])); D['weights'] = dict(meta=meta, t={k: [list(z[k].shape), b64(z[k].astype(np.float32))] for k in z.files if k != 'meta'})
D['delay_frac'] = meta['calibration'].get('delay_frac', 0.0)
open(os.path.join(W, 'tsunaminet.html'), 'w').write(
    open(os.path.join(W, 'page.html')).read().replace('/*TNETJS*/', open(os.path.join(W, 'tnet.js')).read())
    .replace('/*DATA*/', json.dumps(D, separators=(',', ':'))).replace('/*APPJS*/', open(os.path.join(W, 'app.js')).read()))
print('page MB', os.path.getsize(os.path.join(W, 'tsunaminet.html')) / 1e6)
