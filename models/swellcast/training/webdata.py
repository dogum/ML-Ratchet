"""Build the SwellCast web page: embed coastlines, a fixed stratified random draw of exam events (windows quantised exactly as
the page stores them, with Python network + classical predictions on those quantised numbers), real quiet 2025 windows for the
lab, results tables and the float16 weight pack. Writes web/swellcast.html and checks JS == Python with node."""
import os, sys, json, base64, subprocess, numpy as np
import geo, catalog, fetch, evaluate as E, events as EV, realdata as RD, classical as CL, swellsim as SS
sys.path.insert(0, os.path.join(E.ROOT, 'release')); import swellcast as SC
ROOT = E.ROOT; WEB = os.path.join(ROOT, 'web')

def coast():
    out = []
    for p in geo._decode_topo(os.path.join(ROOT, 'data/geo/land-110m.json')):
        for ring in p[:1]:
            r = ring[::2] if len(ring) > 40 else ring
            if len(r) < 4: continue
            out.append(np.round(r, 1).ravel().tolist())
    return out

def quant(x):
    """(3, 80, 34) -> stored ints (E as 1000 log10 E, C/S x 1000) and the dequantised window the page will see"""
    Eq = np.where(np.isfinite(x[0]), np.round(1000 * np.log10(np.maximum(x[0], 1e-3))), np.nan)
    Cq = np.where(np.isfinite(x[1]), np.round(1000 * x[1]), np.nan); Sq = np.where(np.isfinite(x[2]), np.round(1000 * x[2]), np.nan)
    X = np.stack([10 ** (Eq / 1000), Cq / 1000, Sq / 1000]).astype(np.float32)
    enc = lambda a: [None if not np.isfinite(v) else int(v) for v in a.ravel()]
    return dict(E=enc(Eq), C=enc(Cq), S=enc(Sq)), X

def pack(npz):
    z = np.load(npz); meta = json.loads(str(z['meta'])); man = []; buf = []; off = 0
    for k in z.files:
        if k == 'meta': continue
        a = z[k].astype(np.float16); man.append(dict(k=k, shape=list(a.shape), off=off)); off += a.size; buf.append(a.ravel())
    b = np.concatenate(buf).astype('<f2').tobytes(); return dict(b64=base64.b64encode(b).decode(), man=man, meta=meta)

def draw_events():
    ev = EV.load('exam', 'confirmed'); rng = np.random.default_rng(7); out = []
    for b, k in (('EP', 4), ('NA', 4), ('WP', 3), ('SP', 1)):
        pool = [e for e in ev if e['basin'] == b]; idx = rng.choice(len(pool), min(k, len(pool)), replace=False); out += [pool[i] for i in idx]
    return out

if __name__ == '__main__':
    W = os.path.join(ROOT, 'release/swellcast_weights.npz'); model = SC.SwellCast(W); model.W = {k: v.astype(np.float16).astype(np.float32) for k, v in model.W.items()}; S = catalog.stations(); storms = {s['sid']: s for s in catalog.storms()}
    fin = json.load(open(os.path.join(ROOT, 'final.json'))); txt = json.load(open(os.path.join(WEB, 'text.json')))
    events = []
    for e in draw_events():
        (Ew, Cw, Sw), tc = RD.window(e['buoy'], e['tc']); enc, X = quant(np.stack([Ew, Cw, Sw]))
        p = model.read(X); c = CL.read(X); s = storms[e['sid']]
        tr = [[round(t - tc, 1), la, lo, w] for t, la, lo, w in zip(s['t'], s['lat'], s['lon'], s['wind']) if w >= 34 or True][::2]
        events.append(dict(enc, name=e['name'].title(), basin=e['basin'], year=e['year'], buoy=e['buoy'], bname=S[e['buoy']][2], blat=e['buoy_lat'], blon=e['buoy_lon'], tc=tc,
                           D=round(e['D'] / 1e3, 1), t0=e['t0_rel'], dir=round(e['dir'], 1), lat=e['lat'], lon=e['lon'], wind=int(e['wind']), track=tr,
                           py={k: round(float(v), 5) for k, v in p.items()}, cl=dict(D_km=round(c['D_km'], 1), t0=round(c['t0'], 1), dir=None if not np.isfinite(c['dir']) else round(c['dir'], 1))))
    # lab: real quiet windows from 2025 (the held-out false-alarm set), one per chosen buoy, fixed draw
    Xq = np.load(os.path.join(ROOT, 'bank/quiet_test_x.npy')); Yq = np.load(os.path.join(ROOT, 'bank/quiet_test_y.npy')); lab = []
    for b, d0, D0 in (('46026', 205, 3600), ('51001', 300, 5200), ('44008', 165, 2600)):
        idx = np.where(Yq[:, 5] == fetch.BUOYS.index(b))[0]
        if not len(idx): continue
        i = int(np.random.default_rng(3).choice(idx)); enc, X = quant(Xq[i].astype(np.float32)); la, lo = S[b][:2]
        lab.append(dict(enc, buoy=b, blat=la, blon=lo, label=f"{b} {S[b][2][:28]} ({str(np.datetime64(int(Yq[i, 13]), 'h'))[:10]})", dir0=d0, D0=D0))
    data = dict(coast=coast(), events=events, hero=0, lab=lab, det_thr=fin['det_thr'], net=pack(W), **txt)
    html = open(os.path.join(WEB, 'page.html')).read().replace('/*DATA*/', json.dumps(data, separators=(',', ':'))).replace('/*SNET*/', open(os.path.join(WEB, 'snet.js')).read())
    open(os.path.join(WEB, 'swellcast.html'), 'w').write(html); print('page', round(len(html) / 1e6, 2), 'MB')
    # JS == Python check on the embedded events
    js = "const S=require('%s');const D=%s;const net=S.load(D.net);const out=D.events.map(e=>{const n=80*34,E=new Float32Array(n),C=new Float32Array(n),Q=new Float32Array(n);for(let i=0;i<n;i++){E[i]=e.E[i]===null?NaN:Math.pow(10,e.E[i]/1000);C[i]=e.C[i]===null?NaN:e.C[i]/1000;Q[i]=e.S[i]===null?NaN:e.S[i]/1000;}const r=S.read(net,{E,C,S:Q});return [r.D_km,r.t0,r.dir,r.pdet];});console.log(JSON.stringify(out));" % (os.path.join(WEB, 'snet.js'), json.dumps(dict(net=data['net'], events=[dict(E=e['E'], C=e['C'], S=e['S']) for e in events])))
    open(os.path.join(WEB, '_check.js'), 'w').write(js); r = json.loads(subprocess.run(['node', os.path.join(WEB, '_check.js')], capture_output=True, text=True).stdout); os.remove(os.path.join(WEB, '_check.js'))
    dif = [max(abs(a[0] / e['py']['D_km'] - 1), abs(a[1] - e['py']['t0']) / 100, abs(a[3] - e['py']['p_swell'])) for a, e in zip(r, events)]
    print('JS vs Python (both with the float16 weight pack): max rel diff', max(dif)); json.dump(dict(js_vs_py_max_rel=max(dif)), open(os.path.join(WEB, 'jscheck.json'), 'w'))
