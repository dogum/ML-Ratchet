"""JS == Python check: run web/qnet2.js (node) and release/quakenet2.py on the dev 'all' samples' node inputs."""
import sys, json, base64, subprocess, numpy as np
from _paths import H; sys.path.insert(0, H + 'release'); sys.path.insert(0, H + 'src')
import quakenet2 as Q2, train2 as T2
z = np.load(H + 'release/quakenet2_weights.npz'); meta = json.loads(str(z['meta']))
W = {k: dict(b64=base64.b64encode(z[k].astype('<f4').tobytes()).decode(), s=list(z[k].shape)) for k in z.files if k != 'meta'}
tt = np.load(H + 'data/tt_ak135.npz'); TT = dict(depths=tt['depths'].tolist(), P=np.round(tt['P'], 2).tolist())
D = T2.load(H + 'data/dev_samples.npz'); cases = []; py = []
for s in np.where(D['tag'] == 'all')[0][:12]:
    K = int(D['K'][s]); idx = D['idx'][s, :K]
    st = [dict(lat=float(D['n_slat'][i]), lon=float(D['n_slon'][i]), **{f: float(D['n_r_' + f][i]) for f in ('mw', 'mw_sig', 'dist', 'dist_sig', 'logz', 'logz_sig', 'baz', 'kappa')},
               ls=float(D['n_ls'][i]), peak=float(D['n_peak'][i]), tpick=float(np.round(D['n_tpick'][i], 2)), pz=D['n_pz'][i].astype(float).tolist()) for i in idx]
    cases.append(st)
    rd = [dict(R={f: s_[f] for f in ('mw', 'mw_sig', 'dist', 'dist_sig', 'logz', 'logz_sig', 'baz', 'kappa')}, pick=0, ls=s_['ls'], peak=s_['peak'], pz=np.array(s_['pz'], np.float32), emb=None) for s_ in st]
    tp = np.array([s_['tpick'] for s_ in st]); r = Q2.QuakeNet2.__new__(Q2.QuakeNet2); r.meta = meta; r.gnn = Q2.GraphNet({k: z[k].astype(np.float32) for k in z.files if k != 'meta'}, meta)
    out = r.locate_read(np.array([s_['lat'] for s_ in st]), np.array([s_['lon'] for s_ in st]), tp, rd); out['t0'] -= 0
    py.append([out['lat'], out['lon'], out['depth'], out['mw'], out['t0'] - float(np.min(tp)) + float(np.min(tp))])
json.dump(dict(tt=TT, w=dict(meta=meta, w=W), cases=cases), open(H + 'tmp/parity_in.json', 'w'))
js = """const Q=require('%sweb/qnet2.js'); const I=require('%stmp/parity_in.json'); Q.init(I.tt, I.w);
console.log(JSON.stringify(I.cases.map(c=>{const r=Q.locate(c); return [r.lat,r.lon,r.depth,r.mw,r.t0];})));""" % (H, H)
o = json.loads(subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout); o = np.array(o); p = np.array(py)
d = np.abs(o - p); km = np.array([Q2.gc_dist(a[0], a[1], b[0], b[1]) * Q2.KM for a, b in zip(o, p)])
res = dict(n=len(o), max_epi_km=float(km.max()), max_depth_km=float(d[:, 2].max()), max_mw=float(d[:, 3].max()), max_t0_s=float(d[:, 4].max()))
print(json.dumps(res)); json.dump(res, open(H + 'data/js_parity.json', 'w'))
