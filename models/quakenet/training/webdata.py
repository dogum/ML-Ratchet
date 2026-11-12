"""Data for the QuakeNet page: float16 weights, 36 real exam records (picked by a fixed random draw within magnitude/depth
strata, not by how well QuakeNet did), quiet-noise windows for the lab, every exam prediction, the ratchet ladder, the
classical baselines and the real-vs-synthetic comparison."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, base64, numpy as np
sys.path.insert(0, _P.SRC); import qnet as Q, locate as LO, compare as CP, fig_compare as FC, pipeline as PL
from obspy.taup import TauPyModel
D = _P.WORK + '/data/'; OUT = _P.LAB + '/'; CK = sys.argv[1]
J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); S = json.load(open(D + 'stations.json')); cal = json.load(open(_P.WORK + '/sigma_calibration.json'))['factor_used']
r1 = lambda a, n=2: [round(float(v), n) for v in a]

# weights: float16, in the release file's order
W = np.load(_P.WORK + '/release/quakenet_weights.npz'); meta = json.loads(str(W['meta'])); man = []; buf = []; off = 0
for k in W.files:
    if k == 'meta': continue
    a = W[k].astype(np.float16).ravel(); man.append(dict(k=k, shape=list(W[k].shape), off=off)); buf.append(a); off += a.size
json.dump(dict(meta=meta, man=man, b64=base64.b64encode(np.concatenate(buf).tobytes()).decode()), open(OUT + 'weights.json', 'w'))

net = Q.load_net(CK); _, idx, P = Q.real(net, 'exam'); M = [J['meta'][i] for i in idx]; la, lo, e = LO.locate(P, M)
g = lambda k: np.array([m[k] for m in M], float)
exam = dict(mw=r1(g('mw')), mw_p=r1(P['mw']), mw_s=r1(P['mw_sig'] * cal['mw']), dist=r1(g('dist'), 1), dist_p=r1(P['dist'], 1), dist_s=r1(P['dist_sig'] * cal['dist'], 1),
            depth=r1(g('depth'), 0), depth_p=r1(P['depth'], 0), logz_s=r1(P['logz_sig'] * cal['logz']), baz_err=r1(Q.angdiff(P['baz'], g('baz')), 1), loc_km=r1(e, 0),
            elat=r1(g('lat'), 1), elon=r1(g('lon'), 1), plat=r1(la, 1), plon=r1(lo, 1), slat=r1([S[m['station']]['lat'] for m in M], 1), slon=r1([S[m['station']]['lon'] for m in M], 1))

# showcase records: fixed random draw within strata
rng = np.random.default_rng(7); mw, dep = g('mw'), g('depth'); pick = []
for cond, n in (((dep <= 70) & (mw < 6.3), 12), ((dep <= 70) & (mw >= 6.3), 10), ((dep > 70) & (dep <= 300), 7), (dep > 300, 7)):
    pick += sorted(rng.choice(np.where(cond)[0], n, replace=False).tolist(), key=lambda j: -mw[j])
TP = TauPyModel('ak135'); recs = []
for j in pick:
    i = int(idx[j]); m = J['meta'][i]; x = z['x'][i].astype(np.float32); q = float(np.abs(x).max()); tt = lambda ph: (lambda a: a[0].time if a else None)(TP.get_travel_times(max(m['depth'], 0), m['dist'], [ph]))
    tp = tt('P'); ts = tt('S'); rec = dict(id=m['id'], place=m['place'], time=__import__('datetime').datetime.utcfromtimestamp(m['time']).strftime('%Y-%m-%d %H:%M'), mw=m['mw'], magType=m['magType'], depth=m['depth'], dist=round(m['dist'], 2), baz=round(m['baz'], 1),
        station=m['station'], slat=S[m['station']]['lat'], slon=S[m['station']]['lon'], elat=m['lat'], elon=m['lon'], ls=float(z['ls'][i]), peak=float(z['peak'][i]), q=q,
        tS=round(ts - tp, 1) if ts and tp else None, tR=round(m['dist'] * 111.19 / 3.7 - tp, 1) if tp else None,
        py={k: round(float(P[k][j]), 3) for k in ('mw', 'mw_sig', 'dist', 'dist_sig', 'depth', 'logz_sig', 'baz', 'kappa')}, plat=round(float(la[j]), 2), plon=round(float(lo[j]), 2), loc_km=round(float(e[j])),
        x=base64.b64encode(np.round(x / q * 32767).astype(np.int16).tobytes()).decode())
    recs.append(rec)

# quiet noise for the lab: physical units, same band; 6 windows of 2460 s
import simbank as SB; NP = SB.noise_pool(); rn = np.random.default_rng(3); noise = []
for k in rn.choice(len(NP), 6, replace=False):
    x = NP[k][:, 300:300 + PL.L].astype(np.float32); q = float(np.abs(x).max()); noise.append(dict(q=q, x=base64.b64encode(np.round(x / q * 32767).astype(np.int16).tobytes()).decode()))

ladder = []
for r in json.load(open(_P.WORK + '/ratchet.json')):
    ladder.append(dict(ckpt=r['ckpt'], note=r['note'], accepted=r['accepted'], objective=round(r['objective'], 3),
                       **{w: {k: round(r[w][k], 3) for k in ('mw_mae', 'mw_bias', 'dist_mae', 'depth_med_km', 'baz_med_deg')} for w in ('val', 'dev', 'exam')}))
B = json.load(open(_P.WORK + '/baseline_exam.json')); base = {k: B[k] for k in ('dev', 'exam', 'ms_offset_from_dev')}

cmp = []
for i in (84, 1086):
    m = J['meta'][i]; Mt, _ = CP.usgs_mt(m['id']); X = CP.synth_for(m, Mt); R = z['x'][i].astype(float) * 10 ** z['ls'][i]; C = FC.corrected(m, X); L = min(R.shape[1], X.shape[1])
    s = np.abs(R[0, :L]).max(); cmp.append(dict(place=m['place'], mw=m['mw'], depth=m['depth'], dist=round(m['dist']), station=m['station'],
        real=r1(R[0, :L:2] / s, 3), ak135=r1(X[0, :L:2] / s, 3), corrected=r1(C[0, :L:2] / s, 3)))
phys = json.load(open(_P.WORK + '/physics_check.json'))
json.dump(dict(exam=exam, recs=recs, noise=noise, ladder=ladder, base=base, cmp=cmp, cal=cal, phys=phys, land=json.load(open(OUT + 'land.json'))), open(OUT + 'data.json', 'w'), separators=(',', ':'))
print('records', len(recs), 'exam', len(idx))
