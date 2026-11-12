"""The real records, read once by the pipeline (P pick within 30 s of the ak135 prediction from the catalogue origin),
with the USGS answers. 40 events drawn at random are the dev set; their records never enter the exam."""
import _paths as _P  # portable paths (see _paths.py)
import sys, glob, json, numpy as np
sys.path.insert(0, _P.SRC); import pipeline as PL
D = _P.WORK + '/data/'
X, LS, PK, PI, meta = [], [], [], [], []
for p in sorted(glob.glob(D + 'wf/ev_*.npz')):
    z = np.load(p); x = z['x'].astype(np.float64); m = json.loads(str(z['meta']))
    if not np.isfinite(x).all() or (x.std(1) == 0).any(): continue
    W, ls, pk, peak = PL.read(x, 300); e = m['event']
    X.append(W.astype(np.float16)); LS.append(ls); PK.append(peak); PI.append(pk - 300)
    meta.append(dict(id=e['id'], station=m['station'], mw=e['mag'], dist=m['dist'], depth=e['depth'], baz=m['baz'], time=e['time'], lat=e['lat'], lon=e['lon'], place=e['place'], magType=e['magType']))
ids = sorted({m['id'] for m in meta}); rng = np.random.default_rng(2026); dev = set(rng.choice(ids, 40, replace=False).tolist())
split = ['dev' if m['id'] in dev else 'exam' for m in meta]
np.savez(D + 'real.npz', x=np.stack(X), ls=np.array(LS, np.float32), peak=np.array(PK, np.float32), pickoff=np.array(PI))
json.dump(dict(meta=meta, split=split), open(D + 'real_meta.json', 'w'))
print(len(meta), 'records', len(ids), 'events', sum(s == 'dev' for s in split), 'dev records')
