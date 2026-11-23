"""Demo examples: 4 real exam earthquakes from the 400-event subset (fixed draw, seed 5, within depth strata: 2 shallow,
2 deep, among events with 5+ stations), every station record they have (first 10 in the frozen order)."""
import sys, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src'); import realnodes as RN
SP = json.load(open(H + 'splits.json')); R = RN.records(('exam',)); by = {}
for sp, k, s, n, f in R:
    if k in SP['exam_multi']: by.setdefault(k, []).append((s, f))
ok = sorted(k for k, v in by.items() if len(v) >= 5); rng = np.random.default_rng(5)
sh = [k for k in ok if SP['events'][k]['depth'] <= 70]; dp = [k for k in ok if SP['events'][k]['depth'] > 70]
pick = rng.choice(sh, 2, replace=False).tolist() + rng.choice(dp, 2, replace=False).tolist(); from _paths import Q; ST = json.load(open(Q + 'stations.json'))
for k in pick:
    e = SP['events'][k]; X, meta = [], []
    for s, f in by[k]:
        z = np.load(f); m = json.loads(str(z['meta'])); X.append(z['x'].astype(np.float32)); meta.append(dict(station=s, lat=ST[s]['lat'], lon=ST[s]['lon'], t_start=m['p_time'] - 300, p_guess=300))
    np.savez_compressed(H + f'release/examples/{k}.npz', zne=np.stack(X), meta=json.dumps(dict(event=dict(id=k, time=e['time'], lat=e['lat'], lon=e['lon'], depth=e['depth'], mw=e['mw'], magType=e['magType'], place=e['place']), stations=meta)))
    print(k, e['place'], e['depth'], len(X))
