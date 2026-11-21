"""Dominant LANDFIRE FBFM40 fuel group inside each FEDS-MTBS final perimeter (pre-fire version, 64x64 nearest samples).
-> proc/landfire/fuelgroups.parquet (resumable via a jsonl cache)."""
import os, sys, json, numpy as np, pandas as pd, pyarrow.parquet as pq, shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lf
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
P = f'{_DATA}/data/proc'
CACHE = f'{P}/landfire/fuelgroups.jsonl'
f = pd.read_parquet(f'{P}/feds/fires_raw.parquet')
_t = pq.read_table(f'{P}/feds/final_perims.parquet')
geo = dict(zip(_t.column('Event_ID').to_pylist(), _t.column('wkb').to_pylist())); del _t
f = f[~f.Event_ID.str[:2].isin(['AK', 'HI', 'PR'])].copy()
f['year'] = f.Ig_Date.dt.year
f['pri'] = ~f.year.between(2017, 2021)
f = f.sort_values(['pri', 'year', 'Event_ID'])
done = set()
if os.path.exists(CACHE):
    done = {json.loads(l)['Event_ID'] for l in open(CACHE)}
N = 64
with open(CACHE, 'a') as out:
    for k, r in enumerate(f.itertuples()):
        if r.Event_ID in done: continue
        ver, post = lf.lf_version(r.year)
        w, h = r.xmax - r.xmin, r.ymax - r.ymin
        res = max(w, h) / N + 1.0
        x0, y1 = r.xmin, r.ymax
        try:
            a = lf.export('FBFM40', ver, x0, y1, N, N, res=res)
        except Exception as e:
            print('FAIL', r.Event_ID, e, flush=True); continue
        m = rasterize([shapely.from_wkb(geo[r.Event_ID])], out_shape=(N, N), transform=from_origin(x0, y1, res, res),
                      all_touched=True).astype(bool)
        g = lf.fuel_group_codes(a[m]) if m.any() else np.array([])
        vals, cnt = np.unique(g, return_counts=True)
        fr = {str(v or 'other'): float(c) / max(1, len(g)) for v, c in zip(vals, cnt)}
        burn = {k2: v for k2, v in fr.items() if k2 not in ('NB', 'other')}
        dom = max(burn, key=burn.get) if burn else 'NB'
        out.write(json.dumps(dict(Event_ID=r.Event_ID, lf_ver=ver, lf_postfire=post, fuel_group=dom, frac=fr)) + '\n')
        out.flush()
        if k % 200 == 0: print(k, r.Event_ID, flush=True)
rows = [json.loads(l) for l in open(CACHE)]
pd.DataFrame(rows).drop(columns='frac').to_parquet(f'{P}/landfire/fuelgroups.parquet')
print('DONE', len(rows))
