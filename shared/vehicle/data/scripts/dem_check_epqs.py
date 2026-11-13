"""Validate the 2 m DEM mosaic against USGS EPQS point queries (gentle: 1 request / 1.5 s, cached)."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import json, os, time, urllib.request, numpy as np, pandas as pd, glob
import sys; sys.path.insert(0, _VEH)
from realdata.geo import Dem
ROOT = _VEH + '/data'
cache_f = f'{ROOT}/raw/elev/epqs_cache.json'
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
dem = Dem()
rng = np.random.default_rng(7)
d = pd.read_parquet(glob.glob(f'{ROOT}/proc/ved_stage1/b03/*.parquet')[5], columns=['lat', 'lon'])
pts = d.sample(40, random_state=7)[['lat', 'lon']].round(6).values
out = []
for lat, lon in pts:
    k = f'{lat:.6f},{lon:.6f}'
    if k not in cache:
        u = f'https://epqs.nationalmap.gov/v1/json?x={lon:.6f}&y={lat:.6f}&units=Meters&wkid=4326'
        for attempt in range(4):
            try:
                cache[k] = json.load(urllib.request.urlopen(u, timeout=30)); break
            except Exception as e:
                print('retry', e); time.sleep(10 * (attempt + 1))
        time.sleep(1.5)
        json.dump(cache, open(cache_f, 'w'))
    r = cache.get(k)
    if not r: continue
    out.append((lat, lon, float(r['value']), r.get('resolution'), float(dem.sample_ll(np.array([lat]), np.array([lon]))[0])))
o = pd.DataFrame(out, columns=['lat', 'lon', 'epqs', 'res', 'dem2m'])
o['diff'] = o.dem2m - o.epqs
print(o.to_string())
print('median |diff| %.3f m, max %.3f m, resolutions %s' % (o['diff'].abs().median(), o['diff'].abs().max(), o.res.value_counts().to_dict()))
o.to_csv(f'{ROOT}/proc/checks/dem_vs_epqs.csv', index=False)
