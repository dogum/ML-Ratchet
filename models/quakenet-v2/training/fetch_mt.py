"""USGS moment tensors (and derived centroid depth) for the 40 dev events only, for the twin test."""
import json, urllib.request, time, numpy as np
from _paths import H; SP = json.load(open(H + 'splits.json')); out = {}
for k in SP['dev']:
    try:
        js = json.load(urllib.request.urlopen(f'https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={k}&format=geojson', timeout=60))
        for p in js['properties']['products'].get('moment-tensor', []):
            q = p['properties']
            if 'tensor-mrr' in q:
                out[k] = dict(M=[float(q['tensor-' + c]) for c in ('mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp')], depth=float(q.get('derived-depth', js['geometry']['coordinates'][2]))); break
    except Exception as e: print(k, str(e)[:80])
    time.sleep(0.5)
json.dump(out, open(H + 'data/usgs_mt_dev.json', 'w')); print(len(out), 'of', len(SP['dev']), 'dev events have a USGS moment tensor')
