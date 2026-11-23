"""For each candidate event: which DART sites were deployed at the origin time and what detided 1-minute data they
have in the 24 h after it. Writes data/event_sites.json (no waveform judgement is made here: coverage only)."""
import os, json, datetime as dt, numpy as np
from _paths import DATA
import dart
E = json.load(open(os.path.join(DATA, 'events_raw.json')))
S = json.load(open(os.path.join(DATA, 'dart_sites.json')))
def active(site, t):
    day = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%d')
    return any(a <= day and (b == '' or day <= b) for a, b in site['spans'])
out = []
for e in E:
    u = e.get('usgs')
    if not u: continue
    t0 = u['time_ms'] / 1000.0
    rec = dict(id=e['id'], t0=t0, sites=[])
    for s in S:
        if not active(s, t0): continue
        r = dart.detided_minutes(s['station'], t0)
        if r is None: continue
        eta, mask, info = r
        rec['sites'].append(dict(site=s['site'], station=s['station'], lat=s['lat'], lon=s['lon'], cov24=float(mask.mean()),
                                 cov6=float(mask[:361].mean()), cov1min_first3h=float(mask[:181].mean()), tide_rms=info['tide_rms_fit'], types=info['types']))
    print(e['id'], e['year'], e['locationName'], len(rec['sites']), sum(x['cov6'] > 0.3 for x in rec['sites']), flush=True)
    out.append(rec)
json.dump(out, open(os.path.join(DATA, 'event_sites.json'), 'w'), indent=0)
