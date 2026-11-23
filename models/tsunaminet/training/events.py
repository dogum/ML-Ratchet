"""Build the event table and freeze the dev/exam split (splits.json) BEFORE any training.
Eligible: the NGDC Pacific candidates (fetch_meta.py) that NGDC lists with at least one deep-ocean (DART) gauge reading.
Named exam events (the brief lists them): Tohoku 2011, Maule 2010, Iquique 2014, Illapel 2015, Kamchatka 2025.
Dev: 5 events drawn with seed 20261122 from the rest, 2 with Mw >= 8.0 and 3 below (strata). Everything else is exam."""
import os, json, numpy as np
from _paths import DATA, ROOT
import tsim
E = json.load(open(os.path.join(DATA, 'events_raw.json')))
geo = tsim.Geometry()
NAMED = {5413: 'Tohoku 2011', 4682: 'Maule 2010', 5557: 'Iquique 2014', 5590: 'Illapel 2015', 6044: 'Kamchatka 2025'}
rows = []
for e in E:
    u = e.get('usgs')
    if not u or len(e['ngdc_dart']) == 0: continue
    w = u.get('wphase'); f = u.get('ffm')
    mw = w['mw'] if w else float(u['mag'])
    planes = [w['np1'], w['np2']] if w else [[f['strike'], f['dip'], 90.0]]
    interface = any(p[1] <= 35 and 45 <= (p[2] % 360) <= 135 for p in planes)
    fr = geo.frame(u['hypo_lon'], u['hypo_lat'])
    ch = geo.chains[fr['chain']][fr['j']]
    r = dict(id=e['id'], name=f"{e['year']}-{e['month']:02d}-{e['day']:02d} {e['locationName'].title()}", year=e['year'], t0=u['time_ms'] / 1000.0,
             lon=u['hypo_lon'], lat=u['hypo_lat'], depth=u['hypo_depth'], mw=mw, mw_source='USGS W-phase Mww' if w else 'USGS ' + u['magType'],
             ffm_lon=f['centroid_lon'] if f else None, ffm_lat=f['centroid_lat'] if f else None, ffm_mw=f['mw'] if f else None,
             interface=bool(interface), mechanism=planes, usgs_id=u['usgs_id'], zone=ch['zone'], column=ch['k'], frame_dist_km=fr['dist_km'],
             ngdc_dart=len(e['ngdc_dart']), max_water_height=e['maxWaterHeight'])
    if f: r['ffm_s100'] = float(geo.project_point(fr, f['centroid_lon'], f['centroid_lat']))
    rows.append(r)
rng = np.random.default_rng(20261122)
pool = [r for r in rows if r['id'] not in NAMED]
big = [r['id'] for r in pool if r['mw'] >= 8.0]; small = [r['id'] for r in pool if r['mw'] < 8.0]
dev = list(rng.choice(big, 2, replace=False)) + list(rng.choice(small, 3, replace=False))
dev = [int(x) for x in dev]
for r in rows: r['split'] = 'dev' if r['id'] in dev else 'exam'
json.dump(rows, open(os.path.join(DATA, 'events.json'), 'w'), indent=1)
split = dict(rule=__doc__.strip(), seed=20261122, frozen='2026-11-22, before any network training or real-waveform comparison',
             dev=[dict(id=r['id'], name=r['name'], mw=r['mw']) for r in rows if r['split'] == 'dev'],
             exam=[dict(id=r['id'], name=r['name'], mw=r['mw']) for r in rows if r['split'] == 'exam'])
json.dump(split, open(os.path.join(ROOT, 'splits.json'), 'w'), indent=1)
for r in rows: print(r['split'], r['id'], r['name'], r['mw'], 'interface' if r['interface'] else 'OTHER', r['zone'], round(r['frame_dist_km']), r.get('ffm_s100'))
print(len(rows), 'dev', dev)
