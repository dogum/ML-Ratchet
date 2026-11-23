"""Release examples: a fixed seeded draw of exam events within strata (interface thrust or not x Mw >= 8 or not),
one per stratum, NOT picked by result. Stores the processed minute series (pipeline.py output) for every DART that
had Green's functions, plus the target list. Run after the exam."""
import os, json, numpy as np
from _paths import DATA, ROOT
import tsim, realdata
geo = tsim.Geometry(); gf = tsim.GFBank()
EV = [e for e in json.load(open(os.path.join(DATA, 'events.json'))) if e['split'] == 'exam']
rng = np.random.default_rng(424242); pick = []
for itf in (True, False):
    for big in (True, False):
        pool = [e for e in EV if e['interface'] == itf and (e['mw'] >= 8.0) == big]
        if pool: pick.append(pool[rng.integers(len(pool))])
out = os.path.join(ROOT, 'release', 'examples'); os.makedirs(out, exist_ok=True)
for ev in pick:
    p = realdata.prepare(ev, gf, geo)
    sites = [s['site'] for s in p['all']]
    meta = dict(id=ev['id'], name=ev['name'], lon=ev['lon'], lat=ev['lat'], mw=ev['mw'], interface=ev['interface'], ffm_s100=ev.get('ffm_s100'),
                t0=ev['t0'], sites=sites, targets=[dict(site=t['site'], a=int(t['a']), b=int(t['b']), obs=t['amp']) for t in p['targets']])
    np.savez_compressed(os.path.join(out, f"{ev['id']}_{ev['name'][:10].replace(' ', '_').replace('-', '')}.npz"), meta=json.dumps(meta),
                        eta=np.stack([s['eta'] for s in p['all']]), mask=np.stack([s['mask'] for s in p['all']]))
    print(ev['name'], len(sites), len(p['targets']))
