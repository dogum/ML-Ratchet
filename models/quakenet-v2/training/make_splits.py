"""Freeze the dev/exam split before anything is trained. QuakeNet v2 keeps QuakeNet v1.2's split exactly: the 40 dev
events and the 1,734 exam events of QuakeNet v1.2's real_meta.json (Q in _paths.py; drawn there by realbank.py, seed 2026).
Extra station records are fetched for all 40 dev events and for a fixed random subset of 400 exam events (seed 7,
drawn within depth strata: shallow <= 70 km and deep, in proportion). Station order inside an event is fixed here:
QuakeNet's original records first (sorted), then the new stations in a seeded random order; "N stations" always means
the first N in this order."""
import json, collections, numpy as np
from _paths import Q, H
J = json.load(open(Q + 'real_meta.json')); meta, split = J['meta'], J['split']
ev = collections.OrderedDict()
for m, s in zip(meta, split):
    e = ev.setdefault(m['id'], dict(id=m['id'], split=s, time=m['time'], lat=m['lat'], lon=m['lon'], depth=m['depth'], mw=m['mw'], magType=m['magType'], place=m['place'], stations=[]))
    assert e['split'] == s; e['stations'].append(m['station'])
for e in ev.values(): e['stations'] = sorted(e['stations'])
dev = sorted(k for k, e in ev.items() if e['split'] == 'dev'); exam = sorted(k for k, e in ev.items() if e['split'] == 'exam')
rng = np.random.default_rng(7)
sh = [k for k in exam if ev[k]['depth'] <= 70]; dp = [k for k in exam if ev[k]['depth'] > 70]
n_sh = int(round(400 * len(sh) / len(exam))); multi = sorted(rng.choice(sh, n_sh, replace=False).tolist() + rng.choice(dp, 400 - n_sh, replace=False).tolist())
out = dict(note=__doc__, source='QuakeNet v1.2 real_meta.json (realbank.py, seed 2026)', dev=dev, exam=exam, exam_multi=multi,
           exam_multi_seed=7, max_stations=10, events={k: ev[k] for k in dev + exam})
json.dump(out, open(H + 'splits.json', 'w'), indent=0)
print(len(dev), 'dev events', len(exam), 'exam events', len(multi), 'exam events get extra stations (', n_sh, 'shallow)')
