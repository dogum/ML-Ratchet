"""Classical baseline v2 (chosen on dev): for pieces shorter than 300 days use the multi-start background fit
(data/classical_{dev,exam}_v2.json); whole light curve and 365-day pieces keep the v1 reading. Updates real_meta.json."""
import _paths as _P  # portable paths (see _paths.py)
import json
D = _P.WORK + '/data/'
meta = json.load(open(D + 'real_meta.json')); n = 0
for split in ('dev', 'exam'):
    V = json.load(open(f'{D}classical_{split}_v2.json'))
    for m in meta:
        k = f"{m['kic']}_{m['L']}"
        if m['split'] == split and m['L'] in (90, 54, 27) and k in V:
            m.setdefault('cl1_numax', m['cl_numax']); m.setdefault('cl1_dnu', m['cl_dnu']); m['cl_numax'], m['cl_dnu'] = V[k]; n += 1
json.dump(meta, open(D + 'real_meta.json', 'w')); print('updated', n)
