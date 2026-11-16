"""StarQuake v1.1 release: same network weights; the meta gains the hybrid's parameters (fitted on simulations,
work/hybrid/params.json) and the dev-fitted widen-only factors (work/hybrid/widen.json). v1.0's factors stay as
widen_net for the network-only reading."""
import _paths as _P  # portable paths (see _paths.py)
import json, os, numpy as np
W = os.path.join(_P.MODEL, 'starquake_weights.npz'); z = dict(np.load(W)); meta = json.loads(str(z.pop('meta')))
P = json.load(open(_P.WORK + '/hybrid/params.json')); F = json.load(open(_P.WORK + '/hybrid/widen.json'))
meta['widen_net'] = meta.get('widen_net', meta['widen']); meta['widen'] = F['hybrid']; meta['version'] = '1.1'
meta['hybrid'] = {'edges_days': [0, 40, 70, 150, 600], 'numax': {k: P['numax'][k] for k in ('k', 'w', 's')}, 'dnu': {k: P['dnu'][k] for k in ('k', 'w', 's')},
                  'fitted_on': 'simulated giants in real quiet-dwarf noise (noise_train pool, v2 physics, 3,000 readings, seed 11)'}
np.savez_compressed(W, **z, meta=json.dumps(meta)); print(json.dumps(meta['hybrid']), meta['widen'], meta['widen_net'])
