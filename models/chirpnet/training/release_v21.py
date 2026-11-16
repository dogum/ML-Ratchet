"""ChirpNet v2.1 release: the neutron-star weights file keeps the same network; its meta gains the matched filter's
widen-only 68% factor, fitted on the odd-numbered dev injections (logs/bns_mf_dev.json, training/bns_mf_dev.py)."""
import _paths as _P  # portable paths (see _paths.py)
import json, os, numpy as np
W = os.path.join(_P.MODEL, 'chirpnet_bns_weights.npz'); z = dict(np.load(W)); meta = json.loads(str(z.pop('meta')))
r = json.load(open(os.path.join(_P.SRC, 'logs', 'bns_mf_dev.json')))
meta['mf_widen'] = round(float(r['mf_widen']), 3); meta['version'] = '2.1'
meta['mf'] = 'TaylorF2 3.5PN matched filter, 30-1000 Hz over 128 s (coarse view 60-1000 Hz, 64 s); chirpnet.BNSMatchedFilter'
np.savez_compressed(W, **z, meta=json.dumps(meta)); print(meta['mf_widen'])
