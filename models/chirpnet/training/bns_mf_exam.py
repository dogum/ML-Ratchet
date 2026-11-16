"""GW170817 exam row for ChirpNet v2.1 (run once, after the release choice): the released neutron-star network scans two
minutes around the merger exactly as demo.py --bns does, then the matched filter refines the hit's chirp mass on the raw
strain (the 2,048 s LIGO-P1700349 files, whose noise spectrum is estimated from all of it). Reports the v2.0 reading
(the network alone) and the v2.1 reading side by side against LIGO's detector-frame 1.1975.

    python bns_mf_exam.py      # writes logs/bns_mf_exam.json
"""
import _paths as _P  # portable paths (see _paths.py)
import json
import os
import sys

import numpy as np

sys.path.insert(0, _P.MODEL)
import chirpnet as C  # noqa: E402

GPS, MC = 1187008882.43, 1.1975
F = {d: os.path.join(_P.WORK, 'data_bns', f'{d[0]}-{d}_LOSC_CLN_4_V1-1187007040-2048.hdf5') for d in ('H1', 'L1')}

if __name__ == '__main__':
    bns = C.ChirpNetBNS(os.path.join(_P.MODEL, 'chirpnet_bns_weights.npz'))
    (h1, t0), (l1, _) = C.load_gwosc(F['H1']), C.load_gwosc(F['L1'])
    W = C.whiten_bns(h1), C.whiten_bns(l1); k0 = int((GPS - 60 - t0) * C.FSB); k1 = k0 + 120 * C.FSB
    hits = bns.scan(W[0][k0:k1], W[1][k0:k1], gps_start=t0 + k0 / C.FSB, whitened=True)
    hit = next(h for h in hits if abs(h['gps'] - GPS) < 2.0)
    ref = bns.refine(hit, h1, l1, t0)
    res = {'truth_mc_det': MC, 'v2.0': {'score': hit['score'], 'chirp_mass': hit['chirp_mass'], 'chirp_mass_68': hit['chirp_mass_68'], 'gps': hit['gps']},
           'v2.1': {'chirp_mass': ref['chirp_mass'], 'chirp_mass_68': ref['chirp_mass_68'], 'mf_network_snr': ref['mf_network_snr'], 'mf_eta': ref['mf_eta'], 'widen': bns.meta.get('mf_widen')}}
    for k in ('v2.0', 'v2.1'):
        r = res[k]; r['rel_err'] = r['chirp_mass'] / MC - 1; r['truth_in_68'] = bool(r['chirp_mass_68'][0] <= MC <= r['chirp_mass_68'][1])
    json.dump(res, open(os.path.join(_P.SRC, 'logs', 'bns_mf_exam.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
