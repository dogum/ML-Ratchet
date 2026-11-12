"""Dev calibration of ectopic coupling (physics v4): pre-RR / local mean RR of S and V beats on the 6 real dev records vs a
simulated bank. Writes calib_coupling.json. Usage: calib_coupling.py '<bank glob>' <tag>"""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, os, numpy as np
sys.path.insert(0, _P.SRC); import bnet as BN, train as TR
Z = BN.real_bank('mitdb'); _, idx = BN.split_idx('dev'); y = Z['y'][idx]; rr = Z['rr'][idx]; S = TR.load(sys.argv[1]); q = [5, 25, 50, 75, 95]
pct = lambda a: np.round(np.exp(np.percentile(a, q)), 3).tolist()
out = {c: dict(dev=pct(rr[y == k, 0]), sim=pct(S['rr'][S['y'] == k, 0])) for k, c in enumerate('NSV')}
fn = _P.WORK + '/calib_coupling.json'; D = json.load(open(fn)) if os.path.exists(fn) else {}; D[sys.argv[2]] = dict(bank=sys.argv[1], percentiles=q, pre_rr_ratio=out); json.dump(D, open(fn, 'w'), indent=1); print(json.dumps(out))
