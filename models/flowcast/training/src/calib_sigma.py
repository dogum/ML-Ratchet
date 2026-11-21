"""Widen-only interval calibration on DEV (52 basins, WY2001-2014): smallest k >= 1 (per interval) so that the
pooled dev coverage of the 50 % and 90 % intervals reaches the nominal value. usage: python3 calib_sigma.py <ckpt>"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
from evalreal import widen

name = sys.argv[1]
P = np.load(f'{ROOT}/preds/{name}_dev_dev.npz')
idx = pd.date_range(*DEVWIN)
O, Q = [], []
for b in P.files:
    o = wd.streamflow(b, 'caravan').Q_mm.reindex(idx).values
    v = np.isfinite(o)
    O.append(o[v]); Q.append(P[b][v, 1:])
O = np.concatenate(O); Q = np.concatenate(Q)
res = {}
for lab, lo, hi, nom in (('k50', 1, 3, 0.5), ('k90', 0, 4, 0.9)):
    k = 1.0
    while k < 5:
        qq = widen(Q, (k, k))
        c = np.mean((O >= qq[:, lo]) & (O <= qq[:, hi]))
        if c >= nom:
            break
        k += 0.02
    res[lab] = round(k, 2); res[lab + '_cov_raw'] = float(np.mean((O >= Q[:, lo]) & (O <= Q[:, hi]))); res[lab + '_cov'] = float(c)
print(res)
jdump(res, f'{ROOT}/results/sigma_{name}.json')
