"""Rung v2-r18 (27 Nov 2026): do the networks read the four work-share inputs?
Holds aero/Ep, rolling/Ep, grade-down/Ep and inertia/Ep (phys_vec F[3], F[4], F[6], F[7]; X columns 93, 94, 96, 97) at
their dev median and reports the median and 90th-percentile |change of log prediction|; grade-up/Ep (column 95) is the
control. r15 = the shipped v2.1 networks, r18 = the robust-norm retrain. Dev sets only.
python probe_r18.py  -> training/r18_probe.json"""
import _paths as _P
import json, os
import numpy as np
import evaluate as EV
from model_np import Ensemble

FOUR, CTRL = [93, 94, 96, 97], [95]
out = {}
for name in ('r15', 'r18'):
    m = Ensemble([f'{_P.WORK}/ckpt/{name}_s{i}.npz' for i in range(5)])
    for ds in ('ved_dev', 'epa_dev'):
        B, T, X, E, M = EV.real_set(ds, True, True)
        p0 = m.predict(X, E)[0]
        for lab, cc in (('four', FOUR), ('uphill_ctrl', CTRL)):
            X1 = X.copy(); X1[:, cc] = np.median(X[:, cc], axis=0)
            d = np.abs(np.log(m.predict(X1, E)[0] / p0))
            out[f'{name}/{ds}/{lab}'] = dict(median=float(np.median(d)), p90=float(np.quantile(d, 0.9)))
            print(name, ds, lab, 'median |dlog| %.5f  p90 %.5f' % (np.median(d), np.quantile(d, 0.9)), flush=True)
json.dump(out, open(os.path.join(_P.SRC, 'r18_probe.json'), 'w'), indent=1)
