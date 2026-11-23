"""Widen-only error-bar calibration on the real DEV airfoils (one factor per output: Cl, ln Cd, Cm).

factor_k = max(1, 68th percentile of |error| / sigma) over dev points (Cl and Cm pre-stall, Cd all drag points).
usage: python calibrate.py <weights.npz> <out.npz>   (copies the weights and writes the factors into meta)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(2, os.path.abspath(ROOT))   # ROOT: airfoilnet.py in the repository layout
import realeval as rev  # noqa
import airfoilnet as an  # noqa

units = json.load(open(os.path.join(ROOT, 'data', 'units.json')))
splits = json.load(open(os.path.join(ROOT, 'splits.json')))
z = np.load(os.path.join(ROOT, 'work', 'shapes.npz'))
pmap = {k: p for k, p in zip(z['tested_keys'], z['tested_p'])}
net = an.AirfoilNet(sys.argv[1])
R = {'cl': [], 'lncd': [], 'cm': []}
for aid in splits['dev']:
    for run in units[aid]['runs']:
        p = pmap[f"{aid}|{run['model']}"]
        if run['kind'] == 'lift':
            L = rev.upsweep(run['data'])
            if len(L) < 4:
                continue
            a_s, _, _ = rev.measured_stall(L)
            pre = L[:, 0] <= a_s
            r = net.predict(p, run['Re'], L[pre, 0], raw_sigma=True)
            R['cl'] += list(np.abs(r['cl'] - L[pre, 1]) / r['cl_sd'])
            cm = np.array([np.nan if v is None else v for v in L[pre, 2]], float)
            ok = np.isfinite(cm)
            R['cm'] += list(np.abs(r['cm'][ok] - cm[ok]) / r['cm_sd'][ok])
        else:
            D = np.array(run['data'], float); D = D[D[:, 2] > 0]
            r = net.predict(p, run['Re'], D[:, 0], raw_sigma=True)
            R['lncd'] += list(np.abs(np.log(r['cd']) - np.log(D[:, 2])) / r['lncd_sd'])
fac = [max(1.0, float(np.percentile(R[k], 68.27))) if R[k] else 1.0 for k in ('cl', 'lncd', 'cm')]
raw_cov = {k: float(np.mean(np.array(R[k]) <= 1)) for k in R}
print('dev raw coverage', raw_cov, 'widen factors', fac, 'n', {k: len(v) for k, v in R.items()})
zz = dict(np.load(sys.argv[1]))
meta = json.loads(str(zz['meta'])); meta['widen'] = fac; meta['dev_raw_coverage'] = raw_cov
zz['meta'] = np.array(json.dumps(meta))
np.savez(sys.argv[2], **zz)
json.dump(dict(widen=fac, dev_raw_coverage=raw_cov, n={k: len(v) for k, v in R.items()}),
          open(os.path.join(ROOT, 'work', 'calibration.json'), 'w'), indent=1)
