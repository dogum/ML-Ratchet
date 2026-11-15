"""Diagnostic: distribution of the physics-prior error in simulation (frozen val) vs real dev (cross-year fits)."""
import numpy as np, sys
from common import *
val = sys.argv[1] if len(sys.argv) > 1 else 'val_v1'
R = dict(np.load(f'{DATA}/banks/real_dev.npz')); S = dict(np.load(f'{DATA}/banks/{val}.npz'))
def q(a): a = a[np.isfinite(a)]; return np.round(np.percentile(np.abs(a), [50, 90, 99]), 3)
def nm(p, y): m = np.isfinite(y); return round(float(np.sum(np.abs(p - y)[m]) / np.sum(y[m])), 3)
print('demand |log y/prior| p50/p90/p99  real', q(R['yd']), ' sim', q(S['yd']))
for ba in BAS:
    m = R['meta_ba'] == ba
    print(f"  {ba} demand {q(R['yd'][m])} wind prior nMAE {nm(R['pw'][m], R['yw'][m])} solar {nm(R['ps'][m], R['ys'][m]) if R['hass'][m].any() else '-'}")
r = S['meta_region']
for k, p, y in (('wind', 'pw', 'yw'), ('solar', 'ps', 'ys')):
    v = [nm(S[p][r == i], S[y][r == i]) for i in np.unique(r)]
    print(f'sim {k} prior nMAE per region p10/p50/p90', np.round(np.percentile(v, [10, 50, 90]), 3))
