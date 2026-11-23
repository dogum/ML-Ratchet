"""Physics v2, measured on the dev twins (twin2.py; paired real minus twin, 399 dev records):
  - QuakeNet v1.2 reads real records farther than their twins: distance +1.5 deg (median), with extra scatter (real
    robust sd of the paired difference 3.8 deg; we add N(+1.5, 3.0) deg per station);
  - real back-azimuth readings scatter more: robust sd 8.6 deg real vs 5.5 twin -> extra N(0, 6.5) deg per station;
  - real P picks are tighter than simulated ones: robust sd 8.9 s real vs 11.8 s twin -> picker error x 0.75.
v2 applies these to the simulated per-station readings (the 'instrument' between the record and the graph network)
and recomputes the anchors. Builds bank/<out>_samples.npz from bank/<src>_samples.npz with a fixed seed.
Usage: physics_v2.py tr1 tr2 | val1 val2"""
import sys, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import locate2 as LC, prep as PR, train2 as T2
DSHIFT, DSIG, BSIG, PICK = 1.5, 3.0, 6.5, 0.75
src, out = sys.argv[1], sys.argv[2]; D = T2.load(H + f'bank/{src}_samples.npz'); rng = np.random.default_rng(2026 + len(out))
n = len(D['n_r_dist']); D['n_r_dist'] = D['n_r_dist'] + rng.normal(DSHIFT, DSIG, n); D['n_r_baz'] = (D['n_r_baz'] + rng.normal(0, BSIG, n)) % 360; D['n_pickerr'] = D['n_pickerr'] * PICK
nd = {k[2:]: v for k, v in D.items() if k.startswith('n_')}
for s in range(len(D['K'])):
    K = int(D['K'][s]); idx = D['idx'][s, :K]; A = LC.combine_v12(D['n_slat'][idx], D['n_slon'][idx], PR.Rof(nd, idx))
    D['anchor'][s] = [A[k] for k in ('lat', 'lon', 'logz', 'mw', 'logz_sig', 'mw_sig')]; D['anchor_cov'][s] = A['cov']
np.savez(H + f'bank/{out}_samples.npz', **D); print(out, len(D['K']), 'samples')
