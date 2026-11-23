"""Physics v3 = v2 plus the depth and magnitude mismatch measured on the dev twins (paired real minus twin, 399 records):
  - shallow quakes (<= 70 km): v1.2 reads real records deeper than their twins, log10 depth +0.37 (robust sd 0.39);
    deep quakes +0.04 (0.19). Simulated readings get an event-wide shift N(+0.37, 0.25) (deep: N(+0.04, 0.10)) plus a
    per-station N(0, 0.25) (deep: N(0, 0.12)), because the cause (surface waves of shallow sources) is shared by all stations.
  - Mw: real readings sit +0.057 (shallow) / +0.019 (deep) above the twins: event-wide shift N(that, 0.05) + station N(0, 0.05).
Anchors recomputed. Usage: physics_v3.py val2 val3 | tr2 tr3"""
import sys, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import locate2 as LC, prep as PR, train2 as T2
src, out = sys.argv[1], sys.argv[2]; D = T2.load(H + f'bank/{src}_samples.npz'); rng = np.random.default_rng(3030 + len(out))
ev = D['n_ev']; ne = int(ev.max()) + 1; zt = D['events'][:, 2]; sh = zt <= 70; n = len(ev)
cz = np.where(sh, rng.normal(0.37, 0.25, ne), rng.normal(0.04, 0.10, ne)); cm = np.where(sh, rng.normal(0.057, 0.05, ne), rng.normal(0.019, 0.05, ne))
D['n_r_logz'] = D['n_r_logz'] + cz[ev] + np.where(sh[ev], rng.normal(0, 0.25, n), rng.normal(0, 0.12, n)); D['n_r_mw'] = D['n_r_mw'] + cm[ev] + rng.normal(0, 0.05, n)
nd = {k[2:]: v for k, v in D.items() if k.startswith('n_')}
for s in range(len(D['K'])):
    K = int(D['K'][s]); idx = D['idx'][s, :K]; A = LC.combine_v12(D['n_slat'][idx], D['n_slon'][idx], PR.Rof(nd, idx))
    D['anchor'][s] = [A[k] for k in ('lat', 'lon', 'logz', 'mw', 'logz_sig', 'mw_sig')]; D['anchor_cov'][s] = A['cov']
np.savez(H + f'bank/{out}_samples.npz', **D); print(out, len(D['K']), 'samples')
