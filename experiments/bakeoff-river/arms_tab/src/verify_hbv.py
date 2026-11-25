"""Verify the torch HBV (hbv_torch.py) against the harness's vendored WaterSim (wsim.py -> watersim.simulate) on
sim val windows and dev basins with random parameter sets (float64). Writes the max/median abs differences to
<out>.json. usage: python3 verify_hbv.py OUT.json"""
import json
import os
import sys
import time

sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import hbv_torch as H  # noqa: E402
import wsim  # noqa: E402

torch.set_num_threads(1)
rng = np.random.default_rng(5)
B = bake.load_bank('val', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
batches = [bake.sim_window(B, i, a) for i, a in [(0, 0), (5, 731), (17, 0), (333, 731), (512, 0), (799, 731)]]
dev = bake.dev_basins()
batches += [bake.dev_batch(dev[k]) for k in (0, 7, 20, 41)]
# make sure some have irrigation, one band and big areas: report what we have
rows = []
for b in batches:
    for rep in range(3):
        z = rng.normal(0, 1.3, len(H.PARAMS))
        p = H.to_params(torch.tensor(z[None], dtype=torch.float64))
        prm = {k: float(v[0]) for k, v in p.items()}
        t0 = time.time()
        qw = wsim.run_batch(b, prm)
        tw = time.time() - t0
        nw, nt = b['n_warm'], b['n_target']
        F = torch.tensor(np.asarray(b['forcing'], np.float64)[None, :nw + nt])
        frac, dz = H.bands([float(b['attrs'][2])])
        t0 = time.time()
        with torch.no_grad():
            qt = H.run(torch.clamp(F[..., 0], min=0.0), F[..., 1], F[..., 2], p, frac, dz,
                       torch.tensor([float(b['attrs'][0])], dtype=torch.float64),
                       torch.tensor([float(b['attrs'][9])], dtype=torch.float64), nw)[0].numpy()
        tt = time.time() - t0
        d = np.abs(qt - qw)
        rows.append({'uid': b['uid'], 'rep': rep, 'max_abs': float(d.max()), 'max_rel': float(d.max() / max(qw.max(), 1e-9)),
                     'mean_q': float(qw.mean()), 'irr': float(b['attrs'][9]), 'range_m': float(b['attrs'][2]),
                     'area': float(b['attrs'][0]), 't_ws': tw, 't_torch': tt})
        print(rows[-1], flush=True)
res = {'n': len(rows), 'max_abs_mm_day': max(r['max_abs'] for r in rows),
       'median_max_abs': float(np.median([r['max_abs'] for r in rows])),
       'max_rel_to_peak': max(r['max_rel'] for r in rows), 'rows': rows}
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print('MAX ABS', res['max_abs_mm_day'], 'MEDIAN', res['median_max_abs'])
