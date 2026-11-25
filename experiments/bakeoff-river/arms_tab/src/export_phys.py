"""Export a physics arm's parameter mapping to numpy and fit its error bands on held-out sim-TRAIN runs.

usage: python3 export_phys.py dpl RUN_DIR            (MLP from RUN_DIR/ckpt.pt -> RUN_DIR/dpl_weights.npz)
       python3 export_phys.py formula RUN_DIR        (formulas from RUN_DIR/formulas.json -> RUN_DIR/bands.npz)
Bands: train runs 11,400 .. 11,999 (never used for gradient steps), window rows 0..1461 (365 warm-up + 1,096 days),
each simulated by the vendored WaterSim with the mapped parameters (the same code path as predict).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake  # noqa: E402
import physarm as pa  # noqa: E402
import wsim  # noqa: E402

kind, run_dir = sys.argv[1], sys.argv[2]
c0 = time.process_time()
B = bake.load_bank('train', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
info = {}
if kind == 'dpl':
    import torch
    s = torch.load(os.path.join(run_dir, 'ckpt.pt'), weights_only=False)
    sd_ = s['net']
    nl = len([k for k in sd_ if k.endswith('.weight')])
    Ws, bs = [], []
    for i in range(nl - 1):
        Ws.append(sd_[f'layers.{i}.weight'].numpy()); bs.append(sd_[f'layers.{i}.bias'].numpy())
    Ws.append(sd_['out.weight'].numpy()); bs.append(sd_['out.bias'].numpy())
    arr = {'n_layers': np.array(nl), 'mu': s['mu'], 'sd': s['sd']}
    for i, (W, b) in enumerate(zip(Ws, bs)):
        arr[f'W{i}'] = W; arr[f'b{i}'] = b
    # numpy == torch check on the static vectors of 200 train runs
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_dpl_net as tn  # noqa: E402
    net = tn.Net(s['cfg']['hidden']).double(); net.load_state_dict(sd_)
    x = (np.asarray(B['st'][:200], np.float64) - s['mu']) / s['sd']
    with torch.no_grad():
        zt = net(torch.tensor(x)).numpy()
    zn = pa.mlp(x, Ws, bs)
    info['numpy_vs_torch_maxabs_z'] = float(np.abs(zt - zn).max())

    def mapping(b):
        x = (np.asarray(b['static'], np.float64) - arr['mu']) / arr['sd']
        return {k: float(v[0]) for k, v in pa.to_params(pa.mlp(x, Ws, bs)).items()}
    out_file = os.path.join(run_dir, 'dpl_weights.npz')
elif kind in ('formula', 'formula_tuned'):
    import formula_eval as fe  # noqa: E402
    if kind == 'formula_tuned':                  # write the tuned constants into RUN_DIR/formulas.json
        import torch
        import lawsearch as ls
        s = torch.load(os.path.join(run_dir, 'ckpt.pt'), weights_only=False)
        F = json.load(open(s['cfg']['formulas']))
        Xs = np.asarray(B['st'][:11400], np.float64)
        for j, n in enumerate(pa.NAMES):
            c = s['net'][f'consts.{j}'].numpy().tolist()
            F['params'][n]['consts_search'] = F['params'][n]['consts']
            F['params'][n]['consts'] = c
            F['params'][n]['str'] = ls.to_str(ls.from_json(F['params'][n]['tree']), np.array(c), F['vars'])
            v = ls.evaluate(ls.from_json(F['params'][n]['tree']), Xs, np.array(c))
            info[f'nonfinite_{n}'] = int((~np.isfinite(v)).sum())
        for n in pa.NAMES:
            F['params'][n].pop('front', None)
        json.dump(F, open(os.path.join(run_dir, 'formulas.json'), 'w'), indent=1)
    F = json.load(open(os.path.join(run_dir, 'formulas.json')))

    def mapping(b):
        return fe.params_from_formulas(F, b['static'])
    arr = {}
    out_file = os.path.join(run_dir, 'bands.npz')
else:
    raise ValueError(kind)
qh, qo, pms = [], [], []
for i in range(11400, 12000):
    b = bake.sim_window(B, i, 0, 1096)
    q = np.maximum(wsim.run_batch(b, mapping(b)), 0.0)
    qh.append(q); qo.append(bake.sim_target(B, i, b)); pms.append(np.full(len(q), np.exp(b['static'][11])))
qh, qo, pms = np.concatenate(qh), np.concatenate(qo), np.concatenate(pms)
bd = pa.fit_bands(qh, qo, pms)
arr.update(bd)
Q = pa.apply_bands(qh, pms, bd['edges'], bd['rq'])
info.update({'band_fit_cov90': float(np.mean((qo >= Q[:, 0]) & (qo <= Q[:, 4]))),
             'band_fit_cov50': float(np.mean((qo >= Q[:, 1]) & (qo <= Q[:, 3]))),
             'rq': bd['rq'].round(3).tolist(), 'cpu_s': time.process_time() - c0})
np.savez(out_file, **arr)
json.dump(info, open(os.path.join(run_dir, 'export.json'), 'w'), indent=1)
print(json.dumps(info), flush=True)
