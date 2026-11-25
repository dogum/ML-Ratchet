"""LawFinder-style symbolic search for attribute -> WaterSim parameter formulas (vendored lawsearch.py, unchanged).

The teacher is the physics-in-the-loop arm's trained mapping (an MLP trained through the simulator on the sim train
bank): for every sim-train run (index < 11,400) it gives the 17 unbounded parameter values z from the 18-value static
vector. For each parameter, lawsearch evolves expression trees over the 18 named static variables (+ - * / ^, exp,
log, sqrt, inv, sq; constants fitted by Levenberg-Marquardt) and keeps a complexity-vs-error Pareto front. The chosen
formula is the simplest one whose R^2 is within `tol` of the best R^2 on the front (complexity <= maxsize).

usage: python3 search_formula.py TEACHER_RUN_DIR OUT_DIR CONFIG_JSON
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake  # noqa: E402
import lawsearch as ls  # noqa: E402
import physarm as pa  # noqa: E402
import train_dpl_net as tn  # noqa: E402
import feats  # noqa: E402

torch.set_num_threads(1)
teacher, out_dir, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {'seconds': 60, 'maxsize': 15, 'tol': 0.02, 'n_fit': 1500, 'n_refit': 4000, 'seed': 0}
cfg.update(json.load(open(cfg_path)))
os.makedirs(out_dir, exist_ok=True)
c0 = time.process_time()
s = torch.load(os.path.join(teacher, 'ckpt.pt'), weights_only=False)
net = tn.Net(s['cfg']['hidden']).double(); net.load_state_dict(s['net'])
B = bake.load_bank('train', ['st'])
X = np.asarray(B['st'][:11400], np.float64)
with torch.no_grad():
    Zt = net(torch.tensor((X - s['mu']) / s['sd'])).numpy()
Zt = np.clip(Zt, -6, 6)
names = feats.STATIC_NAMES
rng = np.random.default_rng(cfg['seed'])
ref = rng.choice(len(X), cfg['n_refit'], replace=False)
res_path = os.path.join(out_dir, 'formulas.json')
res = json.load(open(res_path)) if os.path.exists(res_path) else {'vars': names, 'params': {}, 'cfg': cfg}
for j, pn in enumerate(pa.NAMES):
    if pn in res['params']:
        continue
    y = Zt[:, j]
    var = float(np.var(y))
    S = ls.Search(names, maxsize=cfg['maxsize'], seed=cfg['seed'] + j, n_fit=cfg['n_fit'])
    S.fit(X, y, seconds=cfg['seconds'])
    front = S.front(X[ref], y[ref])
    rows = [{'complexity': int(c), 'r2': float(1 - l / max(var, 1e-12)), 'tree': ls.to_json(t),
             'consts': [float(v) for v in cc], 'str': ls.to_str(t, cc, names)} for c, l, t, cc in front]
    best = max(r['r2'] for r in rows) if rows else 0.0
    pick = next(r for r in rows if r['r2'] >= best - cfg['tol'])
    # constant fallback for non-finite evaluations
    pick = dict(pick); pick['fallback'] = float(np.median(y)); pick['teacher_var'] = var; pick['front'] = rows
    res['params'][pn] = pick
    print(f'{pn}: var {var:.3f} pick c{pick["complexity"]} r2 {pick["r2"]:.3f}: {pick["str"]}  (best r2 {best:.3f})',
          flush=True)
    res['cpu_s'] = res.get('cpu_s', 0) + time.process_time() - c0; c0 = time.process_time()
    json.dump(res, open(res_path + '.tmp', 'w'), indent=1); os.replace(res_path + '.tmp', res_path)
print('total cpu s', res['cpu_s'])
