"""Assemble a formula-arm folder from chosen laws: laws.json + sim-train bands + support modules + predict.py.

    from build_formula import build
    build(outdir, laws={'ice': law, 'hev': law, 'bev': law}, rung_id='r1', note='...')
A law is a lawsearch JSON dict (vars, tree, consts, complexity, expr). Bands: quantiles 5/25/50/75/95 % of log(y / E_law)
on the simulated train bank (tier draws 11, 12) per architecture x light/heavy duty. card.md / ratchet.json are written
provisionally (evaluate.py needs them); the released arm gets its final ones from finalize_formula.py.
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake  # noqa: E402
import lawvars as LV  # noqa: E402
import lawmodel as LM  # noqa: E402

ENET_FEATURES = '/home/claude/bakeoff/vehicle/harness/vendor/energynet/features.py'
_TRAIN = {}


def train_views(seeds=(11, 12)):
    if 'v' not in _TRAIN:
        B = bake.load_bank('train')
        out = []
        for s in seeds:
            T = bake.draw_tiers(B, np.random.default_rng(s)); D = bake.sim_table(B, T)
            V = LV.from_table(D['X'])
            out.append((V, D['y'], D['arch'], D['cls']))
        _TRAIN['v'] = out
    return _TRAIN['v']


def bands_for(laws):
    L = [[[] for _ in range(2)] for _ in range(7)]
    for V, y, a, c in train_views():
        E = LM.energy(laws, V)
        ok = y > 0.02
        lr = np.log(y[ok] / E[ok]); aa = a[ok]; hd = (c[ok] >= 4).astype(int)
        for i in range(7):
            for h in range(2):
                L[i][h].append(lr[(aa == i) & (hd == h)])
    q = np.zeros((7, 2, 5)); n = np.zeros((7, 2), int)
    for i in range(7):
        for h in range(2):
            x = np.concatenate(L[i][h]); n[i, h] = len(x)
            if len(x) < 50:
                x = np.concatenate(L[i][0])
            q[i, h] = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {'what': 'quantiles 5/25/50/75/95 % of log(y / E_law) on the sim train bank (tier draws 11, 12) per '
                    'architecture x duty (light, heavy); fewer than 50 rows -> the light-duty quantiles',
            'n': n.tolist(), 'logq': q.round(6).tolist()}


def build(outdir, laws, rung_id='r0', note=''):
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(os.path.join(HERE, 'formula_predict_template.py'), os.path.join(outdir, 'predict.py'))
    shutil.copy(os.path.join(HERE, 'lawsearch.py'), os.path.join(outdir, 'lawsearch.py'))
    shutil.copy(os.path.join(HERE, 'lawvars.py'), os.path.join(outdir, 'lawvars.py'))
    shutil.copy(ENET_FEATURES, os.path.join(outdir, 'enet_features.py'))
    keep = ('vars', 'tree', 'consts', 'complexity', 'expr', 'source')
    json.dump({'rung': rung_id, 'note': note, 'laws': {g: {k: L[k] for k in keep if k in L} for g, L in laws.items()}},
              open(os.path.join(outdir, 'laws.json'), 'w'), indent=1)
    json.dump(bands_for(laws), open(os.path.join(outdir, 'bands.json'), 'w'), indent=1)
    if not os.path.exists(os.path.join(outdir, 'ratchet.json')):
        json.dump({'arm_id': 'formula_cand_' + rung_id, 'label': 'formula candidate ' + rung_id, 'family': 'symbolic',
                   'group': 'phys', 'tracks': ['table', 'trace'], 'numpy_only': True, 'real_labels_in_training': False,
                   'compute': {'train_cpu_hours': 0}, 'feature_engineering': 'law variables', 'extra_files': [],
                   'rungs': [{'id': rung_id, 'change': note, 'decision': 'released', 'why': 'candidate'}],
                   'released': rung_id}, open(os.path.join(outdir, 'ratchet.json'), 'w'), indent=1)
        open(os.path.join(outdir, 'card.md'), 'w').write('# formula candidate ' + rung_id + '\n\n' + note + '\n' * 3)
    return outdir
