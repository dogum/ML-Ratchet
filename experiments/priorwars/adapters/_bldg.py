"""Shared loader for the BuildingDetective adapters. Read-only use of the model's training workspace and released weights.

Paths (repository): feats/f2/ (bank features) and feats/real/ (BDG2 and RECS features) are in the model's training
workspace (pw_paths.model_work), rebuilt by models/buildingdetective/training (featurize.py, realfeats.py). The
released networks are models/buildingdetective/weights/<mode>.npz, which the sandbox held as models/final/<mode>.npz.
"""
import sys
sys.dont_write_bytecode = True
import functools
import glob
import json
import os
import numpy as np
import pw_paths as _P

BD = _P.model_work('buildingdetective')
WEIGHTS = _P.model_dir('buildingdetective') + '/weights'
USES = ['office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse', 'residential_single',
        'residential_multi', 'food', 'lab', 'other']
HEAT = ['heat_pump', 'resistance', 'fuel', 'district', 'none']
REG = ['logA', 'year', 'logUA', 'logUAm2', 'logtau', 'Tb_heat', 'Tb_cool', 'logS_heat', 'logS_cool', 'base_share',
       'op_hours', 'weekend']


@functools.lru_cache(maxsize=4)
def net(mode):
    """the released network (models/final/<mode>.npz), forward pass copied from detective.Net (numpy only)"""
    from scipy.special import erf
    z = np.load(f'{WEIGHTS}/{mode}.npz', allow_pickle=False)
    n = int(z['n_layers'])
    W = [z[f'W{i}'] for i in range(n)]
    b = [z[f'b{i}'] for i in range(n)]

    def raw(X):
        h = (np.atleast_2d(X).astype(np.float64) - z['xm']) / z['xs']
        for Wi, bi in zip(W, b):
            h = h @ Wi.T + bi
            h = 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))
        r = h @ z['Wr'].T + z['br']
        nr = len(z['ym'])
        return r[:, :nr] * z['ys'] + z['ym'], h @ z['Wc'].T + z['bc']
    return raw, (z['xm'], z['xs'])


def bank(mode, shards):
    X, Y, K = [], [], []
    for s in shards:
        f = f'{BD}/feats/f2/{mode}_{s:04d}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f)
        X.append(z['X']); Y.append(z['Y']); K.append(z['K'])
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(float), np.concatenate(K).astype(int)


def real(split, name):
    z = np.load(f'{BD}/feats/real/{split}_{name}.npz')
    return {k: z[k] for k in z.files}, json.load(open(f'{BD}/feats/real/{split}_{name}.json'))
