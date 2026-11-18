"""Simulated bank: generated buildings x real weather (BDG2 site-years 2016/2017 + TMYx cities), 1 h, one year.

Resumable shards of 200 buildings: bank/<tag>/shard_XXXX.npz (+ .json info). Meters stored as float16 / scale.
Use mix: 30 % homes (single / multi 50:50), 70 % commercial (10 types, balanced). Shard ids >= 900 are the
FROZEN simulated validation set of this physics version.
usage: python3 simbank.py <tag> <shard0> <shard1>
Tags in PHYSICS use BuildingSim physics versions (rung r10, NOTEBOOK 2026-11-17): 'p12' = generator 1.4 (intensity
per use; replaces W1) and simulate 1.2 (1.1 district plants + 1.2 electricity), no 1.3; W2 campus assignment only.
Every other tag is the v1.0 path (physics 1.0 with W1/W2).
"""
import json
import os
import sys
import time

import numpy as np

import common as C
from wrappers import load_cal
from simone import simulate_one, wx_name

NPER = 200
NL = 8784
COM = [u for u in C.USES if not u.startswith('residential')]
PHYSICS = {'p12': dict(gen='1.4', sim='1.2')}


def build_shard(tag, sid):
    import buildingsim as bs
    d = os.path.join(C.ROOT, 'bank', tag)
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f'shard_{sid:04d}.npz')
    if os.path.exists(fn):
        return
    cal = load_cal()
    rng = np.random.default_rng(10_000 + sid)
    nres = int(round(NPER * 0.3))
    ph = PHYSICS.get(tag, {})
    specs = bs.generate_buildings(nres, seed=20_000 + sid, primary_use=['residential_single', 'residential_multi'],
                                  physics=ph.get('gen'))
    specs += bs.generate_buildings(NPER - nres, seed=30_000 + sid, primary_use=COM, physics=ph.get('gen'))
    rng.shuffle(specs)
    X = np.full((len(specs), 4, NL), np.nan, np.float16)
    S = np.zeros((len(specs), 4))
    Y = np.full((len(specs), len(C.REG)), np.nan, np.float32)
    K = np.zeros((len(specs), 3), np.int16)
    infos = []
    wxd = os.path.join(C.ROOT, 'bank', 'weather')
    os.makedirs(wxd, exist_ok=True)
    t0 = time.time()
    for i, s in enumerate(specs):
        try:
            r = simulate_one(s, rng, cal, physics=ph.get('sim'))
        except Exception as e:                     # keep going; record the failure
            infos.append(dict(id=s.get('id'), error=repr(e)[:300]))
            K[i] = -1
            continue
        for j, ch in enumerate(C.CHANNELS):
            if ch in r['meters']:
                v = r['meters'][ch]
                sc = max(float(np.nanmax(np.abs(v))), 1e-9)
                S[i, j] = sc
                X[i, j, :len(v)] = (v / sc).astype(np.float16)
        Y[i] = [r['y'][k] for k in C.REG]
        K[i] = [C.USES.index(r['cls']['use']), C.HEAT.index(r['cls']['heat']), C.COOL.index(r['cls']['cool'])]
        info = r['info']
        info['n_hours'] = len(r['T'])
        infos.append(info)
        wf = os.path.join(wxd, wx_name(r['wx_key']).replace(':', '_') + '.npz')
        if not os.path.exists(wf):
            np.savez(wf, T=r['T'], GHI=r['GHI'], utc=r['index'].asi8, tz=r['tz'], lat=r['lat'])
    np.savez_compressed(fn + '.tmp.npz', X=X, S=S, Y=Y, K=K)
    json.dump(infos, open(fn.replace('.npz', '.json'), 'w'))
    os.replace(fn + '.tmp.npz', fn)
    print(tag, sid, 'done %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    tag, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for sid in range(a, b):
        build_shard(tag, sid)
