"""Rung r5 physics (v4): regenerate the PHEV examples of a bank with wrappers.phev_patch (labels re-simulated too).
python rebuild_phev.py <fleet> <K> <seed>  -> data/bank_<fleet>_v4/ (all other rows copied unchanged)"""
import sys, os, json, glob, time
from common import *
import numpy as np
import bank, wrappers, fleet
fl, K, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src = f'{EN}/data/bank_{fl}'; dst = f'{EN}/data/bank_{fl}_v4'; os.makedirs(dst, exist_ok=True)
for cp in sorted(glob.glob(f'{EN}/data/fleet_{fl}/chunk_*.json')):
    ci = int(cp[-8:-5]); op = f'{dst}/chunk_{ci:03d}.npz'
    if os.path.exists(op): continue
    t0 = time.time()
    D = dict(np.load(f'{src}/chunk_{ci:03d}.npz'))
    rng = np.random.default_rng([seed, ci, 4])
    keep = ~np.isin(D['arch'], [4, 5])
    rows = []
    for vi, V in enumerate(json.load(open(cp))):
        s = V['spec']
        if not s['architecture'].startswith('phev'): continue
        p = wrappers.phev_patch(s)
        lab = fleet.lab_labels(p) if s['class'] not in HD else None
        rows += bank.vehicle_examples(p, lab or {}, K, rng, ci * 100 + vi)
    N = {k: D[k][keep] for k in D}
    if rows:
        R = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        N = {k: np.concatenate([N[k], R[k].astype(N[k].dtype)]) for k in N}
    np.savez_compressed(op + '.tmp.npz', **N); os.replace(op + '.tmp.npz', op)
    print('chunk', ci, int((~keep).sum()), '->', len(rows), round(time.time() - t0, 1), flush=True)
