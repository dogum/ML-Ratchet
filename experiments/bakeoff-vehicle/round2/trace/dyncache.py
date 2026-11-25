"""Tier-independent 5-s block features of every trip of a round-2 bank (tracelib.dyn_features), computed once.

python dyncache.py <bank> [<bank> ...]   -> round2/work/dyncache/<bank>.npz (dyn float32 (blocks, 12), nb, dist, dur)
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths as P  # noqa: E402
sys.path.insert(0, os.path.join(P.BAKEOFF, 'arms_trace', 'common'))
import numpy as np  # noqa: E402
import tracelib as TL  # noqa: E402

D = os.path.join(P.WORK, 'dyncache')


def path(name):
    return os.path.join(D, name + '.npz')


def load(name):
    z = np.load(path(name))
    return {k: z[k] for k in z.files}


def build(name):
    os.makedirs(D, exist_ok=True)
    if os.path.exists(path(name)):
        return
    t0 = time.time()
    d = P.bank_dir(name)
    off = np.load(os.path.join(d, 'rows.npz'))['trace_off'].astype(np.int64)
    tv = np.load(os.path.join(d, 'trace_v.npy'), mmap_mode='r')
    tx = np.load(os.path.join(d, 'trace_x.npy'), mmap_mode='r')
    n = len(off) - 1
    L = np.diff(off); nb = -(-L // TL.BLOCK)
    bo = np.concatenate([[0], np.cumsum(nb)])
    dyn = np.empty((bo[-1], TL.N_DYN), np.float32); dist = np.zeros(n); dur = np.zeros(n)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        tr = np.concatenate([np.asarray(tv[a:b], np.float32)[:, None], np.asarray(tx[a:b], np.float32)], 1)
        dyn[bo[i]:bo[i + 1]] = TL.dyn_features(tr)
        dist[i], dur[i] = TL.trip_dist_dur(tr)
    np.savez(path(name) + '.tmp.npz', dyn=dyn, nb=nb.astype(np.int64), dist=dist, dur=dur)
    os.replace(path(name) + '.tmp.npz', path(name))
    print(name, n, 'trips', len(dyn), 'blocks', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        build(nm)
