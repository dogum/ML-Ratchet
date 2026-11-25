"""Build the tabular training set for the GBM and GP arms from the sim TRAIN bank only.

One window per train run (365 warm-up + 365 target + 1 rows, random offset, seed 11), features from feats.py, and
K random target days per window (seed 12). Saves X (float32), y = qo / p_mean, run, pm to <out>.npz.
usage: python3 build_tab.py OUT.npz [K]
"""
import os
import sys
import time

sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake  # noqa: E402
import numpy as np  # noqa: E402
import feats  # noqa: E402

out = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NWIN = int(sys.argv[3]) if len(sys.argv) > 3 else 1      # windows per run (window 2 uses seed 13 offsets)
B = bake.load_bank('train', ['F', 'qo', 'attrs', 'st', 'start'])
n = len(B['qo'])
rng = np.random.default_rng(11)
offs = rng.integers(0, 1462 - 731 + 1, n)
offs_all = [offs] + [np.random.default_rng(11 + 2 * w).integers(0, 1462 - 731 + 1, n) for w in range(1, NWIN)]
rng2 = np.random.default_rng(12)
Xs, ys, runs, pms = [], [], [], []
t0 = time.time()
for k in range(0, n, 400):
    idx = np.arange(k, min(n, k + 400))
    for w, of in enumerate(offs_all):
        T = bake.train_arrays(B, idx, of[idx])
        X, pm = feats.features(T['forcing'], T['doy'], T['static'], 365, 365)
        y = T['target'] / pm[:, None]
        for j in range(len(idx)):
            d = np.sort(rng2.choice(365, K, replace=False))
            Xs.append(X[j, d]); ys.append(y[j, d].astype(np.float32))
            runs.append(np.full(K, idx[j], np.int32)); pms.append(np.full(K, pm[j], np.float32))
    print(f'{k + len(idx)}/{n} {time.time() - t0:.0f}s', flush=True)
np.savez(out, X=np.concatenate(Xs), y=np.concatenate(ys), run=np.concatenate(runs), pm=np.concatenate(pms),
         names=np.array(feats.NAMES))
print('saved', out, time.process_time(), 'cpu s')
