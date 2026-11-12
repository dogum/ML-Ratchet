"""Build a bank of simulated examples in real noise, in resumable chunks, stored as float16.

    python simbank.py train1 50000 1      # training bank, seed 1
    python simbank.py val1 5000 99        # the FROZEN validation set: build once per physics version
"""
import os
import sys

import numpy as np

import _paths as _P
import noise
import simulator as S

CHUNK = 1000


def main(name: str, n: int, seed: int):
    out = _P.work("bank", name)
    os.makedirs(out, exist_ok=True)
    pool = noise.load_pool()
    for c in range(0, n, CHUNK):
        f = os.path.join(out, f"{c // CHUNK:05d}.npz")
        if os.path.exists(f):
            continue  # resumable
        rng = np.random.default_rng([seed, c])
        X, Y = [], []
        for _ in range(min(CHUNK, n - c)):
            p = S.draw_params(rng)
            bg = pool[rng.integers(len(pool))]
            X.append(noise.inject(S.simulate(p, rng), bg, snr=rng.uniform(5, 30), rng=rng))
            Y.append(S.targets(p))
        np.savez(f, x=np.asarray(X, np.float16), y=np.asarray(Y, np.float32), physics=S.PHYSICS_VERSION)
        print(f"{name}: {c + len(X)}/{n}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
