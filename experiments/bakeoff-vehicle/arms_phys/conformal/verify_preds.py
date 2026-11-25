"""Check that an existing dev-prediction file (an arm's own evaluate.py output) was made by the arm as registered:
re-run the arm on the first 24 records of each dev set (the register.py smoke batches) and compare.
usage: python3 verify_preds.py <base_dir> <track> <dev_preds.parquet>   -> prints JSON {ok, max_rel_diff, n_checked}"""
import json
import os
import sys

HARNESS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'harness'))
sys.path.insert(0, HARNESS)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
from armlib import Arm, strip_batch  # noqa: E402

base, track, path = sys.argv[1:4]
Fr = bake.dev_frame()
P = pd.read_parquet(path)
out = {'ok': False, 'max_rel_diff': None, 'n_checked': 0}
if len(P) == len(Fr) and all(k in P for k in S.OUT_KEYS):
    arm = Arm(base)
    off = {name: int(Fr.index[Fr.recset == name][0]) for name in bake.SETS}
    md = 0.0; n = 0
    for name in bake.SETS:
        for nm, sl, b in bake._real_batches('dev', track, sets=(name,), chunk=24):
            p, Q, nc, _ = S.check_output(arm.run([strip_batch(b)])[0], b['n'])
            i0 = off[name] + sl.start
            ref = P.iloc[i0:i0 + b['n']]
            md = max(md, float(np.max(np.abs(p / ref.pred.values - 1))),
                     float(np.max(np.abs(Q / ref[['q05', 'q25', 'q50', 'q75', 'q95']].values - 1))))
            n += b['n']
            break
    out = {'ok': bool(md < 1e-6), 'max_rel_diff': md, 'n_checked': n}
print(json.dumps(out))
