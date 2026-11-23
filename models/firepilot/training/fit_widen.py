"""Widen-only sigma factor for the value net's predicted saved fraction, fitted on the real dev gate only.

  python3 fit_widen.py <rung>  -> results/sigma_widen.json

For the chosen plan on each gate case the net predicts (pred, sigma); the truth-world saved fraction is the target.
factor = smallest k >= 1 such that |saved - pred| <= 1.645 k sigma on >= 90 % of gate cases (widen-only rule).
"""
import os
import sys
import json
import numpy as np

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
from fp.env import PHYSICS as VERSION  # noqa: E402  (same physics-versioned file eval_rung.py writes)

ROOT = _paths.WORK
rung = sys.argv[1]
ev = [json.loads(l) for l in open(os.path.join(ROOT, 'results', f'eval_gate_{VERSION}_{rung}.jsonl'))]
ev = list({e['name']: e for e in ev}.values())
y = np.array([e['saved'] for e in ev])
m = np.array([e['pred'] for e in ev])
s = np.array([e['sigma'] for e in ev])
z = np.abs(y - m) / s
k90 = float(np.quantile(z, 0.9) / 1.645)
widen = max(1.0, k90)
out = dict(rung=rung, n=len(ev), bias=float(np.mean(y - m)), cov90_raw=float(np.mean(z <= 1.645)),
           cov50_raw=float(np.mean(z <= 0.674)), k90=k90, widen=widen,
           cov90_widened=float(np.mean(z <= 1.645 * widen)), cov50_widened=float(np.mean(z <= 0.674 * widen)))
json.dump(out, open(os.path.join(ROOT, 'results', 'sigma_widen.json'), 'w'), indent=1)
print(out)
