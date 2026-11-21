"""BasinDetective shared settings. Import this first in every script (sets threads + numba cache).

D is the workspace root (WORK in _paths.py; the sandbox's water/detective/). Paths changed on import into ML Ratchet
(18 Nov 2026); nothing else in this file did."""
import os
import shutil
import sys

from _paths import SRC, WORK

D = WORK
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', f'{D}/.numba')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for _d in ('bank', 'feats', 'models', 'logs', 'web', 'release/examples'):
    os.makedirs(f'{D}/{_d}', exist_ok=True)
for _k in ('gauge_v2.json', 'pbias_v2.json', 'drift_v3.json', 'annual_v4.json'):   # shipped dev-calibrated knobs
    if not os.path.exists(f'{D}/models/{_k}'):
        shutil.copyfile(f'{SRC}/{_k}', f'{D}/models/{_k}')

# Time windows (B's frozen protocol)
CAL = ('1999-10-01', '2008-09-30')          # gauged input: forcing + Q
VAL = ('1989-10-01', '1999-09-30')          # exam target window (forcing is an input; Q is the truth)
EARLY = ('1980-10-01', '1989-09-30')        # dev-only non-exam target window (dev gate, sigma fit, twins)

# Regression targets of the network (all from the target window unless noted)
DELTA_T = ['d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd']       # out-of-period change, real truth
PARAM_T = ['log_kslow', 'gw_share', 'log_soilcap', 'log_pmult']             # hidden, sim/twin truth only
REG_REG = 'log_dor'                                                         # log10 reservoir storage / annual flow
REG_CLS = 'reg'                                                             # regulated: dor > 0.1
REG_THR = 0.1
EPSQ = 0.01   # mm/day offset in log flow transforms


def lq(x):
    import numpy as np
    return np.log(np.asarray(x, float) + EPSQ)
