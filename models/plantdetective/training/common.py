import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from _paths import SRC, MODEL, WORK, GRID   # noqa: F401  (adds GRID/gridsim and GRID to sys.path)
ROOT = WORK                      # was the sandbox workspace grid/plantdetective: out/, web/, release/ now live in WORK
DATA = f'{ROOT}/data'


def epoch_s(idx):
    """UTC seconds of a DatetimeIndex (resolution-safe)."""
    import pandas as pd
    return ((idx - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta('1s')).values.astype('int64') \
        if getattr(idx, 'tz', None) is not None else \
        ((idx - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')).values.astype('int64')


def fix_times(t):
    """Early bank files stored microseconds // 1e9 (kiloseconds, floored); recover hourly UTC seconds."""
    import numpy as np
    t = np.asarray(t, 'int64')
    if t.max() < 10**8:
        t0 = int(np.ceil(t[0] * 1000 / 3600.0) * 3600)
        return t0 + 3600 * np.arange(len(t), dtype='int64')
    return t
