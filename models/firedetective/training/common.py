"""Paths and helpers shared by FireDetective scripts."""
import os, sys, json
from _paths import SRC, MODEL, WORK, FIRE_CODE, FIRE, CHECKS, latest   # import: replaces the sandbox paths below
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', f'{WORK}/.numba')   # was <sandbox>/fire/firedetective/.numba
# sys.path: _paths puts shared/fire, shared/fire/firesim and training/ on it (was <sandbox>/fire/firesim,
# <sandbox>/fire and <sandbox>/fire/firedetective/src)
import numpy as np
import pandas as pd

ROOT = WORK   # was '<sandbox>/fire/firedetective'
DATA = f'{ROOT}/data'
BANK = f'{DATA}/bank'
REAL = f'{DATA}/real'
for d in (DATA, BANK, REAL, f'{ROOT}/logs', f'{ROOT}/tmp', f'{ROOT}/out'):
    os.makedirs(d, exist_ok=True)


def to5070(lon, lat):
    from pyproj import Transformer
    return Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)


def jdump(obj, fn):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (pd.Timestamp,)): return str(o)
        return str(o)
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=conv)
    os.replace(tmp, fn)


def local_day(t_utc, lon):
    """Local solar calendar day for a UTC timestamp."""
    t = pd.Timestamp(t_utc)
    if t.tz is not None:
        t = t.tz_convert(None)
    return (t + pd.Timedelta(hours=lon / 15.0)).normalize()
