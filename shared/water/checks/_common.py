"""Shared helpers for the WaterSim checks: imports, JSON writing, dev-only data access."""
import json
import os
import platform
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # shared/water
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                    # data root (git-ignored data/)
sys.path.insert(0, os.path.join(_CODE, 'watersim'))
sys.path.insert(0, _CODE)
import numpy as np  # noqa: E402

CHECKS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CHECKS, 'out')

# Non-exam periods (B's splits.json: exam years are 1989-10-01..1999-09-30 and 2015-10-01..2023-09-30)
CAL = ('1999-10-01', '2008-09-30')
VAL_EARLY = ('1980-10-01', '1989-09-30')
VAL_LATE = ('2008-10-01', '2014-09-30')
EXAM_WINDOWS = [('1989-10-01', '1999-09-30'), ('2015-10-01', '2023-09-30')]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and (o != o or o in (float('inf'), float('-inf'))):
        return None
    return o


def write(name, payload):
    payload = dict(payload)
    payload.setdefault('generated', time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))
    payload.setdefault('python', platform.python_version())
    path = os.path.join(os.environ.get('WATERSIM_CHECKS_OUT') or CHECKS, name)   # tests write to a temp folder
    with open(path, 'w') as f:
        json.dump(_clean(payload), f, indent=1)
    print('wrote', path)
    return path


def dev_basins():
    """CAMELS-US dev basins from B's frozen splits (never exam)."""
    import waterdata as wd
    ids = wd.basins('camels', split='dev')
    sp = json.load(open(os.path.join(_DATA, 'data', 'proc', 'splits.json')))
    return ids, sp


def assert_not_exam_window(start, end):
    import pandas as pd
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for a, b in EXAM_WINDOWS:
        if not (e < pd.Timestamp(a) or s > pd.Timestamp(b)):
            raise RuntimeError(f'window {start}..{end} overlaps an exam period {a}..{b}')
