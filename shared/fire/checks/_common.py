"""Shared helpers for the FireSim checks."""
import json
import os
import platform
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # shared/fire (code)
FIRE_ROOT = os.environ.get('ML_RATCHET_FIRE') or _CODE               # data root: data/, checks/out/, firesim/refdata/, firesim/.refenv/
sys.path.insert(0, os.path.join(_CODE, 'firesim'))
sys.path.insert(0, _CODE)
import numpy as np  # noqa: E402

CHECKS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIRE_ROOT, 'checks', 'out')
os.makedirs(OUT, exist_ok=True)


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
    path = os.path.join(os.environ.get('FIRESIM_CHECKS_OUT') or CHECKS, name)   # tests write to a temp folder
    with open(path, 'w', newline='\n') as f:
        json.dump(_clean(payload), f, indent=1)
    print('wrote', path)
    return path


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)
