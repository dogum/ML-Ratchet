"""Shared helpers for the BuildingSim checks: imports, weather, JSON writing."""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json
import os
import platform
import sys
import time

sys.path.insert(0, f'{BLDG_CODE}/buildingsim')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402

CHECKS = os.path.dirname(os.path.abspath(__file__))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and (o != o):
        return None
    return o


def write(name, payload):
    payload = dict(payload)
    payload.setdefault('generated', time.strftime('%Y-%m-%d %H:%M'))
    payload.setdefault('python', platform.python_version())
    # Re-runs write to checks/out/ under the data root (git-ignored), never over the shipped v1.0 JSONs next to this
    # file; BUILDINGSIM_PUBLISH=1 writes next to this file instead (the original behaviour, to publish new results).
    d = CHECKS if os.environ.get('BUILDINGSIM_PUBLISH') == '1' else os.path.join(BLDG_ROOT, 'checks', 'out')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, 'w') as f:
        json.dump(_clean(payload), f, indent=1)
    print('wrote', path)
    return path
