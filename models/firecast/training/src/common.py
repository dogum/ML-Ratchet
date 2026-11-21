"""FireCast shared helpers: paths, thread limits, Phase 1 imports."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # training/ (for _paths)
from _paths import WORK, TRAIN, MODEL, FIRE, FIRE_DATA, PROC, SRC, shipped   # noqa: E402  (also puts shared/fire on sys.path)
os.environ.setdefault('NUMBA_CACHE_DIR', f'{WORK}/.numba')
import numpy as np  # noqa: E402

ROOT = WORK                      # every output (ckpt/, logs/, web/, release/, ratchet.json)
OUT = f'{WORK}/out'              # result JSONs (shipped copies: training/results/, read through shipped())
BANK = f'{WORK}/bank'            # simulated banks
PHYS = f'{TRAIN}/phys'           # shipped physics versions p0, p1 (read-only)
CONFIGS = f'{TRAIN}/configs'     # shipped rung configs (read-only)

RES = 90.0
CELL_KM2 = RES * RES / 1e6
WIN_KM = 10.0


def fd():
    import firedata
    return firedata


def fs():
    import firesim
    return firesim


def jdump(obj, fn):
    def _c(o):
        if isinstance(o, dict):
            return {str(k): _c(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_c(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return _c(o.tolist())
        return o
    tmp = fn + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_c(obj), f, indent=1)
    os.replace(tmp, fn)


def mem_mb():
    try:
        import glob
        f = glob.glob('/sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes')[0]
        return int(open(f).read()) / 1e6
    except Exception:
        return float('nan')
