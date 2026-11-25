"""Round 2 of the River Bake-off: the thin path layer every round-2 script imports first.

    import r2                      # puts harness/ on sys.path, sets one thread, picks the bank
    r2.use_bank('p1')              # point bake.load_bank at round2/work/banks/p1 (or any built bank name)

The harness (harness/config.py, bake.py, evaluate.py, register.py) is used unchanged. Only config.BANK is redirected,
because the frozen v1.0 bank directory (harness/bank/) holds just its MANIFEST.json in this repository and is not
git-ignored. Banks are written under round2/work/banks/<name>/ (git-ignored by the repo's `work/` rule).
The bank name can also come from the environment: R2_BANK=<name>.
"""
import os
import sys

ROUND2 = os.path.dirname(os.path.abspath(__file__))
BAKEOFF = os.path.dirname(ROUND2)
HARNESS = os.path.join(BAKEOFF, 'harness')
WORK = os.path.join(ROUND2, 'work')
BANKS = os.path.join(WORK, 'banks')
RESULTS = os.path.join(ROUND2, 'results')
PHYS = os.path.join(ROUND2, 'phys')
for p in (HARNESS, ROUND2):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(WORK, '.numba'))
import config as C  # noqa: E402  (sets one thread, data root, WaterSim on sys.path)

V1_RESULTS = os.path.join(BAKEOFF, 'results')
V1_EXAM_ID = '20261123T074907'


def bank_dir(name):
    return os.path.join(BANKS, name)


def use_bank(name=None):
    """Redirect the harness bank (config.BANK) to round2/work/banks/<name>. Returns the directory."""
    name = name or os.environ.get('R2_BANK')
    if not name:
        raise ValueError('no bank name (pass one or set R2_BANK)')
    d = bank_dir(name)
    if not os.path.exists(os.path.join(d, 'MANIFEST.json')):
        raise FileNotFoundError(f'bank {name} is not finalized ({d})')
    C.BANK = d
    return d


if os.environ.get('R2_BANK'):
    use_bank(os.environ['R2_BANK'])
