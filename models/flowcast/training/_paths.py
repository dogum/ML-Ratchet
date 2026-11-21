"""Where FlowCast's training scripts read and write.

The scripts were first run on 18 Nov 2026 in a cloud sandbox workspace (water/flowcast/, code in src/, the Watershed
Phase 1 tree at water/: watersim/, waterdata/, checks/, data/). Every run there was CPU-only, one shared core. In
this repository:

    SRC         training/src/ (the scripts, as run; src/common.py imports this module)
    TRAIN       training/ — the shipped, read-only inputs and records: configs/, phys/ (physics versions p0-p4),
                ratchet.json and results/ (the result JSONs as shipped)
    MODEL       models/flowcast/ (flowcast.py and the released weights)
    WORK        models/flowcast/work/ (git-ignored), or $ML_RATCHET_WORK/flowcast/ — the old workspace root for
                everything a script writes: bank/ (simulated banks, ~0.4 GB per 12k bank), ckpt/, results/, preds/,
                web/, release/, .numba/, ratchet.json
    WATER       the shared water platform code: shared/water/ (watersim/, waterdata/, checks/). Both shared/water and
                shared/water/watersim are put on sys.path so `import watersim` and `import waterdata` work.
    WATER_DATA  the water data root: $ML_RATCHET_WATER, default shared/water. Scripts read data/proc/... under it
                (waterdata itself reads the same variable). The real time series (Caravan, NWIS, NWM) are not in
                the repository: rebuild them with `mlratchet data water` before any script that touches real flow.

Re-running a script writes new copies under WORK; the shipped JSONs in TRAIN stay as they were.
Inference (flowcast.py), demo.py and the instrument export need none of this.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src').replace(os.sep, '/')
TRAIN = os.path.dirname(SRC)
MODEL = os.path.dirname(TRAIN)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'flowcast')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
WATER = os.path.join(REPO, 'shared', 'water').replace(os.sep, '/')
WATER_DATA = os.path.abspath(os.environ.get('ML_RATCHET_WATER') or WATER).replace(os.sep, '/')
for p in (WATER, os.path.join(WATER, 'watersim')):       # import watersim and waterdata from the repository
    if p not in sys.path:
        sys.path.insert(1, p)
for d in ('bank', 'ckpt', 'results', 'preds', 'web', 'release', '.numba'):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)


def latest(rel):
    """A result written under WORK by a re-run if there is one, else the shipped copy in TRAIN."""
    w = os.path.join(WORK, rel)
    return w if os.path.exists(w) else os.path.join(TRAIN, rel)
