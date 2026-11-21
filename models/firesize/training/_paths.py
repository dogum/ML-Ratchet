"""Where FireSize's training scripts read and write.

The scripts were first run on 18-19 Nov 2026 in a cloud sandbox (Wildfire Phase 2, project 2): the workspace was
fire/firesize/ (code in src/, outputs in data/, models/, results/, web/, release/, logs/, tmp/), next to the Wildfire
Phase 1 tree fire/ (firesim/, firedata/, data/proc/, data/scripts/). Every run there was CPU-only, one thread. In this
repository:

    SRC         training/src/ (every script as run; src/common.py imports this module)
    TRAIN       training/ — the shipped, read-only records: rungs.json (the rung list ratchet.py reads),
                ratchet.json, results/ (exam, post-exam, widening, twin and dev-gate JSONs) and
                results/calibration/ (the suppression fits calib_*.json, written to data/ in the sandbox)
    MODEL       models/firesize/ (firesize.py, the released weights, ros_lut.npz)
    WORK        models/firesize/work/ (git-ignored), or $ML_RATCHET_WORK/firesize/ — the old workspace root for
                everything a script writes: data/ (90 m windows win/, 250 m national grids nat/, gridMET points,
                ignitions, simulated banks bank/<physics>/, feature parquets, calib_*.json, ros_lut.npz),
                models/ (one npz + json per rung), results/, web/, release/, logs/, tmp/, .numba/, ratchet.json
    FIRE        the shared fire platform code: shared/fire/ (firesim/, firedata/, data/scripts/). shared/fire and
                shared/fire/firesim are put on sys.path so `import firedata` and `import firesim` work.
    FIRE_DATA   the fire data root: $ML_RATCHET_FIRE, default shared/fire. Scripts read data/proc/... under it
                (splits.json, firesize_split.parquet, fpafod/fod.parquet, weather/gridmet/fod_*.parquet,
                baselines/firesize_exam.json). The large tables are not in the repository: rebuild them with
                `mlratchet data fire` before any script that touches FPA-FOD records.

Re-running a script writes new copies under WORK; the shipped JSONs in TRAIN stay as they were.
Inference (firesize.py), demo.py and the instrument export need none of this.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src').replace(os.sep, '/')
TRAIN = os.path.dirname(SRC)
MODEL = os.path.dirname(TRAIN)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'firesize')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
FIRE = os.path.join(REPO, 'shared', 'fire').replace(os.sep, '/')
FIRE_DATA = os.path.abspath(os.environ.get('ML_RATCHET_FIRE') or FIRE).replace(os.sep, '/')
for p in (FIRE, os.path.join(FIRE, 'firesim').replace(os.sep, '/')):   # import firedata and firesim from the repo
    if p not in sys.path:
        sys.path.insert(1, p)
for d in ('data', 'models', 'results', 'web', 'release', 'logs', 'tmp', '.numba'):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)


def latest(rel):
    """A file written under WORK by a re-run if there is one, else the shipped copy in TRAIN."""
    w = os.path.join(WORK, rel)
    return w if os.path.exists(w) else os.path.join(TRAIN, rel)
