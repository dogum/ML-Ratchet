"""Where FirePilot's training scripts read and write.

The scripts were first run in a cloud sandbox (Wildfire Phase 2, 18-19 Nov 2026). The workspace was the sandbox
folder fire/firepilot/ (the fp package in fp/, scripts in training/, page sources in web/; it wrote data/, models/,
results/, logs/, tmp/, release/ and .numba/ there), next to the Wildfire Phase 1 tree fire/ (firesim/, firedata/,
checks/, data/). In this repository:

    SRC         this folder (training/): every script as run, and the fp package (training/fp/)
    MODEL       models/firepilot/ (firepilot.py, the released weights, examples/, web/)
    REPO        the repository root
    WORK        models/firepilot/work/ (git-ignored), or $ML_RATCHET_WORK/firepilot/ — the old workspace root for
                everything a script writes: data/ (banks <kind>_<physics>/), models/ (rung weights <rung>.npz),
                results/ (eval_*.jsonl, exam/, the result JSONs), logs/, tmp/, release/, web/, .numba/, ratchet.json
    FIRE        shared/fire/ — the Wildfire platform code: FireSim (shared/fire/firesim/firesim), FireData
                (shared/fire/firedata) and the physics checks (shared/fire/checks, with replay_calibration.json).
                This module puts shared/fire and shared/fire/firesim on sys.path, so `import firesim` and
                `import firedata` work.
    FIRE_ROOT   the platform data root: $ML_RATCHET_FIRE, default shared/fire (FireData reads data/ under it)
    CALIBRATION FIRE/checks/replay_calibration.json, the dev-calibrated ROS factors (physics p1)

The released results are kept as shipped in this folder: ratchet.json and results/*.json (exam_summary,
fidelity_dev_p0/p1, sigma_widen, upstream_U1_leak, page_extra), plus the 22 Nov 2026 physics p2 re-score
(exam_summary_p2, sigma_widen_p2, fidelity_dev_p2). Re-running a script writes new copies under WORK;
latest(rel) returns the WORK copy if there is one, else the shipped copy.

What runs where: inference (../firepilot.py) and the demo need only numpy and scipy; the pipeline adds pandas and
FireSim for the forecast run. The training scripts need FireSim (numba), pandas, torch (train_value.py) and the
real FireData tables (FEDS-MTBS progression, the dev and exam landscapes and HRRR weather), which are not in the
repository: `mlratchet data fire` rebuilds what can be rebuilt. None of them was re-run on import (21 Nov 2026).
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'firepilot')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
FIRE = os.path.join(REPO, 'shared', 'fire').replace(os.sep, '/')
FIRE_ROOT = os.path.abspath(os.environ.get('ML_RATCHET_FIRE') or FIRE).replace(os.sep, '/')
FIRE_DATA = FIRE_ROOT + '/data'
CALIBRATION = FIRE + '/checks/replay_calibration.json'

for p in (FIRE, os.path.join(FIRE, 'firesim').replace(os.sep, '/'), SRC):
    if p not in sys.path:
        sys.path.insert(1, p)
for sub in ('data', 'models', 'results', 'logs', 'tmp', 'release', 'web', '.numba'):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', WORK + '/.numba')


def latest(rel):
    """A result written under WORK by a re-run if there is one, else the shipped copy in training/."""
    w = os.path.join(WORK, rel)
    return w if os.path.exists(w) else os.path.join(SRC, rel)
