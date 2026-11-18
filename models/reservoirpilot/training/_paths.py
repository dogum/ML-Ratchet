"""Where reservoirpilot's training scripts read and write.

The scripts were first run in a cloud sandbox (Watershed Phase 2, 18 Nov 2026): the workspace was the sandbox
home folder water/reservoirpilot/ (code in src/, the page in web/), next to the Watershed repo water/
(watersim/, waterdata/, data/). In this repository:

    SRC         this folder (training/): every script as run, plus the page's engine.js and template.html
    MODEL       models/reservoirpilot/
    REPO        the repository root
    WORK        models/reservoirpilot/work/ (git-ignored), or $ML_RATCHET_WORK/reservoirpilot/ — the old workspace
                root: data/ (banks bank_g*.npz, frozen sets val_g*.pkl, selection sets sel_g*.pkl, real_dev.pkl,
                real_exam.pkl, dam_stats.parquet, bc_g*.npz), runs/<rung>/weights.npz, exam/, web/, tmp/, logs/,
                ratchet.json, .numba/. Rung weight paths in ratchet.json ('runs/...') are relative to WORK, so run
                the scripts from WORK (run_bank.sh and run_train.sh cd there).
    WATER       shared/water/ — the Watershed platform code (WaterSim in watersim/watersim, WaterData in waterdata/,
                the baseline scripts in data/scripts/). Always taken from the repository; this module puts
                shared/water, shared/water/watersim and shared/water/data/scripts on sys.path.
    WATER_ROOT  the platform data root: $ML_RATCHET_WATER, default shared/water (as WaterData reads it)
    WATER_DATA  WATER_ROOT/data (proc/reservoirs, proc/baselines, ...)

The released results are kept as shipped in this folder: ratchet.json, exam_results.json, checks_equiv.json,
check_js.json and twin_r1_dpo_hd_g1_g1.json. Re-running the scripts writes new copies under WORK.

What runs where: inference (../reservoirpilot.py), the demo and the pipeline need only numpy and pandas. The training
scripts need torch, numba (WaterSim) and the real ResOpsUS / CDEC records under WATER_DATA/proc/reservoirs, which are
not in the repository (`mlratchet data water` rebuilds them). None of them was re-run on import.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'reservoirpilot')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
WATER = os.path.join(REPO, 'shared', 'water').replace(os.sep, '/')
WATER_ROOT = os.path.abspath(os.environ.get('ML_RATCHET_WATER') or WATER).replace(os.sep, '/')
WATER_DATA = WATER_ROOT + '/data'

for p in (os.path.join(WATER, 'data', 'scripts'), os.path.join(WATER, 'watersim'), WATER):
    if p not in sys.path:
        sys.path.insert(1, p)
for sub in ('data', 'runs', 'exam', 'web', 'tmp', 'logs'):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)
