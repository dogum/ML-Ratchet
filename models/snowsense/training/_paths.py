"""Where SnowSense's training scripts read and write.

The scripts were first run in a claude.ai cloud sandbox (18 Nov 2026, Watershed Phase 2), with the workspace at
water/snowsense/ in the sandbox home (code in src/, then results/, banks/, ckpt/, release/, web/) and the Watershed
platform beside it in water/ (watersim/, waterdata/, data/). Those absolute paths are gone. In this repository:

    SRC    this folder (training/)
    MODEL  models/snowsense/
    REPO   the repository root
    WORK   models/snowsense/work/ (git-ignored), or $ML_RATCHET_WORK/snowsense/ - the old workspace root. The scripts
           write results/, banks/, ckpt/, release/, web/ and ratchet.json under it, never into the model folder.
    WATER  the Watershed platform code: shared/water/ (WaterSim in shared/water/watersim/watersim, WaterData in
           shared/water/waterdata). Both folders are put on sys.path here.
    WATER_DATA  the data root: $ML_RATCHET_WATER, default shared/water. WaterData reads $ML_RATCHET_WATER/data/proc
           itself; the scripts read the snow baselines from WATER_DATA/data/proc/baselines/snow_scores.parquet.
           The SNOTEL time series (data/proc/snotel/) are not in the repository: rebuild them with
           `mlratchet data water` (or shared/water/data/scripts/fetch_snotel.py and build_snotel.py).

The shipped result JSONs (ratchet.json, final_summary.json, exam_by_rung.json, twin_r6.json, k_r6.json,
prior_p3.json, prior_p4.json) are kept here as shipped; re-running a script writes a new copy under WORK.
The scripts need torch and, for speed, numba (the .venv-train environment has both). Inference (snowsense.py) and the
demo need none of this.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'snowsense')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
WATER = os.path.join(REPO, 'shared', 'water').replace(os.sep, '/')
WATER_DATA = os.path.abspath(os.environ.get('ML_RATCHET_WATER') or WATER).replace(os.sep, '/')
SNOW_SCORES = f'{WATER_DATA}/data/proc/baselines/snow_scores.parquet'
for p in (os.path.join(WATER, 'watersim'), WATER):   # import watersim and waterdata
    if p not in sys.path:
        sys.path.insert(1, p)
for d in ('results', 'banks', 'ckpt', 'release', 'release/examples', 'web', '.numba'):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
