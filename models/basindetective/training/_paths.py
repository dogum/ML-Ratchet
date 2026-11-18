"""Where BasinDetective's training scripts read and write.

The scripts were first run on 18 Nov 2026 in a cloud sandbox workspace (water/detective/, code in src/, next to the
Watershed platform in water/). Nothing was re-run on import. In this repository:

    SRC         this folder (training/): the scripts, the shipped logs and the shipped dev-calibration knobs
    MODEL       models/basindetective/ (the released package basindetective/, weights/, examples/)
    REPO        the repository root
    WORK        models/basindetective/work/ (git-ignored), or $ML_RATCHET_WORK/basindetective/: the old workspace
                root. bank/<phys>/ (simulated banks), feats/ (real dev/exam features, twins), models/ (runs, ensembles,
                knob JSONs), logs/, web/, release/, ratchet.json, exam.json
    WATER_CODE  the shared water platform code, always from the repository: shared/water/ (watersim/, waterdata/,
                checks/ with dev_calibrated_params.json)
    WATER       the water data root: $ML_RATCHET_WATER, default shared/water (data/proc/: attributes, splits,
                baselines; the real time series under data/proc/caravan, nwis, ... arrive with `mlratchet data water`
                and are not in the repository)

The released ratchet log, exam, twin, sigma and calibration JSONs are kept as shipped in this folder (ratchet.json,
exam.json, twin_final.json, calib_*.json, ../weights/basindetective_v1_sigma.json). Re-running a script writes a new
copy under WORK. The four dev-calibrated knob files the bank generator reads (gauge_v2.json, pbias_v2.json,
drift_v3.json, annual_v4.json) are copied from here into WORK/models/ on first use (common.py), unless a re-run of the
calibration scripts has already written them there.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'basindetective')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
WATER_CODE = os.path.join(REPO, 'shared', 'water').replace(os.sep, '/')
WATER = os.path.abspath(os.environ.get('ML_RATCHET_WATER') or WATER_CODE).replace(os.sep, '/')
WATER_PROC = f'{WATER}/data/proc'
DEV_PARAMS = f'{WATER_CODE}/checks/dev_calibrated_params.json'   # five dev-calibrated model setups per dev basin

for p in (WATER_CODE, f'{WATER_CODE}/watersim'):   # import watersim and waterdata
    if p not in sys.path:
        sys.path.insert(1, p)
