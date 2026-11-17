"""Where faultfinder's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (bldg/faultfinder/, platform in bldg/). In this repository:

    SRC    this folder (training/); configs/ holds the rung configs and ladder.json
    MODEL  models/faultfinder/ — faultfinder.py (the numpy reader the export check and webdata.py import)
    WORK   models/faultfinder/work/ (git-ignored), or $ML_RATCHET_WORK/faultfinder/ — the old workspace root (ROOT in
           common.py): data/ (noise_v1.npz, banks/, frozen/), models/ (checkpoints, scores), exam/, twin/, web/,
           release/, ratchet.json
    BLDG   the shared building platform: shared/building/, or $ML_RATCHET_BUILDING. It holds the BuildingSim
           (buildingsim/) and BuildingData (buildingdata/, data/, data/proc/baselines/) packages. It is not in the
           repository yet: it arrives with the platform zip (buildingsim-buildingdata-v1.0), so the training scripts
           cannot run until it does. Inference (faultfinder.py), the demo and the instrument exporter do not need it.
    LAB    models/faultfinder/lab/ — the original page (training/ffnet.js is its network port)

The released ratchet log and exam results are kept as shipped in this folder (ratchet.json, exam.json); re-running
the scripts writes new copies under WORK.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'faultfinder')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
BLDG = os.path.abspath(os.environ.get('ML_RATCHET_BUILDING') or os.path.join(REPO, 'shared', 'building'))
LAB = os.path.join(MODEL, 'lab').replace(os.sep, '/')
BLDG_CODE = os.path.join(REPO, 'shared', 'building')   # code always from the repository; BLDG is the data root
for p in (os.path.join(BLDG_CODE, 'buildingsim'), BLDG_CODE):   # import buildingsim and buildingdata
    if p not in sys.path:
        sys.path.insert(1, p)
