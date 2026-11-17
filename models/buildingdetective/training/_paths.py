"""Where BuildingDetective's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (bldg/detective/, code in src/, the shared BuildingSim and
BuildingData packages in bldg/). In this repository:

    SRC    this folder (training/)
    MODEL  models/buildingdetective/
    REPO   the repository root
    WORK   models/buildingdetective/work/ (git-ignored), or $ML_RATCHET_WORK/buildingdetective/ - the old ROOT:
           bank/ (simulated buildings and weather), feats/, models/ (checkpoints, incumbent, final, calibration.json),
           web/, release/, tmp/, and the ratchet.json / exam.json the scripts write. The logs as shipped are kept in
           training/ (ratchet.json, exam.json, twintest.json, calibration.json); copy training/calibration.json to
           WORK/models/calibration.json before simulating (wrappers.py reads the W1/W2/W3 factors there).
    BLDG   the shared building platform: shared/building/, or $ML_RATCHET_BUILDING. It holds the buildingsim and
           buildingdata packages, checks/ (check_bdg2.py, check_comstock.py), buildingsim/tools/ (build_priors.py)
           and data/proc/ (BDG2, RECS/CBECS, HUE, ResStock/ComStock samples). It arrives with the BuildingSim /
           BuildingData platform zip (buildingsim-buildingdata-v1.0.zip), which is not in the repository yet; until
           then only the inference code and the shipped logs are usable.
    verify_js.py and buildpage.py read the JS port from WORK/web/detective.js (shipped as training/detective.js).
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'buildingdetective')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
BLDG = os.path.abspath(os.environ.get('ML_RATCHET_BUILDING') or os.path.join(REPO, 'shared', 'building'))
BLDG_CODE = os.path.join(REPO, 'shared', 'building')   # code always from the repository; BLDG is the data root
for p in (os.path.join(BLDG_CODE, 'buildingsim'), BLDG_CODE):   # import buildingsim and buildingdata (as in the sandbox)
    if p not in sys.path:
        sys.path.insert(1, p)
