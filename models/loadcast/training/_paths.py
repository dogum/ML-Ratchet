"""Where loadcast's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (bldg/loadcast/, code in src/). In this repository:

    SRC    this folder (training/)
    MODEL  models/loadcast/
    WORK   models/loadcast/work/ (git-ignored), or $ML_RATCHET_WORK/loadcast/ — the old workspace root: data/ (dev
           knobs, noise bank, banks, widen and dev-eval files), ckpt/, exam/, web/, release/, tmp/, ratchet.json
    BLDG   the shared building platform: shared/building/, or $ML_RATCHET_BUILDING. It holds the BuildingSim
           (buildingsim/) and BuildingData (buildingdata/, data/, data/scripts/) packages. It is not in the
           repository yet: it arrives with the platform zip (buildingsim-buildingdata-v1.0), so the training scripts
           cannot run until it does. Inference (loadcast.py) and the demo do not need it.
    LAB    models/loadcast/lab/ — the original page (its inline script is the browser port of the network)

The released ratchet log, exam, twin and widening JSONs are kept as shipped in this folder (ratchet.json,
exam_r3.json, twin_r3.json, widen_r3.json, calib_summary.json); re-running the scripts writes new copies under WORK.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'loadcast')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
BLDG = os.path.abspath(os.environ.get('ML_RATCHET_BUILDING') or os.path.join(REPO, 'shared', 'building'))
LAB = os.path.join(MODEL, 'lab').replace(os.sep, '/')
BLDG_CODE = os.path.join(REPO, 'shared', 'building')   # code always from the repository; BLDG is the data root
for p in (os.path.join(BLDG_CODE, 'buildingsim'), BLDG_CODE):   # import buildingsim and buildingdata
    if p not in sys.path:
        sys.path.insert(1, p)
# A frozen copy of the BuildingSim package (the folder that contains buildingsim/) can be put first with
# $LC_BUILDINGSIM, e.g. while the shared package is being edited (added 17 Nov 2026; data still come from BLDG).
if os.environ.get('LC_BUILDINGSIM') and os.environ['LC_BUILDINGSIM'] not in sys.path:
    sys.path.insert(0, os.environ['LC_BUILDINGSIM'])
