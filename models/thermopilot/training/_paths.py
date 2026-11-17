"""Where thermopilot's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (bldg/thermopilot/, next to bldg/buildingsim and
bldg/data). In this repository:

    SRC   this folder (training/): tp/ (simulator wrappers, teacher, MPC, features, policy), scripts/, configs/,
          tp.js, page_src.html, build_page.py, and the shipped logs (ratchet.json, results/exam_r7.json,
          results/wpuq/)
    WORK  models/thermopilot/work/ (git-ignored), or $ML_RATCHET_WORK/thermopilot/ — what the scripts write:
          data/cases/*.pkl (exam/dev/val cases), data/raw/nasa_yvr/ (NASA POWER download for the HUE homes),
          results/<set>/ (per-episode baselines and policies), banks, checkpoints, ratchet.json, exam_*.json,
          the rebuilt page and release builds. Re-runs write their logs here; the shipped copies in SRC are the
          record of the v1.0 build.
    BLDG  the shared building platform (BuildingSim + BuildingData): shared/building/, or $ML_RATCHET_BUILDING.
          It holds the `buildingsim` and `buildingdata` packages and data/proc (BDG2, RECS/CBECS, HUE, WPuQ,
          tariffs). It arrives with the platform zip (buildingsim-buildingdata-v1.0) and is not in the repository
          yet, so nothing in training/ runs until it is there. Inference and demo.py do not need it.
    PROC  BLDG/data/proc — BuildingData's processed tables (was the sandbox's bldg/data/proc)
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'thermopilot')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
BLDG = os.path.abspath(os.environ.get('ML_RATCHET_BUILDING') or os.path.join(REPO, 'shared', 'building'))
PROC = os.path.join(BLDG, 'data', 'proc')
BLDG_CODE = os.path.join(REPO, 'shared', 'building')   # code always from the repository; BLDG is the data root
for p in (os.path.join(BLDG_CODE, 'buildingsim'), BLDG_CODE):   # import buildingsim and buildingdata
    if p not in sys.path:
        sys.path.insert(1, p)
