"""Where demandcast's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (grid/demandcast/, code in src/, data in data/). In this
repository:

    SRC    this folder (training/)
    WORK   models/demandcast/work/ (git-ignored), or $ML_RATCHET_WORK/demandcast/ — weather bundles (data/wx), fleets,
           priors, calibration, noise, banks, checkpoints, exam results, web builds, release builds
    GRID   the shared grid platform (GridSim + GridData): shared/grid/, or $ML_RATCHET_GRID
           GRID/gridsim holds the gridsim package; GRID holds griddata and data/
    CALIB  models/demandcast/calib/ — the released dev calibration (demand_dev.json, renew_dev.json)
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'demandcast')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
GRID = os.path.abspath(os.environ.get('ML_RATCHET_GRID') or os.path.join(REPO, 'shared', 'grid'))
CALIB = os.path.join(MODEL, 'calib')
for p in (os.path.join(GRID, 'gridsim'), GRID):   # import gridsim and griddata
    if p not in sys.path:
        sys.path.insert(1, p)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
