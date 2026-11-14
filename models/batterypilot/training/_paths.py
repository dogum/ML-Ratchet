"""Where batterypilot's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (grid/batterypilot/, with its modules in src/). In this
repository:

    SRC    this folder (training/) — the modules the scripts import (battery, features, model, strategies, ...)
    WORK   models/batterypilot/work/ (git-ignored), or $ML_RATCHET_WORK/batterypilot/ — data/ (calibration, banks,
           exam tables, scores), ckpt/ (checkpoints), web/ (page data), ratchet.json (a fresh log)
    GRID   the shared grid platform (GridSim + GridData): shared/grid/, or $ML_RATCHET_GRID
           GRID/gridsim holds the gridsim package; GRID holds griddata and data/
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'batterypilot')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
GRID = os.path.abspath(os.environ.get('ML_RATCHET_GRID') or os.path.join(REPO, 'shared', 'grid'))
for p in (os.path.join(GRID, 'gridsim'), GRID):   # import gridsim and griddata
    if p not in sys.path:
        sys.path.insert(1, p)
if SRC not in sys.path:
    sys.path.insert(1, SRC)
