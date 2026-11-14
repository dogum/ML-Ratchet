"""Where gridpulse's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (grid/gridpulse/, code in src/). In this repository:

    SRC   this folder (training/)
    WORK  models/gridpulse/work/ (git-ignored), or $ML_RATCHET_WORK/gridpulse/ — noise slices, banks, checkpoints,
          logs, figures, widen.json, exam.json, release builds
    GRID  the shared grid platform (GridSim + GridData): shared/grid/, or $ML_RATCHET_GRID
    LAB   models/gridpulse/lab/ — the original page; gp.js (the browser GridSim port and network) lives there
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'gridpulse')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
GRID = os.path.abspath(os.environ.get('ML_RATCHET_GRID') or os.path.join(REPO, 'shared', 'grid'))
LAB = os.path.join(MODEL, 'lab').replace(os.sep, '/')
for p in (os.path.join(GRID, 'gridsim'), GRID):   # import gridsim and griddata
    if p not in sys.path:
        sys.path.insert(1, p)
