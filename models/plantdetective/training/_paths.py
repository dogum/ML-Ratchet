"""Where plantdetective's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (grid/plantdetective/). In this repository:

    SRC      this folder (training/)
    MODEL    models/plantdetective/ — the release (plantdetective.py, plantdetective_weights.npz, examples/)
    WORK     models/plantdetective/work/ (git-ignored), or $ML_RATCHET_WORK/plantdetective/ — banks, sets, networks,
             cached real units (WORK/data), grade files and tables (WORK/out), page builds (WORK/web)
    GRID     the shared grid platform (GridSim + GridData): shared/grid/, or $ML_RATCHET_GRID
             GRID/gridsim holds the package gridsim, GRID holds the package griddata, GRID/data the processed data
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'plantdetective')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
GRID = os.path.abspath(os.environ.get('ML_RATCHET_GRID') or os.path.join(REPO, 'shared', 'grid'))
for p in (os.path.join(GRID, 'gridsim'), GRID):   # import gridsim and griddata
    if p not in sys.path:
        sys.path.insert(1, p)
