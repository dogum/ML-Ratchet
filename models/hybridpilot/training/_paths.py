"""Where hybridpilot's training scripts read and write.

The scripts were first run in a cloud sandbox workspace (road/controller/). In this repository:

    SRC      this folder (training/)
    WORK     models/hybridpilot/work/ (git-ignored), or $ML_RATCHET_WORK/hybridpilot/ — banks, checkpoints, data
    VEHICLE  the shared vehicle platform (RoadSim + RoadData): shared/vehicle/, or $ML_RATCHET_VEHICLE
    OUT      WORK/dist — release builds
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'hybridpilot')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
VEHICLE = os.path.abspath(os.environ.get('ML_RATCHET_VEHICLE') or os.path.join(REPO, 'shared', 'vehicle'))
OUT = os.path.join(WORK, 'dist')
for p in (os.path.join(VEHICLE, 'roadsim'), VEHICLE):   # import roadsim and realdata
    if p not in sys.path:
        sys.path.insert(1, p)
