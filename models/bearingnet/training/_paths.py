"""Where this model's training scripts read and write.

These scripts were first run in a cloud sandbox, with code, data, banks and
checkpoints all under one workspace folder. In this repository the code lives
here in training/, and everything the scripts download or produce (data,
simulation banks, checkpoints, figures, release builds) goes in a work folder:

    models/bearingnet/work/            by default (git-ignored)
    $ML_RATCHET_WORK/bearingnet/       if ML_RATCHET_WORK is set (e.g. a bigger disk)

SRC is this folder, LAB is the page sources (models/bearingnet/lab/src), and
OUT is where release builds are written (WORK/dist).
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
LAB = os.path.join(MODEL, 'lab', 'src')
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'bearingnet')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
OUT = os.path.join(WORK, 'dist')
