"""Where TsunamiNet's training scripts read and write.

These scripts were first run in a cloud sandbox (bb/tsunami in the sandbox home, with src/, data/, work/ and logs/ under one
folder). In the repository the code lives here in training/, and everything the scripts download or produce (DART
files, ETOPO grid, Green's functions, banks, checkpoints, exam outputs) goes to a work folder:

    models/tsunaminet/work/            by default (git-ignored)
    $ML_RATCHET_WORK/tsunaminet/       if ML_RATCHET_WORK is set

ROOT (ratchet.json, splits.json, calibration.json, exam/, release/, web/) is the work folder too.
"""
import os
SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'tsunaminet') if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
ROOT = WORK
DATA = os.path.join(WORK, 'data'); BANK = os.path.join(WORK, 'work'); LOGS = os.path.join(WORK, 'logs')
for d in (DATA, BANK, LOGS, os.path.join(WORK, 'release'), os.path.join(WORK, 'release', 'examples'), os.path.join(WORK, 'exam'),
          os.path.join(WORK, 'web'), os.path.join(DATA, 'checks'), os.path.join(BANK, 'ckpt')):   # every folder the scripts write into
    os.makedirs(d, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(WORK, 'numba_cache'))
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('NUMBA_NUM_THREADS', '1')
