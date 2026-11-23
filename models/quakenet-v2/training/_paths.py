"""Where QuakeNet v2's training scripts read and write.

The scripts were first run in a cloud sandbox with everything under one folder (bb/quake2/ in the sandbox home: src/, data/,
bank/, ckpt/, release/, web/) and QuakeNet v1.2's work folder at quake/data/ (Green's functions, noise
windows, real records, catalogue). In this repository:

    H  = models/quakenet-v2/work/        (or $ML_RATCHET_WORK/quakenet-v2/), with data/, bank/, ckpt/, release/, web/ inside
    Q  = models/quakenet/work/data/      (or $ML_RATCHET_WORK/quakenet/data/): QuakeNet v1.2's data, rebuilt by its own scripts

Both end with a slash, as the scripts expect. This folder and training/v12/ (QuakeNet v1.2's physics modules) go on sys.path.
"""
import os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
_W = os.environ.get('ML_RATCHET_WORK')
H = os.path.join(os.path.abspath(os.path.join(_W, 'quakenet-v2') if _W else os.path.join(MODEL, 'work')), '')
Q = os.path.join(os.path.abspath(os.path.join(_W, 'quakenet', 'data') if _W else os.path.join(MODEL, '..', 'quakenet', 'work', 'data')), '')
for p in (SRC, os.path.join(SRC, 'v12')):
    if p not in sys.path: sys.path.insert(0, p)
for d in ('data', 'bank', 'ckpt', 'release', os.path.join('release', 'examples'), 'web', 'tmp'):   # what the scripts write into
    os.makedirs(os.path.join(H, d), exist_ok=True)
