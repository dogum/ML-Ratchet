"""Where this model's training scripts read and write.

Code lives here in training/. Everything the scripts download or produce (UIUC data, simulation banks,
checkpoints, page builds) goes in a work folder:

    models/airfoilnet/work/          by default (git-ignored)
    $ML_RATCHET_WORK/airfoilnet/     if ML_RATCHET_WORK is set

The scripts were run from a flat build folder (src/, data/, work/, release/, web/); see training/README.md.
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'airfoilnet')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
os.makedirs(WORK, exist_ok=True)   # a fresh work folder (make_shapes.py is the first step that writes here)


def work(*parts, mkdir=False):
    p = os.path.join(WORK, *parts)
    if mkdir:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
