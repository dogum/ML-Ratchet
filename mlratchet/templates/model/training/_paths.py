"""Where this model's training scripts read and write.

Code lives here in training/. Everything the scripts download or produce (data,
simulation banks, checkpoints, figures, release builds) goes in a work folder:

    models/__ID__/work/            by default (git-ignored)
    $ML_RATCHET_WORK/__ID__/       if ML_RATCHET_WORK is set (e.g. a bigger disk)

SRC is this folder, LAB is the page sources (models/__ID__/lab/src), and
OUT is where release builds are written (WORK/dist).
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
LAB = os.path.join(MODEL, 'lab', 'src')
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], '__ID__')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work'))
OUT = os.path.join(WORK, 'dist')


def work(*parts, mkdir=False):
    """Path inside the work folder; ``mkdir=True`` creates its parent folder."""
    p = os.path.join(WORK, *parts)
    if mkdir:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
