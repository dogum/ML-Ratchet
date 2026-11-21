"""Where the Prior Wars scripts read and write.

The scripts first ran on 18-19 Nov 2026 in a cloud sandbox: the experiment in priorwars/ (scripts/, work/, weights/,
a venv), next to the five model projects as sibling folders (grid/gridpulse, water/detective, bldg/detective,
grid/plantdetective, road/energynet). Those sandbox paths are still quoted in PREREG.md, which is kept byte for byte
because prereg_log.json hashes it. In this repository:

    HERE        experiments/priorwars/: the scripts and adapters, and the shipped record (PREREG.md,
                prereg_log.json, results.json, RESULTS.md, scores/<model>/)
    REPO        the repository root
    WORK        experiments/priorwars/work/ (git-ignored), or $ML_RATCHET_WORK/priorwars/: everything a rerun writes
                (<model>/ with tune_dev.json, tune_sim.json, pred/, exam_scores.json, exam_boot.npz, log.txt;
                prereg_log.json; logs/; results.json; page/)
    WEIGHTS     the TabPFN-v2 checkpoints: $TABPFN_WEIGHTS, default WORK/weights. Never commit them.
    PREREG      WORK/PREREG.md once a rerun has started its own pre-registration there, else the shipped one
    PREREG_LOG  the same rule for prereg_log.json

The adapters reach each released model through model_dir(), model_work() and use_training():

    model_dir(id)     models/<id>/ (released weights and inference code)
    model_work(id)    that model's own training workspace: $ML_RATCHET_WORK/<id>, else models/<id>/work. Its
                      banks, real features and checkpoints are rebuilt there by the model's own training scripts;
                      none of them is in the repository.
    use_training(id)  put models/<id>/training on sys.path, so `import common` etc. find that model's training code

This module is named pw_paths, not _paths: the model training folders each have their own _paths.py, and their
common.py imports it by that name.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
REPO = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'priorwars')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(HERE, 'work')).replace(os.sep, '/')
WEIGHTS = os.path.abspath(os.environ.get('TABPFN_WEIGHTS') or os.path.join(WORK, 'weights')).replace(os.sep, '/')


def latest(name):
    """A file written under WORK by a rerun if there is one, else the shipped copy in HERE."""
    w = os.path.join(WORK, name)
    return w if os.path.exists(w) else os.path.join(HERE, name)


PREREG = latest('PREREG.md')
PREREG_LOG = latest('prereg_log.json')


def model_dir(mid):
    return f'{REPO}/models/{mid}'


def model_work(mid):
    w = os.path.join(os.environ['ML_RATCHET_WORK'], mid) if os.environ.get('ML_RATCHET_WORK') else \
        os.path.join(REPO, 'models', mid, 'work')
    return os.path.abspath(w).replace(os.sep, '/')


def use_training(mid):
    p = f'{REPO}/models/{mid}/training'
    if p not in sys.path:
        sys.path.append(p)
    return p
