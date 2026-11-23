"""Assemble release/ (and for_repo/models/airfoilnet/) from the final weights and the training scripts.
usage: python make_release.py <final weights.npz>
"""
import os, sys, json, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
REL = os.path.join(ROOT, 'release')
shutil.copy(sys.argv[1], os.path.join(REL, 'airfoilnet_weights.npz'))
TR = os.path.join(REL, 'training'); os.makedirs(TR, exist_ok=True)
for f in ['mfoil.py', 'MFOIL_PYPI_LICENSE.txt', 'mfoil_fast.py', 'xsolver.py', 'verify_fast.py', 'verify_physics.py',
          'uiuc_lsat.py', 'dataset.py', 'make_splits.py', 'shapes.py', 'make_shapes.py', 'simbank.py', 'run_bank.sh',
          'chain_A.sh', 'neuralfoil_np.py', 'baselines.py', 'realeval.py', 'twin_dev.py', 'calib_sim.py', 'train.py',
          'ratchet.py', 'calibrate.py', 'dev_baselines.py', 'exam.py', 'make_examples.py', 'make_release.py']:
    shutil.copy(os.path.join(HERE, f), os.path.join(TR, f))
for f in ['webdata.py', 'template.html', 'shot.js', 'page_text.json']:
    shutil.copy(os.path.join(ROOT, 'web', f), os.path.join(TR, 'web_' + f))
shutil.copy(os.path.join(ROOT, 'splits.json'), os.path.join(TR, 'splits.json'))
W = os.path.join(WORK)
for f in ['verify_fast.json', 'verify_physics.json', 'twin_dev.json', 'calib_sim.json', 'calibration.json', 'ratchet.json',
          'exam.json', 'dev_baselines.json', 'shapes_report.json']:
    if os.path.exists(os.path.join(W, f)):
        shutil.copy(os.path.join(W, f), os.path.join(TR, f))
for r in sorted(glob.glob(os.path.join(W, 'runs', '*', 'result.json'))):
    rid = r.split(os.sep)[-2]
    shutil.copy(r, os.path.join(TR, f'rung_{rid}_result.json'))
open(os.path.join(TR, '_paths.py'), 'w').write('''"""Where this model's training scripts read and write.

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


def work(*parts, mkdir=False):
    p = os.path.join(WORK, *parts)
    if mkdir:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
''')
print('release assembled in', REL)
