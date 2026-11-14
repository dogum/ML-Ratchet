"""Release choice (before the exam): copy the chosen rung's networks to weights/, fit the dev error-bar factors,
copy the raw ladder log next to the scripts and regenerate ratchet.yaml. python release.py <rung>"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, shutil, glob, subprocess
sys.path.insert(0, _P.SRC)
from ratchet2 import RUNGS

name = sys.argv[1]
r = {x['name']: x for x in RUNGS}[name]
W = os.path.join(_P.MODEL, 'weights')
for f in glob.glob(os.path.join(W, '*.npz')):
    os.remove(f)
for m in (r.get('members') or [name]):
    shutil.copy(os.path.join(_P.WORK, 'ckpt', m + '.npz'), os.path.join(W, m + '.npz'))
shutil.copy(os.path.join(_P.WORK, 'ratchet2.json'), os.path.join(_P.SRC, 'ratchet2.json'))
for f in glob.glob(os.path.join(_P.WORK, 'twin_read_*.json')):
    shutil.copy(f, os.path.join(_P.SRC, os.path.basename(f)))
py = sys.executable
subprocess.check_call([py, os.path.join(_P.SRC, 'sigma_dev.py'), name])
subprocess.check_call([py, os.path.join(_P.SRC, 'write_ratchet_yaml.py'), name])
print('released', name, sorted(os.listdir(W)))
