"""Build a candidate arm folder for one rung under <arm>/work/cand/<rung>/ and score it with the harness's evaluate.py
(frozen sim val + dev gate) into <arm>/work/eval/<rung>_table_*.

usage: python3 cand.py ARM_DIR RUNG --code f1 f2 ... --files SRC:DST ... [--no-eval] [--sets simval,dev]
ARM_DIR's code files (paths relative to ARM_DIR) are copied as they are now; --files copies SRC (relative to ARM_DIR or
absolute) to DST inside the candidate. The candidate gets a minimal ratchet.json / card.md (from ARM_DIR's ratchet.json
when present) so that evaluate.py can load it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('arm'); ap.add_argument('rung')
ap.add_argument('--code', nargs='*', default=['predict.py'])
ap.add_argument('--files', nargs='*', default=[])
ap.add_argument('--no-eval', action='store_true')
ap.add_argument('--link', nargs='*', default=[], help='directories of ARM_DIR symlinked into the candidate (e.g. pylib, weights)')
ap.add_argument('--sets', default='simval,dev')
ap.add_argument('--meta', default=None, help='json string merged into the candidate ratchet.json')
a = ap.parse_args()
arm = os.path.abspath(a.arm)
cd = os.path.join(arm, 'work', 'cand', a.rung)
if os.path.exists(cd):
    shutil.rmtree(cd)
os.makedirs(cd)
for f in a.code:
    os.makedirs(os.path.dirname(os.path.join(cd, f)) or cd, exist_ok=True)
    shutil.copy2(os.path.join(arm, f), os.path.join(cd, f))
for fs in a.files:
    s, d = fs.split(':')
    s = s if os.path.isabs(s) else os.path.join(arm, s)
    os.makedirs(os.path.dirname(os.path.join(cd, d)), exist_ok=True)
    shutil.copy2(s, os.path.join(cd, d))
for dname in a.link:
    os.symlink(os.path.join(arm, dname), os.path.join(cd, dname))
aid = os.path.basename(arm)
meta = {'arm_id': f'{aid}_{a.rung}_cand', 'family': 'x', 'group': 'table', 'tracks': ['table'], 'numpy_only': True,
        'compute': {'train_cpu_hours': 0}, 'feature_engineering': 'candidate', 'released': a.rung,
        'rungs': [{'id': a.rung, 'change': 'candidate', 'decision': 'released', 'why': 'candidate'}]}
if a.meta:
    meta.update(json.loads(a.meta))
json.dump(meta, open(os.path.join(cd, 'ratchet.json'), 'w'), indent=1)
open(os.path.join(cd, 'card.md'), 'w').write(f'candidate {a.rung} of {aid} (rung evaluation only; not an artefact)\n' * 2)
print('candidate', cd, flush=True)
if not a.no_eval:
    out = os.path.join(arm, 'work', 'eval')
    rc = subprocess.call([sys.executable, '/home/claude/bakeoff/vehicle/harness/evaluate.py', cd, '--track', 'table',
                          '--tag', a.rung, '--out', out, '--sets', a.sets])
    sys.exit(rc)
