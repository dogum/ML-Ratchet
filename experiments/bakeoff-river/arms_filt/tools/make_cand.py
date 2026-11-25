"""Make an evaluation-only candidate folder for an EnKF rung: enkf_watersim/work/cand_<rung>/ (copies of the arm code
+ the rung's hyper.json), then the harness evaluate.py scores it (sim val 800 runs + dev gate, gauged leads 1/3/7).
usage: python3 make_cand.py <rung> '<json hyper overrides on top of tools/hyper_base.json>' [--eval]"""
import json, os, shutil, subprocess, sys
ARM = '/home/claude/bakeoff/river/arms_filt/enkf_watersim'
rung, over = sys.argv[1], json.loads(sys.argv[2])
H = json.load(open(os.path.join(ARM, 'hyper.json')))
H.update(over); H['rung'] = rung
d = os.path.join(ARM, 'work', f'cand_{rung}')
os.makedirs(d, exist_ok=True)
src = sys.argv[sys.argv.index('--src') + 1] if '--src' in sys.argv else ARM
for f in ('predict.py', 'enkf_core.py'):
    shutil.copy(os.path.join(src, f), d)
json.dump(H, open(os.path.join(d, 'hyper.json'), 'w'), indent=1)
json.dump({'arm_id': 'enkf_watersim', 'family': 'assim', 'group': 'filt', 'protocols': ['gauged'], 'numpy_only': False,
           'numpy_only_reason': 'candidate', 'compute': {'train_cpu_hours': 0}, 'feature_engineering': '-',
           'rungs': [{'id': rung, 'change': '-', 'decision': 'released', 'why': 'candidate'}], 'released': rung},
          open(os.path.join(d, 'ratchet.json'), 'w'))
open(os.path.join(d, 'card.md'), 'w').write('# candidate folder for evaluation only (not an arm)\n' * 3)
print(d, json.dumps(H))
if '--eval' in sys.argv:
    subprocess.run(['nice', '-n', '5', sys.executable, '/home/claude/bakeoff/river/harness/evaluate.py', d, '--tag', rung,
                    '--out', os.path.join(ARM, 'work', 'eval')], check=True)
