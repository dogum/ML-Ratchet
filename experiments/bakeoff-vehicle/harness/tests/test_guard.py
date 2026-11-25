"""Harness guard tests (no exam data is read; nothing is left behind). usage: python3 tests/test_guard.py
Writes results/test_guard.json. Each test prints PASS or FAIL."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, H)
import config as C  # noqa: E402
import bake  # noqa: E402
import exam  # noqa: E402
import register as REG  # noqa: E402
import build_real  # noqa: E402

PY = sys.executable
results = {}


def run(args, env=None):
    p = subprocess.run([PY] + args, cwd=H, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def check(name, cond, detail=''):
    results[name] = {'pass': bool(cond), 'detail': detail[-600:]}
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '\n' + detail[-1500:]), flush=True)


def raises(f, exc=PermissionError):
    try:
        f()
    except exc as e:
        return True, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f'{type(e).__name__}: {e}'
    return False, 'no exception'


assert not os.path.exists(bake.LOCK) and not os.path.exists(bake.DONE), 'run these tests only before the exam'
assert not os.path.exists(os.path.join(C.REAL, 'exam')), 'exam records must not exist before the exam'

# 1. the data guard
for nm, f in [('exam_batches table', lambda: next(bake.exam_batches('table'))),
              ('exam_batches trace', lambda: next(bake.exam_batches('trace'))),
              ('exam_frame', lambda: bake.exam_frame()),
              ('real_frame exam', lambda: bake.real_frame('exam')),
              ('real_batches exam', lambda: next(bake.real_batches('exam', 'table'))),
              ('_load_real exam', lambda: bake._load_real('exam', 'ved')),
              ('real_trace exam', lambda: bake.real_trace('exam', 'ved', 0)),
              ('build_real.build exam', lambda: build_real.build('exam'))]:
    ok, msg = raises(f)
    check(f'guard: {nm} refused without the exam lock', ok, msg)
rc, out = run(['build_real.py', 'exam'])
check('build_real.py exam refused from the command line', rc != 0 and 'only dev' in out, out)
os.environ['BAKEOFF_EXAM_RUN'] = 'fake-run'
ok, msg = raises(lambda: bake.exam_frame())
check('guard: env var alone (no lock file) is refused', ok, msg)
os.environ.pop('BAKEOFF_EXAM_RUN')
rc, out = run(['-c', 'import bake; bake.exam_frame()'], env=dict(os.environ, BAKEOFF_EXAM_RUN='x'))
check('guard: a separate process with a fake run id is refused', rc != 0 and 'PermissionError' in out, out)
check('no exam records were built by the tests', not os.path.exists(os.path.join(C.REAL, 'exam')))

tmpd = tempfile.mkdtemp(dir=os.path.join(H, 'tmp'))
real_lock, real_done = bake.LOCK, bake.DONE
bake.LOCK = os.path.join(tmpd, 'EXAM_RUNNING.json')
bake.DONE = os.path.join(tmpd, 'EXAM_DONE.json')
json.dump({'run_id': 'run-A'}, open(bake.LOCK, 'w'))
ok, msg = raises(lambda: bake._unlock('run-B'))
check('guard: a lock naming a different run id is refused', ok, msg)
ok, msg = raises(lambda: exam.drive('exam'), exam.Refused)
check('exam: --run refused while a run is in progress (lock present)', ok and 'in progress' in msg, msg)
os.remove(bake.LOCK)
json.dump({'run_id': 'run-A'}, open(bake.DONE, 'w'))
ok, msg = raises(lambda: exam.preflight(None, 'exam'), exam.Refused)
check('exam: refused once EXAM_DONE exists (runs once)', ok and 'once' in msg, msg)
ok = REG.exam_started()
check('register: refused once the exam has started or finished', ok and REG.register(os.path.join(H, 'arms_ref', 'climatology')) == 2)
bake.LOCK, bake.DONE = real_lock, real_done
shutil.rmtree(tmpd)

# 2. preflight passes with the registered arms
rc, out = run(['exam.py', '--check'])
check('exam --check passes with the registered arms', rc == 0 and 'CHECK PASSED' in out, out)

# 3. an unregistered arm folder makes the exam refuse; naming an unregistered arm is refused
tdir = os.path.join(H, 'arms_ref', '_guardtest_unregistered')
shutil.copytree(os.path.join(H, 'arms_ref', 'climatology'), tdir, ignore=shutil.ignore_patterns('work'))
meta = json.load(open(os.path.join(tdir, 'ratchet.json')))
meta['arm_id'] = 'guardtest_unregistered'
json.dump(meta, open(os.path.join(tdir, 'ratchet.json'), 'w'))
rc, out = run(['exam.py', '--check'])
check('exam --check refuses an unregistered arm folder', rc == 3 and 'unregistered arm folder' in out, out)
rc, out = run(['exam.py', '--check', '--arm', 'guardtest_unregistered'])
check('exam --check --arm <unregistered> is refused', rc == 3 and 'not registered' in out, out)
shutil.rmtree(tdir)
adir = os.path.join(C.BAKEOFF, 'arms_table', '_guardtest_unregistered_arm')
shutil.copytree(os.path.join(H, 'arms_ref', 'climatology'), adir, ignore=shutil.ignore_patterns('work'))
rc, out = run(['exam.py', '--check'])
check('exam --check refuses an unregistered folder under arms_table/', rc == 3 and 'unregistered arm folder' in out, out)
shutil.rmtree(adir)

# 4. a registered arm whose artefact changed is refused
card = os.path.join(H, 'arms_ref', 'climatology', 'card.md')
orig = open(card, 'rb').read()
st = os.stat(card)
try:
    with open(card, 'ab') as f:
        f.write(b'\nedited after registration\n')
    rc, out = run(['exam.py', '--check'])
    check('exam --check refuses a registered arm whose artefact changed', rc == 3 and 'artefact changed' in out, out)
    rc, out = run(['register.py', '--verify', 'climatology'])
    check('register --verify reports the change', rc == 1 and 'card.md' in out, out)
    rc, out = run(['register.py', 'arms_ref/climatology'])
    check('register.py refuses a silent re-registration (needs --replace)', rc == 2 and '--replace' in out, out)
finally:
    open(card, 'wb').write(orig)
    os.utime(card, (st.st_atime, st.st_mtime))
extra = os.path.join(H, 'arms_ref', 'climatology', 'extra_weights.npz')
try:
    open(extra, 'wb').write(b'x')
    rc, out = run(['exam.py', '--check'])
    check('exam --check refuses an arm with an added file', rc == 3 and 'extra_weights.npz' in out, out)
finally:
    os.remove(extra)
rc, out = run(['exam.py', '--check'])
check('exam --check passes again after restoring the bytes', rc == 0, out)

# 5. the worker refuses an unregistered arm
ok, msg = raises(lambda: exam.run_worker('no_such_arm', 'rehearsal', 'x'), (exam.Refused, KeyError))
check('exam worker refuses an unregistered arm id', ok, msg)

# 6. the file audit catches an arm that reads a file outside its hashed artefact
tdir = os.path.join(H, 'arms_ref', '_guardtest_audit')
shutil.copytree(os.path.join(H, 'arms_ref', 'climatology'), tdir, ignore=shutil.ignore_patterns('work'))
meta = json.load(open(os.path.join(tdir, 'ratchet.json')))
meta['arm_id'] = 'guardtest_audit'
json.dump(meta, open(os.path.join(tdir, 'ratchet.json'), 'w'))
os.remove(os.path.join(tdir, 'clim.json'))
outside = os.path.join(H, 'arms_ref', 'climatology', 'clim.json')
src = open(os.path.join(tdir, 'predict.py')).read().replace("os.path.join(HERE, 'clim.json')", repr(outside))
open(os.path.join(tdir, 'predict.py'), 'w').write(src)
rec, problems, warns = REG.check_arm(tdir)
check('register: audit refuses an arm that reads a file outside its artefact',
      any('not in the hashed artefact' in p for p in problems), str(problems))
shutil.rmtree(tdir)

# 7. contract checks: bad outputs, real labels in training, a batch-dependent arm
tdir = os.path.join(H, 'arms_ref', '_guardtest_bad')
os.makedirs(tdir)
meta = {'arm_id': 'guardtest_bad', 'family': 'test', 'group': 'test', 'tracks': ['table'], 'numpy_only': True,
        'compute': {'train_cpu_hours': 0}, 'feature_engineering': 'none', 'released': 'r0',
        'rungs': [{'id': 'r0', 'change': 'x', 'decision': 'released', 'why': 'test'}]}
json.dump(meta, open(os.path.join(tdir, 'ratchet.json'), 'w'))
open(os.path.join(tdir, 'card.md'), 'w').write('test arm that returns negative energy and depends on its batch ' * 2)
open(os.path.join(tdir, 'predict.py'), 'w').write(
    'import numpy as np\n'
    'def predict(b):\n'
    '    e = np.asarray(b["E_phys"], float)\n'
    '    return {k: e - 1e9 for k in ("pred", "q05", "q25", "q50", "q75", "q95")}\n')
rec, problems, warns = REG.check_arm(tdir)
check('register: refuses an arm whose outputs are not > 0', any('<= 0' in p for p in problems), str(problems))
open(os.path.join(tdir, 'predict.py'), 'w').write(
    'import numpy as np\n'
    'def predict(b):\n'
    '    e = np.asarray(b["E_phys"], float) * (1 + 0.01 * np.arange(b["n"]))\n'
    '    return {k: e for k in ("pred", "q05", "q25", "q50", "q75", "q95")}\n')
rec, problems, warns = REG.check_arm(tdir)
check('register: warns about an arm whose outputs depend on the batch', any('depend on the batch' in w for w in warns), str(warns))
meta['real_labels_in_training'] = True
json.dump(meta, open(os.path.join(tdir, 'ratchet.json'), 'w'))
rec, problems, warns = REG.check_arm(tdir)
check('register: refuses an arm that declares real labels in training', any('real_labels_in_training' in p for p in problems), str(problems))
shutil.rmtree(tdir)

json.dump(results, open(os.path.join(C.RESULTS, 'test_guard.json'), 'w'), indent=1)
n = sum(v['pass'] for v in results.values())
print(f'{n}/{len(results)} passed')
sys.exit(0 if n == len(results) else 1)
