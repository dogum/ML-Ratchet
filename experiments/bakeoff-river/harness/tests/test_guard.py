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
ex = bake._exam_ids()[0]

# 1. the data guard
for nm, f in [('exam_basins', lambda: bake.exam_basins()),
              ('exam_batch', lambda: bake.exam_batch(ex, 'main')),
              ('exam_obs', lambda: bake.exam_obs(ex, 'recent')),
              ('exam_batches', lambda: next(bake.exam_batches('main'))),
              ('real_batch exam basin dev window', lambda: bake.real_batch(ex, 'dev')),
              ('real_batch dev basin main window', lambda: bake.real_batch(bake.dev_basins()[0], 'main')),
              ('dev_batch exam basin', lambda: bake.dev_batch(ex)),
              ('dev_batches exam basin', lambda: next(bake.dev_batches(basins=[ex]))),
              ('dev_obs exam basin', lambda: bake.dev_obs(ex))]:
    ok, msg = raises(f)
    check(f'guard: {nm} refused without the exam lock', ok, msg)
os.environ['BAKEOFF_EXAM_RUN'] = 'fake-run'
ok, msg = raises(lambda: bake.exam_basins())
check('guard: env var alone (no lock file) is refused', ok, msg)
os.environ.pop('BAKEOFF_EXAM_RUN')
rc, out = run(['-c', 'import bake; bake.exam_obs("%s", "main")' % ex], env=dict(os.environ, BAKEOFF_EXAM_RUN='x'))
check('guard: a separate process with a fake run id is refused', rc != 0 and 'PermissionError' in out, out)

# a lock that names another run id is refused (temp lock path, so the real results/ is untouched)
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
rc, out = run(['exam.py', '--rehearsal', '--arm', 'guardtest_unregistered'])
check('exam --rehearsal --arm <unregistered> is refused', rc == 3, out)
shutil.rmtree(tdir)

# 4. a registered arm whose artefact changed is refused (by exam --check, register.py --verify, register.py)
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
    # a new weights file is a change too
    extra = os.path.join(H, 'arms_ref', 'climatology', 'extra_weights.npz')
finally:
    open(card, 'wb').write(orig)
    os.utime(card, (st.st_atime, st.st_mtime))
try:
    open(extra, 'wb').write(b'x')
    rc, out = run(['exam.py', '--check'])
    check('exam --check refuses an arm with an added file', rc == 3 and 'extra_weights.npz' in out, out)
finally:
    os.remove(extra)
rc, out = run(['exam.py', '--check'])
check('exam --check passes again after restoring the bytes', rc == 0, out)

# 5. the worker refuses an unregistered arm even if preflight was skipped
ok, msg = raises(lambda: exam.run_worker('no_such_arm', 'rehearsal', 'x'), exam.Refused)
check('exam worker refuses an unregistered arm id', ok, msg)

# 6. the file audit catches an arm that reads a file outside its hashed artefact
tdir = os.path.join(H, 'arms_ref', '_guardtest_audit')
shutil.copytree(os.path.join(H, 'arms_ref', 'climatology'), tdir, ignore=shutil.ignore_patterns('work'))
meta = json.load(open(os.path.join(tdir, 'ratchet.json')))
meta['arm_id'] = 'guardtest_audit'
json.dump(meta, open(os.path.join(tdir, 'ratchet.json'), 'w'))
outside = os.path.join(H, 'arms_ref', 'climatology', 'clim_params.json')
src = open(os.path.join(tdir, 'predict.py')).read().replace(
    "os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clim_params.json')", repr(outside))
open(os.path.join(tdir, 'predict.py'), 'w').write(src)
rec, problems, warns = REG.check_arm(tdir)
check('register: audit refuses an arm that reads a file outside its artefact',
      any('not in the hashed artefact' in p for p in problems), str(problems))
shutil.rmtree(tdir)

# 7. the causality check catches a gauged arm that peeks at future flow
tdir = os.path.join(H, 'arms_ref', '_guardtest_peek')
os.makedirs(tdir)
meta = {'arm_id': 'guardtest_peek', 'family': 'test', 'group': 'test', 'protocols': ['gauged'], 'numpy_only': True,
        'compute': {'train_cpu_hours': 0}, 'feature_engineering': 'none', 'released': 'r0',
        'rungs': [{'id': 'r0', 'change': 'x', 'decision': 'released', 'why': 'test'}]}
json.dump(meta, open(os.path.join(tdir, 'ratchet.json'), 'w'))
open(os.path.join(tdir, 'card.md'), 'w').write('test arm that cheats: copies the observed flow of the same day ' * 2)
open(os.path.join(tdir, 'predict.py'), 'w').write(
    'import numpy as np\n'
    'def predict(b):\n'
    '    q = np.nan_to_num(np.asarray(b["qobs"], float))[b["n_warm"]:b["n_warm"] + b["n_target"]]\n'
    '    return {k: q for k in ("mean", "q05", "q25", "q50", "q75", "q95")}\n')
rec, problems, warns = REG.check_arm(tdir)
check('register: causality test refuses a gauged arm that reads same-day observed flow',
      any('may not see yet' in p for p in problems), str(problems))
shutil.rmtree(tdir)

json.dump(results, open(os.path.join(C.RESULTS, 'test_guard.json'), 'w'), indent=1)
n = sum(v['pass'] for v in results.values())
print(f'{n}/{len(results)} passed')
sys.exit(0 if n == len(results) else 1)
