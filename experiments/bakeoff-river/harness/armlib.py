"""Loading and running an arm (one arm per process: arms may have sibling modules with the same names).

An arm is a folder with predict.py (predict(batch) -> dict; optional predict_many(list) -> list), card.md and
ratchet.json. See HARNESS.md, "The arm contract"."""
import hashlib
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402

IGNORE_DIRS = {'work', '__pycache__', '.numba', 'logs', '.ipynb_checkpoints', '.venv', 'venv', '.pytest_cache'}
IGNORE_SUFFIX = ('.log', '.pyc', '.swp', '.tmp')
REQUIRED = ['arm_id', 'family', 'group', 'protocols', 'numpy_only', 'compute', 'feature_engineering', 'rungs',
            'released']


def read_meta(arm_dir):
    with open(os.path.join(arm_dir, 'ratchet.json')) as f:
        return json.load(f)


def validate_meta(meta, arm_dir):
    errs = []
    for k in REQUIRED:
        if k not in meta:
            errs.append(f'ratchet.json: missing "{k}"')
    if errs:
        return errs
    if not set(meta['protocols']) <= {'ungauged', 'gauged'} or not meta['protocols']:
        errs.append('protocols must be a non-empty subset of ["ungauged", "gauged"]')
    if not isinstance(meta['numpy_only'], bool):
        errs.append('numpy_only must be true/false')
    if meta['numpy_only'] is False and not meta.get('numpy_only_reason'):
        errs.append('numpy_only is false: give numpy_only_reason')
    if not isinstance(meta['compute'], dict) or 'train_cpu_hours' not in meta['compute']:
        errs.append('compute.train_cpu_hours is required (single-thread CPU-hours of training, all rungs)')
    rel = [r for r in meta['rungs'] if r.get('decision') == 'released']
    if len(rel) != 1 or rel[0].get('id') != meta['released']:
        errs.append('exactly one rung must have decision "released" and its id must equal "released"')
    for r in meta['rungs']:
        for k in ('id', 'change', 'decision', 'why'):
            if k not in r:
                errs.append(f'rung {r.get("id")}: missing "{k}"')
        if r.get('decision') not in ('base', 'kept', 'rejected', 'released'):
            errs.append(f'rung {r.get("id")}: decision must be base/kept/rejected/released')
    if not os.path.exists(os.path.join(arm_dir, 'card.md')) or os.path.getsize(os.path.join(arm_dir, 'card.md')) < 50:
        errs.append('card.md missing or empty')
    if not os.path.exists(os.path.join(arm_dir, 'predict.py')):
        errs.append('predict.py missing')
    return errs


def artefact_files(arm_dir, meta=None):
    """Every file that makes up the frozen artefact: the arm folder (minus work/, caches, logs) + meta['extra_files']."""
    arm_dir = os.path.abspath(arm_dir)
    files = []
    for root, dirs, fs in os.walk(arm_dir):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        for f in sorted(fs):
            if f.endswith(IGNORE_SUFFIX):
                continue
            files.append(os.path.join(root, f))
    for x in (meta or {}).get('extra_files', []):
        p = os.path.abspath(os.path.join(arm_dir, x))
        if os.path.isdir(p):
            for root, dirs, fs in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
                files += [os.path.join(root, f) for f in sorted(fs) if not f.endswith(IGNORE_SUFFIX)]
        else:
            files.append(p)
    return sorted(set(files))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def artefact_hash(arm_dir, meta=None):
    meta = meta if meta is not None else read_meta(arm_dir)
    arm_dir = os.path.abspath(arm_dir)
    per = {}
    for p in artefact_files(arm_dir, meta):
        per[os.path.relpath(p, arm_dir).replace(os.sep, '/')] = sha256_file(p)   # '/' keys on every OS (Windows import, 24 Nov 2026)
    tot = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return tot, per


def harness_hash():
    """Hash of the harness code + bank manifest (recorded at registration and in the exam)."""
    per = {}
    for f in sorted(os.listdir(C.HARNESS)):
        if f.endswith('.py'):
            per[f] = sha256_file(os.path.join(C.HARNESS, f))
    for rel in ('vendor/VENDOR.json', 'bank/MANIFEST.json', 'showcase.json'):
        p = os.path.join(C.HARNESS, rel)
        if os.path.exists(p):
            per[rel] = sha256_file(p)
    return hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest(), per


class Arm:
    """Loads predict.py of one arm. Call .run(batches) to get validated outputs."""

    def __init__(self, arm_dir):
        self.dir = os.path.abspath(arm_dir)
        self.meta = read_meta(self.dir)
        if self.dir not in sys.path:
            sys.path.insert(0, self.dir)
        name = 'bakeoff_arm_' + ''.join(ch if ch.isalnum() else '_' for ch in self.meta.get('arm_id', 'x'))
        spec = importlib.util.spec_from_file_location(name, os.path.join(self.dir, 'predict.py'))
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.mod
        spec.loader.exec_module(self.mod)
        if not hasattr(self.mod, 'predict'):
            raise AttributeError('predict.py has no predict(batch)')
        self.seconds = 0.0
        self.calls = 0

    def run(self, batches):
        """List of batches -> list of raw outputs (predict_many when the arm has it)."""
        t0 = time.process_time()
        if hasattr(self.mod, 'predict_many') and len(batches) > 1:
            outs = list(self.mod.predict_many(batches))
        else:
            outs = [self.mod.predict(b) for b in batches]
        self.seconds += time.process_time() - t0
        self.calls += len(batches)
        if len(outs) != len(batches):
            raise ValueError('predict_many returned a different number of outputs')
        return outs


def load_predict(arm_dir):
    """For wrapper arms (e.g. a conformal layer around another arm): returns the wrapped arm's predict function and
    its module. The wrapper must list the wrapped folder in its own ratchet.json "extra_files" (so it is hashed)."""
    a = Arm(arm_dir)
    return a.mod.predict, a.mod


def strip_batch(b):
    """A defensive copy (arms must not mutate inputs that the harness reuses)."""
    c = dict(b)
    for k in ('forcing', 'attrs', 'static', 'qobs', 'dates', 'doy'):
        if k in c and isinstance(c[k], np.ndarray):
            c[k] = c[k].copy()
    c['classes'] = list(b['classes'])
    return c
