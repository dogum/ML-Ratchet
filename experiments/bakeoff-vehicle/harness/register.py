"""Register (freeze) an arm for the exam.

usage: python3 register.py <arm_dir> [--replace]      (--replace: re-register a changed arm BEFORE the exam; logged)
       python3 register.py --verify <arm_dir|arm_id>  (recompute the hash and compare; changes nothing)
       python3 register.py --list

What it does, in order (any failure = not registered):
 1. ratchet.json has the required fields (armlib.REQUIRED: tracks, one released rung, compute.train_cpu_hours,
    real_labels_in_training false or absent), card.md exists, predict.py defines predict(batch).
 2. Hashes the frozen artefact: every file in the arm folder except work/, __pycache__/, .numba/, logs/, *.log,
    plus ratchet.json['extra_files'] (shared code outside the folder).
 3. Smoke run, per declared track: the first 24 dev records of each dev set (VED, EPA, HD) and 48 frozen sim-val
    trips; every output is validated (keys, shapes, finite, > 0).
 4. Audit: every file opened and every module imported from under /home/claude/bakeoff during the smoke run must be
    inside the hashed set or the harness itself.
 5. Determinism (the same batch twice) and row independence (the batch reversed): warnings only, recorded.
 6. Writes registry/<arm_id>.json (hash, per-file hashes, harness hash, smoke timings, exam CPU estimate).
The exam (exam.py) re-hashes every registered arm before and after scoring it and refuses a changed one.
Registration is refused once the exam has started (results/EXAM_RUNNING.json or EXAM_DONE.json exists).
"""
import argparse
import builtins
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import bake  # noqa: E402
import score as S  # noqa: E402
import armlib  # noqa: E402
from armlib import Arm, strip_batch  # noqa: E402

SMOKE_ROWS = 24          # per dev set (ved, epa, hd) and track
SMOKE_SIMVAL = 48
EXAM_ROWS = {'ved': 25793, 'epa': 93000, 'hd': 2800}   # approximate exam record counts (DATA.md; EPA ~9x dev), for the CPU estimate


class Audit:
    """Records files opened under the bake-off tree while active."""

    def __init__(self):
        self.paths = set()

    def __enter__(self):
        self._open, self._io_open = builtins.open, io.open
        aud = self

        def wrapped(file, *a, **k):
            try:
                p = os.path.abspath(os.fsdecode(file)) if isinstance(file, (str, bytes, os.PathLike)) else None
            except Exception:
                p = None
            if p and p.startswith(C.BAKEOFF):
                aud.paths.add(p)
            return aud._open(file, *a, **k)
        builtins.open = wrapped
        io.open = wrapped
        return self

    def __exit__(self, *exc):
        builtins.open, io.open = self._open, self._io_open


def exam_started():
    return os.path.exists(bake.LOCK) or os.path.exists(bake.DONE)


def _smoke_batches(track):
    """the first SMOKE_ROWS records of each dev set, as one batch per set (+ the first SMOKE_SIMVAL sim-val trips)."""
    out = []
    for name in bake.SETS:
        for nm, sl, b in bake._real_batches('dev', track, sets=(name,), chunk=SMOKE_ROWS):
            out.append((name, b))
            break
    gen = bake.simval_batches(track, chunk=SMOKE_SIMVAL, nmax=SMOKE_SIMVAL)
    b, y = next(gen)
    out.append(('simval', b))
    return out


def _reverse(b):
    r = dict(b)
    for k in ('uid', 'trace_key', 'arch', 'cls', 'X', 'E_phys', 'spec', 'rl'):
        if k in r:
            r[k] = np.asarray(b[k])[::-1].copy()
    if 'traces' in r:
        r['traces'] = [np.array(t, copy=True) for t in b['traces'][::-1]]
    return r


def check_arm(arm_dir, verbose=True):
    """Steps 1-6. Returns (record, problems, warnings)."""
    problems, warns = [], []
    arm_dir = os.path.abspath(arm_dir)
    meta = armlib.read_meta(arm_dir)
    problems += armlib.validate_meta(meta, arm_dir)
    if problems:
        return None, problems, warns
    h, per = armlib.artefact_hash(arm_dir, meta)
    hashed = {os.path.abspath(os.path.join(arm_dir, k)) for k in per}
    mods_before = set(sys.modules)
    rec = {'arm_id': meta['arm_id'], 'arm_dir': arm_dir, 'hash': h, 'files': per, 'tracks': meta['tracks'],
           'meta': {k: meta.get(k) for k in ('label', 'family', 'family_label', 'group', 'reference', 'numpy_only',
                                              'numpy_only_reason', 'compute', 'released', 'feature_engineering')},
           'smoke': {}}
    est = 0.0
    with Audit() as aud:
        t0 = time.process_time()
        arm = Arm(arm_dir)
        rec['smoke']['load_cpu_s'] = round(time.process_time() - t0, 2)
        for track in meta['tracks']:
            sm = {}
            for name, b in _smoke_batches(track):
                c0 = arm.seconds
                o = arm.run([strip_batch(b)])[0]
                try:
                    p, Q, nc, comp = S.check_output(o, b['n'])
                except S.OutputError as e:
                    problems.append(f'{track}/{name}: {e}')
                    continue
                dt = (arm.seconds - c0) / b['n']
                sm[name] = {'cpu_s_per_row': round(dt, 5), 'n_cross': nc, 'components': comp is not None}
                if name in EXAM_ROWS:
                    est += dt * EXAM_ROWS[name]
                o2 = arm.run([strip_batch(b)])[0]
                d = max(float(np.max(np.abs(np.asarray(o[k], float) - np.asarray(o2[k], float)) / np.asarray(o[k], float)))
                        for k in S.OUT_KEYS)
                sm[name]['determinism_maxreldiff'] = d
                if d > 1e-9:
                    warns.append(f'{track}/{name}: not deterministic: max relative diff {d:.3g} between two identical calls')
                o3 = arm.run([_reverse(b)])[0]
                d = max(float(np.max(np.abs(np.asarray(o[k], float) - np.asarray(o3[k], float)[::-1]) / np.asarray(o[k], float)))
                        for k in S.OUT_KEYS)
                sm[name]['row_order_maxreldiff'] = d
                if d > 1e-6:
                    warns.append(f'{track}/{name}: outputs depend on the batch (rows reversed: max relative diff {d:.3g}); '
                                 f'each row must be predicted from its own inputs')
            rec['smoke'][track] = sm
    mods = []
    for name in set(sys.modules) - mods_before:
        f = getattr(sys.modules[name], '__file__', None)
        if f:
            mods.append(os.path.abspath(f))
    used = {p for p in aud.paths | set(mods) if p.startswith(C.BAKEOFF)}
    refdir = os.path.join(C.HARNESS, 'arms_ref') + os.sep
    outside = sorted(p for p in used if p not in hashed
                     and not (p.startswith(C.HARNESS + os.sep) and not p.startswith(refdir)))
    rec['smoke']['files_used'] = sorted(os.path.relpath(p, C.BAKEOFF) for p in used)
    if outside:
        problems.append('files used at predict time but not in the hashed artefact (add them to the arm folder or '
                        'to ratchet.json "extra_files"): ' + ', '.join(os.path.relpath(p, C.BAKEOFF) for p in outside))
    rec['smoke']['exam_cpu_estimate_min'] = round(est / 60, 1)
    rec['harness_hash'] = armlib.harness_hash()[0]
    return rec, problems, warns


def reg_path(arm_id):
    return os.path.join(C.REGISTRY, f'{arm_id}.json')


def register(arm_dir, replace=False):
    if exam_started():
        print('REFUSED: the exam has started or finished; no registrations after that.')
        return 2
    meta = armlib.read_meta(arm_dir)
    p = reg_path(meta['arm_id'])
    old = json.load(open(p)) if os.path.exists(p) else None
    if old and os.path.abspath(arm_dir) != old['arm_dir']:
        print(f'REFUSED: arm_id {meta["arm_id"]} is already registered from {old["arm_dir"]}')
        return 2
    rec, problems, warns = check_arm(arm_dir)
    for w in warns:
        print('WARNING:', w)
    if problems:
        print('NOT REGISTERED:')
        for x in problems:
            print(' -', x)
        return 1
    if old and old['hash'] != rec['hash'] and not replace:
        print(f'REFUSED: {meta["arm_id"]} is registered with a different hash ({old["hash"][:12]} vs now '
              f'{rec["hash"][:12]}). Pass --replace to re-register before the exam (logged).')
        return 2
    rec['registered_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    rec['history'] = (old or {}).get('history', []) + ([{'hash': old['hash'], 'registered_at': old['registered_at'],
                                                         'replaced_at': rec['registered_at']}] if old and old['hash'] != rec['hash'] else [])
    os.makedirs(C.REGISTRY, exist_ok=True)
    tmp = p + '.tmp'
    json.dump(rec, open(tmp, 'w'), indent=1, default=float)
    os.replace(tmp, p)
    with open(os.path.join(C.REGISTRY, 'log.jsonl'), 'a') as f:
        f.write(json.dumps({'arm_id': rec['arm_id'], 'hash': rec['hash'], 'at': rec['registered_at'],
                            'replaced': bool(old and old['hash'] != rec['hash'])}) + '\n')
    print(f'REGISTERED {rec["arm_id"]} hash {rec["hash"][:16]} ({len(rec["files"])} files); smoke {rec["smoke"]}')
    return 0


def registered(include_excluded=False):
    """registry records; arms whose folder is listed in registry/excluded.json (with a reason) are left out unless
    include_excluded (Phase 3 fix: excluded.json lives in registry/ and is not an arm record)."""
    out = {}
    if os.path.isdir(C.REGISTRY):
        xp = os.path.join(C.REGISTRY, 'excluded.json')
        excl = json.load(open(xp)) if os.path.exists(xp) else {}
        for f in sorted(os.listdir(C.REGISTRY)):
            if f.endswith('.json') and f != 'excluded.json':
                r = json.load(open(os.path.join(C.REGISTRY, f)))
                if r.get('arm_dir') in excl and not include_excluded:
                    continue
                out[r['arm_id']] = r
    return out


def verify(rec):
    """(ok, current_hash, changed_files) for a registry record."""
    meta = armlib.read_meta(rec['arm_dir'])
    h, per = armlib.artefact_hash(rec['arm_dir'], meta)
    changed = sorted(set(k for k in set(per) | set(rec['files']) if per.get(k) != rec['files'].get(k)))
    return h == rec['hash'], h, changed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', nargs='?')
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, r in registered().items():
            ok, h, ch = verify(r)
            print(f'{k:24s} {r["hash"][:12]} {"OK" if ok else "CHANGED " + str(ch)} {r["arm_dir"]}')
        sys.exit(0)
    if a.verify:
        regs = registered()
        r = regs.get(a.arm) or next((x for x in regs.values() if x['arm_dir'] == os.path.abspath(a.arm)), None)
        if r is None:
            print('NOT REGISTERED'); sys.exit(1)
        ok, h, ch = verify(r)
        print('OK' if ok else f'CHANGED since registration: {ch}')
        sys.exit(0 if ok else 1)
    sys.exit(register(a.arm, a.replace))
