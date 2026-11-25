"""Register (freeze) an arm for the exam.

usage: python3 register.py <arm_dir> [--replace]      (--replace: re-register a changed arm BEFORE the exam; logged)
       python3 register.py --verify <arm_dir|arm_id>  (recompute the hash and compare; changes nothing)
       python3 register.py --list

What it does, in order (any failure = not registered):
 1. ratchet.json has the required fields (armlib.REQUIRED; one released rung; compute.train_cpu_hours), card.md
    exists, predict.py defines predict(batch).
 2. Hashes the frozen artefact: every file in the arm folder except work/, __pycache__/, .numba/, logs/, *.log,
    plus ratchet.json['extra_files'] (shared code outside the folder).
 3. Smoke run: 3 dev basins (ungauged and/or gauged at leads 1, 3, 7, by the arm's protocols) and 6 frozen sim-val
    windows; every output is validated (keys, shapes, finite values).
 4. Audit: while the smoke run executes, every file opened and every module imported from under
    /home/claude/bakeoff must be inside the hashed set or the harness itself. Otherwise the artefact is incomplete.
 5. Causality (gauged arms): qobs after row r0 is replaced by noise; predictions for target rows t < r0 + lead must
    not change (|diff| <= 1e-6 mm/day).
 6. Determinism: the same batch twice gives the same output (warning only).
 7. Writes registry/<arm_id>.json (hash, per-file hashes, harness hash, smoke timings, ratchet summary).
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

SMOKE_BASINS = 3
SMOKE_SEED = 20261123


def smoke_basins():
    ids = bake.dev_basins()
    rng = np.random.default_rng(SMOKE_SEED)
    return [ids[i] for i in sorted(rng.choice(len(ids), SMOKE_BASINS, replace=False))]


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


def check_arm(arm_dir, verbose=True):
    """Steps 1-6. Returns (record, problems, warnings)."""
    problems, warns = [], []
    arm_dir = os.path.abspath(arm_dir)
    meta = armlib.read_meta(arm_dir)
    problems += armlib.validate_meta(meta, arm_dir)
    if problems:
        return None, problems, warns
    h, per = armlib.artefact_hash(arm_dir, meta)
    hashed = set(os.path.join(arm_dir, k) if not os.path.isabs(k) else k for k in per)
    hashed = {os.path.abspath(p) for p in hashed}
    mods_before = set(sys.modules)
    rec = {'arm_id': meta['arm_id'], 'arm_dir': arm_dir, 'hash': h, 'files': per, 'protocols': meta['protocols'],
           'meta': {k: meta.get(k) for k in ('label', 'family', 'family_label', 'group', 'reference', 'numpy_only',
                                              'numpy_only_reason', 'compute', 'released', 'feature_engineering')},
           'smoke': {}}
    timing = {}
    with Audit() as aud:
        t0 = time.process_time()
        arm = Arm(arm_dir)
        rec['smoke']['load_cpu_s'] = round(time.process_time() - t0, 2)
        jobs = ([('ungauged', None)] if 'ungauged' in meta['protocols'] else []) + \
               ([('gauged', L) for L in C.LEADS] if 'gauged' in meta['protocols'] else [])
        B = bake.load_bank('val', ['F', 'qo', 'attrs', 'st', 'lat', 'lon', 'start'])
        for protocol, lead in jobs:
            tn = protocol if lead is None else f'{protocol}_L{lead}'
            bs = [bake.dev_batch(b, protocol, lead) for b in smoke_basins()]
            c0 = arm.seconds
            outs = arm.run([strip_batch(b) for b in bs])
            nse = []
            for bid, b, o in zip(smoke_basins(), bs, outs):
                m, Q, nc = S.check_output(o, b['n_target'])
                nse.append(S.score_real(bake._obs(bid, 'dev'), m, Q)['nse'])
            timing[tn] = round((arm.seconds - c0) / len(bs), 3)
            sv = [bake.sim_window(B, i, a, 365, protocol, lead) for i in (0, 1, 2) for a in bake.SIMVAL_OFFSETS]
            souts = arm.run([strip_batch(b) for b in sv])
            for b, o in zip(sv, souts):
                S.check_output(o, b['n_target'])
            rec['smoke'][tn] = {'dev_nse': [round(float(x), 4) for x in nse]}
            if protocol == 'gauged':
                b = bs[0]
                r0 = b['n_warm'] + 400
                o1 = arm.run([strip_batch(b)])[0]
                bp = strip_batch(b)
                rng = np.random.default_rng(1)
                bp['qobs'][r0:] = (rng.lognormal(1.0, 1.5, len(bp['qobs']) - r0)).astype(np.float32)
                o2 = arm.run([bp])[0]
                m1, _, _ = S.check_output(o1, b['n_target'])
                m2, _, _ = S.check_output(o2, b['n_target'])
                lim = r0 + lead - b['n_warm']             # target index t (row n_warm + t) must not see row >= r0
                d = float(np.max(np.abs(m1[:lim] - m2[:lim])))
                rec['smoke'][tn]['causality_maxdiff'] = d
                if d > 1e-6:
                    problems.append(f'{tn}: prediction for target rows before r0+lead changed by {d:.3g} when '
                                    f'qobs after r0 was perturbed: the arm reads observed flow it may not see yet')
                if np.max(np.abs(m1[lim:] - m2[lim:])) == 0:
                    warns.append(f'{tn}: perturbing qobs changed nothing at all: is qobs used?')
        b = bake.dev_batch(smoke_basins()[0], jobs[0][0], jobs[0][1])
        o1, o2 = arm.run([strip_batch(b)])[0], arm.run([strip_batch(b)])[0]
        d = max(float(np.max(np.abs(np.asarray(o1[k], float) - np.asarray(o2[k], float)))) for k in S.OUT_KEYS)
        rec['smoke']['determinism_maxdiff'] = d
        if d > 1e-9:
            warns.append(f'not deterministic: max diff {d:.3g} between two identical calls')
    rec['smoke']['predict_cpu_s_per_basin_period'] = timing
    # audit: files and modules from the bake-off tree must be hashed (or be the harness itself)
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
    # estimated exam cost: periods main (619) + recent (603) per job
    est = sum(v for v in timing.values()) * 1222
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


CLOUD_BAKEOFF = '/home/claude/bakeoff/river/'      # where the bake-off ran (22-24 Nov 2026); registry records keep it


def local_dir(d):
    """an arm folder as the registry records it (cloud path) -> the same folder in this checkout (records unchanged)."""
    if d.startswith(CLOUD_BAKEOFF):
        return os.path.abspath(os.path.join(C.BAKEOFF, d[len(CLOUD_BAKEOFF):]))
    return d


def registered():
    """registry records, arm_dir resolved into this checkout (local_dir); the files on disk are not changed."""
    out = {}
    if os.path.isdir(C.REGISTRY):
        for f in sorted(os.listdir(C.REGISTRY)):
            if f.endswith('.json'):
                r = json.load(open(os.path.join(C.REGISTRY, f)))
                r['arm_dir'] = local_dir(r['arm_dir'])
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
