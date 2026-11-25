"""Register (freeze) a round-2 arm into round 2's own registry (round2/registry/); v1.0's registry is untouched.

usage: python register2.py <arm_dir> [--replace]
       python register2.py --list            (re-hash every round-2 arm and show OK / CHANGED)

The checks are the harness's own (harness/register.py check_arm: ratchet.json, hash of the folder + extra_files,
smoke run on 3 dev basins + 6 frozen sim-val windows, file audit, determinism). Differences from register.py:
  - the record stores the arm folder relative to experiments/bakeoff-river/ (arm_dir_rel), so the registry works
    from any checkout; arm_dir is resolved at load time;
  - the smoke run's sim-val windows come from the round-2 bank named by R2_BANK (default b2 if built, else p1);
  - refused once exam 2 has started (round2/results/EXAM2_RUNNING.json or EXAM2_DONE.json exists).
"""
import argparse
import json
import os
import sys
import time

import r2
import config as C  # noqa: E402

REGISTRY = os.path.join(r2.ROUND2, 'registry')
LOCK = os.path.join(r2.RESULTS, 'EXAM2_RUNNING.json')
DONE = os.path.join(r2.RESULTS, 'EXAM2_DONE.json')


def _bank():
    name = os.environ.get('R2_BANK') or ('b2' if os.path.exists(os.path.join(r2.bank_dir('b2'), 'MANIFEST.json'))
                                         else 'p1')
    try:
        r2.use_bank(name)
    except FileNotFoundError:
        pass   # no local bank (e.g. a fresh checkout): only --list / verify work without one
    return name


def registered():
    out = {}
    if os.path.isdir(REGISTRY):
        for f in sorted(os.listdir(REGISTRY)):
            if f.endswith('.json') and f != 'excluded.json':
                r = json.load(open(os.path.join(REGISTRY, f)))
                r['arm_dir'] = os.path.normpath(os.path.join(r2.BAKEOFF, r['arm_dir_rel']))
                out[r['arm_id']] = r
    return out


def verify(rec):
    import armlib
    meta = armlib.read_meta(rec['arm_dir'])
    h, per = armlib.artefact_hash(rec['arm_dir'], meta)
    per = {k.replace('\\', '/'): v for k, v in per.items()}
    files = {k.replace('\\', '/'): v for k, v in rec['files'].items()}
    changed = sorted(k for k in set(per) | set(files) if per.get(k) != files.get(k))
    return not changed, h, changed


def register(arm_dir, replace=False):
    if os.path.exists(LOCK) or os.path.exists(DONE):
        print('REFUSED: exam 2 has started or finished; no registrations after that.')
        return 2
    bank = _bank()
    import register as REG  # harness
    import armlib
    arm_dir = os.path.abspath(arm_dir)
    meta = armlib.read_meta(arm_dir)
    p = os.path.join(REGISTRY, f'{meta["arm_id"]}.json')
    old = registered().get(meta['arm_id'])
    rec, problems, warns = REG.check_arm(arm_dir)
    for w in warns:
        print('WARNING:', w)
    if problems:
        print('NOT REGISTERED:')
        for x in problems:
            print(' -', x)
        return 1
    rec['files'] = {k.replace('\\', '/'): v for k, v in rec['files'].items()}
    if old and old['hash'] != rec['hash'] and not replace:
        print(f'REFUSED: {meta["arm_id"]} is registered with a different hash; pass --replace (logged).')
        return 2
    rec['arm_dir_rel'] = os.path.relpath(arm_dir, r2.BAKEOFF).replace('\\', '/')
    rec.pop('arm_dir', None)
    rec['smoke']['files_used'] = [x.replace('\\', '/') for x in rec['smoke']['files_used']]
    rec['meta']['compare_to'] = meta.get('compare_to')
    rec['smoke_simval_bank'] = bank
    rec['registered_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    rec['history'] = (old or {}).get('history', []) + ([{'hash': old['hash'], 'registered_at': old['registered_at'],
                                                         'replaced_at': rec['registered_at']}]
                                                       if old and old['hash'] != rec['hash'] else [])
    os.makedirs(REGISTRY, exist_ok=True)
    json.dump(rec, open(p + '.tmp', 'w'), indent=1, default=float)
    os.replace(p + '.tmp', p)
    with open(os.path.join(REGISTRY, 'log.jsonl'), 'a') as f:
        f.write(json.dumps({'arm_id': rec['arm_id'], 'hash': rec['hash'], 'at': rec['registered_at'],
                            'replaced': bool(old and old['hash'] != rec['hash'])}) + '\n')
    print(f'REGISTERED {rec["arm_id"]} hash {rec["hash"][:16]} ({len(rec["files"])} files); '
          f'exam CPU estimate {rec["smoke"]["exam_cpu_estimate_min"]} min')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', nargs='?')
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, r in registered().items():
            ok, h, ch = verify(r)
            print(f'{k:24s} {r["hash"][:12]} {"OK" if ok else "CHANGED " + str(ch)} {r["arm_dir_rel"]}')
        sys.exit(0)
    sys.exit(register(a.arm, a.replace))
