"""Compare re-run check JSONs (checks/out/ under the data root) with the shipped v1.0 JSONs next to this file.

usage: python shared/building/checks/compare.py [name.json ...]     (default: every JSON present in both places)

Every numeric leaf is compared (relative tolerance 1e-6; both values below 1e-9 count as equal: solver round-off).
Timings (a last key ending in _s, total_s_*, s_per_*, x_realtime*) are reported, not compared; strings are ignored except for
missing keys. Writes checks/out/compare.json.
"""
import json
import math
import re
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('ML_RATCHET_BUILDING') or os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'checks', 'out')
TIMING = re.compile(r'(_s$|^total_s_|^x_realtime|^s_per_)')
SKIP = ('generated', 'python')
RTOL, NOISE = 1e-6, 1e-9


def leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f'{path}.{k}' if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f'{path}[{i}]')
    else:
        yield path, x


def compare(name):
    a = dict(leaves(json.load(open(os.path.join(HERE, name)))))
    b = dict(leaves(json.load(open(os.path.join(OUT, name)))))
    eq, diff, timing, missing, extra = 0, [], [], [], []
    for k, va in a.items():
        if k.split('.')[0] in SKIP:
            continue
        if k not in b:
            missing.append(k)
            continue
        vb = b[k]
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
        if not (num(va) and num(vb)):
            eq += va == vb or isinstance(va, str)
            if va != vb and not isinstance(va, str):
                diff.append((k, va, vb))
            continue
        if TIMING.search(k.split('.')[-1]):
            timing.append((k, va, vb))
            continue
        if (abs(va) < NOISE and abs(vb) < NOISE) or math.isclose(va, vb, rel_tol=RTOL, abs_tol=0.0):
            eq += 1
        else:
            diff.append((k, va, vb))
    extra = [k for k in b if k not in a and k.split('.')[0] not in SKIP]
    return dict(equal=eq, differ=len(diff), diffs=diff, timings=timing, missing=missing, extra=extra)


def main(names):
    names = names or sorted(f for f in os.listdir(HERE) if f.endswith('.json') and os.path.exists(os.path.join(OUT, f)))
    rep = {}
    for n in names:
        r = rep[n] = compare(n)
        print(f'{n}: {r["equal"]} equal, {r["differ"]} differ, {len(r["timings"])} timings, '
              f'{len(r["missing"])} missing, {len(r["extra"])} extra')
        for k, va, vb in r['diffs'][:12]:
            rel = (vb - va) / abs(va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va else None
            print(f'    {k}: shipped {va} rerun {vb}' + (f' ({rel:+.2e} rel)' if rel is not None else ''))
        if len(r['diffs']) > 12:
            print(f'    ... {len(r["diffs"]) - 12} more')
        for k in r['missing'][:5]:
            print(f'    missing in rerun: {k}')
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, 'compare.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1:])
