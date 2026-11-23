"""Parse the UIUC Low-Speed Airfoil Tests (Selig et al.) tabulated polars, volumes 1-5.

Each run: airfoil label, builder, comment, average Reynolds number, and rows of
  lift files: alpha, Cl, Cm        (volume 1 Cm is 'meaningless', per the volume's own note)
  drag files: alpha, Cl, Cd, ...   (wake-rake drag, pre-stall only)
Output: data/lsat_runs.json (one record per run) and an inventory print.
"""
import os, re, json, glob, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')


def _entries_concat(path):
    """vol1-3 files are concatenations: '::::::::::::::\\nNAME\\n::::::::::::::\\n<body>'"""
    txt = open(path, errors='ignore').read()
    parts = re.split(r'^:{6,}\s*$', txt, flags=re.M)
    out = []
    # parts: [preamble, name, body, name, body...]
    i = 1
    while i + 1 < len(parts):
        name = parts[i].strip(); body = parts[i + 1]
        out.append((name, body)); i += 2
    return out


def _parse_body(body):
    lines = body.splitlines()
    meta = {'airfoil': None, 'builder': None, 'comment': None}
    runs = []
    k = 0
    while k < len(lines):
        L = lines[k].strip()
        if L.lower().startswith('airfoil:'):
            meta['airfoil'] = L.split(':', 1)[1].strip()
        elif L.lower().startswith('builder:'):
            meta['builder'] = L.split(':', 1)[1].strip()
        elif L.lower().startswith('comment:'):
            meta['comment'] = L.split(':', 1)[1].strip()
        elif L.lower().startswith('average reynolds'):
            k += 1
            re_ = float(lines[k].split()[0])
            while not lines[k].lower().strip().startswith('number of angles'):
                k += 1
            k += 1
            n = int(float(lines[k].split()[0]))
            k += 1
            hdr = lines[k].lower()
            kind = 'lift' if 'cm' in hdr and 'cd' not in hdr else ('drag' if 'cd' in hdr else 'unknown')
            rows = []
            for j in range(n):
                k += 1
                p = lines[k].split()
                try:
                    rows.append([float(v) for v in p[:3]])
                except (ValueError, IndexError):
                    break
            runs.append(dict(Re=re_, kind=kind, data=rows))
        k += 1
    return meta, runs


def parse_all():
    recs = []
    for vol in (1, 2, 3):
        d = os.path.join(ROOT, f'volume0{vol}', f'volume0{vol}')
        for fn in (f'LIFT0{vol}.TXT', f'DRAG0{vol}.TXT'):
            for name, body in _entries_concat(os.path.join(d, fn)):
                meta, runs = _parse_body(body)
                for r in runs:
                    recs.append(dict(vol=vol, file=name, **meta, **r))
    for vol, pat in ((4, 'volume04/volume04/*.txt'), (5, 'volume05/volume05/*.dat')):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            meta, runs = _parse_body(open(f, errors='ignore').read())
            for r in runs:
                recs.append(dict(vol=vol, file=os.path.basename(f), **meta, **r))
    return recs


if __name__ == '__main__':
    recs = parse_all()
    out = os.path.join(ROOT, '..', 'lsat_runs.json')
    json.dump(recs, open(out, 'w'))
    from collections import defaultdict
    inv = defaultdict(lambda: defaultdict(list))
    for r in recs:
        inv[(r['vol'], r['airfoil'], r['comment'])][r['kind']].append(int(r['Re']))
    for k in sorted(inv, key=lambda x: (x[0], str(x[1]))):
        print(k, {kk: sorted(v) for kk, v in inv[k].items()})
    print(len(recs), 'runs')
