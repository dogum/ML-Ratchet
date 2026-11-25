"""vendor/ = byte copies of the repo: writes vendor/VENDOR.json (sha256 of every vendored file and its repo twin)."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
MAP = {'vehicle/roadsim': 'shared/vehicle/roadsim', 'vehicle/realdata': 'shared/vehicle/realdata',
       'vehicle/checks': 'shared/vehicle/checks', 'energynet': 'models/energynet'}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
out, bad = {}, []
for v, r in MAP.items():
    base = os.path.join(C.VENDOR, v)
    for root, dirs, fs in os.walk(base):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in sorted(fs):
            p = os.path.join(root, f); rel = os.path.relpath(p, base)
            q = os.path.join(C.REPO, r, rel)
            a = sha(p); b = sha(q) if os.path.exists(q) else None
            out[os.path.join(v, rel)] = {'sha256': a, 'repo': os.path.join(r, rel), 'equal': a == b}
            if a != b: bad.append(rel)
json.dump({'n_files': len(out), 'all_equal': not bad, 'different': bad,
           'data': {'proc': os.path.realpath(os.path.join(C.VEHICLE, 'data', 'proc')),
                    'raw/ved': {f: sha(os.path.join(C.HARNESS, 'data', 'raw', 'ved', f)) for f in sorted(os.listdir(os.path.join(C.HARNESS, 'data', 'raw', 'ved')))}},
           'files': out}, open(os.path.join(C.VENDOR, 'VENDOR.json'), 'w'), indent=1)
print(len(out), 'files; all equal' if not bad else f'DIFFERENT: {bad}')
