"""Freeze PREREG.md for the given models: log the hash of the whole file as it stands.
python freeze.py <model> [<model> ...]"""
import hashlib, json, os, sys, time
import pw_paths as _P
# A rerun keeps its own pre-registration in WORK (PREREG.md and prereg_log.json); the shipped pair in this folder is
# the 18-19 Nov 2026 record and is never written to.
txt = open(f'{_P.WORK}/PREREG.md', encoding='utf-8', newline='').read()
fn = f'{_P.WORK}/prereg_log.json'
L = json.load(open(fn)) if os.path.exists(fn) else []
# every earlier freeze must still hold
for e in L:
    assert hashlib.sha256(txt[:e['n_chars']].encode()).hexdigest() == e['sha256'], f'earlier freeze broken: {e}'
models = sys.argv[1:]
for e in L:
    for m in models:
        assert m not in e['models'], f'{m} already frozen'
e = {'models': models, 'n_chars': len(txt), 'sha256': hashlib.sha256(txt.encode()).hexdigest(),
     'time': time.strftime('%Y-%m-%d %H:%M:%S %Z')}
L.append(e)
json.dump(L, open(fn, 'w', newline='\n'), indent=1)
print(json.dumps(e))
