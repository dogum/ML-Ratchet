"""Round-2 banks with the harness's own recipe (harness/build_bank.py, unchanged), written under round2/work/banks/.

python build.py worker <w> <n_workers> <bank> [<bank> ...]   one process: chunks w, w+n, ... of every listed bank
python build.py finalize <bank>                              -> rows.npz, trace_*.npy, fleet.json, MANIFEST.json
python build.py check <bank> <harness split>                 compare MANIFEST sha256s with harness/bank/<split>

Bank names and recipes: paths.BANK_RECIPES ('val', 'train101', 'train102', 'train103'). A bank name ending in
'_<preset tag>' (e.g. 'train101_v4') is the same recipe with EnergyNet's physics pin (training/physics.PHYSICS) swapped
for the named RoadSim preset (paths.PRESETS), everything else (fleet draws, trips, seeds) unchanged.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # noqa: E402
import config as C  # noqa: E402

PRESETS = {'v4': 'roadsim-physics-v4'}


def recipe(name):
    base, _, tag = name.partition('_')
    cfg = dict(P.BANK_RECIPES[base])
    return cfg, PRESETS.get(tag) if tag else None


def setup(name):
    cfg, preset = recipe(name)
    import physics as PHY
    if preset:
        PHY.PHYSICS = preset
        C.PHYSICS = preset
    import build_bank as BB
    C.BANK = P.BANKS
    BB.SPLITS[name] = cfg
    return BB


if __name__ == '__main__':
    os.makedirs(P.BANKS, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == 'worker':
        w, nw = int(sys.argv[2]), int(sys.argv[3])
        jobs = []
        for name in sys.argv[4:]:
            cfg, _ = recipe(name)
            nch = (cfg['n'] + 49) // 50
            jobs += [(name, ci) for ci in range(nch)]
        for name, ci in jobs[w::nw]:
            BB = setup(name)
            BB.one_chunk(name, ci)
        print('WORKER_DONE', w, flush=True)
    elif cmd == 'finalize':
        BB = setup(sys.argv[2])
        BB.finalize(sys.argv[2])
    elif cmd == 'check':
        name, split = sys.argv[2], sys.argv[3]
        a = json.load(open(os.path.join(P.BANKS, name, 'MANIFEST.json')))
        b = json.load(open(os.path.join(P.HARNESS, 'bank', split, 'MANIFEST.json')))
        ok = True
        for f, v in b['files'].items():
            same = a['files'][f]['sha256'] == v['sha256']
            ok &= same
            print(f, 'EQUAL' if same else 'DIFFERENT', a['files'][f]['bytes'], v['bytes'])
        print('rows', a['n_rows'], b['n_rows'], 'MANIFEST CHECK', 'PASS' if ok else 'FAIL')
