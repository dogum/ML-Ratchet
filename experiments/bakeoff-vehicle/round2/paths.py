"""Round 2 path layer (24 Nov 2026). Import this before any harness module.

The v1.0 harness (../harness) and the arms' training scripts name cloud paths (/home/claude/...) or write into
harness/. Round 2 never edits them: it imports the harness modules and points their outputs at round2/work/
(git-ignored) instead.

    R2        experiments/bakeoff-vehicle/round2
    WORK      round2/work            banks, checkpoints, eval outputs, dev records (git-ignored)
    BANKS     round2/work/banks/<name>/{rows.npz, trace_v.npy, trace_x.npy, fleet.json, MANIFEST.json}
    REGISTRY  round2/registry        round-2 arm records (v1.0's harness/registry is never written)
    RESULTS   round2/results         round-2 exam results, lock and DONE files
"""
import os
import sys

R2 = os.path.dirname(os.path.abspath(__file__))
BAKEOFF = os.path.dirname(R2)
HARNESS = os.path.join(BAKEOFF, 'harness')
WORK = os.path.join(R2, 'work')
BANKS = os.path.join(WORK, 'banks')
REGISTRY = os.path.join(R2, 'registry')
RESULTS = os.path.join(R2, 'results')
ARMS = os.path.join(R2, 'arms')

os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(WORK, '.numba'))
os.environ.setdefault('ML_RATCHET_WORK', os.path.join(WORK, 'enet_work'))
for p in (HARNESS, R2):
    if p not in sys.path:
        sys.path.insert(0, p)
import config as C  # noqa: E402  (harness config: sets the one-thread env, sys.path for RoadSim and EnergyNet)

# the real dev records (harness/real/dev in the cloud) are rebuilt by harness/build_real.py into round2/work/real
C.REAL = os.path.join(WORK, 'real')
C.WORK = os.path.join(WORK, 'enet_work')
C.LOGS = os.path.join(WORK, 'logs')
os.makedirs(C.LOGS, exist_ok=True)
# harness outputs that v1.0 wrote into results/ (dev reference rows, rehearsals) go to round2/work/harness_results;
# v1.0's results/ (incl. EXAM_DONE.json) is never written. exam2.py sets its own results folder.
C.RESULTS = os.path.join(WORK, 'harness_results')
os.makedirs(os.path.join(C.RESULTS, 'dev'), exist_ok=True)

# Bank names (round 2). 'val' is v2.1's frozen val2[h]; train101-103 are v2.1's three train banks (fleet train2, seed
# 201; trip seeds 101-103). '<name>_v4' are the same draws under the round-2 physics preset (if one is kept).
BANK_RECIPES = {
    'val': dict(fleet='val2', n=400, fseed=202, K=10, bseed=777),
    'train101': dict(fleet='train2', n=2400, fseed=201, K=12, bseed=101),
    'train102': dict(fleet='train2', n=2400, fseed=201, K=12, bseed=102),
    'train103': dict(fleet='train2', n=2400, fseed=201, K=12, bseed=103),
}


def bank_dir(name):
    return os.path.join(BANKS, name)


def use_bank(name, as_split):
    """Point bake.load_bank(as_split) at round-2 bank `name` (as_split 'train' or 'val'). Clears bake's cache."""
    import bake
    d = bank_dir(name)
    if not os.path.exists(os.path.join(d, 'MANIFEST.json')):
        raise FileNotFoundError(d)
    bake._BANK.pop(as_split, None)
    _LINKS[as_split] = name
    return d


_LINKS = {'train': 'train101', 'val': 'val'}


def _patch_bake():
    """bake.load_bank / fleet / bank_info read os.path.join(C.BANK, split). Round 2 maps each split to a named bank."""
    import bake
    orig_join = os.path.join

    def load_bank(split):
        if split not in ('train', 'val'):
            raise ValueError(split)
        if split in bake._BANK:
            return bake._BANK[split]
        import numpy as np
        d = bank_dir(_LINKS[split])
        if not os.path.exists(orig_join(d, 'MANIFEST.json')):
            raise FileNotFoundError(d)
        B = dict(np.load(orig_join(d, 'rows.npz')))
        B['trace_v'] = np.load(orig_join(d, 'trace_v.npy'), mmap_mode='r')
        B['trace_x'] = np.load(orig_join(d, 'trace_x.npy'), mmap_mode='r')
        B['_split'] = split
        B['_bank'] = _LINKS[split]
        bake._BANK[split] = B
        return B

    def fleet(B):
        import json
        with open(orig_join(bank_dir(B.get('_bank', _LINKS[B['_split']])), 'fleet.json')) as f:
            return json.load(f)

    def bank_info(split):
        import json
        with open(orig_join(bank_dir(_LINKS[split]), 'MANIFEST.json')) as f:
            return json.load(f)

    bake.load_bank = load_bank
    bake.fleet = fleet
    bake.bank_info = bank_info
    C.BANK = BANKS


def map_arm_dir(d):
    """registry arm_dir -> this checkout's folder (records hold the absolute path where they were registered: a cloud
    path for v1.0, a worktree path for round 2). Everything after 'bakeoff-vehicle/' (or the cloud 'bakeoff/vehicle/')
    is joined onto this checkout's experiments/bakeoff-vehicle."""
    if os.path.isdir(d):
        return d
    s = d.replace('\\', '/')
    for tag in ('experiments/bakeoff-vehicle/', 'bakeoff/vehicle/'):
        if tag in s:
            return os.path.normpath(os.path.join(BAKEOFF, s.split(tag, 1)[1]))
    return d


def patch_registry(REG):
    """harness register.registered() with arm_dir mapped onto this checkout (records on disk are unchanged)."""
    orig = REG.registered

    def registered(include_excluded=False):
        out = orig(include_excluded)
        for r in out.values():
            r['arm_dir_recorded'] = r['arm_dir']
            r['arm_dir'] = map_arm_dir(r['arm_dir'])
        return out
    REG.registered = registered


def setup():
    """Call once in a process that uses bake.* on round-2 banks."""
    _patch_bake()
