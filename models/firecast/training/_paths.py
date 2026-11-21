"""Where FireCast's training scripts read and write.

The scripts were first run on 17-19 Nov 2026 in a cloud sandbox workspace (fire/firecast/, code in src/, outputs in
out/, banks in bank/, the Wildfire Phase 1 tree at fire/: firesim/, firedata/, checks/, data/). Every run there was
CPU-only, on one or two shared cores. In this repository:

    SRC         training/src/ (the scripts, as run; src/common.py imports this module)
    TRAIN       training/ — the shipped, read-only inputs and records: configs/, phys/ (physics versions p0, p1),
                ratchet.json and results/ (the result JSONs as shipped; the sandbox wrote them to out/)
    MODEL       models/firecast/ (firecast.py and the released weights)
    WORK        models/firecast/work/ (git-ignored), or $ML_RATCHET_WORK/firecast/ — the old workspace root for
                everything a script writes: out/ (results, out/exam/), bank/ (simulated banks, p0 and p1 shards),
                ckpt/, logs/, web/, release/, .numba/, ratchet.json
    FIRE        the shared fire platform code: shared/fire/ (firesim/, firedata/, checks/). shared/fire and
                shared/fire/firesim are put on sys.path so `import firesim` and `import firedata` work.
    FIRE_DATA   the fire data root: $ML_RATCHET_FIRE, default shared/fire. Scripts read data/proc/... under it
                (FEDS fire lines, the fire-day tables, the exam baseline rows). The large real tables are not in the
                repository: rebuild them with `mlratchet data fire` before any script that touches real fires.

Re-running a script writes new copies under WORK; the shipped JSONs in TRAIN stay as they were. `shipped(name)`
returns a result a re-run wrote under WORK/out if there is one, else the shipped copy in TRAIN/results.
Inference (firecast.py), demo.py and the instrument export need none of this.
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src').replace(os.sep, '/')
TRAIN = os.path.dirname(SRC)
MODEL = os.path.dirname(TRAIN)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'firecast')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
FIRE = os.path.join(REPO, 'shared', 'fire').replace(os.sep, '/')
FIRE_DATA = os.path.abspath(os.environ.get('ML_RATCHET_FIRE') or FIRE).replace(os.sep, '/')
PROC = f'{FIRE_DATA}/data/proc'
for p in (FIRE, os.path.join(FIRE, 'firesim').replace(os.sep, '/'), SRC):   # import firesim, firedata, src modules
    if p not in sys.path:
        sys.path.insert(1, p)
for d in ('out', 'out/exam', 'bank', 'ckpt', 'logs', 'web', 'release', '.numba'):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)


def shipped(name):
    """A result a re-run wrote under WORK/out if there is one, else the shipped copy in TRAIN/results."""
    w = os.path.join(WORK, 'out', name)
    return w if os.path.exists(w) else os.path.join(TRAIN, 'results', name)


if __name__ == '__main__':      # the shell wrappers ask for WORK
    print(WORK)
