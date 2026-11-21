"""Assemble release/ (numpy/scipy-only planner + weights + demo cases + docs + training scripts) and zip it.

  python3 build_release.py <final_model>
The standalone module release/firepilot.py is generated from fp/core.py and fp/planner.py (the FireSim-dependent
parts, Context.from_scenario and cem_search, are dropped), so release == training code by construction.
"""
import os
import re
import sys
import json
import shutil
import zipfile

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
ROOT = _paths.WORK
REL = os.path.join(ROOT, 'release')


def strip_block(src, start_pat):
    """Remove a top-level def/class or an indented method starting at start_pat until the next block at the same
    or lower indent."""
    i = src.index(start_pat)
    line_start = src.rfind('\n', 0, i) + 1
    indent = i - line_start
    # include preceding decorator lines
    prev = src.rfind('\n', 0, line_start - 1) + 1
    if src[prev:line_start].strip().startswith('@'):
        line_start = prev
    j = i
    lines = src[i:].split('\n')
    n = len(lines[0]) + 1
    for ln in lines[1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        n += len(ln) + 1
    return src[:line_start] + src[i + n:]


def module():
    core = open(os.path.join(_paths.SRC, 'fp', 'core.py')).read()
    pl = open(os.path.join(_paths.SRC, 'fp', 'planner.py')).read()
    pl = pl.replace('from .core import Frame, head_azimuth, build_times, FL_HOLD_M, HORIZON_H\n', '')
    pl = strip_block(pl, 'def from_scenario(cls, S):')
    pl = strip_block(pl, 'def cem_search(')
    pl = pl.replace('import numpy as np\nfrom scipy import ndimage as ndi\n', '')
    pl = pl[pl.index('"""', 3) + 3:]   # drop planner docstring
    head = ('"""FirePilot v1.0: a learned containment-line planner (numpy + scipy only).\n\n'
            'Generated from the training code (fp/core.py + fp/planner.py). See MODEL_CARD.md.\n'
            'Main entry points: Context(...) or Context.from_store(npz, rec), ValueNet(path), plan_value(ctx, net, rng),\n'
            'plan_proxy(ctx, rng), BASELINES, build_times(...).\n"""\n')
    body = core[core.index('import numpy as np'):]
    return head + body + '\n\n# ---- planner ----\n' + pl


if __name__ == '__main__':
    final = sys.argv[1]
    os.makedirs(REL, exist_ok=True)
    open(os.path.join(REL, 'firepilot.py'), 'w').write(module())
    shutil.copy(os.path.join(ROOT, 'models', final + '.npz'), os.path.join(REL, 'firepilot_weights.npz'))
    tr = os.path.join(REL, 'training')
    if os.path.exists(tr):
        shutil.rmtree(tr)
    os.makedirs(os.path.join(tr, 'fp'))
    for f in os.listdir(_paths.SRC):
        if f.endswith(('.py', '.sh')):
            shutil.copy(os.path.join(_paths.SRC, f), tr)
    for f in os.listdir(os.path.join(_paths.SRC, 'fp')):
        if f.endswith('.py'):
            shutil.copy(os.path.join(_paths.SRC, 'fp', f), os.path.join(tr, 'fp'))
    for f in ('ratchet.json',):
        shutil.copy(_paths.latest(f), REL)
    os.makedirs(os.path.join(REL, 'results'), exist_ok=True)
    for f in os.listdir(os.path.join(_paths.SRC, 'results')):
        if f.endswith('.json'):
            shutil.copy(_paths.latest('results/' + f), os.path.join(REL, 'results'))
    zp = os.path.join(ROOT, 'firepilot-v1.0.zip')   # sandbox: the outputs folder
    os.makedirs(os.path.dirname(zp), exist_ok=True)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, dn, fn in os.walk(REL):
            dn[:] = [d for d in dn if d != '__pycache__']
            for f in fn:
                p = os.path.join(dp, f)
                z.write(p, os.path.join('firepilot-v1.0', os.path.relpath(p, REL)))
    print(zp, os.path.getsize(zp) // 1024, 'KB')
