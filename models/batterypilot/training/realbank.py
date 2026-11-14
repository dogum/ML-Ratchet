"""COMPARISON ONLY (breaks the 'trained on physics' rule; never released): a bank of REAL ERCOT prices from the
training-input years 2018, 2020, 2022 at the 12 exam nodes, in the same format as the simulated banks.

    python3 realbank.py
"""
import os, sys, json
import numpy as np
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from realdata import node_frame, fill, ERCOT_NODES
from simbank import finish

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
od = f'{ROOT}/data/banks/real_train'
os.makedirs(od, exist_ok=True)
i = 0
for y in (2018, 2020, 2022):
    for n in ERCOT_NODES:
        fn = f'{od}/p{i:04d}.npz'
        if not os.path.exists(fn):
            f, miss = fill(node_frame('ERCOT', n, f'{y}-01-01', f'{y + 1}-01-01'))
            f = f.dropna(subset=['rt', 'da'])
            out = finish(f)
            np.savez(fn, meta=json.dumps(dict(node=n, year=y, real=True)), **out)
            print(i, n, y, flush=True)
        i += 1
