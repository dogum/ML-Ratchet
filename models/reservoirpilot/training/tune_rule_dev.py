"""Dev-tuned classical rule (added on import, 18 Nov 2026; the exam had already been seen).

The same 3-parameter rule as rung c1 (policy.TunedSOP: SOP + flood pool + linear hedging), the same grid and the same
objective as classical.py (median over tasks of the mean daily cost J), but tuned on the 15 real ResOpsUS dev dams
instead of simulated tasks. The dev tasks are WaterData's own (evaluate.prep('dev'): calibration half -> demand, Qmax;
second half scored), which dev dams allow. No exam dam is read. The grid optimum is frozen in dev_tuned_rule.json and
WORK/runs/tuned_sop_dev/weights.npz before any exam score of it is computed (rescore_exam.py).

usage: ML_RATCHET_WATER=... python tune_rule_dev.py
"""
import itertools
import json
import os
import sys

import numpy as np

import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from evaluate import prep  # noqa: E402
from policy import TunedSOP, rollout_np  # noqa: E402
from task import cost  # noqa: E402

GRID = ([0.4, 0.5, 0.6, 0.75, 0.85, 0.95, 1.01], [3, 10, 30], [1, 10, 30, 90])   # classical.py's grid, unchanged

if __name__ == '__main__':
    tasks = prep('dev')
    assert sorted(t['res_id'] for t in tasks) == sorted(json.load(open(f'{_paths.WATER_DATA}/proc/splits.json'))['reservoirs']['dev'])
    best, table = None, []
    for f, tau, H in itertools.product(*GRID):
        R, S = rollout_np(TunedSOP(f, tau, H), tasks)
        J = float(np.median([np.mean(cost(r, t['D'], t['Qmax'])) for r, t in zip(R, tasks)]))
        table.append([f, tau, H, J])
        if best is None or J < best[0]:
            best = (J, f, tau, H)
        print(f, tau, H, round(J, 5), flush=True)
    J, f, tau, H = best
    print('best', best)
    out = {'rule': 'policy.TunedSOP (rung c1 form): SOP + flood pool (f, tau) + linear hedging (H)',
           'tuned_on': '15 ResOpsUS dev dams, WaterData task (calibration half -> D, Qmax; second half scored)',
           'objective': 'median over dams of mean daily cost J (classical.py objective); grid of classical.py, first minimum',
           'f': f, 'tau': tau, 'H': H, 'dev_median_J': J,
           'compare': {'sim_g1_c1': {'f': 0.75, 'tau': 3, 'H': 10, 'dev_median_J': 0.04220199702172155},
                       'sim_g4_c2': {'f': 0.6, 'tau': 3, 'H': 10, 'dev_median_J': 0.047465296679753764}},
           'grid': table, 'note': 'Added on import, 18 Nov 2026, after the exam had been seen; a baseline, not a rung.'}
    json.dump(out, open(os.path.join(_paths.SRC, 'dev_tuned_rule.json'), 'w'), indent=1)
    os.makedirs(f'{_paths.WORK}/runs/tuned_sop_dev', exist_ok=True)
    np.savez(f'{_paths.WORK}/runs/tuned_sop_dev/weights.npz', fset='tuned_sop', f=f, tau=tau, H=H)
    sys.exit(0)
