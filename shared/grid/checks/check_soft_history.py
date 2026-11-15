"""Soft history constraints (HANDOFF P0.6): random outages that start while a unit is on, unit-level MILP.

With hard minimum up/down rows, `dispatch(outages='random', cluster=False)` goes infeasible when a forced outage starts
while a unit is on (its forced shutdown breaks the min-down row) and the day is re-solved without carried history, or
fails. `soft_history=True` puts a penalised slack on those rows instead. For random fleets (generate_fleet, 3 GW peak,
5 days, physics of $GRIDSIM_PHYSICS) this check counts days re-solved without history, failures, the slack used, and
the mean price both ways.  usage: python check_soft_history.py [first_seed] [n_seeds]  -> soft_history.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
import json, sys, time
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np, pandas as pd
import gridsim as gs


def one(seed, soft, days=5):
    idx = pd.date_range('2018-07-01', periods=24 * days, freq='h', tz='UTC')
    fl = gs.generate_fleet(seed=seed, peak_MW=3000)
    dem = pd.Series(2200 + 500 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi), index=idx)
    t = time.time()
    try:
        r = gs.dispatch(fl, dem, cluster=False, outages='random', seed=seed, method='milp', soft_history=soft,
                        prices={'NG': 3.0}, time_limit=120.0)
        return dict(seed=seed, soft=soft, days_relaxed=int(r.n_days_relaxed), failed=False,
                    slack={k: round(v, 3) for k, v in r.history_slack.items()},
                    mean_price=round(float(np.nanmean(r.price)), 3), seconds=round(time.time() - t, 1))
    except Exception as e:
        return dict(seed=seed, soft=soft, failed=True, error=str(e)[:200], seconds=round(time.time() - t, 1))


if __name__ == '__main__':
    s0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rows = []
    for seed in range(s0, s0 + n):
        for soft in (False, True):
            rows.append(one(seed, soft)); print(rows[-1], flush=True)
    out = dict(physics=gs.PHYSICS_VERSION, rows=rows,
               summary={k: dict(days_relaxed=sum(r.get('days_relaxed', 0) for r in rows if r['soft'] == s),
                                failures=sum(r['failed'] for r in rows if r['soft'] == s), runs=n)
                        for k, s in (('hard', False), ('soft', True))})
    json.dump(out, open(f"{GRID_CODE}/checks/soft_history.json", "w"), indent=1)
    print(out['summary'])
