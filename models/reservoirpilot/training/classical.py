"""Tune the 3-parameter TunedSOP comparator (flood pool + hedging) by grid search on SIMULATED training tasks only.
usage: python3 classical.py g1  -> runs/tuned_sop_g1/weights.npz"""
import sys, os, json, itertools
import numpy as np
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import TunedSOP, rollout_np
from task import cost
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)


def bank_tasks(gen, n=200, seed=5):
    import pandas as pd
    z = np.load(f'{ROOT}/data/bank_{gen}.npz'); rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(z['q']), n, replace=False):
        doy = z['doy'][i].astype(int)
        # rebuild a date index with the same day-of-year sequence (start year arbitrary but leap-consistent)
        start = pd.Timestamp('2001-01-01') + pd.Timedelta(days=int(doy[0]) - 1)
        dates = pd.date_range(start, periods=len(doy))
        C = z['C'][i].astype(float)
        out.append({'I': z['I'][i].astype(float), 'dates': dates, 'Dcurve': C[0], 'Icurve': np.ones(366),
                    'Qmax': float(z['q'][i]), 'cap': float(z['cap'][i]), 'S0': float(z['S0'][i]),
                    'D': C[0][dates.dayofyear - 1]})
    return out


if __name__ == '__main__':
    gen = sys.argv[1]; tasks = bank_tasks(gen)
    best = None
    for f, tau, H in itertools.product([0.4, 0.5, 0.6, 0.75, 0.85, 0.95, 1.01], [3, 10, 30], [1, 10, 30, 90]):
        R, S = rollout_np(TunedSOP(f, tau, H), tasks)
        J = np.median([np.mean(cost(r, t['D'], t['Qmax'])) for r, t in zip(R, tasks)])
        if best is None or J < best[0]:
            best = (J, f, tau, H)
        print(f, tau, H, round(J, 5), flush=True)
    J, f, tau, H = best; print('best', best)
    os.makedirs(f'{ROOT}/runs/tuned_sop_{gen}', exist_ok=True)
    np.savez(f'{ROOT}/runs/tuned_sop_{gen}/weights.npz', fset='tuned_sop', f=f, tau=tau, H=H)
