"""Wrappers around GridSim (no edits to gridsim/).  Upstream candidates.

1. random_outages_as_derates: gridsim.dispatch(outages='random') turns forced/planned outages into navail = 0.  When an
   outage starts while a unit is on, the min-down row  sum(sd over last DT h) <= navail - u  cannot hold (the forced
   shutdown itself is counted), so the day is infeasible even after the history relaxation (seen in 1 of 1 test jobs,
   also with must-run nuclear).  Here the same Markov/planned outage states are applied as a derate of ~0 instead
   (unit stays 'available' but can produce nothing), which keeps min up/down feasible.
"""
from common import *
import numpy as np
import gridsim  # noqa
gd_ = sys.modules['gridsim.dispatch']

_orig = gd_.availability


def _availability(cf, times, mode='expected', seed=0, mttr_h=60.0):
    if mode != 'random':
        return _orig(cf, times, mode, seed, mttr_h)
    avail, _ = _orig(cf, times, 'random', seed, mttr_h)
    n_on = cf.online_count(times)
    der = np.where(n_on > 0, avail / np.maximum(n_on, 1), 1.0)
    return n_on, np.maximum(der, 1e-4)


def install():
    gd_.availability = _availability
