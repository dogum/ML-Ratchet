"""Run strategies on a price path and settle them (RT prices, rainflow degradation with temperature)."""
import numpy as np
import pandas as pd

from battery import Battery, settle
from strategies import perfect_foresight, daily_spread, threshold_rule, rolling_da_lp, offer_policy_run
from model import PolicyNet, run_policy


def run_strategy(name, path, bat, net=None, thresholds=None):
    rt = np.asarray(path['rt'], float); da = np.asarray(path['da'], float)
    T = np.asarray(path['T'], float)
    cl = bat.c_lin(T)
    day = np.asarray(path['day']); hour = np.asarray(path['hour'])
    if name == 'pf':
        ch, dis = perfect_foresight(rt, bat, cl)
    elif name == 'spread':
        ch, dis = daily_spread(da, day, bat, cl)
    elif name == 'threshold':
        lo, hi = thresholds
        ch, dis = threshold_rule(rt, bat, lo, hi)
    elif name == 'rolling':
        ch, dis = rolling_da_lp(da, hour, day, bat, cl)
    elif name == 'bb':      # v1.2 reference: the day-ahead value-curve backbone alone, as an offer curve
        from backbone import value_curves
        V = value_curves(da, hour, day, cl, bat.d, P=bat.P, eta_rt=bat.eta_rt)
        ch, dis, _ = offer_policy_run(bat, rt, V, cl)
    elif name == 'bp':
        ch, dis, _, _ = run_policy(net, path, bat, cl)
    else:
        raise ValueError(name)
    s = settle(bat, rt, ch, dis, 0.0, T)
    s['ch'], s['dis'] = ch, dis
    return s


def evaluate_path(path, durations=(1, 2, 4), strategies=('pf', 'rolling', 'spread', 'bp'), net=None,
                  thresholds=None, keep_traces=False):
    rows = []
    traces = {}
    for d in durations:
        bat = Battery(1.0, d)
        for st in strategies:
            if st == 'threshold' and thresholds is None:
                continue
            s = run_strategy(st, path, bat, net, thresholds)
            rows.append(dict(duration=d, strategy=st, gross=s['gross'], deg=s['deg'], net=s['net'],
                             cycles=s['cycles']))
            if keep_traces:
                traces[(d, st)] = s
    return pd.DataFrame(rows), traces
