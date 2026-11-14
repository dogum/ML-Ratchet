"""Physics/implementation checks: (1) PF daily, spread, threshold reproduce GridData's dev-year baseline numbers;
(2) rainflow vs the `rainflow` package; (3) energy conservation of the offer executor; (4) annual LP >= daily LP."""
import json, sys, time
import numpy as np, pandas as pd
import os  # noqa: E402
sys.path.insert(0, '.')
import _paths  # noqa: E402  (GridData on sys.path; outputs to WORK)
from battery import Battery, rainflow_cycles, step_offer, settle
from strategies import pf_daily, daily_spread, threshold_rule, perfect_foresight, lp_schedule
from realdata import node_frame
import griddata as gd
out = {}
bd = gd.baselines('battery_daily')
bat = Battery(1, 2, degradation=False)
res = {}
for node in ['HB_HOUSTON', 'HB_WEST']:
    f = node_frame('ERCOT', node, '2019-01-01', '2019-04-01')
    # GridData: dev-period RT quartiles over 2019+2021
    dev = pd.concat([node_frame('ERCOT', node, f'{y}-01-01', f'{y+1}-01-01').rt for y in (2019, 2021)])
    lo, hi = np.nanpercentile(dev, 25), np.nanpercentile(dev, 75)
    ref = bd[(bd.node == node)].copy(); ref['dayn'] = (pd.to_datetime(ref.day) - pd.Timestamp('2000-01-01')).dt.days
    rows = []
    for dd, g in f.groupby('day'):
        if len(g) < 23 or g.rt.isna().any() or g.da.isna().any(): continue
        dayid = g.day.values
        c, d = pf_daily(g.rt.values, dayid, bat); pf = (g.rt.values * (d - c)).sum()
        c, d = daily_spread(g.da.values, dayid, bat); spr = (g.rt.values * (d - c)).sum()
        c, d = threshold_rule(g.rt.values, bat, lo, hi); thr = (g.rt.values * (d - c)).sum()
        r = ref[ref.dayn == dd]
        if len(r): rows.append((pf, float(r.pf_rt.iloc[0]), spr, float(r.rule_spread.iloc[0]), thr, float(r.rule_threshold.iloc[0])))
    a = np.array(rows)
    res[node] = dict(days=len(a), pf_max_abs_diff=float(np.abs(a[:, 0] - a[:, 1]).max()),
                     spread_max_abs_diff=float(np.abs(a[:, 2] - a[:, 3]).max()),
                     threshold_max_abs_diff=float(np.abs(a[:, 4] - a[:, 5]).max()),
                     pf_sum=[float(a[:, 0].sum()), float(a[:, 1].sum())])
out['reproduce_griddata_dev_2019Q1'] = res
print(res)
# rainflow vs package
import rainflow
rng = np.random.default_rng(0)
errs = []
for k in range(200):
    x = np.cumsum(rng.normal(size=rng.integers(5, 400)))
    mine = sorted([(round(r, 9), c) for r, c, _, _ in rainflow_cycles(x)])
    ref = sorted([(round(r, 9), c) for r, c in rainflow.count_cycles(x)])
    # compare total damage-like sums for several exponents (package merges equal ranges)
    for kk in (1.0, 1.3, 2.0):
        a_ = sum(c * r ** kk for r, c in mine); b_ = sum(c * r ** kk for r, c in ref)
        errs.append(abs(a_ - b_) / max(b_, 1e-12))
out['rainflow_vs_package'] = dict(n_series=200, max_rel_diff_damage=float(max(errs)))
print(out['rainflow_vs_package'])
# offer executor: energy conservation and limits on random curves
b4 = Battery(1, 4); soc = 0.3; tot = 0
for t in range(5000):
    v = np.sort(rng.normal(30, 20, 5))[::-1]
    p = rng.normal(30, 30)
    s, c, d = step_offer(b4, soc, v, p, 5.0)
    assert -1e-12 <= s <= 1 + 1e-12 and c <= b4.P + 1e-12 and d <= b4.P + 1e-12 and (c == 0 or d == 0)
    tot = max(tot, abs((s - soc) * b4.E - (b4.eta_c * c - d / b4.eta_d)))
    soc = s
out['offer_executor_energy_balance_max_err_MWh'] = tot
# annual chunked LP vs one LP and vs daily
f = node_frame('ERCOT', 'HB_HOUSTON', '2019-06-01', '2019-09-01')
rt = f.rt.interpolate().values
b2 = Battery(1, 2)
t0 = time.time(); c, d = perfect_foresight(rt, b2, b2.c_lin(), chunk=24 * 30); t1 = time.time()
c2, d2, _, obj = lp_schedule(rt, b2, b2.c_lin()); t2 = time.time()
v1 = (rt * (d - c)).sum() - b2.c_lin() * d.sum(); v2 = obj
cd, dd = pf_daily(rt, f.day.values, b2, b2.c_lin()); v3 = (rt * (dd - cd)).sum() - b2.c_lin() * dd.sum()
out['annual_lp'] = dict(chunked=float(v1), single=float(v2), rel_gap=float((v2 - v1) / v2), daily_window=float(v3),
                        secs_chunked=t1 - t0, secs_single=t2 - t1)
print(out['annual_lp'])
s = settle(b2, rt, c2, d2)
out['rainflow_vs_linear_proxy_summer2019_PF'] = dict(rainflow_deg=s['deg'], linear_proxy=float(b2.c_lin() * d2.sum()), cycles=s['cycles'])
print(out['rainflow_vs_linear_proxy_summer2019_PF'])
out['c_lin_$per_MWh'] = {f'{h}h': Battery(1, h).c_lin() for h in (1, 2, 4)}
out['c_lin_temp_factor_35C_vs_25C'] = float(Battery(1, 2).temp_factor(35) / Battery(1, 2).temp_factor(25))
os.makedirs(os.path.join(_paths.WORK, 'checks'), exist_ok=True)   # was ../checks/ in the sandbox; the release copy is training/checks/
json.dump(out, open(os.path.join(_paths.WORK, 'checks', 'battery_checks.json'), 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
