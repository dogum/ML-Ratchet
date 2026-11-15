"""Verify the self-commitment DP (training/selfcommit.py) against an independent MILP (scipy.optimize.milp / HiGHS) on the
same single-unit problem (discrete output levels, start cost, minimum up/down, ramp and start-up/shut-down capability),
on 20 random 96-hour cases with ERCOT 2022 prices.  Writes WORK/checks/check_dp.json (the shipped copy is training/checks/check_dp.json)."""
import sys, json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # training/
from common import *
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from selfcommit import schedule
from realdata import prices

pr = prices('2018-01-01', '2023-01-01'); pr = pr[pr.index.year == 2022]
rng = np.random.default_rng(0)
res = []
for case in range(20):
    T = 96; s0 = int(rng.integers(0, len(pr) - T))
    price = pr.price.values[s0:s0 + T]; gas = pr.gas.values[s0:s0 + T]
    L = 7; pmax = 300.0; pmin = float(rng.uniform(0.3, 0.6)) * pmax
    dl = (pmax - pmin) / (L - 1); rb = int(rng.integers(1, 4)); R = rb * dl
    hr, beta = float(rng.uniform(7, 11)), float(rng.uniform(0.05, 0.3))
    F = lambda p: p * hr * (1 + beta * (pmax / max(p, 1e-3) - 1))  # noqa
    UT, DT = int(rng.integers(1, 6)), int(rng.integers(1, 6)); SC = float(rng.uniform(2e3, 3e4)); vom = 3.0
    on, P = schedule(price, gas, pmax, pmin, F, vom, SC, UT, DT, R, L=L, start_mult=(1.0, 1.0, 1.0))
    lev = np.linspace(pmin, pmax, L); fuel = np.array([F(p) for p in lev])
    prof = (price[:, None] - vom) * lev[None] - gas[:, None] * fuel[None]
    dp_obj = float(sum(prof[t, np.argmin(np.abs(lev - P[t]))] for t in range(T) if on[t]) - SC * np.sum(np.diff(np.r_[0, on.astype(int)]) == 1))
    nz = T * L; iz = lambda t, l: t * L + l; isu = lambda t: nz + t; isd = lambda t: nz + T + t  # noqa
    nv = nz + 2 * T
    c = np.zeros(nv); c[:nz] = -prof.ravel(); c[nz:nz + T] = SC
    A, lo, hi = [], [], []
    def row(d, l_, h_):
        r = np.zeros(nv)
        for k, v in d.items(): r[k] += v
        A.append(r); lo.append(l_); hi.append(h_)
    on_ = lambda t: {iz(t, l): 1.0 for l in range(L)}  # noqa
    Pt = lambda t, s=1.0: {iz(t, l): s * lev[l] for l in range(L)}  # noqa
    def add(*ds):
        o = {}
        for d in ds:
            for k, v in d.items(): o[k] = o.get(k, 0) + v
        return o
    cap = max(pmin, R)
    for t in range(T):
        row(on_(t), 0, 1)
        prev = on_(t - 1) if t > 0 else {}
        row(add({isu(t): 1.0, isd(t): -1.0}, {k: -v for k, v in on_(t).items()}, prev), 0, 0)   # su - sd = on_t - on_t-1
        row(add({isu(tt): 1.0 for tt in range(max(0, t - UT + 1), t + 1)}, {k: -v for k, v in on_(t).items()}), -np.inf, 0)
        row(add({isd(tt): 1.0 for tt in range(max(0, t - DT + 1), t + 1)}, on_(t)), -np.inf, 1)
        if t > 0:   # ramp up: P_t - P_t-1 <= R on_t-1 + cap su_t ; down: P_t-1 - P_t <= R on_t + cap sd_t
            row(add(Pt(t), Pt(t - 1, -1.0), {k: -R for k in on_(t - 1)}, {isu(t): -cap}), -np.inf, 0)
            row(add(Pt(t - 1), Pt(t, -1.0), {k: -R for k in on_(t)}, {isd(t): -cap}), -np.inf, 0)
        else:
            row(add(Pt(0), {isu(0): -cap}), -np.inf, 0)
    # shut-down at the end of the horizon is free in the DP (no terminal constraint) -- same here
    integ = np.r_[np.ones(nz), np.zeros(2 * T)]
    r = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integ, bounds=Bounds(0, np.r_[np.ones(nz), np.full(2 * T, 1)]),
             options=dict(mip_rel_gap=1e-7))
    milp_obj = -r.fun
    res.append(dict(case=case, dp=dp_obj, milp=milp_obj, rel=(dp_obj - milp_obj) / max(abs(milp_obj), 1.0)))
    print(case, round(dp_obj), round(milp_obj), f'{res[-1]["rel"]:.2e}')
out = dict(cases=res, max_abs_rel_diff=float(max(abs(x['rel']) for x in res)),
           note='profit of the DP schedule vs the MILP optimum of the same discrete single-unit problem; 0 = identical')
os.makedirs(f'{WORK}/checks', exist_ok=True)
json.dump(out, open(f'{WORK}/checks/check_dp.json', 'w'), indent=1)
print('max |rel diff|', out['max_abs_rel_diff'])
