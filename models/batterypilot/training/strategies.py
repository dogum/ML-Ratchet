"""Dispatch strategies BatteryPilot is compared with (numpy/scipy; shipped in the release).

All strategies settle at the real-time (RT) price of each hour.  Information each one uses:
- perfect foresight (PF): the whole year of RT prices (upper bound; annual LP, energy may be carried across days)
- pf_daily: GridData's definition (empty at local midnight each day) for checking against gd.baselines
- daily spread: tomorrow's day-ahead (DA) prices only (charge the d cheapest DA hours, discharge the d dearest later)
- threshold: fixed RT price quartiles of the node's dev period (charge below the 25th, discharge above the 75th)
- rolling DA LP: re-plans when new DA prices clear (14:00 local), optimising over the known DA horizon (to the end
  of the next day) with DA prices as the forecast of RT; executes the plan as a fixed schedule.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from battery import Battery


def lp_schedule(price, bat: Battery, c_lin=0.0, soc0=0.0, soc_end_min=0.0, soc_end_max=1.0, terminal_value=0.0,
                zero_idx=None):
    """Maximise sum price*(dis-ch) - c_lin*dis + terminal_value*E*soc_end  s.t. SOC dynamics and limits.
    price, c_lin: arrays of length n (c_lin may be scalar).  Returns (ch, dis, soc[n+1], objective)."""
    p = np.asarray(price, float)
    n = len(p)
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,))
    E = bat.E
    # variables: ch (n), dis (n), soc (n+1)
    c = np.r_[p, -(p - cl), np.zeros(n + 1)]
    c[-1] -= terminal_value * E
    I = sp.identity(n, format='csr')
    Sd = sp.hstack([sp.csr_matrix((n, 1)), sp.identity(n)]) - sp.hstack([sp.identity(n), sp.csr_matrix((n, 1))])
    A = sp.hstack([-bat.eta_c / E * I, 1.0 / (bat.eta_d * E) * I, Sd]).tocsr()
    b = np.zeros(n)
    lb = np.r_[np.zeros(2 * n), np.zeros(n + 1)]
    ub = np.r_[np.full(2 * n, bat.P), np.ones(n + 1)]
    lb[2 * n] = ub[2 * n] = soc0
    lb[-1] = max(lb[-1], soc_end_min); ub[-1] = min(ub[-1], soc_end_max)
    if zero_idx is not None:                   # SOC fixed to 0 at these step boundaries (daily windows)
        lb[2 * n + np.asarray(zero_idx)] = 0.0; ub[2 * n + np.asarray(zero_idx)] = 0.0
    r = linprog(c, A_eq=A, b_eq=b, bounds=np.c_[lb, ub], method='highs')
    if not r.success:
        raise RuntimeError(r.message)
    x = r.x
    ch, dis, soc = x[:n], x[n:2 * n], x[2 * n:]
    # remove simultaneous charge/discharge (never optimal with eta<1 unless prices < 0; net it out)
    return np.clip(ch, 0, bat.P), np.clip(dis, 0, bat.P), soc, -r.fun


def perfect_foresight(rt, bat, c_lin=0.0, soc0=0.0, chunk=24 * 92):
    """Annual perfect-foresight LP on RT (solved in overlapping quarterly chunks with a week of overlap so HiGHS stays
    fast; the overlap makes the chunk ends near-optimal).  Returns (ch, dis)."""
    rt = np.asarray(rt, float); n = len(rt)
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,))
    if n <= chunk + 24 * 7:
        ch, dis, _, _ = lp_schedule(rt, bat, cl, soc0)
        return ch, dis
    ch = np.zeros(n); dis = np.zeros(n); s = soc0; t = 0
    while t < n:
        e = min(n, t + chunk + 24 * 7)
        keep = min(chunk, n - t) if e < n else n - t
        c1, d1, soc, _ = lp_schedule(rt[t:e], bat, cl[t:e], s)
        ch[t:t + keep] = c1[:keep]; dis[t:t + keep] = d1[:keep]
        s = float(np.clip(soc[keep], 0, 1))
        t += keep
    return ch, dis


def pf_daily(rt, day_id, bat, c_lin=0.0):
    """GridData's PF definition: each local day separately, SOC 0 at start and end (one LP with SOC pinned to 0 at
    every day boundary; identical to solving the days separately)."""
    rt = np.asarray(rt, float); day_id = np.asarray(day_id)
    b = np.r_[0, np.where(np.diff(day_id) != 0)[0] + 1, len(rt)]
    ch, dis, _, _ = lp_schedule(rt, bat, c_lin, 0.0, 0.0, 0.0, zero_idx=b)
    return ch, dis


def pf_daily_slow(rt, day_id, bat, c_lin=0.0):
    """Reference implementation: one LP per day."""
    rt = np.asarray(rt, float); ch = np.zeros(len(rt)); dis = np.zeros(len(rt))
    cl = np.broadcast_to(np.asarray(c_lin, float), (len(rt),))
    for d in np.unique(day_id):
        i = np.where(day_id == d)[0]
        c1, d1, _, _ = lp_schedule(rt[i], bat, cl[i], 0.0, 0.0, 0.0)
        ch[i] = c1; dis[i] = d1
    return ch, dis


def daily_spread(da, day_id, bat, c_lin=0.0):
    """Charge P in the d cheapest DA hours of the local day, discharge in the d dearest later hours (energy limited),
    only if the DA spread pays for losses and degradation.  d = duration (hours, rounded)."""
    da = np.asarray(da, float); n = len(da)
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,))
    ch = np.zeros(n); dis = np.zeros(n)
    k = int(round(bat.d))
    for d in np.unique(day_id):
        i = np.where(day_id == d)[0]
        if len(i) < 2 * k:
            continue
        p = da[i]
        o = np.argsort(p); cheap = np.sort(o[:k])
        later = [j for j in np.argsort(-p) if j > cheap.max()][:k]
        if len(later) < k:              # GridData's rule needs k discharge hours after the charge hours
            continue
        stored = k * bat.P * bat.eta_c                    # MWh stored by charging P for k hours
        out = stored * bat.eta_d                          # grid MWh available to sell
        q = out / len(later)
        if p[later].mean() * out - cl[i].mean() * out > p[cheap].mean() * k * bat.P:
            ch[i[cheap]] = bat.P
            dis[i[np.array(later)]] = min(q, bat.P)
    return ch, dis


def threshold_rule(rt, bat, lo, hi):
    """Charge when RT < lo, discharge when RT > hi, at full power, state limited (GridData's rule)."""
    rt = np.asarray(rt, float); n = len(rt)
    ch = np.zeros(n); dis = np.zeros(n); soc = 0.0
    for t, p in enumerate(rt):
        if p < lo and soc < 1.0:
            q = min(bat.P, (1.0 - soc) * bat.E / bat.eta_c); ch[t] = q; soc += q * bat.eta_c / bat.E
        elif p > hi and soc > 0.0:
            q = min(bat.P, soc * bat.E * bat.eta_d); dis[t] = q; soc -= q / (bat.eta_d * bat.E)
        soc = min(max(soc, 0.0), 1.0)
    return ch, dis


def rolling_da_lp(da, local_hour, day_id, bat, c_lin=0.0, soc0=0.0, publish_hour=14):
    """Rolling-horizon LP with DA prices as the RT forecast.  Re-plans at the first hour and whenever a new day's DA
    prices become known (at `publish_hour` local); the horizon runs to the end of the last day with known DA.
    The executed schedule is fixed (not price responsive), so SOC follows the plan exactly."""
    da = np.asarray(da, float); n = len(da)
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,))
    ch = np.zeros(n); dis = np.zeros(n)
    days = np.asarray(day_id)
    # end index (exclusive) of each day
    uniq, first = np.unique(days, return_index=True)
    order = np.argsort(first); uniq = uniq[order]; first = first[order]
    ends = np.r_[first[1:], n]
    day_pos = {d: j for j, d in enumerate(uniq)}
    replan = [0] + [t for t in range(1, n) if local_hour[t] == publish_hour]
    soc = soc0
    for j, t0 in enumerate(replan):
        dj = day_pos[days[t0]]
        known_end = ends[min(dj + 1, len(ends) - 1)] if (t0 > 0 or local_hour[t0] >= publish_hour) else ends[dj]
        t1 = replan[j + 1] if j + 1 < len(replan) else n
        h = slice(t0, known_end)
        p = da[h]
        if np.isnan(p).any():
            p = np.where(np.isnan(p), np.nanmedian(p) if np.isfinite(p).any() else 0.0, p)
        c1, d1, s, _ = lp_schedule(p, bat, cl[h], soc)
        m = t1 - t0
        net = d1[:m] - c1[:m]                     # remove simultaneous ch/dis
        ch[t0:t1] = np.clip(-net, 0, None); dis[t0:t1] = np.clip(net, 0, None)
        ds = (bat.eta_c * ch[t0:t1] - dis[t0:t1] / bat.eta_d) / bat.E
        soc = float(np.clip(soc + ds.sum(), 0, 1))
    return ch, dis


def offer_policy_run(bat, rt, V, c_lin, soc0=0.0, policy=None):
    """Execute value curves V (n, K) as hourly offer curves against RT prices.  Returns (ch, dis, soc)."""
    from battery import step_offer
    n = len(rt); ch = np.zeros(n); dis = np.zeros(n); soc = np.zeros(n + 1); soc[0] = soc0
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,))
    for t in range(n):
        s, c, d = step_offer(bat, soc[t], V[t], rt[t], cl[t])
        soc[t + 1] = s; ch[t] = c; dis[t] = d
    return ch, dis, soc
