"""BatteryPilot v1.3 -- a causal battery-arbitrage policy trained only on simulated grids (numpy/scipy only).

    import batterypilot as bp
    net = bp.PolicyNet()                                     # batterypilot_weights.npz next to this file (v1.3)
    net_v11 = bp.PolicyNet(bp.WEIGHTS_V1_1)                  # the v1.1 release; bp.WEIGHTS_V1_0 = v1.0
    bat = bp.Battery(P=1.0, duration_h=2.0)                 # 1 MW / 2 MWh usable, 85 % round trip
    ch, dis, soc, V = bp.run_policy(net, path, bat, bat.c_lin(path['T']))
    res = bp.settle(bat, path['rt'], ch, dis, 0.0, path['T'])   # gross, degradation, net ($), cycles

`path` holds hourly arrays: rt, da ($/MWh), df (BA day-ahead demand forecast, MW), T (degC), hour, day (local day
number), dow, doy.  Every hour the policy reads only information available before the hour and returns the marginal
value of stored energy at 5 state-of-charge knots; the battery then follows that curve as an offer curve against the
hour's real-time price.  v1.3 is a hybrid: a backward dynamic programme over the known day-ahead prices (the rolling
day-ahead optimiser's plan, `value_curves`) gives a backbone curve, and the network reads it with the 135 features
and outputs a residual on it.  See MODEL_CARD.md.

Weights: batterypilot_weights.npz is v1.3 (rung R8 of the v1.3 ladder, trained on GridSim physics 1.2, which adds
real-time scarcity the day-ahead market cannot see).  batterypilot_weights_v1_1.npz is v1.1 (physics 1.1) and
batterypilot_weights_v1_0.npz is v1.0 (physics 1.0), kept so their exam numbers stay reproducible.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


# ======================================================================= battery physics (battery.py)




R_GAS = 8.314


class Battery:
    def __init__(self, P=1.0, duration_h=2.0, eta_rt=0.85, soc_lo=0.05, soc_hi=0.95,
                 cost_per_kWh=130.0, N100=6000.0, k_wohler=1.3, Ea=31700.0, b_crate=370.3, kappa_hvac=0.25,
                 degradation=True):
        self.P = float(P)
        self.d = float(duration_h)
        self.E = self.P * self.d                     # usable stored MWh
        self.eta_rt = float(eta_rt)
        self.eta_c = self.eta_d = float(np.sqrt(eta_rt))
        self.soc_lo, self.soc_hi = soc_lo, soc_hi
        self.E_nom = self.E / (soc_hi - soc_lo)       # nameplate MWh
        self.cost = cost_per_kWh
        self.N100, self.k = N100, k_wohler
        self.Ea, self.b = Ea, b_crate
        self.kappa = kappa_hvac
        self.degradation = degradation
        self.crate = self.P / self.E_nom            # C-rate at full power (1/h)

    # ---------- ageing
    def phi(self, depth):
        """Fraction of cell life used by one full cycle of the given depth (fraction of nameplate)."""
        return np.asarray(depth, float) ** self.k / self.N100

    def temp_factor(self, T_amb_C=25.0, crate=None):
        """Ageing-rate multiplier vs 25 C cell temperature and C/2 (Wang et al. 2011 form)."""
        c = self.crate if crate is None else crate
        Tc = 25.0 + self.kappa * (np.asarray(T_amb_C, float) - 25.0) + 3.0 * c
        f = np.exp(-(self.Ea - self.b * c) / (R_GAS * (Tc + 273.15)))
        f0 = np.exp(-(self.Ea - self.b * 0.5) / (R_GAS * (25.0 + 1.5 + 273.15)))
        return f / f0

    def c_lin(self, T_amb_C=25.0):
        """Linear degradation proxy, $ per grid-side MWh discharged, at a full usable-depth cycle."""
        if not self.degradation:
            return np.zeros_like(np.asarray(T_amb_C, float)) if np.ndim(T_amb_C) else 0.0
        depth = self.E / self.E_nom
        per_cycle = 1000.0 * self.cost * self.E_nom * self.phi(depth)      # $ per full usable cycle at 25 C
        return per_cycle * self.temp_factor(T_amb_C) / (self.E * self.eta_d)

    def degradation_cost(self, soc, T_amb_C=None, return_cycles=False):
        """Rainflow degradation cost ($) of a SOC trace (fraction of usable, length n+1), with each cycle's cost
        assigned to the step where it closes.  Returns (total $, per-step $ array of length n)."""
        soc = np.asarray(soc, float)
        n = len(soc) - 1
        per = np.zeros(max(n, 0))
        if not self.degradation or n <= 0:
            return 0.0, per
        x = self.soc_lo + soc * (self.soc_hi - self.soc_lo)        # fraction of nameplate
        cyc = rainflow_cycles(x)
        if T_amb_C is None:
            fT = np.ones(n)
        else:
            fT = self.temp_factor(np.asarray(T_amb_C, float)[:n])
        # throughput-weighted temperature factor over each cycle span
        thr = np.abs(np.diff(x))
        cthr = np.r_[0, np.cumsum(thr)]
        cft = np.r_[0, np.cumsum(thr * fT)]
        tot = 0.0
        for rng, cnt, i0, i1 in cyc:
            a, b = min(i0, i1), max(i0, i1)
            w = cthr[b] - cthr[a]
            f = (cft[b] - cft[a]) / w if w > 1e-12 else 1.0
            c = cnt * self.phi(rng) * f * 1000.0 * self.cost * self.E_nom
            tot += c
            per[min(max(b - 1, 0), n - 1)] += c
        if return_cycles:
            return tot, per, cyc
        return tot, per


# ---------------------------------------------------------------- rainflow (ASTM E1049-85, 3-point)
def turning_points(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.arange(len(x)), x
    d = np.diff(x)
    keep = [0]
    last = 0.0
    for i in range(1, len(x) - 1):
        if d[i - 1] != 0:
            last = d[i - 1]
        if last * d[i] < 0:
            keep.append(i)
    keep.append(len(x) - 1)
    idx = np.array(keep)
    return idx, x[idx]


def rainflow_cycles(x):
    """Rainflow counting -> list of (range, count, i_start, i_end); count 1.0 full cycle, 0.5 half cycle."""
    idx, tp = turning_points(x)
    out = []
    stack = []   # (value, index)
    for v, i in zip(tp, idx):
        stack.append((v, i))
        while len(stack) >= 3:
            X = abs(stack[-1][0] - stack[-2][0])
            Y = abs(stack[-2][0] - stack[-3][0])
            if X < Y:
                break
            if len(stack) == 3:
                out.append((Y, 0.5, stack[0][1], stack[1][1]))
                stack.pop(0)
            else:
                out.append((Y, 1.0, stack[-3][1], stack[-2][1]))
                last = stack.pop()
                stack.pop(); stack.pop()
                stack.append(last)
    for j in range(len(stack) - 1):
        out.append((abs(stack[j + 1][0] - stack[j][0]), 0.5, stack[j][1], stack[j + 1][1]))
    return [c for c in out if c[0] > 0]


# ---------------------------------------------------------------- settlement
def settle(bat: Battery, price, ch, dis, soc0=0.0, T_amb_C=None):
    """Revenue of a grid-side schedule at the given prices; returns dict with gross, degradation (rainflow), net,
    SOC trace, throughput (equivalent full cycles)."""
    price = np.asarray(price, float); ch = np.asarray(ch, float); dis = np.asarray(dis, float)
    ds = (bat.eta_c * ch - dis / bat.eta_d) / bat.E
    soc = soc0 + np.r_[0, np.cumsum(ds)]
    assert soc.min() > -1e-6 and soc.max() < 1 + 1e-6, (soc.min(), soc.max())
    assert ch.max(initial=0) <= bat.P + 1e-9 and dis.max(initial=0) <= bat.P + 1e-9
    gross_t = price * (dis - ch)
    deg, deg_t = bat.degradation_cost(np.clip(soc, 0, 1), T_amb_C)
    return dict(gross=float(gross_t.sum()), deg=float(deg), net=float(gross_t.sum() - deg), gross_t=gross_t,
                deg_t=deg_t, soc=soc, cycles=float((dis / bat.eta_d).sum() / bat.E))


# ---------------------------------------------------------------- the offer-curve executor (policy physics)
SOC_KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def s_star(v, y, knots=SOC_KNOTS):
    """For a decreasing piecewise-linear value curve lambda(s) with values v (..., K) at the knots, return the SOC
    up to which lambda(s) >= y: sum over segments of the fraction of each segment where lambda >= y."""
    v = np.asarray(v, float); y = np.asarray(y, float)
    ds = np.diff(knots)
    a, b = v[..., :-1], v[..., 1:]
    frac = np.clip((a - y[..., None]) / np.maximum(a - b, 1e-9), 0.0, 1.0)
    return (frac * ds).sum(-1)


def step_offer(bat: Battery, soc, v, price, c_lin):
    """One hour of offer-curve execution.  v: marginal value of stored energy ($ per stored MWh) at SOC_KNOTS
    (decreasing), submitted before the hour; price: the hour's real-time price (clears against the offer).
    Discharge while price x eta_d - degradation beats the value of the stored MWh; charge while the stored MWh is
    worth more than its cost price / eta_c.  Returns (new soc, ch MW, dis MW)."""
    y_d = bat.eta_d * (price - c_lin)
    y_c = price / bat.eta_c
    s_hi = s_star(v, y_d)
    s_lo = s_star(v, y_c)
    tgt = np.clip(soc, s_lo, np.maximum(s_hi, s_lo))
    tgt = np.clip(tgt, soc - bat.P / (bat.eta_d * bat.E), soc + bat.P * bat.eta_c / bat.E)
    tgt = np.clip(tgt, 0.0, 1.0)
    d = tgt - soc
    ch = np.where(d > 0, d * bat.E / bat.eta_c, 0.0)
    dis = np.where(d < 0, -d * bat.E * bat.eta_d, 0.0)
    return tgt, ch, dis


# ======================================================================= causal features (features.py)



H_DA = 36      # day-ahead horizon read (hours from t)
H_RT = 24      # real-time history (hours before t)
H_DF = 24      # demand forecast horizon
PUBLISH_HOUR = 14
N_FEAT = 2 * H_DA + H_RT + 6 + H_DF + 1 + 5 + 3


def zq(p, ps):
    return np.arcsinh(p / ps)


def price_scale(da, t_idx, hist=168):
    """Median DA over the previous `hist` hours (causal), floor $5."""
    da = np.asarray(da, float)
    out = np.empty(len(t_idx))
    for i, t in enumerate(t_idx):
        w = da[max(0, t - hist):t]
        out[i] = np.nanmedian(w) if len(w) and np.isfinite(w).any() else (da[t] if np.isfinite(da[t]) else 25.0)
    return np.maximum(out, 5.0)


def rolling_scale(da, hist=168):
    """Causal rolling median of DA for every hour (fast via daily blocks)."""
    da = np.asarray(da, float)
    n = len(da)
    out = np.empty(n)
    # recompute once per 24 h (exact median of the trailing week at each day boundary), hold within the day
    last = 25.0
    for t in range(n):
        if t % 24 == 0:
            w = da[max(0, t - hist):t]
            if len(w) >= 24 and np.isfinite(w).any():
                last = float(np.nanmedian(w))
            elif np.isfinite(da[t:t + 24]).any():
                last = float(np.nanmedian(da[t:t + 24]))
        out[t] = last
    return np.maximum(out, 5.0)


def da_known_mask(hour, day, t, offsets):
    """1 where the DA price of hour t+o is known at decision time t (same local day, or the next day after the
    14:00 publication)."""
    tgt = t[:, None] + offsets[None, :]
    tgt = np.clip(tgt, 0, len(day) - 1)
    dd = day[tgt] - day[t][:, None]
    known = (dd <= 0) | ((dd == 1) & (hour[t][:, None] >= PUBLISH_HOUR))
    return known


def build(path, t_idx, duration_h, c_lin, ps=None):
    """Feature matrix (len(t_idx), N_FEAT) for decision hours t_idx of a path dict/DataFrame with arrays
    rt, da, df, T, hour, day, dow, doy.  Hours outside the path are padded with edge values."""
    rt = np.asarray(path['rt'], float); da = np.asarray(path['da'], float)
    df = np.asarray(path['df'], float); T = np.asarray(path['T'], float)
    hour = np.asarray(path['hour']); day = np.asarray(path['day'])
    dow = np.asarray(path['dow']); doy = np.asarray(path['doy'])
    n = len(rt)
    t = np.asarray(t_idx)
    if ps is None:
        ps = rolling_scale(da)[t]
    ps = np.asarray(ps, float)
    if np.ndim(ps) == 0:
        ps = np.full(len(t), float(ps))
    cl = np.broadcast_to(np.asarray(c_lin, float), (len(t),)) if np.ndim(c_lin) == 0 else np.asarray(c_lin)[t] \
        if np.asarray(c_lin).shape == (n,) else np.asarray(c_lin, float)
    o_da = np.arange(H_DA)
    idx = np.clip(t[:, None] + o_da[None, :], 0, n - 1)
    known = da_known_mask(hour, day, t, o_da)
    zda = np.where(known, zq(da[idx], ps[:, None]), 0.0)
    o_rt = np.arange(-H_RT, 0)
    ridx = np.clip(t[:, None] + o_rt[None, :], 0, n - 1)
    zrt = zq(rt[ridx], ps[:, None])
    o6 = np.arange(-6, 0)
    i6 = np.clip(t[:, None] + o6[None, :], 0, n - 1)
    sp6 = zq(rt[i6], ps[:, None]) - zq(da[i6], ps[:, None])
    # demand forecast relative to the trailing 30-day max of the forecast (tightness proxy)
    dfi = np.clip(t[:, None] + np.arange(H_DF)[None, :], 0, n - 1)
    if 'dmax' in path:
        dmax = np.asarray(path['dmax'], float)[t]
    else:
        dmax = _rollmax(df, 720)[t]
    dfn = df[dfi] / np.maximum(dmax[:, None], 1.0) - 0.8
    Tl = (T[np.clip(t - 1, 0, n - 1)] - 20.0) / 10.0
    hh = 2 * np.pi * hour[t] / 24.0; dy = 2 * np.pi * doy[t] / 365.25
    cal = np.c_[np.sin(hh), np.cos(hh), np.sin(dy), np.cos(dy), (dow[t] >= 5).astype(float)]
    bat = np.c_[np.log2(np.full(len(t), duration_h, float)), cl / ps, np.log(ps / 30.0)]
    X = np.concatenate([zda, known.astype(float), zrt, sp6, dfn, Tl[:, None], cal, bat], 1)
    return np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32), ps


def _rollmax(x, w):
    """Trailing max over the last w+1 values (inclusive), numpy only."""
    x = np.asarray(x, float)
    xp = np.r_[np.full(w, -np.inf), x]
    return np.lib.stride_tricks.sliding_window_view(xp, w + 1).max(1)


# ======================================================================= day-ahead backbone (backbone.py, v1.3)
# The rolling day-ahead LP's plan written as a value curve: a backward dynamic programme over the known DA horizon
# (DA prices as the forecast, no value left at the horizon end, re-planned at the first hour and at 14:00 local, as
# the LP) gives the marginal value of stored energy at every state of charge.  Hybrid weights (v1.3) read this curve
# and output a residual on it.


def _cross(lam, y, grid):
    """SOC up to which the decreasing curve lam (on grid) is >= y (piecewise linear)."""
    a, b = lam[:-1], lam[1:]
    frac = np.clip((a - y) / np.maximum(a - b, 1e-12), 0.0, 1.0)
    frac = np.where(a - b <= 1e-12, (a >= y).astype(float), frac)
    return float((frac * np.diff(grid)).sum())


def backward_pass(price, c_lin, E, P, eta_c, eta_d, N=200, terminal=0.0):
    """Value curves for a horizon of forecast prices: lam (n+1, N+1), lam[k] = slope of the value-to-go from step k
    ($ per stored MWh) on the SOC grid; lam[n] = terminal."""
    n = len(price)
    grid = np.linspace(0.0, 1.0, N + 1)
    lam = np.empty((n + 1, N + 1))
    lam[n] = terminal
    W = terminal * grid * E
    up = P * eta_c / E; dn = P / (eta_d * E)
    for k in range(n - 1, -1, -1):
        p = price[k]; c = c_lin[k]
        y_c = p / eta_c; y_d = eta_d * (p - c)
        nxt = lam[k + 1]
        s_lo = _cross(nxt, y_c, grid); s_hi = _cross(nxt, y_d, grid)
        tgt = np.clip(grid, s_lo, max(s_hi, s_lo))
        tgt = np.clip(tgt, grid - dn, grid + up)
        tgt = np.clip(tgt, 0.0, 1.0)
        d = tgt - grid
        r = np.where(d > 0, -p * d * E / eta_c, (p - c) * (-d) * E * eta_d)
        Wn = r + np.interp(tgt, grid, W)
        g = np.gradient(Wn, grid) / E
        lam[k] = np.minimum.accumulate(g)
        W = Wn
    return lam, grid


def value_curves(da, local_hour, day_id, c_lin, duration_h, P=1.0, eta_rt=0.85, knots=None, N=200):
    """Backbone value curves at every hour of a path: (n, K) $ per stored MWh at the SOC knots (causal: at hour t
    only the DA prices known at t, the rolling LP's publication rule)."""
    if knots is None:
        knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    da = np.asarray(da, float); n = len(da)
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,)).astype(float)
    E = P * duration_h; eta = np.sqrt(eta_rt)
    days = np.asarray(day_id)
    uniq, first = np.unique(days, return_index=True)
    order = np.argsort(first); uniq = uniq[order]; first = first[order]
    ends = np.r_[first[1:], n]
    day_pos = {d: j for j, d in enumerate(uniq)}
    replan = [0] + [t for t in range(1, n) if local_hour[t] == PUBLISH_HOUR]
    out = np.zeros((n, len(knots)))
    for j, t0 in enumerate(replan):
        dj = day_pos[days[t0]]
        known_end = ends[min(dj + 1, len(ends) - 1)] if (t0 > 0 or local_hour[t0] >= PUBLISH_HOUR) else ends[dj]
        t1 = replan[j + 1] if j + 1 < len(replan) else n
        p = da[t0:known_end]
        if np.isnan(p).any():
            p = np.where(np.isnan(p), np.nanmedian(p) if np.isfinite(p).any() else 0.0, p)
        lam, grid = backward_pass(p, cl[t0:known_end], E, P, eta, eta, N=N)
        m = t1 - t0
        for i in range(min(m, len(p))):
            out[t0 + i] = np.interp(knots, grid, lam[i + 1])
    return out


# ======================================================================= network (model.py)




K = len(SOC_KNOTS)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batterypilot_weights.npz')            # v1.3
WEIGHTS_V1_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batterypilot_weights_v1_1.npz')  # v1.1
WEIGHTS_V1_0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batterypilot_weights_v1_0.npz')  # v1.0


class PolicyNet:
    def __init__(self, weights=None):
        if weights is None:            # the current release (v1.3), next to this file
            weights = WEIGHTS
        w = dict(np.load(weights)) if isinstance(weights, (str, os.PathLike)) else dict(weights)
        self.mu, self.sd = w['x_mu'], w['x_sd']
        self.hybrid = bool(int(w['hybrid'])) if 'hybrid' in w else False    # v1.3: residual on the DA backbone
        self.min_step = float(w['min_step']) if 'min_step' in w else 0.0
        self.layers = []
        i = 0
        while f'W{i}' in w:
            self.layers.append((w[f'W{i}'], w[f'b{i}']))
            i += 1

    def z_knots(self, X):
        h = (X - self.mu) / self.sd
        for j, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if j < len(self.layers) - 1:
                h = silu(h)
        if self.hybrid:
            # backbone knots (the last K inputs) plus the residual, each knot at least min_step below the last
            u = X[:, -K:] + h
            if self.min_step > 0:
                z = u.copy()
                for k in range(1, K):
                    z[:, k] = np.minimum(z[:, k - 1] - self.min_step, u[:, k])
                return np.clip(z, -12.0, 12.0)
            return np.clip(np.minimum.accumulate(u, axis=1), -12.0, 12.0)
        z0 = h[:, :1]
        steps = softplus(h[:, 1:]) + 0.01
        return np.clip(np.concatenate([z0, z0 - np.cumsum(steps, 1)], 1), -12.0, 12.0)

    def values(self, X, ps):
        """Value curve in $/MWh of stored energy at SOC_KNOTS, decreasing."""
        return np.sinh(self.z_knots(X)) * np.asarray(ps, float)[:, None]


def run_policy(net: PolicyNet, path, bat: Battery, c_lin, soc0=0.0):
    """Run BatteryPilot causally over a whole path (dict/DataFrame with rt, da, df, T, hour, day, dow, doy).
    The value curves depend only on information before each hour, so they are computed for all hours at once; the
    offer-curve execution against each hour's RT price is sequential.  Returns (ch, dis, soc, V)."""
    n = len(path['rt'])
    t = np.arange(n)
    ps = rolling_scale(np.asarray(path['da'], float))
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,)).copy()
    X, ps = build(path, t, bat.d, cl, ps=ps)
    if net.hybrid:
        lam = value_curves(np.asarray(path['da'], float), np.asarray(path['hour']), np.asarray(path['day']), cl,
                           bat.d, P=bat.P, eta_rt=bat.eta_rt)
        X = np.concatenate([X, np.arcsinh(lam / ps[:, None])], 1).astype(np.float32)
    V = net.values(X, ps)
    rt = np.asarray(path['rt'], float)
    ch = np.zeros(n); dis = np.zeros(n); soc = np.zeros(n + 1); soc[0] = soc0
    for i in range(n):
        s, c, d = step_offer(bat, soc[i], V[i], rt[i], cl[i])
        soc[i + 1] = s; ch[i] = c; dis[i] = d
    return ch, dis, soc, V


# ======================================================================= baselines (strategies.py)






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
    n = len(rt); ch = np.zeros(n); dis = np.zeros(n); soc = np.zeros(n + 1); soc[0] = soc0
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,))
    for t in range(n):
        s, c, d = step_offer(bat, soc[t], V[t], rt[t], cl[t])
        soc[t + 1] = s; ch[t] = c; dis[t] = d
    return ch, dis, soc
