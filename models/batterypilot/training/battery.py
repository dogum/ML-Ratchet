"""BatteryPilot battery physics (numpy/scipy only; shipped in the release).

Units: power MW (grid side), energy MWh, prices $/MWh, one step = one hour.
State of charge (SOC) is the fraction of the USABLE energy E = P x duration (stored, DC side), as in GridData's
baseline (1 MW / 2 MWh).  The usable window is soc_lo..soc_hi of nameplate (5..95 %), so nameplate = E / 0.9.

Physics
- energy balance: s[t+1] = s[t] + (eta_c ch[t] - dis[t] / eta_d) / E, eta_c = eta_d = sqrt(eta_rt)
- rate limits: 0 <= ch, dis <= P (grid side); SOC limits 0 <= s <= 1 (usable window)
- cycle ageing (Woehler / rainflow, Xu et al. 2018 framework): one full cycle of depth delta (fraction of nameplate)
  uses delta**k / N100 of the cell life (life = 20 % capacity fade).  Half cycles count 0.5.
- temperature and C-rate (Wang et al. 2011, LiFePO4 A123 cells, the chemistry CellSense was built on):
  rate factor exp(-(Ea - b C) / (R T)), normalised to 25 C and C/2.  Cell temperature in an HVAC container:
  T_cell = 25 + kappa (T_ambient - 25) + 3 C (degC), kappa = 0.25 (assumption).
- degradation cost = life used x replacement cost ($/kWh nameplate x nameplate).
  The dispatch optimisers use the matching linear proxy: c_lin $ per grid-side MWh discharged (a full-depth cycle).
Calendar ageing happens whatever the dispatch, so it is not charged to any strategy.
"""
from __future__ import annotations

import numpy as np

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
