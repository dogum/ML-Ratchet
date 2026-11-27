"""ReservoirPilot v1.0.1: a causal learned reservoir release policy (numpy only).

The policy decides today's release from what an operator knows today: storage, capacity, today's inflow, the recent
inflow (5, 20, 60-day exponential means), the demand calendar D(doy), the safe channel capacity Qmax and (feature set
'clim') the calibration-period inflow climatology. It never sees future inflow.

A small network (15 inputs, 2 x 64 tanh, 2 outputs) gives two numbers in (0, 1) every day:
  h  the share of today's demand to release (h < 1 = hedging: ration now to keep water for later)
  e  an extra release on top, as a share of 1.25 x (Qmax - D) (e > 0 = make flood space before it is needed)
  requested release R = h D + 1.25 e max(Qmax - D, 0); the mass balance then caps it by the water stored and
  spills anything above capacity (spill counts as release, as in the benchmark).
(The 'rule' head, a rung that was beaten, instead sets a classical flood-pool + hedging rule's 3 parameters.)
K = 0.0864 hm3 per (m3/s * day). Units: storage hm3, flows m3/s.

Usage:
    import reservoirpilot as rp
    pol = rp.Pilot.load()                                 # reservoirpilot_weights.npz next to this file
    task = rp.make_task(cal_df, eval_df, capacity_hm3)    # WaterData's task construction from a daily record
    R, S = pol.run(task)
    rp.cost_components(R, S, task)
or step by step:  pol.reset(task); r = pol.act(t, storage_today)  for t = 0, 1, ...
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, 'reservoirpilot_weights.npz')
K = 0.0864
EMA_TAUS = (5.0, 20.0, 60.0)


# ------------------------------------------------------------------------------------------- task construction
def doy_curve(values, dates, smooth=31):
    """Day-of-year median (1..366), interpolated, circularly smoothed with a 31-day moving mean (as WaterData)."""
    import pandas as pd
    s = pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(dates)).dropna()
    d = s.groupby(s.index.day_of_year).median().reindex(range(1, 367)).interpolate(limit_direction='both')
    v = np.concatenate([d.values[-15:], d.values, d.values[:15]])
    return np.convolve(v, np.ones(smooth) / smooth, mode='valid')


def make_task(cal, ev, capacity_hm3, causal_capacity=True):
    """cal / ev: DataFrames (daily DatetimeIndex) with storage_hm3, inflow_cms, release_cms. The calibration record
    defines demand D(doy) (median release), Qmax (99th percentile release), the storage and inflow climatologies.
    The scored record supplies the inflow and the start storage only.

    Capacity is the larger of capacity_hm3 and the highest storage in the calibration record (v1.0.1, 27 Nov 2026).
    causal_capacity=False restores v1.0's exam definition, which also took the maximum over the scored window, so a
    later observation could raise the capacity the first release sees (audit, 25 Nov 2026: 21.853 -> 21.792 m3/s on a
    synthetic case). The 27 Nov 2026 re-score of every arm on the causal task moved the pilot's median J from 0.01543
    to 0.01544 on 244 dams; every verdict stayed (NOTEBOOK)."""
    Dc = doy_curve(cal.release_cms, cal.index); Sc = doy_curve(cal.storage_hm3, cal.index)
    Ic = doy_curve(cal.inflow_cms, cal.index)
    seen = cal.storage_hm3.values if causal_capacity else np.r_[cal.storage_hm3.values, ev.storage_hm3.values]
    cap = max(float(capacity_hm3), float(np.nanmax(seen)))
    doy = ev.index.day_of_year.values
    return {'I': ev.inflow_cms.values.astype(float), 'dates': ev.index, 'Dcurve': Dc, 'Icurve': Ic, 'Scurve': Sc,
            'D': Dc[doy - 1], 'Qmax': float(np.nanquantile(cal.release_cms, 0.99)), 'cap': cap,
            'S0': float(ev.storage_hm3.iloc[0])}


def fwd_mean(curve, doy, w):
    c = np.concatenate([curve, curve[:w]]); cs = np.concatenate([[0.0], np.cumsum(c)]); i = np.asarray(doy) - 1
    return (cs[i + w] - cs[i]) / w


def calendar(task, n=None):
    doy = np.asarray(task['dates'].day_of_year)[:n]
    Dc, Ic = np.asarray(task['Dcurve'], float), np.asarray(task['Icurve'], float)
    return {'D': Dc[doy - 1], 'Dn30': fwd_mean(Dc, doy, 30), 'Dn90': fwd_mean(Dc, doy, 90),
            'Ic30': fwd_mean(Ic, doy, 30), 'Ic90': fwd_mean(Ic, doy, 90)}


# ------------------------------------------------------------------------------------------- network
def features(S, I, emas, D, Dn30, Dn90, Ic30, Ic90, q, cap, fset):
    qK = q * K
    f = [S / cap, np.arcsinh(I / q)] + [np.arcsinh(e / q) for e in emas] + [
        D / q, Dn30 / q, np.log(cap / qK) / 3.0, np.arcsinh((cap - S) / qK) / 3.0,
        np.arcsinh((S + I * K - D * K - cap) / qK), np.arcsinh(S / (Dn30 * K + 1e-3 * qK) / 10.0)]
    if fset == 'clim':
        f += [np.arcsinh(Ic30 / q), np.arcsinh(Ic90 / q),
              np.arcsinh((S + 90 * K * (Ic90 - Dn90) - cap) / qK) / 3.0,
              np.arcsinh((S + 30 * K * (Ic30 - Dn30)) / qK) / 3.0]
    return np.stack(f, -1)


def hd_release(o, D, q):
    sig = lambda z: 1 / (1 + np.exp(-z))
    h = sig(o[..., 0]); e = sig(o[..., 1])
    return h * D + 1.25 * e * np.maximum(q - D, 0.0), h, e


def rule_release(o, S, I, D, q, cap, base):
    f0, t0, H0 = base
    sig = lambda z: 1 / (1 + np.exp(-z))
    cf = np.log((f0 - 0.3) / 0.75 / (1 - (f0 - 0.3) / 0.75))
    f = 0.3 + 0.75 * sig(o[..., 0] + cf)
    tau = t0 * np.exp(np.clip(o[..., 1], -3, 3)); H = H0 * np.exp(np.clip(o[..., 2], -3, 3))
    rh = D * np.minimum(1.0, (S + I * K) / np.maximum(H * D * K, 1e-12))
    extra = np.minimum(q, D + (S - f * cap) / (tau * K))
    return np.maximum(rh, np.where(D < q, extra, rh)), f, tau, H


class Pilot:
    def __init__(self, w, fset, head='hd', base=(0.75, 3.0, 10.0)):
        self.w = w; self.fset = fset; self.head = head; self.base = tuple(base)
        self.nl = sum(1 for k in w if k.startswith('W'))

    @classmethod
    def load(cls, path=WEIGHTS):
        z = np.load(path)
        w = {k: z[k].astype(np.float64) for k in z.files if k[0] in 'Wbms'}
        head = str(z['head']) if 'head' in z.files else 'hd'
        return cls(w, str(z['fset']), head, z['base'] if 'base' in z.files else (0.75, 3.0, 10.0))

    def mlp(self, x):
        x = (x - self.w['mu']) / self.w['sd']
        for i in range(self.nl):
            x = x @ self.w[f'W{i}'] + self.w[f'b{i}']
            if i < self.nl - 1:
                x = np.tanh(x)
        return x

    def decide(self, S, I, emas, cal, t, q, cap):
        q = max(q, 1e-9)
        x = features(S, I, emas, cal['D'][t], cal['Dn30'][t], cal['Dn90'][t], cal['Ic30'][t], cal['Ic90'][t],
                     q, cap, self.fset)
        o = self.mlp(x)
        if self.head == 'hd':
            r, h, e = hd_release(o, cal['D'][t], q)
            return r, h, e, 0.0
        return rule_release(o, S, I, cal['D'][t], q, cap, self.base)

    # step-by-step interface
    def reset(self, task):
        self.task = task; self.cal = calendar(task, len(task['I']))
        e0 = self.cal['Ic30'][0] if self.fset == 'clim' else task['I'][0]
        self.emas = [float(e0)] * 3

    def act(self, t, S, return_params=False):
        I = float(self.task['I'][t])
        self.emas = [e + (I - e) / tau for e, tau in zip(self.emas, EMA_TAUS)]
        out = self.decide(np.float64(S), I, self.emas, self.cal, t, self.task['Qmax'], self.task['cap'])
        return tuple(float(v) for v in out[:3]) if return_params else float(out[0])

    def run(self, task, return_params=False, seen=None):
        """Operate over the whole record with the benchmark's mass balance. Returns release, end-of-day storage
        (and, with return_params, per day: requested release, h, e for the 'hd' head).
        seen: optional inflow series the policy reads instead of task['I'] (e.g. a noisy gauge); the mass balance
        always uses task['I']."""
        I, cap = task['I'], task['cap']; n = len(I)
        S = np.empty(n + 1); R = np.empty(n); P = np.empty((n, 3)); S[0] = task['S0']
        self.reset(task if seen is None else dict(task, I=np.asarray(seen, float)))
        for t in range(n):
            r, a, b = self.act(t, S[t], True)
            avail = S[t] + I[t] * K
            r = min(max(r, 0.0), max(avail, 0) / K); s1 = avail - r * K
            if s1 > cap:
                r += (s1 - cap) / K; s1 = cap
            S[t + 1] = max(s1, 0.0); R[t] = r; P[t] = (r, a, b)
        return (R, S[1:], P) if return_params else (R, S[1:])


def sop(task):
    """Standard operating policy (the benchmark's SOP): release demand when the water is there, spill when full."""
    I, D, cap = task['I'], task['D'], task['cap']; n = len(I); S = np.empty(n); R = np.empty(n); s = task['S0']
    for t in range(n):
        avail = s + I[t] * K; r = min(D[t], max(avail, 0) / K); s1 = avail - r * K
        if s1 > cap:
            r += (s1 - cap) / K; s1 = cap
        s = max(s1, 0.0); R[t] = r; S[t] = s
    return R, S


def tuned_rule(task, f=0.6, tau=3.0, H=10.0, seen=None):
    """The classical comparator of the exam (no network): SOP with a flood pool and linear hedging, 3 parameters
    grid-tuned on simulated tasks (training/classical.py, policy.TunedSOP). Defaults are rung c2's, tuned on the g4
    generator (the exam's 'tuned_rule_g4', median J 0.0285); rung c1's g1 tuning is f=0.75, tau=3, H=10.
      R = D min(1, (S + I K) / (H D K))                 hedge when less than H days of demand are available
      R = max(R, min(Qmax, D + (S - f cap) / (tau K)))  draw down toward f x capacity, never above Qmax
    seen: optional inflow series the rule reads (as in Pilot.run); the mass balance always uses task['I']."""
    I, D, q, cap = task['I'], task['D'], task['Qmax'], task['cap']; n = len(I)
    Is = I if seen is None else np.asarray(seen, float)
    S = np.empty(n); R = np.empty(n); s = float(task['S0'])
    for t in range(n):
        r = D[t] * min(1.0, (s + Is[t] * K) / max(H * D[t] * K, 1e-12))
        if D[t] < q:
            r = max(r, min(q, D[t] + (s - f * cap) / (tau * K)))
        avail = s + I[t] * K; r = min(max(r, 0.0), max(avail, 0) / K); s1 = avail - r * K
        sp = max(s1 - cap, 0.0); r += sp / K; s1 -= sp
        s = max(s1, 0.0); R[t] = r; S[t] = s
    return R, S


def cost_components(R, S, task):
    """WaterData's reservoir cost J = mean daily (shortage/D)^2 + 10 (excess over Qmax / Qmax)^2, its parts, and the
    storage behaviour J does not charge for (mean storage, end - start, days near empty / full), as in the exam."""
    D, q, cap = task['D'], task['Qmax'], task['cap']
    short = (np.maximum(0, D - R) / np.maximum(D, 1e-6)) ** 2; flood = 10 * (np.maximum(0, R - q) / q) ** 2
    return {'J': float(np.mean(short + flood)), 'J_short': float(np.mean(short)), 'J_flood': float(np.mean(flood)),
            'shortage_ratio': float(np.sum(np.maximum(0, D - R)) / max(np.sum(D), 1e-12)),
            'shortage_days_frac': float(np.mean(R < 0.999 * D)),
            'flood_days_per_yr': float(np.mean(R > q) * 365.25),
            'max_excess_ratio': float(np.max(np.maximum(0, R - q)) / q),
            'storage_end_minus_start_frac': float((S[-1] - task['S0']) / cap),
            'storage_mean_frac': float(np.mean(S) / cap),
            'empty_days_frac': float(np.mean(S < 0.01 * cap)), 'full_days_frac': float(np.mean(S > 0.999 * cap))}
