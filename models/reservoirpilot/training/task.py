"""ReservoirPilot task definition: B's frozen reservoir exam, reproduced exactly.

Everything here mirrors shared/water/data/scripts/run_baselines_reservoir.py (ResOpsUS) and
run_baselines_reservoir_recent.py (CDEC). B's functions (doy_curve, cost, simulate, dp) are imported, not copied,
so the scoring is identical. A learned policy is run with `run_policy`, which uses the same daily mass balance as
B's `simulate` (S' = S + (I - R) K, release capped by what is stored, spill above capacity counts as release).

27 Nov 2026 (audit follow-up, owner-approved re-score): `resops_task(..., causal=True)`, or the environment variable
RESERVOIRPILOT_TASK=causal, builds the causal version of the same task. Three changes, nothing else:
  1. capacity = max(GRanD capacity, highest storage in the calibration half); the exam definition also took the
     maximum over the scored half, so a later observation could raise the capacity the first release sees;
  2. the record is split first and each half is interpolated (limit 5 days) on its own, so no gap in one half is
     filled from the other;
  3. the split day belongs to the scored half only (it was in both).
The CDEC recent task already takes everything from the 1980-2020 ResOpsUS record before its window: unchanged.
WaterData's run_baselines_reservoir.py has the same switch (--causal), for the baseline-table cross-check.
"""
import os, sys
import numpy as np, pandas as pd

import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
import waterdata as wd                                            # noqa: E402
from run_baselines_reservoir import doy_curve, cost, simulate, dp  # noqa: E402,F401

K = 0.0864
ROOT = _paths.WATER_DATA   # WaterData's data root
CAUSAL = os.environ.get('RESERVOIRPILOT_TASK', '') == 'causal'


def _interp(d):
    d = d.copy()
    d[['inflow_cms', 'release_cms']] = d[['inflow_cms', 'release_cms']].interpolate(limit=5)
    d['storage_hm3'] = d.storage_hm3.interpolate(limit=5)
    return d


def resops_task(rid, meta=None, causal=None):
    """B's ResOpsUS task for one dam (None if B skipped it). Returns a dict with the calibration half (cal, a
    DataFrame of observed storage/inflow/release) and the scored window: I, D, Qmax, cap, S0, Ss, dates, obs R/S.
    causal (default: RESERVOIRPILOT_TASK=causal): the 27 Nov 2026 causal definition (module docstring)."""
    causal = CAUSAL if causal is None else causal
    if meta is None:
        meta = wd.reservoir_meta().set_index('res_id')
    d = wd.reservoir(rid).loc['1980-01-01':]
    d = d[['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
    if len(d) < 3650:
        return None
    d = d.asfreq('D')
    half = d.index[len(d) // 2]                      # the same split day either way (asfreq before interpolation)
    if causal:
        cal = _interp(d.loc[:half - pd.Timedelta('1D')])
        ev = _interp(d.loc[half:]).dropna()
    else:
        d = _interp(d)
        cal, ev = d.loc[:half], d.loc[half:].dropna()
    blk = (ev.index.to_series().diff() != pd.Timedelta('1D')).cumsum()
    ev = ev[blk == blk.value_counts().idxmax()]
    if len(ev) < 1000:
        return None
    ev = ev.iloc[:3653]
    cap = float(meta.loc[rid, 'cap_hm3'])
    if causal:
        cap = float(np.nanmax([cap, float(cal.storage_hm3.max())]))
    else:
        cap = max(cap, float(d.storage_hm3.max()))
    Dc = doy_curve(cal.release_cms.dropna()); Sc = doy_curve(cal.storage_hm3.dropna())
    Qmax = float(cal.release_cms.quantile(0.99))
    Ic = doy_curve(cal.inflow_cms.dropna())                        # inflow climatology (calibration half only)
    doy = ev.index.dayofyear
    return {'res_id': rid, 'I': ev.inflow_cms.values.astype(float), 'D': Dc.reindex(doy).values,
            'Ss': Sc.reindex(doy).values, 'Qmax': Qmax, 'cap': cap, 'S0': float(ev.storage_hm3.iloc[0]),
            'dates': ev.index, 'R_obs': ev.release_cms.values, 'S_obs': ev.storage_hm3.values,
            'Dcurve': Dc.values, 'Icurve': Ic.values, 'Scurve': Sc.values,
            'cal_I_mean': float(cal.inflow_cms.mean()), 'cal': cal}


def cdec_task(res_id, cdec_id, meta=None):
    """B's recent CDEC task (WY2021-2025), demand/Qmax/rule from the dam's whole ResOpsUS record 1980..2020."""
    if meta is None:
        meta = wd.reservoir_meta().set_index('res_id')
    h = wd.reservoir(res_id).loc['1980-01-01':'2020-09-30'][['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
    if len(h) < 3650:
        return None
    ev = wd.reservoir_cdec(cdec_id, '2020-10-01', '2025-09-30').asfreq('D').interpolate(limit=5).dropna()
    if len(ev) < 1000:
        return None
    blk = (ev.index.to_series().diff() != pd.Timedelta('1D')).cumsum()
    ev = ev[blk == blk.value_counts().idxmax()]
    cap = max(float(meta.loc[res_id, 'cap_hm3']), float(h.storage_hm3.max()))
    Dc = doy_curve(h.release_cms); Sc = doy_curve(h.storage_hm3); Qmax = float(h.release_cms.quantile(0.99))
    Ic = doy_curve(h.inflow_cms)
    doy = ev.index.dayofyear
    return {'res_id': res_id, 'cdec_id': cdec_id, 'I': ev.inflow_cms.values.astype(float),
            'D': Dc.reindex(doy).values, 'Ss': Sc.reindex(doy).values, 'Qmax': Qmax, 'cap': cap,
            'S0': float(ev.storage_hm3.iloc[0]), 'dates': ev.index, 'R_obs': ev.release_cms.values,
            'S_obs': ev.storage_hm3.values, 'Dcurve': Dc.values, 'Icurve': Ic.values, 'Scurve': Sc.values,
            'cal_I_mean': float(h.inflow_cms.mean())}


def components(R, S, D, Qmax, cap, S0, I=None):
    """Every piece of B's cost J plus storage behaviour, so a low J cannot hide a loophole."""
    R = np.asarray(R, float); D = np.asarray(D, float)
    short = (np.maximum(0, D - R) / np.maximum(D, 1e-6)) ** 2
    flood = 10 * (np.maximum(0, R - Qmax) / Qmax) ** 2
    out = {'J': float(np.mean(short + flood)), 'J_short': float(np.mean(short)), 'J_flood': float(np.mean(flood)),
           'shortage_ratio': float(np.sum(np.maximum(0, D - R)) / np.sum(D)),
           'shortage_days_frac': float(np.mean(R < 0.999 * D)),
           'flood_days_per_yr': float(np.mean(R > Qmax) * 365.25),
           'max_excess_ratio': float(np.max(np.maximum(0, R - Qmax)) / Qmax),
           'release_mean_over_D': float(np.mean(R) / max(np.mean(D), 1e-9))}
    if S is not None:
        S = np.asarray(S, float)
        out.update({'S_mean_frac': float(np.mean(S) / cap), 'S_min_frac': float(np.min(S) / cap),
                    'S_end_minus_start_frac': float((S[-1] - S0) / cap),
                    'empty_days_frac': float(np.mean(S < 0.01 * cap)),
                    'full_days_frac': float(np.mean(S > 0.999 * cap))})
    return out


def run_policy(policy, task):
    """Run a causal policy on a task with B's mass balance. policy is a PolicyState-like object with
    .reset(task) and .act(t, S) -> requested release (m3/s); it may read task['I'][:t+1] only."""
    I, D, Qmax, cap = task['I'], task['D'], task['Qmax'], task['cap']
    n = len(I); S = np.empty(n + 1); R = np.empty(n); S[0] = task['S0']
    policy.reset(task)
    for t in range(n):
        avail = S[t] + I[t] * K
        r = float(policy.act(t, S[t]))
        r = min(max(r, 0.0), max(avail, 0) / K)
        s1 = avail - r * K
        if s1 > cap:
            r += (s1 - cap) / K; s1 = cap
        S[t + 1] = max(s1, 0.0); R[t] = r
    return R, S[1:]
