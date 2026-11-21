"""ReservoirPilot classical baselines on ResOpsUS (exam + dev reservoirs).

Per reservoir: the simultaneous storage/inflow/release record from 1980 is split in half. First half = calibration
(demand curve, rule curve, channel capacity); second half = scored. All policies are driven by OBSERVED inflow, start at
the observed storage on the first scored day, and use S(t+1) = S(t) + (I - R) * 0.0864 hm3 (evaporation ignored),
0 <= S <= capacity (water above capacity is spilled and counts as release).

Task definition (frozen here, used by Phase 2):
  demand D(doy)  = median observed release on that day of year in the calibration half (31-day smoothing)
  Qmax           = 99th percentile of observed release in the calibration half (safe channel capacity)
  daily cost     = (max(0, D - R) / D)^2 + 10 * (max(0, R - Qmax) / Qmax)^2   ; J = mean daily cost (lower is better)
Policies: historical (observed release), SOP (standard operating policy: release D if water is available, spill above
capacity), rule (storage-target rule: R = D + (S - S*(doy)) / 30 d, clipped to [0, Qmax], S* = median calibration-half
storage by day of year), dp (perfect-foresight dynamic programming over 101 storage levels: a lower bound on J with the
end storage >= start storage).
Also imitation skill: NSE/KGE of simulated release and storage vs observed for SOP and rule.
"""
import json, os, sys
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd
from waterdata import metrics as M

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
K = 0.0864


def doy_curve(s, smooth=31):
    d = s.groupby(s.index.dayofyear).median().reindex(range(1, 367)).interpolate(limit_direction='both')
    v = np.concatenate([d.values[-15:], d.values, d.values[:15]])
    return pd.Series(np.convolve(v, np.ones(smooth) / smooth, mode='valid'), index=range(1, 367))


def cost(R, D, Qmax):
    return (np.maximum(0, D - R) / np.maximum(D, 1e-6)) ** 2 + 10 * (np.maximum(0, R - Qmax) / Qmax) ** 2


def simulate(policy, I, D, Qmax, cap, S0, Sstar=None):
    n = len(I); S = np.empty(n + 1); R = np.empty(n); S[0] = S0
    for t in range(n):
        avail = S[t] + I[t] * K                      # hm3 available today
        if policy == 'sop':
            r = D[t]
        else:
            r = np.clip(D[t] + (S[t] - Sstar[t]) / 30.0 / K, 0, Qmax)
        r = min(r, max(avail, 0) / K)                # cannot release more than is stored
        s1 = avail - r * K
        if s1 > cap:
            r += (s1 - cap) / K; s1 = cap
        S[t + 1] = max(s1, 0.0); R[t] = r
    return R, S[1:]


def dp(I, D, Qmax, cap, S0, nlev=201, nr=21):
    """Perfect-foresight DP on a storage grid with continuous next-state (linear interpolation of the value function).
    Release candidates per day: 21 values in [0, D] and 21 in [D, Qmax], plus spill above capacity. Terminal penalty
    1000 * max(0, S0 - S_T)/cap (end at least as full as the start). Approximately optimal: a near-lower bound on J."""
    lev = np.linspace(0, cap, nlev)
    n = len(I)
    V = 1000 * np.maximum(0, S0 - lev) / cap
    pol = np.zeros((n, nlev, ), dtype=np.float32)
    for t in range(n - 1, -1, -1):
        Rc = np.unique(np.concatenate([np.linspace(0, D[t], nr), np.linspace(D[t], max(Qmax, D[t]), nr)]))
        avail = lev[:, None] + I[t] * K                       # (nlev, 1)
        Rm = np.minimum(Rc[None, :], np.maximum(avail, 0) / K)  # cannot release more than stored
        s1 = avail - Rm * K
        spill = np.maximum(0, s1 - cap)
        Rm = Rm + spill / K; s1 = np.clip(s1 - spill, 0, cap)
        c = cost(Rm, D[t], Qmax) + np.interp(s1, lev, V)
        j = np.argmin(c, axis=1)
        V = c[np.arange(nlev), j]
        pol[t] = Rc[j]
    S = S0; Rs = np.empty(n)
    for t in range(n):
        r = np.interp(S, lev, pol[t])
        avail = S + I[t] * K
        r = min(r, max(avail, 0) / K)
        s1 = avail - r * K
        if s1 > cap:
            r += (s1 - cap) / K; s1 = cap
        Rs[t] = r; S = max(s1, 0.0)
    return Rs


def main():
    sp = wd.splits()['reservoirs']
    meta = wd.reservoir_meta().set_index('res_id')
    rows = []
    for rid in sp['dev'] + sp['exam']:
        d = wd.reservoir(rid).loc['1980-01-01':]
        d = d[['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
        if len(d) < 3650:
            continue
        d = d.asfreq('D')
        d[['inflow_cms', 'release_cms']] = d[['inflow_cms', 'release_cms']].interpolate(limit=5)
        d['storage_hm3'] = d.storage_hm3.interpolate(limit=5)
        half = d.index[len(d) // 2]
        cal, ev = d.loc[:half], d.loc[half:].dropna()
        # longest contiguous block in the evaluation half
        blk = (ev.index.to_series().diff() != pd.Timedelta('1D')).cumsum()
        ev = ev[blk == blk.value_counts().idxmax()]
        if len(ev) < 1000:
            continue
        ev = ev.iloc[:3653]                                  # score at most 10 years
        cap = float(meta.loc[rid, 'cap_hm3'])
        cap = max(cap, float(d.storage_hm3.max()))
        Dc = doy_curve(cal.release_cms.dropna()); Sc = doy_curve(cal.storage_hm3.dropna())
        Qmax = float(cal.release_cms.quantile(0.99))
        I = ev.inflow_cms.values.astype(float); D = Dc.reindex(ev.index.dayofyear).values; Ss = Sc.reindex(ev.index.dayofyear).values
        S0 = float(ev.storage_hm3.iloc[0])
        res = {'historical': (ev.release_cms.values, ev.storage_hm3.values)}
        res['sop'] = simulate('sop', I, D, Qmax, cap, S0)
        res['rule'] = simulate('rule', I, D, Qmax, cap, S0, Ss)
        res['dp'] = (dp(I, D, Qmax, cap, S0), None)
        for pol, (R, S) in res.items():
            c = cost(R, D, Qmax)
            r = {'res_id': rid, 'split': 'dev' if rid in sp['dev'] else 'exam', 'policy': pol, 'n_days': len(ev),
                 'eval_start': str(ev.index[0].date()), 'eval_end': str(ev.index[-1].date()),
                 'J': float(np.mean(c)), 'shortage_ratio': float(np.sum(np.maximum(0, D - R)) / np.sum(D)),
                 'flood_days_per_yr': float(np.mean(R > Qmax) * 365.25), 'main_use': meta.loc[rid, 'main_use'],
                 'cap_hm3': cap}
            if pol in ('sop', 'rule'):
                r['release_nse'] = M.nse(ev.release_cms.values, R); r['release_kge'] = M.kge(ev.release_cms.values, R)
                r['storage_nse'] = M.nse(ev.storage_hm3.values, S); r['storage_kge'] = M.kge(ev.storage_hm3.values, S)
            rows.append(r)
        print(rid, len(ev), {p: round(float(np.mean(cost(v[0], D, Qmax))), 4) for p, v in res.items()}, flush=True)
    sc = pd.DataFrame(rows)
    os.makedirs(f'{ROOT}/proc/baselines', exist_ok=True)
    sc.to_parquet(f'{ROOT}/proc/baselines/reservoir_scores.parquet')
    e = sc[sc.split == 'exam']
    summ = {}
    for pol, g in e.groupby('policy'):
        summ[pol] = {'n': int(len(g)), 'J_median': float(g.J.median()), 'J_mean': float(g.J.mean()),
                     'shortage_ratio_median': float(g.shortage_ratio.median()),
                     'flood_days_per_yr_median': float(g.flood_days_per_yr.median())}
        for c in ['release_nse', 'release_kge', 'storage_nse', 'storage_kge']:
            if c in g and g[c].notna().any():
                summ[pol][c + '_median'] = float(g[c].median())
    json.dump(summ, open(f'{ROOT}/proc/baselines/reservoir_summary.json', 'w'), indent=1)
    print(pd.DataFrame(summ).T.round(4))


if __name__ == '__main__':
    main()
