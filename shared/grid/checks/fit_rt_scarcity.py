"""Fit GridSim physics 1.2's real-time scarcity process (gridsim.market) on the TRAINING-INPUT years only.

Data: ERCOT HB_HUBAVG hourly RT (mean of the four 15-min SPPs) and DA settlement point prices, and EIA-930 ERCO
demand with its day-ahead forecast, for 2018, 2020 and 2022 (splits.json: real inputs, prices included, may be used
from train_inputs years). Dev years (2019, 2021) and exam years are not read here; check_rt_scarcity.py checks the
fit on dev afterwards.

An event is a run of hours with RT - DA > u = $100/MWh. Fitted:
  H[24], M[12]   start counts by local hour and month (pooled, +0.5 each, circular [1, 2, 1] smoothing, mean 1)
  beta           effect of the standardised demand-forecast miss z on the start rate (Bernoulli likelihood, grid)
  r0             base start rate, so that the expected starts equal the observed ones
  q              1 / mean event duration (geometric)
  xi, sigma      generalised Pareto fit of the hourly excess RT - DA - u in event hours (scipy, loc 0)
  rho            lag-1 correlation of the normal scores of that excess inside events (Gaussian copula)
Writes gridsim/gridsim/data/rt_scarcity_v1_2.json.

usage: python shared/grid/checks/fit_rt_scarcity.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

GRID_CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f"{GRID_CODE}/gridsim")
sys.path.insert(0, GRID_CODE)
import griddata as gd  # noqa: E402

TRAIN = (2018, 2020, 2022)
U = 100.0
X_MAX = 9000.0 - U
NODE = 'HB_HUBAVG'
OUT = os.path.join(GRID_CODE, 'gridsim', 'gridsim', 'data', 'rt_scarcity_v1_2.json')


def year_frame(year, node=NODE):
    """Hourly frame (UTC) of one year: rt, da, demand error fraction e = (d - df) / df, local hour and month."""
    s, e = pd.Timestamp(f'{year}-01-01', tz='UTC'), pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
    idx = pd.date_range(s, e, freq='h', inclusive='left')
    rt = gd.prices('ERCOT', node, s, e, kind='rt').price
    da = gd.prices('ERCOT', node, s, e, kind='da').price
    f = pd.DataFrame(index=idx)
    f['rt'] = rt[~rt.index.duplicated()].reindex(idx).astype(float).interpolate(limit=6, limit_direction='both')
    f['da'] = da[~da.index.duplicated()].reindex(idx).astype(float).interpolate(limit=6, limit_direction='both')
    y = gd.eia930('ERCO', s, e)
    y = y.set_index('time_utc') if 'time_utc' in y else y
    y = y[~y.index.duplicated()].reindex(idx)
    d = pd.to_numeric(y['d'], errors='coerce').astype(float)
    df = pd.to_numeric(y['df'], errors='coerce').astype(float)
    bad = df.isna() | d.isna() | ((df - d).abs() > 0.25 * d)        # as BatteryPilot's simulator does
    err = ((d - df) / df).where(~bad)
    f['e'] = err.interpolate(limit_direction='both').fillna(0.0)
    loc = idx.tz_convert('America/Chicago')
    f['hour'] = loc.hour
    f['month'] = loc.month
    return f.dropna(subset=['rt', 'da'])


def events(s, u=U):
    """Runs of s > u: (start indices, durations)."""
    m = (np.asarray(s) > u).astype(int)
    d = np.diff(np.r_[0, m, 0])
    st = np.where(d == 1)[0]
    en = np.where(d == -1)[0]
    return st, en - st


def smooth_circ(c):
    c = np.asarray(c, float) + 0.5
    s = 0.25 * np.roll(c, 1) + 0.5 * c + 0.25 * np.roll(c, -1)
    return s


def fit(frames):
    hs, ms, starts, eligible, z_all, durs, exc, pairs = [], [], [], [], [], [], [], []
    for f in frames:
        s = (f.rt - f.da).values
        st, L = events(s)
        act = np.zeros(len(s), bool)
        for a, n in zip(st, L):
            act[a:a + n] = True
        start = np.zeros(len(s), bool)
        start[st] = True
        # eligible hours: not inside a running event (a start hour is eligible)
        elig = ~act | start
        hs.append(f.hour.values); ms.append(f.month.values); starts.append(start); eligible.append(elig)
        z_all.append(f.e.values)
        durs += list(L)
        for a, n in zip(st, L):
            x = s[a:a + n] - U
            exc += list(x)
            pairs += [(x[k], x[k + 1]) for k in range(n - 1)]
    hour = np.concatenate(hs); month = np.concatenate(ms); start = np.concatenate(starts)
    elig = np.concatenate(eligible); e = np.concatenate(z_all)
    err_sd = float(np.std(e))
    z = np.clip(e / err_sd, -4, 4)
    H = smooth_circ(np.bincount(hour[start], minlength=24))
    H = H / H.mean()
    # month weights: starts per eligible hour-of-month exposure is close to uniform hours, so counts suffice
    M = smooth_circ(np.bincount(month[start] - 1, minlength=12))
    M = M / M.mean()
    base = H[hour] * M[month - 1]

    def loglik(beta):
        g = np.exp(beta * z)
        g = g / g[elig].mean()
        w = base * g
        r0 = start.sum() / w[elig].sum()
        p = np.clip(r0 * w, 1e-9, 0.9)
        ll = np.sum(np.where(start, np.log(p), np.log1p(-p))[elig])
        return ll, r0
    grid = np.round(np.arange(-0.2, 1.21, 0.05), 3)
    lls = [loglik(b)[0] for b in grid]
    beta = float(grid[int(np.argmax(lls))])
    ll0 = loglik(0.0)[0]
    _, r0 = loglik(beta)
    durs = np.asarray(durs)
    q = float(1.0 / durs.mean())
    exc = np.asarray(exc)
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    pairs = np.asarray(pairs)
    sc = stats.norm.ppf(np.clip(stats.genpareto.cdf(pairs, xi, scale=sigma), 1e-6, 1 - 1e-6))
    rho = float(np.corrcoef(sc[:, 0], sc[:, 1])[0, 1]) if len(sc) > 2 else 0.0
    return dict(u=U, r0=float(r0), H=[round(float(v), 4) for v in H], M=[round(float(v), 4) for v in M],
                beta=beta, err_sd=round(err_sd, 5), q=round(q, 4), xi=round(float(xi), 4),
                sigma=round(float(sigma), 3), rho=round(rho, 3), x_max=X_MAX), \
        dict(n_events=int(len(durs)), n_event_hours=int(len(exc)), mean_duration_h=round(float(durs.mean()), 3),
             max_duration_h=int(durs.max()), duration_counts={int(k): int(v) for k, v in
                                                            zip(*np.unique(durs, return_counts=True))},
             start_counts_by_hour=np.bincount(hour[start], minlength=24).tolist(),
             start_counts_by_month=np.bincount(month[start] - 1, minlength=12).tolist(),
             loglik_beta=round(float(max(lls)), 2), loglik_beta0=round(float(ll0), 2),
             likelihood_ratio_stat=round(float(2 * (max(lls) - ll0)), 2),
             excess_quantiles={str(qq): round(float(np.quantile(exc, qq)), 1) for qq in (0.25, 0.5, 0.75, 0.9, 0.99)},
             excess_mean=round(float(exc.mean()), 1), n_pairs=int(len(pairs)))


if __name__ == '__main__':
    frames = [year_frame(y) for y in TRAIN]
    p, info = fit(frames)
    out = dict(
        version='1.2', fitted='2026-11-15', fitted_on=f'ERCOT {NODE} RT and DA prices and EIA-930 ERCO demand '
        f'forecast errors of the training-input years {list(TRAIN)} only (splits.json); dev and exam prices unused',
        script='shared/grid/checks/fit_rt_scarcity.py',
        note='CAISO has no fit: its prices start in Aug 2023 (OASIS history limit), so no training-year prices '
             'exist. gridsim.market returns zeros for any region without parameters.',
        regions=dict(ERCO=dict(p, node=NODE, fit=info)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1))
