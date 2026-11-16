"""Fit GridSim physics 1.2's real-time scarcity process (gridsim.market) on the TRAINING-INPUT years only.

Data: ERCOT HB_HUBAVG hourly RT (mean of the four 15-min SPPs) and DA settlement point prices, and EIA-930 ERCO
demand with its day-ahead forecast, for 2018, 2020 and 2022 (splits.json: real inputs, prices included, may be used
from train_inputs years). Dev years (2019, 2021) and exam years are not read here; check_rt_scarcity.py checks the
fit on dev afterwards.

An event is a run of hours with RT - DA > u = $100/MWh. Fitted:
  H[24], M[12]   start counts by local hour and month (pooled, +0.5 each, circular [1, 2, 1] smoothing, mean 1)
  beta           effect of the standardised demand-forecast miss z on the start rate (Bernoulli likelihood)
  gamma_rank,    effect of the DA price's rank within its local day and of its level against the median of the
  gamma_level    previous 7 days (gridsim.market.da_shape); ERCO and ERCO_shape_only
  delta_level    size term: GPD scale sigma exp(delta_level level) (joint MLE with xi and sigma); ERCO only
  r0             base start rate, so that the expected starts equal the observed ones
  q              1 / mean event duration (geometric)
  xi, sigma      generalised Pareto fit of the hourly excess RT - DA - u in event hours (scipy, loc 0)
  rho            lag-1 correlation of the normal scores of that excess inside events (Gaussian copula)
Writes gridsim/gridsim/data/rt_scarcity_v1_2.json with three fits, all on the training-input years only:
  ERCO_first_fit  H, M, beta (beta on a 0.05 grid). BatteryPilot's twin test on dev 2019 then showed what it misses:
                  real unforecast spikes fall in the hours the DA already marks as the day's dearest (the rolling DA
                  optimiser, discharging in those hours, gained $47.8k/MW from hours with RT - DA > $100 in real
                  2019; on twins with this fit only $9.6-11.0k/MW), while this fit placed them by clock hour only.
  ERCO_shape_only M, beta, gamma_rank, gamma_level (Nelder-Mead), with the hour profile H or without it (H = 1),
                  whichever has the lower AIC on the training years (H counted as 23 parameters). The same twin test
                  showed the optimiser still gained only $9.3-10.7k/MW in spike hours with it: on the training years
                  the excess RT - DA - $100 also grows with the DA level (mean $117 in the lowest quartile of level
                  among event hours, $756 in the highest; corr of log excess with level 0.41).
  ERCO            ERCO_shape_only plus the size term delta_level. The default.

usage: python shared/grid/checks/fit_rt_scarcity.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

GRID_CODE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, f"{GRID_CODE}/gridsim")
sys.path.insert(0, GRID_CODE)
import griddata as gd  # noqa: E402
from gridsim.market import da_shape  # noqa: E402

TRAIN = (2018, 2020, 2022)
U = 100.0
X_MAX = 9000.0 - U
NODE = 'HB_HUBAVG'
OUT = os.path.join(GRID_CODE, 'gridsim', 'gridsim', 'data', 'rt_scarcity_v1_2.json')


def year_frame(year, node=NODE):
    """Hourly frame (UTC) of one year: rt, da, demand error fraction e = (d - df) / df, local hour and month, and the
    DA shape (rank within the local day, level against the previous 7 days)."""
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
    f = f.dropna(subset=['rt', 'da'])
    f['rank'], f['level'] = da_shape(f.index, f.da.values, 'ERCO')
    return f


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


def fit_size(exc, lev):
    """GPD with scale sigma exp(delta level): MLE of (xi, sigma, delta), and the AIC against delta = 0."""
    def nll(th):
        xi, ls, d = th
        sc = np.exp(ls + d * lev)
        return -np.sum(stats.genpareto.logpdf(exc, xi, scale=sc))
    xi0, _, s0 = stats.genpareto.fit(exc, floc=0)
    res = optimize.minimize(nll, x0=[xi0, np.log(s0), 0.0], method='Nelder-Mead',
                            options=dict(xatol=1e-5, fatol=1e-5, maxiter=4000))
    ll0 = float(np.sum(stats.genpareto.logpdf(exc, xi0, scale=s0)))
    return dict(xi=round(float(res.x[0]), 4), sigma=round(float(np.exp(res.x[1])), 3),
                delta_level=round(float(res.x[2]), 4), loglik=round(float(-res.fun), 2),
                loglik_no_size_term=round(ll0, 2), aic=round(float(6 + 2 * res.fun), 2),
                aic_no_size_term=round(float(4 - 2 * ll0), 2))


def fit(frames, shape=False, size=False):
    """shape=False: the first fit (H, M, beta).  shape=True: M, beta, gamma_rank, gamma_level, with the hour profile
    H or without it (H = 1), whichever has the lower AIC."""
    hs, ms, starts, eligible, z_all, durs, exc, pairs, rk, lv, elv = [], [], [], [], [], [], [], [], [], [], []
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
        z_all.append(f.e.values); rk.append(f['rank'].values); lv.append(f['level'].values)
        durs += list(L)
        for a, n in zip(st, L):
            x = s[a:a + n] - U
            exc += list(x)
            elv += list(f['level'].values[a:a + n])
            pairs += [(x[k], x[k + 1]) for k in range(n - 1)]
    hour = np.concatenate(hs); month = np.concatenate(ms); start = np.concatenate(starts)
    elig = np.concatenate(eligible); e = np.concatenate(z_all)
    rank = np.concatenate(rk); level = np.concatenate(lv)
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
    extra = {}
    if shape:
        def nll(th, Hv):
            b, a, c = th
            w = Hv[hour] * M[month - 1] * np.exp(b * z + a * (rank - 0.5) + c * level)
            r = start.sum() / w[elig].sum()
            pp = np.clip(r * w, 1e-9, 0.9)
            return -np.sum(np.where(start, np.log(pp), np.log1p(-pp))[elig])
        variants = {}
        for name, Hv, k in (('with_H', H, 3 + 23), ('flat_H', np.ones(24), 3)):
            res = optimize.minimize(lambda th: nll(th, Hv), x0=[beta, 0.0, 0.0], method='Nelder-Mead',
                                    options=dict(xatol=1e-4, fatol=1e-4, maxiter=4000))
            variants[name] = dict(beta=round(float(res.x[0]), 4), gamma_rank=round(float(res.x[1]), 4),
                                  gamma_level=round(float(res.x[2]), 4), loglik=round(float(-res.fun), 2),
                                  aic=round(float(2 * k + 2 * res.fun), 2))
        pick = min(variants, key=lambda n: variants[n]['aic'])
        v = variants[pick]
        beta = v['beta']
        if pick == 'flat_H':
            H = np.ones(24)
        # r0 for the normalised factors gridsim.market uses (each factor divided by its mean over the series)
        g = np.exp(beta * z); g = g / g[elig].mean()
        kf = np.exp(v['gamma_rank'] * (rank - 0.5) + v['gamma_level'] * level); kf = kf / kf.mean()
        r0 = float(start.sum() / (H[hour] * M[month - 1] * g * kf)[elig].sum())
        extra = dict(gamma_rank=v['gamma_rank'], gamma_level=v['gamma_level'])
        shape_info = dict(variants=variants, picked=pick,
                          first_fit=dict(loglik=round(float(max(lls)), 2),
                                         aic=round(float(2 * (1 + 23) - 2 * max(lls)), 2)))
    durs = np.asarray(durs)
    q = float(1.0 / durs.mean())
    exc = np.asarray(exc)
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    pairs = np.asarray(pairs)
    size_extra = {}
    if size:
        sf = fit_size(exc, np.asarray(elv))
        xi, sigma = sf['xi'], sf['sigma']
        size_extra = dict(delta_level=sf['delta_level'])
    sc = stats.norm.ppf(np.clip(stats.genpareto.cdf(pairs, xi, scale=sigma), 1e-6, 1 - 1e-6))
    rho = float(np.corrcoef(sc[:, 0], sc[:, 1])[0, 1]) if len(sc) > 2 else 0.0
    p = dict(u=U, r0=float(r0), H=[round(float(v), 4) for v in H], M=[round(float(v), 4) for v in M],
             beta=beta, err_sd=round(err_sd, 5), q=round(q, 4), xi=round(float(xi), 4),
             sigma=round(float(sigma), 3), rho=round(rho, 3), x_max=X_MAX, **extra, **size_extra)
    info = dict(n_events=int(len(durs)), n_event_hours=int(len(exc)), mean_duration_h=round(float(durs.mean()), 3),
                max_duration_h=int(durs.max()), duration_counts={int(k): int(v) for k, v in
                                                               zip(*np.unique(durs, return_counts=True))},
                start_counts_by_hour=np.bincount(hour[start], minlength=24).tolist(),
                start_counts_by_month=np.bincount(month[start] - 1, minlength=12).tolist(),
                loglik_beta=round(float(max(lls)), 2), loglik_beta0=round(float(ll0), 2),
                likelihood_ratio_stat=round(float(2 * (max(lls) - ll0)), 2),
                excess_quantiles={str(qq): round(float(np.quantile(exc, qq)), 1)
                                  for qq in (0.25, 0.5, 0.75, 0.9, 0.99)},
                excess_mean=round(float(exc.mean()), 1), n_pairs=int(len(pairs)))
    if shape:
        info['shape_fit'] = shape_info
    if size:
        info['size_fit'] = sf
    return p, info


if __name__ == '__main__':
    frames = [year_frame(y) for y in TRAIN]
    p0, info0 = fit(frames, shape=False)
    p2, info2 = fit(frames, shape=True)
    p, info = fit(frames, shape=True, size=True)
    out = dict(
        version='1.2', fitted='2026-11-15', fitted_on=f'ERCOT {NODE} RT and DA prices and EIA-930 ERCO demand '
        f'forecast errors of the training-input years {list(TRAIN)} only (splits.json); dev and exam prices unused',
        script='shared/grid/checks/fit_rt_scarcity.py',
        note='CAISO has no fit: its prices start in Aug 2023 (OASIS history limit), so no training-year prices '
             'exist. gridsim.market returns zeros for any region without parameters.',
        regions=dict(ERCO=dict(p, node=NODE, fit=info),
                     ERCO_shape_only=dict(p2, node=NODE, fit=info2,
                                          note='the second 1.2 fit (DA-shape start term, no size term); no bank '
                                               'was built with it'),
                     ERCO_first_fit=dict(p0, node=NODE, fit=info0,
                                         note='the first 1.2 fit (no DA-shape term, clock-hour profile); '
                                              'BatteryPilot frozen set v5 was built with it')))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({k: p[k] for k in ('r0', 'beta', 'gamma_rank', 'gamma_level', 'xi', 'sigma', 'delta_level',
                                        'rho')}), json.dumps(info['shape_fit']), json.dumps(info['size_fit']))
