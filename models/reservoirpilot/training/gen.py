"""Simulated reservoir tasks for ReservoirPilot, built exactly like B's real task.

One task = 20 years of daily inflow from WaterSim (a generated basin run through one of the four runoff models and
one of the two snow routines; real non-exam forcing from a donor basin or the stochastic weather generator), a
reservoir of random size, and a simulated "historical operator" that runs the first 10 years (calibration half).
From that operator's release and storage we take B's task quantities exactly as B takes them from real records:
  demand D(doy) = doy_curve(release), Qmax = 99th percentile of release, S*(doy) = doy_curve(storage),
  inflow climatology Ic(doy) = doy_curve(inflow), start storage S0 = the operator's storage on the first scored day.
The second 10 years are the scored window. Exam dams' inflows are never used; ranges come from dev + aux dams
(data/dam_stats.parquet).

Generator versions (the ratchet's physics axis; each version gets its own frozen validation set):
  g1  WaterSim inflow, optional channel/upstream smoothing, random operator.
  g2  g1 + back-calculation noise on inflow: real ResOpsUS inflow is usually I = R + dS/dt from a noisy storage
      gauge, so I_obs = I + (e_t - e_{t-1})/K with e ~ N(0, sigma_S cap). This gives negative inflow days
      (dev/aux median 2 %, up to 35 %) without changing volumes.
  g3  g2 + upstream regulation (an SOP/rule reservoir upstream reshapes the seasonal inflow) and gross data spikes.
  g4  g3 + non-stationary halves: the scored decade's inflow is the calibration decade's times exp(N(0, 0.3)).
      Found on dev: dev dams' second-half / first-half mean inflow is 0.81 (median; 0.38-0.98 5-95 %), 29 aux dams
      1.17 (0.79-1.72); a stationary generator gives ~1. Demand and climatology come from the first half, so a
      policy that trusts them over-releases in a drier decade (the dev dams where the pilot lost to SOP).
"""
import math, sys, os
import numpy as np, pandas as pd

import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
os.environ.setdefault('NUMBA_CACHE_DIR', _paths.WORK + '/.numba')
import watersim as ws          # noqa: E402
from task import doy_curve, K  # noqa: E402

MODELS = ('hbv', 'gr4j', 'sacsma', 'vic_lite')
SNOWS = ('degree_day', 'energy_balance')
NYEARS = 20


def smooth(x, k):
    """Linear reservoir y' = (x - y)/k (mass conserving), k in days."""
    y = np.empty_like(x); s = x[:30].mean() * k; a = 1.0 / k
    for t in range(len(x)):
        s += x[t] - a * s
        y[t] = a * s
    return y


def make_inflow(rng):
    """20 years of daily inflow (m3/s) at a dam + dates + info."""
    prior = 'camels' if rng.random() < 0.5 else 'default'
    spec = ws.generate_basins(1, rng=int(rng.integers(2**31)), prior=prior, regulation=False)[0]
    model = MODELS[int(rng.integers(4))]; snow = SNOWS[int(rng.integers(2))]
    if prior == 'camels':
        for _ in range(20):        # watersim's window arithmetic can land a few days into the exam period:
            try:                   # its guard raises PermissionError and we redraw (never read exam years)
                df, meta, info = ws.realdata.random_forcing(int(rng.integers(2**31)), years=NYEARS,
                                                            basin_id=spec['donor_id'])
                break
            except PermissionError:
                continue
        forcing = (df, meta); start = info['sim_start']
    else:
        y0 = int(rng.integers(1955, 1995))
        forcing = ws.generate_weather(spec, f'{y0}-01-01', f'{y0 + NYEARS + 1}-01-10', rng=int(rng.integers(2**31)))
        start = f'{y0 + 1}-01-01'
    out = ws.simulate(spec, forcing, start, model=model, snow=snow, routing='uh', labels=False)
    q = out['ts']['Q_m3s'].values[:int(NYEARS * 365.25)].astype(float)
    dates = out['ts'].index[:len(q)]
    info = {'prior': prior, 'model': model, 'snow': snow, 'donor': spec.get('donor_id', ''), 'smooth_k': 0.0}
    # dam catchments are usually larger than CAMELS basins: extra channel / floodplain attenuation
    if rng.random() < 0.5:
        k = float(np.exp(rng.uniform(np.log(1.5), np.log(25))))
        q = smooth(q, k); info['smooth_k'] = k
    q *= 10 ** rng.uniform(-0.5, 2.5) / max(q.mean(), 1e-6)      # arbitrary scale (features are dimensionless)
    return q, dates, info


def upstream_regulation(q, dates, rng):
    """An upstream dam (SOP or rule curve) reshapes the inflow: stores the wet season, releases a seasonal demand."""
    mI = q.mean(); cy = float(np.exp(rng.uniform(np.log(0.05), np.log(1.5))))
    cap = cy * mI * 365.25 * K; S = 0.6 * cap; out = np.empty_like(q)
    doy = dates.dayofyear.values; ph = rng.uniform(0, 365); amp = rng.uniform(0, 0.9); df = rng.uniform(0.6, 1.0)
    for t in range(len(q)):
        d = df * mI * (1 + amp * math.cos(2 * math.pi * (doy[t] - ph) / 365.25))
        avail = S + q[t] * K
        r = min(d, avail / K); s1 = avail - r * K
        if s1 > cap:
            r += (s1 - cap) / K; s1 = cap
        S = s1; out[t] = r
    return out


def operator(I, dates, cap, rng):
    """A simulated historical operator over the calibration half. Returns release, storage (end of day)."""
    n = len(I); mI = max(I.mean(), 1e-9); doy = dates.dayofyear.values
    Ic = pd.Series(I, index=dates).groupby(doy).mean().reindex(range(1, 367)).interpolate(limit_direction='both')
    ipk = int(Ic.rolling(31, center=True, min_periods=1).mean().values.argmax()) + 1
    kind = rng.choice(['follow', 'rule', 'sop', 'hanasaki'], p=[0.3, 0.3, 0.2, 0.2])
    dfrac = 0.12 + 0.9 * rng.beta(2.5, 1.2); amp = rng.uniform(0, 0.95)
    phase = ipk + rng.normal(0, 25) if rng.random() < 0.5 else rng.uniform(150, 240)
    dem = dfrac * mI * (1 + amp * np.cos(2 * np.pi * (doy - phase) / 365.25))
    qcap = mI * float(np.exp(rng.uniform(np.log(1.5), np.log(12))))
    lo, hi = rng.uniform(0.2, 0.6), rng.uniform(0.7, 0.98); dfull = rng.uniform(60, 240); tadj = rng.uniform(5, 60)
    tau = float(np.exp(rng.uniform(0, np.log(40)))); sig = rng.uniform(0.0, 0.35)
    eps = np.zeros(n); e = 0.0
    for t in range(n):
        e = 0.9 * e + math.sqrt(1 - 0.81) * rng.normal(); eps[t] = e
    noise = np.exp(sig * eps - 0.5 * sig ** 2)
    S = rng.uniform(0.2, 0.95) * cap; ema = I[:30].mean()
    R = np.empty(n); Ss = np.empty(n); sy = S
    for t in range(n):
        ema += (I[t] - ema) / tau
        tgt = cap * (lo + (hi - lo) * 0.5 * (1 + math.cos(2 * math.pi * (doy[t] - dfull) / 365.25)))
        if kind == 'follow':
            r = max(ema + (S - tgt) / (tadj * K), 0.0)
        elif kind == 'rule':
            r = max(dem[t] + (S - tgt) / (tadj * K), 0.0)
        elif kind == 'sop':
            r = dem[t]
        else:                                                    # Hanasaki et al. (2006)
            if doy[t] == 274:
                sy = S
            dm = dfrac * mI
            rp = mI / 2 * (1 + dem[t] / dm) if dm >= 0.5 * mI else mI + dem[t] - dm
            kk = sy / (0.85 * cap); c = cap / (mI * 365.25 * K)
            r = kk * rp if c >= 0.5 else (c / 0.5) ** 2 * kk * rp + (1 - (c / 0.5) ** 2) * I[t]
        r = min(r * noise[t], qcap if kind != 'sop' else 1e30)
        avail = S + I[t] * K
        r = min(max(r, 0.0), max(avail, 0) / K); s1 = avail - r * K
        if s1 > cap:
            r += (s1 - cap) / K; s1 = cap
        S = max(s1, 0.0); R[t] = r; Ss[t] = S
    return R, Ss, str(kind)


def backcalc_noise(I, cap, rng, scale):
    e = rng.normal(0, scale * cap, len(I) + 1)
    return I + np.diff(e) / K


def make_task(rng, gen='g1', eval_len=None):
    I, dates, info = make_inflow(rng)
    if gen in ('g3', 'g4') and rng.random() < 0.3:
        I = upstream_regulation(I, dates, rng); info['upstream'] = 1
    n = len(I); nc = n // 2
    if gen == 'g4':                    # non-stationary halves: scored decade wetter/drier than the calibration one
        m = float(np.exp(rng.normal(0.0, 0.3))); I = I.copy(); I[nc:] *= m; info['eval_factor'] = m
    mI = I[:nc].mean()
    cy = float(np.exp(np.clip(rng.normal(np.log(0.8), 1.5), np.log(0.005), np.log(40))))
    cap = cy * mI * 365.25 * K
    Ical = I[:nc].copy(); Iev = I[nc:].copy()
    R, S, kind = operator(Ical, dates[:nc], cap, rng)
    # the operator keeps running into the scored half for one day to give the start storage (B: observed S0)
    S0 = float(S[-1])
    if gen in ('g2', 'g3', 'g4') and rng.random() < 0.5:
        sc = float(np.exp(rng.uniform(np.log(1e-4), np.log(8e-3))))
        Inoisy = backcalc_noise(I, cap, rng, sc); Ical, Iev = Inoisy[:nc], Inoisy[nc:]; info['bc_noise'] = sc
    if gen in ('g3', 'g4') and rng.random() < 0.1:                      # gross data errors (unit slips, spikes)
        j = rng.integers(0, len(Iev), size=int(rng.integers(1, 4)))
        Iev[j] *= 10 ** rng.uniform(1, 3, size=len(j))
    dc, de = dates[:nc], dates[nc:]
    Dc = doy_curve(pd.Series(R, index=dc)); Sc = doy_curve(pd.Series(S, index=dc))
    Ic = doy_curve(pd.Series(Ical, index=dc))
    Qmax = float(np.quantile(R, 0.99))
    if eval_len is not None:
        Iev = Iev[:eval_len]; de = de[:eval_len]
    info.update({'cap_years': cy, 'operator': kind, 'Qmax_over_I': Qmax / mI, 'D_over_I': float(Dc.mean()) / mI})
    return {'I': Iev.astype(np.float64), 'doy0': int(de[0].dayofyear), 'dates': de, 'Dcurve': Dc.values,
            'Scurve': Sc.values, 'Icurve': Ic.values, 'Qmax': Qmax, 'cap': cap, 'S0': S0,
            'D': Dc.reindex(de.dayofyear).values, 'Ss': Sc.reindex(de.dayofyear).values, 'info': info}


def task_stats(tk):
    """Same dimensionless statistics as dam_stats.py (eval half), for the generator-vs-real comparison."""
    I = tk['I']; mI = max(np.mean(tk['Icurve']), 1e-9); Dc = tk['Dcurve']; Ic = tk['Icurve']
    return {'cap_years': tk['cap'] / (mI * 365.25 * K), 'qmax_over_I': tk['Qmax'] / mI, 'D_over_I': Dc.mean() / mI,
            'D_seas': (Dc.max() - Dc.min()) / max(Dc.mean(), 1e-9), 'I_seas': (Ic.max() - Ic.min()) / max(Ic.mean(), 1e-9),
            'I_cv': np.std(I) / max(np.mean(I), 1e-9), 'I_ac1': np.corrcoef(I[:-1], I[1:])[0, 1],
            'neg_frac': np.mean(I < 0), 'I_p99_over_mean': np.quantile(I, 0.99) / max(np.mean(I), 1e-9),
            'S0_frac': tk['S0'] / tk['cap'], 'Dc_I_corr': np.corrcoef(Dc, Ic)[0, 1]}
