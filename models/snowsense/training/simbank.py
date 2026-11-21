"""Simulated training banks for SnowSense (WaterSim snow-only physics, model='none').

Forcing: the daily P and T of the DEV SNOTEL stations, water years <= 2010 only (WY2011-2025 of dev stations is the
dev gate). Six dev stations (fixed seed) are held out of training and supply the frozen simulated validation set.
Each run: one station record, a climate perturbation (P x exp(N(0, 0.3)), T + N(0, 1.5) C, elevation moved with the
lapse rate), one snow routine (degree-day or energy balance, 50/50), N parameter sets drawn from the prior of the
physics version, simulated in one numba batch. Windows are 1 Aug .. 1 Oct (427 days).

Physics versions (env SS_PHYS, recorded in each bank):
  p1  WaterSim parameter ranges, uniform (log-uniform for ALB_DECAY, Z0, TURB); the network input = the forcing
      that drives the physics (no measurement error).
  p2  p1 + measurement error between the gauge record the network sees and the forcing the snowpack feels:
      wet-day P x lognormal(0.25), P timing slips of a day, T AR(1) error 0.6 C + station bias N(0, 0.5) C, and
      injected data gaps in the network input (P -> 0, T interpolated, flagged).
  p3  p2 + parameter prior fitted on the dev stations (calib_prior.py -> results/prior_p3.json).
  p4  p3 with the dev-fitted prior used as fitted (inflation 1.3 -> 1.0): R4 on dev was too uncertain (68% bands
      covered 84% of snow days, 86% of peaks, 93% of melt-outs) and low on peak SWE (-11%).
Usage: python3 simbank.py <tag> <n_train_windows> [val]
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from common import *

PHYS = os.environ.get('SS_PHYS', 'p1')
PV = int(PHYS[1:])
NVAL_STATIONS = 6


def dev_split():
    ids = sorted(wd().splits()['snotel']['dev'])
    rng = np.random.default_rng(SEED)
    val = sorted(rng.choice(ids, NVAL_STATIONS, replace=False).tolist())
    return [i for i in ids if i not in val], val


def good_wys(df, last_wy=2010):
    """Water years whose window has >= 85% observed T and P and that end by last_wy."""
    out = []
    if len(df) < 2 * 365:
        return out
    for y in range(df.index[0].year + 2, last_wy + 1):
        s = window_start(y)
        w = df.loc[s:s + pd.Timedelta(days=L - 1)]
        if len(w) < L:
            continue
        if w['mT'].mean() < 0.15 and w['mP'].mean() < 0.15:
            out.append(y)
    return out


# ------------------------------------------------------------------ parameter priors
def _lu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_params(rng, snow):
    if PV >= 3:
        return draw_params_fitted(rng, snow)
    if snow == 'degree_day':
        return dict(TT=rng.uniform(-2.5, 2.5), TTI=rng.uniform(0, 4), CFMAX=rng.uniform(1, 8),
                    SFCF=rng.uniform(0.7, 1.4), CFR=rng.uniform(0, 0.1), CWH=rng.uniform(0, 0.2))
    return dict(TT=rng.uniform(-1.5, 2.5), TTI=rng.uniform(0, 4), SFCF=rng.uniform(0.7, 1.4), CWH=rng.uniform(0, 0.2),
                ALB_MAX=rng.uniform(0.75, 0.92), ALB_MIN=rng.uniform(0.30, 0.60), ALB_DECAY=_lu(rng, 3, 30),
                Z0=_lu(rng, 2e-4, 2e-2), TURB=_lu(rng, 0.2, 3.0), GHEAT=rng.uniform(0, 0.5),
                CANOPY=rng.uniform(0.5, 1.0))


_PRIOR = None


def draw_params_fitted(rng, snow):
    """Dev-fitted prior: multivariate normal in transformed space (results/prior_p3.json), inflated, clipped."""
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = json.load(open(f'{ROOT}/results/prior_{os.environ.get("SS_PRIOR", PHYS)}.json'))
    pr = _PRIOR[snow]
    names = pr['names']
    z = rng.multivariate_normal(np.array(pr['mean']), np.array(pr['cov']) * pr['inflate'] ** 2)
    out = {}
    for n, v, t, lo, hi in zip(names, z, pr['transform'], pr['lo'], pr['hi']):
        v = float(np.exp(v)) if t == 'log' else float(v)
        out[n] = float(np.clip(v, lo, hi))
    for n, (lo, hi) in pr.get('uniform', {}).items():
        out[n] = float(rng.uniform(lo, hi))
    return out


# ------------------------------------------------------------------ measurement error (p2+)
def gauge_error(meas, rng):
    """Return the 'true' forcing the snowpack feels, given the gauge record the network sees (both DataFrames)."""
    tru = meas.copy()
    P = meas['P'].values.copy()
    n = len(P)
    wet = P > 0
    P = P * np.where(wet, np.exp(rng.normal(-0.03, 0.25, n)), 1.0)
    # timing slips: part of a day's total really fell the day before or after
    slip = wet & (rng.random(n) < 0.15)
    for i in np.where(slip)[0]:
        j = i + (1 if rng.random() < 0.5 else -1)
        if 0 <= j < n:
            a = rng.uniform(0.2, 0.8) * P[i]
            P[i] -= a
            P[j] += a
    tru['P'] = P
    e = np.empty(n)
    e[0] = rng.normal(0, 0.6)
    phi = 0.7
    eps = rng.normal(0, 0.6 * np.sqrt(1 - phi ** 2), n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + eps[i]
    e += rng.normal(0, 0.5)
    for c in ('T_mean', 'T_min', 'T_max'):
        tru[c] = meas[c].values + e
    return tru


def inject_gaps(X, rng):
    """In-place: random gap runs in the network input of one window (X: (L, NCH) features)."""
    for ch, mch, rate in ((0, 6, 0.004), (1, 7, 0.006)):
        nruns = rng.poisson(rate * L)
        for _ in range(nruns):
            a = rng.integers(0, L)
            b = min(L, a + int(rng.geometric(1 / 6.0)))
            X[a:b, mch] = 1.0
            if ch == 0:
                X[a:b, 0] = 0.0
            else:
                for c in (1, 2, 3):
                    lo = X[a - 1, c] if a > 0 else X[min(b, L - 1), c]
                    hi = X[b, c] if b < L else lo
                    X[a:b, c] = np.linspace(lo, hi, b - a + 2)[1:-1]
    return X


# ------------------------------------------------------------------ one run
_CACHE = {}


def station_record(sid):
    if sid not in _CACHE:
        m = wd().snotel_meta().set_index('station_id').loc[sid]
        df = load_station(sid, None, '2010-10-01')
        _CACHE[sid] = (df, good_wys(df), float(m.lat), float(m.elev_m))
    return _CACHE[sid]


def run(sid, rng, nparam=4, nwin=6):
    """Simulate one perturbed station record with nparam parameter sets; return windows (X, Y, info)."""
    import watersim as ws
    df, wys, lat, elev = station_record(sid)
    if len(wys) < 2 or len(df) < 3 * 365:
        return [], [], []
    snow = 'degree_day' if rng.random() < 0.5 else 'energy_balance'
    a = float(np.exp(rng.normal(0, 0.3)))
    dT = float(rng.normal(0, 1.5))
    meas = df[['P', 'T_mean', 'T_min', 'T_max', 'mP', 'mT']].copy()
    meas['P'] *= a
    for c in ('T_mean', 'T_min', 'T_max'):
        meas[c] += dT
    elev2 = float(np.clip(elev - dT / 0.0065, 300, 4200))
    lat2 = lat + float(rng.normal(0, 1.0))
    tru = gauge_error(meas, rng) if PV >= 2 else meas
    spec = {'id': sid, 'area_km2': 1.0, 'lat': lat2, 'lon': -110.0, 'elev_mean_m': elev2}
    plist = [draw_params(rng, snow) for _ in range(nparam)]
    f = tru[['P', 'T_mean', 'T_min', 'T_max']]
    r = ws.simulate_batch(spec, f, plist, model='none', snow=snow, start=f.index[0], end=f.index[-1],
                          outputs=('SWE',), pet='hargreaves', spinup='1y')
    swe = pd.DataFrame(r['SWE'].T, index=pd.DatetimeIndex(r['index']))
    X, Y, info = [], [], []
    for k in range(nparam):
        for y in rng.choice(wys, min(nwin, len(wys)), replace=False):
            s = window_start(int(y))
            idx = pd.date_range(s, periods=L, freq='D')
            w = meas.reindex(idx)
            x = features(w.P.values, w.T_mean.values, w.T_min.values, w.T_max.values, w.mP.values, w.mT.values, idx,
                         lat2, elev2)
            if PV >= 2:
                x = inject_gaps(x, rng)
            X.append(x)
            Y.append(swe[k].reindex(idx).values.astype(np.float32))
            info.append((sid, int(y), snow, k))
    return X, Y, info


def build(tag, n_target, stations, seed, nparam=4, nwin=6):
    rng = np.random.default_rng(seed)
    X, Y, info = [], [], []
    t0 = time.time()
    runs = 0
    while len(X) < n_target:
        sid = stations[rng.integers(len(stations))]
        x, y, i = run(sid, rng, nparam, nwin)
        # plausibility filter: no snow left on 1 Aug (perennial packs that grow for years are not SNOTEL-like)
        keep = [j for j in range(len(y)) if np.isfinite(y[j]).all() and y[j][0] <= 5.0]
        x, y, i = [x[j] for j in keep], [y[j] for j in keep], [i[j] for j in keep]
        X += x
        Y += y
        info += [(runs,) + t for t in i]
        runs += 1
        if runs % 100 == 0:
            print(tag, runs, len(X), round(time.time() - t0, 1), flush=True)
    X = np.stack(X[:n_target]).astype(np.float16)
    Y = np.stack(Y[:n_target]).astype(np.float32)
    meta = pd.DataFrame(info[:n_target], columns=['run', 'station', 'wy', 'snow', 'k'])
    return X, Y, meta


if __name__ == '__main__':
    tag, n = sys.argv[1], int(sys.argv[2])
    tr, va = dev_split()
    if len(sys.argv) > 3 and sys.argv[3] == 'val':
        # frozen validation: held-out stations, nparam=2, all good WYs per parameter set -> per-series NSE
        X, Y, meta = build(tag, n, va, SEED + 1000 + PV, nparam=2, nwin=8)
    else:
        X, Y, meta = build(tag, n, tr, SEED + PV * 7 + int(os.environ.get('SS_SEEDOFF', '0')))
    np.savez(f'{ROOT}/banks/{tag}.npz', X=X, Y=Y.astype(np.float16), phys=PHYS)
    meta.to_parquet(f'{ROOT}/banks/{tag}_meta.parquet')
    print('done', tag, X.shape, 'SWE>0 frac', float((Y > 1).mean()), 'peak median', float(np.median(Y.max(1))))
