"""Simulated training bank for BasinDetective.

Each simulated basin gets 1 + 10 + 9 years of daily forcing laid out like the real exam:
    [spin-up 1 y][TARGET period 10 y: forcing only][CALIBRATION period 9 y: forcing + Q]
The network sees calibration forcing + noisy calibration Q + target forcing ("what the forcing product shows").
Labels: the target-period change of six signatures (computed from noisy simulated Q with the same code as the real
truth), hidden parameters (slow-store time constant, groundwater share, soil capacity, precipitation multiplier) and
regulation (reservoir storage / mean annual flow).

Physics versions (PHYS):
  v1  WaterSim generator, three forcing sources (camels-prior donor forcing / default prior on a random CAMELS
      basin's free-window forcing / stochastic weather), all four models, both snow routines; the TRUE forcing that
      drives the simulator differs from the SEEN forcing by a persistent multiplier exp(N(0, 0.08)), daily noise
      (cv 0.25), extra snowfall U(0, 0.1) and a temperature offset N(0, 0.7 C); target-period climate perturbation
      in half the basins; gauge noise (watersim.noise.streamflow_obs) and gaps.
  v2  dev-measured gauge records and precipitation bias: gauge noise as fitted on dev twins (calib_gauge.py:
      AR(1) rho 0.999, cv 0.02, 3 significant figures in cfs) and log multiplier ~ N(mu0 + mu_snow fsnow, sd)
      fitted on dev (calib_pbias.py).
  v4  v3 + year-to-year forcing precipitation error (calib_annual.py): TRUE P x exp(e_y), e_y ~ N(0, 0.10) per
      water year, fitted on dev twins to the real annual flow-vs-precipitation correlation.
  v3  v2 + decadal drift of the forcing precipitation (calib_drift.py): TRUE target-period P = SEEN / exp(delta),
      delta ~ N(mu, sd) measured on dev twins.
  v3s, v4s (27 Nov 2026, audit follow-up): v3 / v4 with a symmetric precipitation-change prior, delta ~ N(0, sd)
      (the same sd 0.038; mu 0 instead of 0.029). The draw uses the same generator call, so every other random
      choice is the same as in v3 / v4: the banks are matched basin by basin.

Usage: python3 simbank.py <shard> [<shard> ...]   (shards of 250 basins; >= 900 = frozen validation)
"""
import common  # noqa: F401
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import watersim as ws
from watersim.generator import params_from_attributes, _random_reservoirs
from watersim.labels import CAPACITY_LABEL
import waterdata as wd

from common import D
from features import featurize, sigs, delta_labels

PHYS_TAG = os.environ.get('PHYS', 'v1')       # bank folder name
SYM = PHYS_TAG in ('v3s', 'v4s')                # symmetric drift prior (27 Nov 2026)
PHYS = PHYS_TAG[:-1] if SYM else PHYS_TAG       # the physics version the rest of this file switches on
NPER = 250
SPIN, TGT, CALY = 1, 10, 9
MODELS = ['hbv', 'gr4j', 'sacsma', 'vic_lite']
SNOWS = ['degree_day', 'energy_balance']
ALL_IDS = None
DONORS = None


def pbias_params():
    if PHYS == 'v1':
        return {'mu0': 0.0, 'mu_snow': 0.0, 'sd': 0.08, 'snow_extra': 0.1, 'tsd': 0.7}
    return json.load(open(f'{D}/models/pbias_v2.json'))


def gauge_params():
    if PHYS == 'v1':
        return {'rho': 0.9, 'cv': 0.08, 'sig': None}
    return json.load(open(f'{D}/models/gauge_v2.json'))


def annual_sd():
    if PHYS in ('v1', 'v2', 'v3'):
        return 0.0
    return json.load(open(f'{D}/models/annual_v4.json'))['sd']


def drift_params():
    if PHYS in ('v1', 'v2'):
        return None
    dr = json.load(open(f'{D}/models/drift_v3.json'))
    if SYM:
        dr = dict(dr, mu=0.0)
    return dr


def _window(rng):
    """Random 20-year (+1 spin-up) window inside the free window 1951-01-01..1988-09-30."""
    start_min = pd.Timestamp('1951-01-01')
    last = pd.Timestamp('1988-09-30')
    span = SPIN + TGT + CALY
    latest = last - pd.DateOffset(years=span) + pd.Timedelta(days=1)
    y0 = int(rng.integers(start_min.year + 1, latest.year + 1))
    s0 = pd.Timestamp(f'{y0 - 1}-10-01')
    if s0 < start_min:
        s0 = pd.Timestamp(f'{y0}-10-01')
    s1 = s0 + pd.DateOffset(years=span) - pd.Timedelta(days=1)
    return s0, s1


def _regen_params(spec, rng):
    lat = spec['latent']
    p = {}
    for m in MODELS:
        p[m] = params_from_attributes(spec, m, 'none', lat, rng)
    p['snow_dd'] = params_from_attributes(spec, 'none', 'degree_day', lat, rng)
    p['snow_eb'] = params_from_attributes(spec, 'none', 'energy_balance', lat, rng)
    p['routing'] = spec['params']['routing']
    spec['params'] = p


def one_basin(rng, k):
    global ALL_IDS, DONORS
    if ALL_IDS is None:
        ALL_IDS = wd.basins('camels')
        DONORS = ws.realdata.donor_table()
    u = rng.random()
    src = 'camels' if u < 0.40 else ('realforcing' if u < 0.75 else 'weather')
    if src == 'camels':
        spec = ws.generate_basins(1, rng=rng, prior='camels', attributes=DONORS)[0]
        s0, s1 = _window(rng)
        df, meta = ws.realdata.nonexam_forcing(spec['donor_id'], str(s0.date()), str(s1.date()))
        fid = spec['donor_id']
    else:
        spec = ws.generate_basins(1, rng=rng, prior='default')[0]
        s0, s1 = _window(rng)
        if src == 'realforcing':
            fid = ALL_IDS[int(rng.integers(len(ALL_IDS)))]
            df, meta = ws.realdata.nonexam_forcing(fid, str(s0.date()), str(s1.date()))
            for kk in ('lat', 'lon', 'elev_mean_m', 'elev_min_m', 'elev_max_m'):
                if meta.get(kk) is not None:
                    spec[kk] = float(meta[kk])
            spec['elev_bands'] = meta.get('elev_bands')
            spec['p_mean'] = float(df.P.mean())
            spec['aridity'] = float(df.PET.mean() / max(df.P.mean(), 1e-3))
            _regen_params(spec, rng)
            if spec.get('reservoirs'):
                spec['reservoirs'] = _random_reservoirs(spec, rng)
        else:
            fid = 'weather'
            df = ws.generate_weather(spec, str(s0.date()), str(s1.date()), rng=rng)
            meta = None
    spec['id'] = f'sim_{k}'
    df = df[[c for c in ['P', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind', 'PET', 'LRAD_net', 'pressure']
             if c in df.columns]].copy()
    idx = df.index
    t0 = s0 + pd.DateOffset(years=SPIN)
    c0 = t0 + pd.DateOffset(years=TGT)
    tgt_mask = idx < c0
    # climate perturbation of the target period (real decadal change is in the forcing already; widen it)
    pscale, tshift = 1.0, 0.0
    if rng.random() < 0.5:
        pscale = float(rng.uniform(0.88, 1.12))
        tshift = float(rng.uniform(-1.0, 1.0))
        df.loc[tgt_mask, 'P'] *= pscale
        for c in ('T_mean', 'T_min', 'T_max'):
            df.loc[tgt_mask, c] += tshift
        if 'PET' in df:
            df.loc[tgt_mask, 'PET'] *= (1 + 0.025 * tshift)
    seen = df.copy()
    # TRUE forcing the basin really got: precipitation multiplier, daily error, snow undercatch, T offset
    pb = pbias_params()
    cold = (seen['T_mean'].values < 0).astype(float)
    fsn = float((seen.P.values * cold).sum() / max(seen.P.values.sum(), 1e-9))
    mu = pb['mu0'] + pb['mu_snow'] * fsn
    m = float(np.exp(rng.normal(mu, pb['sd'])))
    dcv = 0.25
    dn = np.exp(rng.normal(0, dcv, len(df)) - 0.5 * dcv ** 2)
    sx = float(rng.uniform(0, pb['snow_extra']))
    true = seen.copy()
    true['P'] = seen.P.values * m * dn * (1 + sx * cold)
    asd = annual_sd()
    if asd > 0:
        from features import water_year
        wyv = water_year(true.index)
        uy = np.unique(wyv)
        ey = dict(zip(uy, rng.normal(0, asd, uy.size)))
        true['P'] = true.P.values * np.exp(np.array([ey[y] for y in wyv]))
    dr = drift_params()
    delta = 0.0
    if dr is not None:
        delta = float(rng.normal(dr['mu'], dr['sd']))
        true.loc[tgt_mask, 'P'] = true.loc[tgt_mask, 'P'] * np.exp(-delta)
    dT = float(rng.normal(0, pb['tsd']))
    for c in ('T_mean', 'T_min', 'T_max'):
        true[c] = seen[c] + dT
    model = MODELS[int(rng.integers(4))]
    snow = SNOWS[int(rng.integers(2))]
    f_in = (true, meta) if meta is not None else true
    out = ws.simulate(spec, f_in, str(t0.date()), str(s1.date()), model=model, snow=snow, routing='uh',
                      spinup='1y', labels=False)
    ts = out['ts']
    Q = ts['Q'].values
    gp = gauge_params()
    gf = float(rng.uniform(0, 0.03)) if rng.random() < 0.6 else 0.0
    if PHYS == 'v1':
        qobs = ws.noise.streamflow_obs(Q, rng, gap_frac=gf)
    else:
        from calib_gauge import gauge
        qobs = gauge(Q, rng, float(out['spec']['area_km2']), gp['rho'], gp['cv'], gp['sig'], gf)
    qobs = pd.Series(qobs, index=ts.index)
    pet_seen = ts['PET'].values if 'PET' not in seen else seen['PET'].reindex(ts.index).values
    sv = pd.DataFrame({'P': seen.P.reindex(ts.index).values, 'T_mean': seen.T_mean.reindex(ts.index).values,
                       'PET': pet_seen}, index=ts.index)
    cal = sv[sv.index >= c0]
    tgt = sv[sv.index < c0]
    x, s_cal, names = featurize(cal, qobs[qobs.index >= c0].values, tgt)
    qt = qobs[qobs.index < c0]
    s_tgt = sigs(qt, tgt.P, tgt.PET, tgt.T_mean)
    lab = delta_labels(s_cal, s_tgt)
    # hidden parameters (calibration period)
    pm = out['params']['model']
    cm = ts.index >= c0
    qn = ts['Q_natural'].values[cm]
    lab['gw_share'] = float(ts['Q_base_routed'].values[cm].sum() / max(qn.sum(), 1e-9))
    if model == 'hbv':
        ks = pm['K2']
    elif model == 'sacsma':
        ks = pm['LZPK']
    elif model == 'gr4j':
        R = max(float(ts['groundwater'].values[cm].mean()), 1e-6)
        X3 = pm['X3']
        ks = float(1 - (1 + (R / X3) ** 4) ** -0.25 + (R / X3) ** 4 * (1 + (R / X3) ** 4) ** -1.25)
    else:
        ks = pm['DS'] * pm['DSMAX'] / (pm['WS'] * pm['W2M'])
    lab['log_kslow'] = float(np.log(1.0 / max(ks, 1e-5)))
    lab['log_soilcap'] = float(np.log(max(CAPACITY_LABEL[model](pm), 1.0)))
    lab['log_pmult'] = float(np.log(true.P.values[(true.index >= c0)].sum() / max(seen.P.values[seen.index >= c0].sum(), 1e-9)))
    reg = out.get('reservoirs') or []
    if reg:
        cap = sum(r['reservoir'].C for r in reg)
        vol = ts['Q_natural'].mean() * 365.25 * float(out['spec']['area_km2']) / 1000.0
        dor = float(cap / max(vol, 1e-9))
    else:
        dor = 0.0
    lab['dor'] = dor
    lab['log_dor'] = float(np.log10(dor)) if dor > 0 else np.nan
    lab['reg'] = float(dor > 0.1)
    # extras for diagnostics
    for kk in ('q_mean', 'runoff_ratio', 'bfi_lh', 'q5', 'q95', 'hfd_mean', 'rec_k_days', 'slope_fdc'):
        lab[f'tgt_{kk}'] = s_tgt.get(kk, np.nan)
        lab[f'cal_{kk}'] = s_cal.get(kk, np.nan)
    info = {'src': src, 'forcing_id': fid, 'model': model, 'snow': snow, 'pscale': pscale, 'tshift': tshift,
            'm': m, 'dT': dT, 'delta': delta, 'start': str(s0.date())}
    return x, names, lab, info


def run_shard(shard):
    path = f'{D}/bank/{PHYS_TAG}/shard_{shard:04d}.npz'
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng([20261118, shard, 7])
    X, L, I = [], [], []
    names = None
    t = time.time()
    for k in range(NPER):
        try:
            x, names, lab, info = one_basin(rng, shard * NPER + k)
        except Exception as e:  # keep going; log the failure
            print('fail', shard, k, repr(e)[:200], flush=True)
            traceback.print_exc(limit=1)
            continue
        X.append(x)
        L.append(lab)
        I.append(info)
    lab = pd.DataFrame(L)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, X=np.array(X, np.float32), names=np.array(names),
                        labels=lab.to_records(index=False), info=np.array([json.dumps(i) for i in I]))
    os.replace(tmp, path)
    print(f'shard {shard} done: {len(X)} basins, {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    for a in sys.argv[1:]:
        if '-' in a:
            lo, hi = map(int, a.split('-'))
            for s in range(lo, hi + 1):
                run_shard(s)
        else:
            run_shard(int(a))
