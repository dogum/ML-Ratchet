"""Simulated long-term water-balance bank from WaterSim (LawFinder, law 1).

One sample = one generated basin (prior 'camels' or 'default', 50/50), one runoff model (hbv/gr4j/sacsma/vic_lite),
one snow routine, a real ERA5-Land forcing segment from a random CAMELS-US basin in a FREE window
(1 spin-up year + 10 scored years), simulated with WaterSim. The law sees only what a real basin offers:
the forcing after WaterSim's forcing-error model, and attribute-based estimates.

Physics versions (phys/<ver>.json):
  w1  WaterSim as shipped (Phase 2 recommended mix, default forcing error, no water-yield multiplier)
  w2  w1 + FlowCast's dev-fitted ERA5-Land water-yield multiplier (true P = ERA5 x m; the law still sees ERA5)

usage: python3 water_bank.py <ver> <train|val> <n> [seed]   (resumable shards of 250)
"""
import sys, os, json, math, time
import numpy as np, pandas as pd
import watersim as ws, waterdata as wd
from watersim import realdata, noise
from watersim.labels import CAPACITY_LABEL

from _paths import ROOT  # the work folder (training/_paths.py)
SHARD = 250
MODELS = ['hbv', 'gr4j', 'sacsma', 'vic_lite']
SNOWS = ['degree_day', 'energy_balance']
YEARS = 10
PHYS = {
    'w1': {'note': 'WaterSim v1 as shipped: priors 50/50, 4 models, 2 snow routines, reservoirs/irrigation from the '
                   'generator, default forcing_error on what the law sees, no water-yield multiplier',
           'undercatch': None},
    'w2': {'note': 'w1 + ERA5-Land water-yield multiplier fitted on the 52 dev basins by FlowCast (calib_water.py): '
                   'log m = -0.088 + 0.058 elev_km + 0.153 frac_snow + 0.397 log(p_mean/3) + N(0, 0.146)',
           'undercatch': {'a': -0.088, 'b_elev': 0.058, 'b_snow': 0.153, 'b_wet': 0.397, 'sd': 0.146,
                          'lo': 0.4, 'hi': 2.5}},
}
FREE = [('1951-01-01', '1988-09-30'), ('1999-10-01', '2015-09-30')]


def features(P, PET, T, dates):
    """Law inputs from a forcing record: identical code for simulated and real basins."""
    P = np.asarray(P, float); PET = np.asarray(PET, float); T = np.asarray(T, float)
    ps = np.nansum(P)
    m = pd.DatetimeIndex(dates).month
    mp = np.array([np.nanmean(P[m == k]) for k in range(1, 13)])
    me = np.array([np.nanmean(PET[m == k]) for k in range(1, 13)])
    sync = float(np.corrcoef(mp, me)[0, 1]) if mp.std() > 0 and me.std() > 0 else 0.0
    return {'aridity': float(np.nansum(PET) / ps), 'frac_snow': float(np.nansum(P[T < 0]) / ps),
            'sync': sync, 'p_mean': float(np.nanmean(P)), 'pet_mean': float(np.nanmean(PET)),
            'p_cv_month': float(mp.std() / mp.mean())}


def soil_proxy(spec):
    """The simulator's attribute-prior soil capacity (mm), averaged over the four runoff models (no latent
    scatter): the same estimate is available for a real basin."""
    v = []
    for mm in MODELS:
        p = ws.params_from_attributes(spec, mm, 'none', None, None)
        v.append(float(CAPACITY_LABEL[mm](p)))
    return float(np.exp(np.mean(np.log(np.maximum(v, 1.0)))))


def free_segment(rng):
    w = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in FREE]
    L = np.array([(b - a).days - (YEARS + 1) * 366 for a, b in w], float)
    k = int(rng.choice(len(w), p=L / L.sum()))
    a, b = w[k]
    s0 = a + pd.Timedelta(days=int(rng.integers(0, int(L[k]))))
    s1 = s0 + pd.DateOffset(years=YEARS + 1) - pd.Timedelta(days=1)
    realdata.check_window(str(s0.date()), str(s1.date()))
    return s0, s1


def p_multiplier(u, elev_m, fs, pm, rng):
    if not u:
        return 1.0
    z = u['a'] + u['b_elev'] * elev_m / 1000 + u['b_snow'] * fs + u['b_wet'] * math.log(pm / 3.0) + rng.normal(0, u['sd'])
    return float(np.clip(math.exp(z), u['lo'], u['hi']))


def make_sample(rng, phys, donors, df_full, meta):
    prior = 'camels' if rng.random() < 0.5 else 'default'
    spec = ws.generate_basins(1, rng=int(rng.integers(2**31)), prior=prior, attributes=donors)[0]
    s0, s1 = free_segment(rng)
    df = df_full.loc[s0:s1].copy()
    if len(df) < (YEARS + 1) * 365 or df[['P', 'T_mean', 'PET']].isna().mean().max() > 0.02:
        return None
    df = df.interpolate(limit_direction='both')
    spec['lat'], spec['lon'] = float(meta['lat']), float(meta['lon'])
    sc = df.iloc[365:]
    f_true = features(sc['P'].values, sc['PET'].values, sc['T_mean'].values, sc.index)
    m = p_multiplier(phys['undercatch'], spec.get('elev_mean_m', 500.0), f_true['frac_snow'], f_true['p_mean'], rng)
    df_sim = df.copy()
    df_sim['P'] = df_sim['P'] * m
    spec['p_mean'] = f_true['p_mean'] * m
    spec['aridity'] = f_true['pet_mean'] / spec['p_mean']
    spec['frac_snow'] = f_true['frac_snow']
    lat = spec['latent']
    prm = {mm: ws.params_from_attributes(spec, mm, 'none', lat, rng) for mm in MODELS}
    prm['snow_dd'] = ws.params_from_attributes(spec, 'none', 'degree_day', lat, rng)
    prm['snow_eb'] = ws.params_from_attributes(spec, 'none', 'energy_balance', lat, rng)
    prm['routing'] = spec['params']['routing']
    spec['params'] = prm
    model = MODELS[int(rng.integers(4))]
    snow = SNOWS[int(rng.integers(2))]
    start = sc.index[0]
    o = ws.simulate(spec, (df_sim, meta), str(start.date()), str(df.index[-1].date()), model=model, snow=snow,
                    routing='uh', labels=True, rng=int(rng.integers(2**31)))
    ts = o['ts']
    q = ts['Q'].values
    if not np.isfinite(q).all():
        return None
    # what the law sees: ERA5 forcing after the forcing-error model
    dfn = noise.forcing_error(sc, rng=int(rng.integers(2**31)))
    f = features(dfn['P'].values, dfn['PET'].values, dfn['T_mean'].values, dfn.index)
    lab = o['labels']
    rec = dict(f)
    rec.update({'rr': float(q.sum() / dfn['P'].sum()), 'rr_true': float(q.sum() / ts['P_forcing'].sum()),
                'soil_proxy': soil_proxy(spec), 'soil_true': lab['soil_capacity_mm'],
                'model': MODELS.index(model), 'snow': SNOWS.index(snow), 'prior': int(prior == 'camels'),
                'pmult': m, 'reg': int(bool(spec.get('reservoirs'))), 'irr': float(spec.get('irrigation_frac', 0) or 0),
                'forest': float(spec.get('forest_frac', np.nan) or np.nan), 'bfi': lab.get('bfi_lyne_hollick', np.nan),
                'k_slow': lab.get('k_slow_true', np.nan)})
    return rec


def build(ver, split, n, seed=1):
    phys = PHYS[ver]
    out = f'{ROOT}/banks/water_{ver}'
    os.makedirs(out, exist_ok=True)
    json.dump({'version': ver, **phys}, open(f'{out}/phys.json', 'w'), indent=1)
    donors = realdata.donor_table()
    ids = wd.basins('camels')
    nsh = int(math.ceil(n / SHARD))
    for k in range(nsh):
        fn = f'{out}/{split}_{k:03d}.parquet'
        if os.path.exists(fn):
            continue
        rng = np.random.default_rng([seed, 0 if split == 'train' else 1, k])
        t0 = time.time(); recs = []
        while len(recs) < SHARD:
            fb = ids[int(rng.integers(len(ids)))]
            df_full = wd.forcing(fb, '1951-01-01', '2015-09-30')
            meta = wd.forcing_meta(fb)
            for _ in range(4):
                try:
                    r = make_sample(rng, phys, donors, df_full, meta)
                except Exception as e:      # rare generator corner cases: log and skip
                    print('skip', repr(e)[:120], flush=True); r = None
                if r is not None:
                    r['fbid'] = fb; recs.append(r)
        pd.DataFrame(recs[:SHARD]).to_parquet(fn + '.tmp')
        os.replace(fn + '.tmp', fn)
        print(f'{ver} {split} shard {k+1}/{nsh} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ver, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    build(ver, split, n, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
