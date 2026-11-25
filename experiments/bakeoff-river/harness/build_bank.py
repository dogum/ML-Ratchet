"""Build the frozen simulated bank (FlowCast's released recipe, physics p1) with per-run labels.

usage: python3 build_bank.py val|train      (resumable: shards of 400 runs; finished shards are skipped)
       python3 build_bank.py finalize       (checks every shard, writes bank/MANIFEST.json with sha256 of each array)

Recipe = models/flowcast/training/src/simbank.py (make_sample + build) copied line for line, with the same seeds
(val: seed 7, train: seed 1; rng = default_rng([seed, 0|1, shard])), so the val set reproduces FlowCast's frozen p1
set. The only change: extra fields are recorded (true parameters, daily states, the run's spec), none of which draws
from the random generator. WaterSim = vendor/watersim (a byte copy of the repo's shared/water/watersim, VENDOR.json).

Stored per split under bank/<split>/ as .npy (memory-mappable), rows = runs:
  F       (n, 1462, 8) float16  forcing the network sees: ERA5-Land through WaterSim's forcing-error model (DYN order)
  qo      (n, 1462)    float32  target: gauge-like observed flow (mm/day) = Q through the streamflow_obs noise model
  q       (n, 1462)    float32  true simulated outlet flow (mm/day)
  states  (n, 1462, 5) float32  swe, soil, groundwater, routing_store (the model's own UH buffer; can be slightly
                                negative when a component is), channel_storage; mm, end of day, as WaterSim reports
  attrs   (n, 11)      float32  basin attributes (config.ATTR_NAMES, sim units)
  st      (n, 18)      float32  FlowCast static vector (attributes + climate indices of F over the 4 stored years)
  theta   (n, K)       float32  true parameters: every model's draw, both snow routines, routing, latent, pmult
  meta    (n, ...)     model, snow, routing, prior, reg, doy0, start (str), fbid (str), lat, lon, pmult
plus specs_<shard>.jsonl.gz (the full WaterSim spec + run settings, enough to re-run WaterSim exactly).
"""
import gzip
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402  (sets env + vendor path)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import waterdata as wd  # noqa: E402
import watersim as ws  # noqa: E402
from watersim import realdata, noise  # noqa: E402
from watersim.simulate import MODULES  # noqa: E402
from watersim import snow as SN  # noqa: E402
from features import climate_stats, spec_attrs, static_vector, forcing_array, attr_vector  # noqa: E402

NDAYS = C.NDAYS
SHARD = C.SHARD
MODELS = ['hbv', 'gr4j', 'sacsma', 'vic_lite']
SNOWS = ['degree_day', 'energy_balance']
ROUTINGS = ['uh', 'muskingum']
PHYS = json.load(open(os.path.join(C.FLOWCAST_DIR, 'training', 'phys', f'{C.PHYS}.json')))
FREE = C.FREE

# parameter table layout (theta columns)
THETA_COLS = ([f'{m}.{p[0]}' for m in MODELS for p in MODULES[m].PARAMS]
              + [f'snow_dd.{p[0]}' for p in SN.DD_PARAMS] + [f'snow_eb.{p[0]}' for p in SN.EB_PARAMS]
              + ['routing.CH_VEL', 'routing.CH_N', 'latent.storage', 'latent.perm', 'latent.quick', 'latent.snow',
                 'pmult'])
STATE_NAMES = ['swe', 'soil', 'groundwater', 'routing_store', 'channel_storage']


def free_segment(rng, years=5):
    w = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in FREE]
    L = np.array([(b - a).days - years * 366 for a, b in w], float)
    k = int(rng.choice(len(w), p=L / L.sum()))
    a, b = w[k]
    s0 = a + pd.Timedelta(days=int(rng.integers(0, int(L[k]))))
    s1 = s0 + pd.Timedelta(days=365 + NDAYS - 1)
    realdata.check_window(str(s0.date()), str(s1.date()))
    return s0, s1


def p_multiplier(phys, attrs, clim, rng):
    u = phys.get('undercatch')
    if not u:
        return 1.0
    z = (u['a'] + u['b_elev'] * (attrs['elev_mean_m'] / 1000.0) + u['b_snow'] * clim['frac_snow']
         + u['b_wet'] * math.log(clim['p_mean'] / 3.0) + rng.normal(0, u['sd']))
    return float(np.clip(math.exp(z), u.get('lo', 0.6), u.get('hi', 2.5)))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def make_sample(rng, phys, donors, df_full, meta, fbid):
    # ---- identical to FlowCast simbank.make_sample (p1: no shrink_total, no p_jitter) ----
    prior = 'camels' if rng.random() < phys.get('camels_share', 0.5) else 'default'
    spec = ws.generate_basins(1, rng=int(rng.integers(2**31)), prior=prior, attributes=donors)[0]
    s0, s1 = free_segment(rng)
    df = df_full.loc[s0:s1].copy()
    if len(df) != 365 + NDAYS or df[['P', 'T_mean', 'PET']].isna().mean().max() > 0.02:
        return None
    df = df.interpolate(limit_direction='both')
    spec['lat'], spec['lon'] = float(meta['lat']), float(meta['lon'])
    F_true = forcing_array(df)
    clim_true = climate_stats(F_true[:, 0], F_true[:, 1], F_true[:, 2])
    attrs = spec_attrs(spec)
    m = p_multiplier(phys, attrs, clim_true, rng)
    df_sim = df.copy()
    if m != 1.0:
        df_sim['P'] = df_sim['P'] * m
    spec['p_mean'] = clim_true['p_mean'] * m
    spec['aridity'] = clim_true['pet_mean'] / spec['p_mean']
    spec['frac_snow'] = clim_true['frac_snow']
    lat = spec['latent']
    prm = {}
    for mm in MODELS:
        prm[mm] = ws.params_from_attributes(spec, mm, 'none', lat, rng)
        if phys.get('shrink_total', 0.5) != 0.5:
            raise NotImplementedError('shrink_total is not part of p1')
    prm['snow_dd'] = ws.params_from_attributes(spec, 'none', 'degree_day', lat, rng)
    prm['snow_eb'] = ws.params_from_attributes(spec, 'none', 'energy_balance', lat, rng)
    prm['routing'] = spec['params']['routing']
    spec['params'] = prm
    model = MODELS[int(rng.integers(4))]
    snow = SNOWS[int(rng.integers(2))]
    routing = 'muskingum' if rng.random() < phys.get('muskingum_share', 0.25) else 'uh'
    start = df.index[365]
    on = phys.get('obs_noise', {})
    sim_seed = int(rng.integers(2**31))
    o = ws.simulate(spec, (df_sim, meta), str(start.date()), str(df.index[-1].date()), model=model, snow=snow,
                    routing=routing, labels=False, rng=sim_seed, obs_noise=on if on else True)
    ts = o['ts']
    q = ts['Q'].values.astype(np.float32)
    qo = ts['Q_obs'].values.astype(np.float32)
    if not np.isfinite(q).all() or len(q) != NDAYS:
        return None
    fe = phys.get('forcing_error', {})
    fe_seed = int(rng.integers(2**31))
    dfn = noise.forcing_error(df.iloc[365:], rng=fe_seed, **fe)
    if phys.get('p_jitter', 0) > 0:
        raise NotImplementedError('p_jitter is not part of p1')
    F = forcing_array(dfn)
    clim = climate_stats(F[:NDAYS - 1, 0], F[:NDAYS - 1, 1], F[:NDAYS - 1, 2])
    st = static_vector(attrs, clim)
    # ---- extras (no random draws below this line) ----
    states = np.stack([ts['SWE'].values, ts['soil'].values, ts['groundwater'].values,
                       ts['routing_store'].values, ts['channel_storage'].values], 1).astype(np.float32)
    th = []
    for mm in MODELS:
        th += [float(prm[mm][p[0]]) for p in MODULES[mm].PARAMS]
    th += [float(prm['snow_dd'][p[0]]) for p in SN.DD_PARAMS] + [float(prm['snow_eb'][p[0]]) for p in SN.EB_PARAMS]
    th += [float(prm['routing']['CH_VEL']), float(prm['routing']['CH_N'])]
    th += [float(lat[k]) for k in ('storage', 'perm', 'quick', 'snow')] + [m]
    runspec = {'spec': _jsonable(spec), 'model': model, 'snow': snow, 'routing': routing, 'start': str(start.date()),
               'end': str(df.index[-1].date()), 'seg0': str(df.index[0].date()), 'fbid': fbid, 'pmult': m,
               'sim_seed': sim_seed, 'forcing_error_seed': fe_seed, 'obs_noise': on if on else True,
               'forcing_error': fe}
    return {'F': F.astype(np.float16), 'q': q, 'qo': qo, 'st': st, 'doy0': int(start.dayofyear),
            'start': str(start.date()), 'model': MODELS.index(model), 'snow': SNOWS.index(snow),
            'prior': int(prior == 'camels'), 'routing': int(routing == 'muskingum'), 'pmult': m,
            'reg': int(bool(spec.get('reservoirs'))), 'fbid': fbid, 'states': states,
            'theta': np.array(th, np.float32), 'attrs': attr_vector(attrs), 'lat': spec['lat'], 'lon': spec['lon'],
            'runspec': runspec}


ARRAYS = {  # name: (trailing shape, dtype)
    'F': ((NDAYS, 8), np.float16), 'qo': ((NDAYS,), np.float32), 'q': ((NDAYS,), np.float32),
    'states': ((NDAYS, 5), np.float32), 'attrs': ((11,), np.float32), 'st': ((18,), np.float32),
    'theta': ((len(THETA_COLS),), np.float32), 'doy0': ((), np.int16), 'model': ((), np.int8),
    'snow': ((), np.int8), 'routing': ((), np.int8), 'prior': ((), np.int8), 'reg': ((), np.int8),
    'pmult': ((), np.float32), 'lat': ((), np.float32), 'lon': ((), np.float32),
    'start': ((), '<U10'), 'fbid': ((), '<U15')}


def split_n_seed(split):
    return (C.N_TRAIN, C.SEED_TRAIN) if split == 'train' else (C.N_VAL, C.SEED_VAL)


def open_arrays(split, mode):
    n, _ = split_n_seed(split)
    d = os.path.join(C.BANK, split)
    os.makedirs(d, exist_ok=True)
    out = {}
    for k, (shp, dt) in ARRAYS.items():
        fn = os.path.join(d, f'{k}.npy')
        if mode == 'w' and not os.path.exists(fn):
            out[k] = np.lib.format.open_memmap(fn, mode='w+', dtype=dt, shape=(n,) + shp)
        else:
            out[k] = np.load(fn, mmap_mode='r+' if mode == 'w' else 'r')
    return out


def progress(split):
    fn = os.path.join(C.BANK, split, 'progress.json')
    return json.load(open(fn)) if os.path.exists(fn) else {'done': {}}


def build(split):
    n, seed = split_n_seed(split)
    donors = realdata.donor_table()
    ids = wd.basins('camels')
    nsh = int(math.ceil(n / SHARD))
    arr = open_arrays(split, 'w')
    pr = progress(split)
    for k in range(nsh):
        if str(k) in pr['done']:
            continue
        rng = np.random.default_rng([seed, 0 if split == 'train' else 1, k])
        t0 = time.time()
        recs = []
        while len(recs) < SHARD:
            fb = ids[int(rng.integers(len(ids)))]
            df_full = wd.forcing(fb, '1951-01-01', '2015-09-30')
            meta = wd.forcing_meta(fb)
            for _ in range(4):
                r = make_sample(rng, PHYS, donors, df_full, meta, fb)
                if r is not None:
                    recs.append(r)
        recs = recs[:min(SHARD, n - k * SHARD)]
        a = k * SHARD
        for key in ARRAYS:
            arr[key][a:a + len(recs)] = np.stack([np.asarray(r[key]) for r in recs]) if ARRAYS[key][0] else \
                np.array([r[key] for r in recs], dtype=ARRAYS[key][1])
        for key in ARRAYS:
            arr[key].flush()
        with gzip.open(os.path.join(C.BANK, split, f'specs_{k:03d}.jsonl.gz'), 'wt') as f:
            for r in recs:
                f.write(json.dumps(r['runspec']) + '\n')
        pr['done'][str(k)] = {'rows': [a, a + len(recs)], 'seconds': round(time.time() - t0, 1),
                              'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        tmp = os.path.join(C.BANK, split, 'progress.json.tmp')
        json.dump(pr, open(tmp, 'w'), indent=1)
        os.replace(tmp, os.path.join(C.BANK, split, 'progress.json'))
        print(f'{split} shard {k + 1}/{nsh} {time.time() - t0:.0f}s', flush=True)


def finalize():
    import hashlib
    man = {'what': 'River Bake-off frozen simulated bank (FlowCast released recipe, physics p1)',
           'physics': C.PHYS, 'phys_json': PHYS, 'watersim_version': ws.__version__,
           'watersim_source': 'vendor/watersim = byte copy of mlr-repo shared/water/watersim/watersim (VENDOR.json)',
           'seeds': {'train': C.SEED_TRAIN, 'val': C.SEED_VAL}, 'ndays': NDAYS, 'dyn': C.DYN,
           'attr_names': C.ATTR_NAMES, 'static_names': C.STATIC_NAMES, 'theta_cols': THETA_COLS,
           'state_names': STATE_NAMES, 'models': MODELS, 'snows': SNOWS, 'routings': ROUTINGS, 'splits': {}}
    for split in ('val', 'train'):
        n, seed = split_n_seed(split)
        pr = progress(split)
        nsh = int(math.ceil(n / SHARD))
        assert len(pr['done']) == nsh, f'{split}: {len(pr["done"])}/{nsh} shards done'
        files = {}
        for k in ARRAYS:
            fn = os.path.join(C.BANK, split, f'{k}.npy')
            h = hashlib.sha256()
            with open(fn, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 24), b''):
                    h.update(chunk)
            files[k] = {'sha256': h.hexdigest(), 'bytes': os.path.getsize(fn)}
        A = open_arrays(split, 'r')
        assert np.isfinite(A['qo'][:]).all() and np.isfinite(np.asarray(A['F'][:, :, 0], np.float32)).all()
        secs = sum(v['seconds'] for v in pr['done'].values())
        man['splits'][split] = {'n': n, 'seed': seed, 'shards': nsh, 'build_seconds': round(secs, 1), 'files': files}
    tmp = os.path.join(C.BANK, 'MANIFEST.json.tmp')
    json.dump(man, open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(C.BANK, 'MANIFEST.json'))
    print(json.dumps({s: {k: v for k, v in d.items() if k != 'files'} for s, d in man['splits'].items()}))


if __name__ == '__main__':
    if sys.argv[1] == 'finalize':
        finalize()
    else:
        build(sys.argv[1])
