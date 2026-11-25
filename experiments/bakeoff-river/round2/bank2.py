"""Round-2 bank builder: FlowCast's p1 recipe (harness/build_bank.py, unchanged) plus the b2 physics options.

usage: python bank2.py build <name> [--workers 6] [--splits val,train]   shard-parallel, resumable
       python bank2.py finalize <name>                                     merge shards -> .npy + MANIFEST.json
       python bank2.py selftest                                            b2 code with every option off == p1 code

<name> is a recipe in round2/phys/<name>.json. Bank 'p1' calls harness/build_bank.make_sample itself, so it is the
frozen v1.0 bank byte for byte (MANIFEST sha256 compared on finalize). Other banks use make_sample_b2 below: a line-for-
line copy of make_sample plus three options, each a separate key of the recipe's "b2" block:

  reservoirs  (rung a) {"consistent": true, "operator": "rp"}:
              consistent: every run whose reservoir_storage_frac attribute is > 0 gets reservoirs. In p1 the camels
                prior copies the donor's storage attribute but draws dams with WaterSim's own 30 % coin, so 70 % of the
                runs whose attribute says "regulated" are simulated without a reservoir.
              operator "rp": the reservoir is run by ReservoirPilot's simulated historical operator
                (models/reservoirpilot/training/gen.py `operator`: follow / rule curve / SOP / Hanasaki with a seasonal
                demand, a release cap and operator noise, ranges fitted to dev + aux ResOpsUS dams) instead of
                WaterSim's default (demand 0.6 x mean inflow, no release cap). Storage = the basin's attribute.
  undercatch  (rung b) the water-yield prior coefficients (same functional form as p1; values from calib_prior.py)
  mosaic      (rung c) {"share": 0.5, "f_lo": 0.2, "f_hi": 0.8}: with probability share the basin is two
              sub-areas run by two different runoff models (fraction f ~ U(f_lo, f_hi) and 1 - f; same forcing, snow
              routine, routing and parameter draws); outlet flow and states are the area-weighted mix.

The main random stream is untouched by the options: they draw from auxiliary generators seeded by the run's sim_seed,
so every bank shares p1's basins, forcing windows, model choices and noise seeds run for run (paired banks).
Gauge noise for a changed flow uses noise.streamflow_obs with default_rng(sim_seed), exactly as simulate() does.

Output: round2/work/banks/<name>/{val,train}/ with the harness's array files (+ 'mosaic' (n, 2): model B index or -1,
f) and specs_<shard>.jsonl.gz, then MANIFEST.json. The harness's bake.load_bank reads it after r2.use_bank(name).
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import r2
import numpy as np
import pandas as pd

sys.path.insert(0, r2.HARNESS)
import build_bank as BB  # noqa: E402  (the v1.0 recipe, used unchanged)
import config as C  # noqa: E402
import waterdata as wd  # noqa: E402
import watersim as ws  # noqa: E402
from watersim import realdata, noise  # noqa: E402
from features import climate_stats, spec_attrs, static_vector, forcing_array, attr_vector  # noqa: E402

NDAYS = BB.NDAYS
SHARD = BB.SHARD
MODELS, SNOWS = BB.MODELS, BB.SNOWS
K_HM3 = 1.0   # the operator below works in basin mm/day and mm (capacity = years x mean inflow x 365.25)


def recipe(name):
    if name == 'p1':
        return dict(BB.PHYS)
    return json.load(open(os.path.join(r2.PHYS, f'{name}.json')))


# ============================================================================ rung (a): reservoir operator
def rp_operator(I, doy, cap, rng):
    """ReservoirPilot's simulated historical operator (gen.py `operator`, same draws and ranges), in basin mm/day.
    I: inflow to the reservoir (mm/day of the whole basin); cap: capacity (mm). Returns release R, storage S."""
    n = len(I); mI = max(float(I.mean()), 1e-9)
    Ic = pd.Series(I).groupby(doy).mean().reindex(range(1, 367)).interpolate(limit_direction='both')
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
    nz = np.exp(sig * eps - 0.5 * sig ** 2)
    S = rng.uniform(0.2, 0.95) * cap; ema = I[:30].mean()
    R = np.empty(n); Ss = np.empty(n); sy = S
    for t in range(n):
        ema += (I[t] - ema) / tau
        tgt = cap * (lo + (hi - lo) * 0.5 * (1 + math.cos(2 * math.pi * (doy[t] - dfull) / 365.25)))
        if kind == 'follow':
            r = max(ema + (S - tgt) / (tadj * K_HM3), 0.0)
        elif kind == 'rule':
            r = max(dem[t] + (S - tgt) / (tadj * K_HM3), 0.0)
        elif kind == 'sop':
            r = dem[t]
        else:
            if doy[t] == 274:
                sy = S
            dm = dfrac * mI
            rp = mI / 2 * (1 + dem[t] / dm) if dm >= 0.5 * mI else mI + dem[t] - dm
            kk = sy / (0.85 * cap); c = cap / (mI * 365.25 * K_HM3)
            r = kk * rp if c >= 0.5 else (c / 0.5) ** 2 * kk * rp + (1 - (c / 0.5) ** 2) * I[t]
        r = min(r * nz[t], qcap if kind != 'sop' else 1e30)
        avail = S + I[t] * K_HM3
        r = min(max(r, 0.0), max(avail, 0) / K_HM3); s1 = avail - r * K_HM3
        if s1 > cap:
            r += (s1 - cap) / K_HM3; s1 = cap
        S = max(s1, 0.0); R[t] = r; Ss[t] = S
    info = {'kind': str(kind), 'dfrac': float(dfrac), 'qcap_over_I': float(qcap / mI), 'sig': float(sig)}
    return R, Ss, dem, info


def apply_reservoir(q_nat, doy, spec, opt, rng):
    """Outlet flow after the operated reservoir(s): bypass (1 - up) x q_nat + release - consumptive withdrawal."""
    res = spec['reservoirs']
    up = float(res[-1]['upstream_frac'])            # the terminal reservoir's share of the basin
    cy = float(spec['reservoir_storage_frac'])      # storage in years of mean flow = the attribute the network sees
    I = up * q_nat
    cap = max(cy * max(float(I.mean()), 1e-6) * 365.25, 1e-6)
    R, Ss, dem, info = rp_operator(I, doy, cap, rng)
    withdraw = bool(res[-1]['policy_params'].get('withdraw', False))
    cons = float(res[-1].get('consumptive_frac', 0.6))
    W = np.minimum(R, dem) * cons if withdraw else np.zeros_like(R)
    q = (1.0 - up) * q_nat + R - W
    info.update({'up': up, 'cap_years': cy, 'withdraw': withdraw, 'consumptive_frac': cons})
    return q, Ss, info


# ============================================================================ the b2 sample (copy of make_sample)
def make_sample_b2(rng, phys, donors, df_full, meta, fbid):
    b2 = phys.get('b2') or {}
    prior = 'camels' if rng.random() < phys.get('camels_share', 0.5) else 'default'
    spec = ws.generate_basins(1, rng=int(rng.integers(2**31)), prior=prior, attributes=donors)[0]
    s0, s1 = BB.free_segment(rng)
    df = df_full.loc[s0:s1].copy()
    if len(df) != 365 + NDAYS or df[['P', 'T_mean', 'PET']].isna().mean().max() > 0.02:
        return None
    df = df.interpolate(limit_direction='both')
    spec['lat'], spec['lon'] = float(meta['lat']), float(meta['lon'])
    F_true = forcing_array(df)
    clim_true = climate_stats(F_true[:, 0], F_true[:, 1], F_true[:, 2])
    attrs = spec_attrs(spec)
    m = BB.p_multiplier(phys, attrs, clim_true, rng)
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
    # ---- b2 options (auxiliary generators only; the main stream continues exactly as in p1) ----
    ropt, mopt = b2.get('reservoirs'), b2.get('mosaic')
    extra = {}
    if ropt and ropt.get('consistent') and not spec.get('reservoirs') and float(spec.get('reservoir_storage_frac') or 0) > 0:
        ra = np.random.default_rng([sim_seed, 10])
        a2 = dict(spec)
        a2['dam_count'] = max(int(spec.get('dam_count') or 0), 1)
        spec['reservoirs'] = ws.generator._random_reservoirs(a2, ra)
        extra['reservoirs_added'] = True
    op_res = bool(ropt and ropt.get('operator') == 'rp' and spec.get('reservoirs'))
    mB, fA = None, 1.0
    if mopt:
        rm = np.random.default_rng([sim_seed, 12])
        if rm.random() < mopt.get('share', 0.5):
            others = [x for x in MODELS if x != model]
            mB = others[int(rm.integers(len(others)))]
            fA = float(rm.uniform(mopt.get('f_lo', 0.2), mopt.get('f_hi', 0.8)))
    changed = op_res or mB is not None
    kw = dict(snow=snow, routing=routing, labels=False, rng=sim_seed)
    if op_res:
        kw['reservoirs'] = []
    o = ws.simulate(spec, (df_sim, meta), str(start.date()), str(df.index[-1].date()), model=model,
                    obs_noise=(on if on else True) if not changed else None, **kw)
    ts = o['ts']
    q = ts['Q'].values.astype(np.float64)
    stA = np.stack([ts['SWE'].values, ts['soil'].values, ts['groundwater'].values,
                    ts['routing_store'].values, ts['channel_storage'].values], 1)
    states = stA
    if mB is not None:
        oB = ws.simulate(spec, (df_sim, meta), str(start.date()), str(df.index[-1].date()), model=mB, obs_noise=None,
                         **kw)
        tB = oB['ts']
        stB = np.stack([tB['SWE'].values, tB['soil'].values, tB['groundwater'].values,
                        tB['routing_store'].values, tB['channel_storage'].values], 1)
        q = fA * q + (1 - fA) * tB['Q'].values.astype(np.float64)
        states = fA * stA + (1 - fA) * stB
        extra['mosaic'] = {'model_b': mB, 'f_a': fA}
    if op_res:
        rr = np.random.default_rng([sim_seed, 11])
        q, _, info = apply_reservoir(q, ts.index.dayofyear.values, spec, ropt, rr)
        extra['operator'] = info
    if changed:
        qo = noise.streamflow_obs(q, np.random.default_rng(sim_seed), **(on if on else {})).astype(np.float32)
    else:
        qo = ts['Q_obs'].values.astype(np.float32)
    q = q.astype(np.float32)
    if not np.isfinite(q).all() or len(q) != NDAYS or not np.isfinite(qo).all():
        return None
    fe = phys.get('forcing_error', {})
    fe_seed = int(rng.integers(2**31))
    dfn = noise.forcing_error(df.iloc[365:], rng=fe_seed, **fe)
    if phys.get('p_jitter', 0) > 0:
        raise NotImplementedError('p_jitter is not part of p1')
    F = forcing_array(dfn)
    clim = climate_stats(F[:NDAYS - 1, 0], F[:NDAYS - 1, 1], F[:NDAYS - 1, 2])
    st = static_vector(attrs, clim)
    states = states.astype(np.float32)
    th = []
    for mm in MODELS:
        th += [float(prm[mm][p[0]]) for p in BB.MODULES[mm].PARAMS]
    th += [float(prm['snow_dd'][p[0]]) for p in BB.SN.DD_PARAMS] + [float(prm['snow_eb'][p[0]]) for p in BB.SN.EB_PARAMS]
    th += [float(prm['routing']['CH_VEL']), float(prm['routing']['CH_N'])]
    th += [float(lat[k]) for k in ('storage', 'perm', 'quick', 'snow')] + [m]
    runspec = {'spec': BB._jsonable(spec), 'model': model, 'snow': snow, 'routing': routing, 'start': str(start.date()),
               'end': str(df.index[-1].date()), 'seg0': str(df.index[0].date()), 'fbid': fbid, 'pmult': m,
               'sim_seed': sim_seed, 'forcing_error_seed': fe_seed, 'obs_noise': on if on else True,
               'forcing_error': fe, 'b2': BB._jsonable(extra)}
    return {'F': F.astype(np.float16), 'q': q, 'qo': qo, 'st': st, 'doy0': int(start.dayofyear),
            'start': str(start.date()), 'model': MODELS.index(model), 'snow': SNOWS.index(snow),
            'prior': int(prior == 'camels'), 'routing': int(routing == 'muskingum'), 'pmult': m,
            'reg': int(bool(spec.get('reservoirs'))), 'fbid': fbid, 'states': states,
            'theta': np.array(th, np.float32), 'attrs': attr_vector(attrs), 'lat': spec['lat'], 'lon': spec['lon'],
            'mosaic': np.array([MODELS.index(mB) if mB else -1, fA], np.float32), 'runspec': runspec}


ARRAYS = dict(BB.ARRAYS)
ARRAYS_B2 = dict(ARRAYS, mosaic=((2,), np.float32))


# ============================================================================ shards
def split_n_seed(split):
    return BB.split_n_seed(split)


def shard_path(name, split, k):
    return os.path.join(r2.bank_dir(name), split, 'shards', f'shard_{k:03d}.npz')


def build_shard(name, split, k):
    phys = recipe(name)
    fn = shard_path(name, split, k)
    if os.path.exists(fn):
        return
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    n, seed = split_n_seed(split)
    donors = realdata.donor_table()
    ids = wd.basins('camels')
    maker = BB.make_sample if name == 'p1' else make_sample_b2
    rng = np.random.default_rng([seed, 0 if split == 'train' else 1, k])
    t0 = time.time()
    recs = []
    while len(recs) < SHARD:
        fb = ids[int(rng.integers(len(ids)))]
        df_full = wd.forcing(fb, '1951-01-01', '2015-09-30')
        meta = wd.forcing_meta(fb)
        for _ in range(4):
            r = maker(rng, phys, donors, df_full, meta, fb)
            if r is not None:
                recs.append(r)
    recs = recs[:min(SHARD, n - k * SHARD)]
    arrs = ARRAYS if name == 'p1' else ARRAYS_B2
    out = {}
    for key, (shp, dt) in arrs.items():
        out[key] = np.stack([np.asarray(r[key]) for r in recs]).astype(dt) if shp else \
            np.array([r[key] for r in recs], dtype=dt)
    np.savez(fn + '.tmp.npz', **out)
    with gzip.open(os.path.join(os.path.dirname(fn), f'specs_{k:03d}.jsonl.gz.tmp'), 'wt') as f:
        for r in recs:
            f.write(json.dumps(r['runspec']) + '\n')
    os.replace(os.path.join(os.path.dirname(fn), f'specs_{k:03d}.jsonl.gz.tmp'),
               os.path.join(os.path.dirname(fn), f'specs_{k:03d}.jsonl.gz'))
    os.replace(fn + '.tmp.npz', fn)
    print(f'{name} {split} shard {k} {time.time() - t0:.0f}s', flush=True)


def build(name, workers, splits):
    """Run shards in `workers` subprocesses (one thread each)."""
    todo = []
    for split in splits:
        n, _ = split_n_seed(split)
        for k in range(int(math.ceil(n / SHARD))):
            if not os.path.exists(shard_path(name, split, k)):
                todo.append((split, k))
    groups = [todo[i::workers] for i in range(workers)]
    procs = []
    log = os.path.join(r2.bank_dir(name), 'build_logs')
    os.makedirs(log, exist_ok=True)
    for i, g in enumerate(groups):
        if not g:
            continue
        arg = ','.join(f'{s}:{k}' for s, k in g)
        procs.append(subprocess.Popen([sys.executable, '-X', 'utf8', os.path.abspath(__file__), 'shards', name, arg],
                                      stdout=open(os.path.join(log, f'w{i}.log'), 'a'), stderr=subprocess.STDOUT))
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f'shard workers failed: {rc}')


def finalize(name):
    phys = recipe(name)
    arrs = ARRAYS if name == 'p1' else ARRAYS_B2
    man = {'what': f'River Bake-off round-2 bank {name}' + (' (= the frozen v1.0 bank, FlowCast p1)' if name == 'p1' else
                                                             ' (p1 recipe + b2 options, round2/bank2.py)'),
           'physics': name, 'phys_json': phys, 'watersim_version': ws.__version__,
           'watersim_source': 'shared/water/watersim (repo)', 'seeds': {'train': C.SEED_TRAIN, 'val': C.SEED_VAL},
           'ndays': NDAYS, 'dyn': C.DYN, 'attr_names': C.ATTR_NAMES, 'static_names': C.STATIC_NAMES,
           'theta_cols': BB.THETA_COLS, 'state_names': BB.STATE_NAMES, 'models': MODELS, 'snows': SNOWS,
           'routings': BB.ROUTINGS, 'splits': {}}
    for split in ('val', 'train'):
        n, seed = split_n_seed(split)
        nsh = int(math.ceil(n / SHARD))
        d = os.path.join(r2.bank_dir(name), split)
        parts = [np.load(shard_path(name, split, k)) for k in range(nsh)]
        files = {}
        for key in arrs:
            a = np.concatenate([p[key] for p in parts])
            assert len(a) == n, (key, len(a))
            fn = os.path.join(d, f'{key}.npy')
            np.save(fn, a)
            h = hashlib.sha256(open(fn, 'rb').read()).hexdigest()
            files[key] = {'sha256': h, 'bytes': os.path.getsize(fn)}
        for k in range(nsh):
            src = os.path.join(d, 'shards', f'specs_{k:03d}.jsonl.gz')
            dst = os.path.join(d, f'specs_{k:03d}.jsonl.gz')
            open(dst, 'wb').write(open(src, 'rb').read())
        man['splits'][split] = {'n': n, 'seed': seed, 'shards': nsh, 'files': files}
    if name == 'p1':
        ref = json.load(open(os.path.join(r2.HARNESS, 'bank', 'MANIFEST.json')))
        cmp = {s: {k: files_k['sha256'] == ref['splits'][s]['files'][k]['sha256']
                   for k, files_k in man['splits'][s]['files'].items()} for s in ('val', 'train')}
        man['matches_v1_manifest'] = cmp
        print('p1 vs v1.0 MANIFEST:', json.dumps(cmp))
    json.dump(man, open(os.path.join(r2.bank_dir(name), 'MANIFEST.json'), 'w'), indent=1, default=float)
    print('finalized', name)


def selftest():
    """make_sample_b2 with no b2 options must equal build_bank.make_sample (same rng stream, same arrays)."""
    donors = realdata.donor_table(); ids = wd.basins('camels')
    phys = dict(BB.PHYS)
    for seed in (5, 6):
        r1 = np.random.default_rng([99, seed]); r2_ = np.random.default_rng([99, seed])
        fb = ids[seed * 7]
        df_full = wd.forcing(fb, '1951-01-01', '2015-09-30'); meta = wd.forcing_meta(fb)
        a = BB.make_sample(r1, phys, donors, df_full, meta, fb)
        b = make_sample_b2(r2_, phys, donors, df_full, meta, fb)
        if a is None:
            assert b is None
            continue
        for k in ARRAYS:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
        assert r1.integers(2**31) == r2_.integers(2**31)
    print('selftest OK: b2 code with every option off reproduces p1 exactly')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('name', nargs='?')
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--splits', default='val,train')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.name, a.workers, a.splits.split(','))
        finalize(a.name)
    elif a.cmd == 'shards':
        for item in a.arg.split(','):
            s, k = item.split(':')
            build_shard(a.name, s, int(k))
    elif a.cmd == 'finalize':
        finalize(a.name)
    elif a.cmd == 'selftest':
        selftest()
