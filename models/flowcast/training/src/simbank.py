"""Simulated training/validation banks for FlowCast.

Each sample = one generated basin (prior 'camels' or 'default', random latent factors), one runoff model
(hbv/gr4j/sacsma/vic_lite), one snow routine, a real ERA5-Land forcing segment from a random CAMELS-US basin in a
FREE window (5 years: 1 spin-up + 4 stored + 1 day), simulated with WaterSim. The network input is the forcing
after a forcing-error model; the target is gauge-like Q_obs. Physics knobs are versioned in phys/<ver>.json.

usage: python3 simbank.py <phys_ver> <split: train|val> <n_samples> [seed]
Resumable: shards of 400 samples, existing shards are skipped.
"""
import sys, os, time, json, math
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import waterdata as wd
import watersim as ws
from watersim import realdata, noise
from calib_flash import extra_correction
from calib_noise import jitter_p

NDAYS = 1462            # 4 years + 1 day (for the time-zone shift)
SHARD = 400
MODELS = ['hbv', 'gr4j', 'sacsma', 'vic_lite']
SNOWS = ['degree_day', 'energy_balance']


def load_phys(ver):
    with open(f'{PHYS}/{ver}.json') as f:
        return json.load(f)


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
    """Systematic ERA5-Land precipitation error: the basin's TRUE precipitation is ERA5 x m (the network still
    sees ERA5). Version p0: m = 1."""
    u = phys.get('undercatch')
    if not u:
        return 1.0
    z = (u['a'] + u['b_elev'] * (attrs['elev_mean_m'] / 1000.0) + u['b_snow'] * clim['frac_snow']
         + u['b_wet'] * math.log(clim['p_mean'] / 3.0) + rng.normal(0, u['sd']))
    return float(np.clip(math.exp(z), u.get('lo', 0.6), u.get('hi', 2.5)))


def make_sample(rng, phys, donors, df_full, meta, fbid):
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
    # climate fields of the spec follow the forcing the basin really gets; params regenerated from them
    spec['p_mean'] = clim_true['p_mean'] * m
    spec['aridity'] = clim_true['pet_mean'] / spec['p_mean']
    spec['frac_snow'] = clim_true['frac_snow']
    lat = spec['latent']
    prm = {}
    for mm in MODELS:
        prm[mm] = ws.params_from_attributes(spec, mm, 'none', lat, rng)
        if phys.get('shrink_total', 0.5) != 0.5:
            prm[mm] = extra_correction(prm[mm], mm, phys['shrink_total'])
    prm['snow_dd'] = ws.params_from_attributes(spec, 'none', 'degree_day', lat, rng)
    prm['snow_eb'] = ws.params_from_attributes(spec, 'none', 'energy_balance', lat, rng)
    prm['routing'] = spec['params']['routing']
    spec['params'] = prm
    model = MODELS[int(rng.integers(4))]
    snow = SNOWS[int(rng.integers(2))]
    routing = 'muskingum' if rng.random() < phys.get('muskingum_share', 0.25) else 'uh'
    start = df.index[365]
    on = phys.get('obs_noise', {})
    o = ws.simulate(spec, (df_sim, meta), str(start.date()), str(df.index[-1].date()), model=model, snow=snow,
                    routing=routing, labels=False, rng=int(rng.integers(2**31)), obs_noise=on if on else True)
    ts = o['ts']
    q = ts['Q'].values.astype(np.float32)
    qo = ts['Q_obs'].values.astype(np.float32)
    if not np.isfinite(q).all() or len(q) != NDAYS:
        return None
    # network input: ERA5 forcing (not the true P) through the forcing-error model
    fe = phys.get('forcing_error', {})
    dfn = noise.forcing_error(df.iloc[365:], rng=int(rng.integers(2**31)), **fe)
    if phys.get('p_jitter', 0) > 0:
        dfn = jitter_p(dfn, rng, phys['p_jitter'])
    F = forcing_array(dfn)
    clim = climate_stats(F[:NDAYS - 1, 0], F[:NDAYS - 1, 1], F[:NDAYS - 1, 2])
    st = static_vector(attrs, clim)
    return {'F': F.astype(np.float16), 'q': q, 'qo': qo, 'st': st, 'doy0': int(start.dayofyear),
            'start': str(start.date()), 'model': MODELS.index(model), 'snow': SNOWS.index(snow),
            'prior': int(prior == 'camels'), 'routing': int(routing == 'muskingum'), 'pmult': m,
            'reg': int(bool(spec.get('reservoirs'))), 'fbid': fbid}


def build(ver, split, n, seed):
    phys = load_phys(ver)
    out = f'{ROOT}/bank/{ver}'
    os.makedirs(out, exist_ok=True)
    donors = realdata.donor_table()
    ids = wd.basins('camels')
    nsh = int(math.ceil(n / SHARD))
    for k in range(nsh):
        fn = f'{out}/{split}_{k:03d}.npz'
        if os.path.exists(fn):
            continue
        rng = np.random.default_rng([seed, 0 if split == 'train' else 1, k])
        t0 = time.time()
        recs = []
        while len(recs) < SHARD:
            fb = ids[int(rng.integers(len(ids)))]
            df_full = wd.forcing(fb, '1951-01-01', '2015-09-30')
            meta = wd.forcing_meta(fb)
            for _ in range(4):
                r = make_sample(rng, phys, donors, df_full, meta, fb)
                if r is not None:
                    recs.append(r)
        recs = recs[:SHARD]
        np.savez(fn + '.tmp.npz', F=np.stack([r['F'] for r in recs]), q=np.stack([r['q'] for r in recs]),
                 qo=np.stack([r['qo'] for r in recs]), st=np.stack([r['st'] for r in recs]),
                 doy0=np.array([r['doy0'] for r in recs]), start=np.array([r['start'] for r in recs]),
                 model=np.array([r['model'] for r in recs]), snow=np.array([r['snow'] for r in recs]),
                 prior=np.array([r['prior'] for r in recs]), routing=np.array([r['routing'] for r in recs]),
                 pmult=np.array([r['pmult'] for r in recs], np.float32), reg=np.array([r['reg'] for r in recs]),
                 fbid=np.array([r['fbid'] for r in recs]))
        os.replace(fn + '.tmp.npz', fn)
        print(f'{split} shard {k+1}/{nsh} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ver, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    build(ver, split, n, seed)
