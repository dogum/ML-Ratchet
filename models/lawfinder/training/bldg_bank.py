"""Simulated single-family homes from BuildingSim (LawFinder, law 4: the energy signature).

One sample = one generated single-family home (ResStock-bootstrapped generator, BuildingData's prior split), one
year at 1 h with the TMYx weather of its climate zone's city, shifted by a random temperature offset U(-3, +3) C so the
degree days are not 17 discrete values. Recorded: annual site energy per floor area (kWh/m2/yr), heating and cooling
degree days (base 18 C, from the run's daily mean temperature: RECS uses the same 65 F base), floor area, year built,
system, fuels.
    b1 = BuildingSim physics 1.0 (as released)      b2 = physics 1.4 (residential heat-pump signature, and
                                                         per-use intensities fitted on the CBECS/RECS prior split)
usage: python3 bldg_bank.py <ver> <train|val> <n> [seed]   (resumable shards of 100)
"""
import os, sys, math, time
import numpy as np, pandas as pd

from _paths import ROOT  # the work folder (training/_paths.py)
SHARD = 100
PHYS = {'b1': '1.0', 'b2': '1.4'}


def degree_days(T_hourly):
    d = pd.Series(np.asarray(T_hourly, float)).groupby(np.arange(len(T_hourly)) // 24).mean().values
    return float(np.sum(np.maximum(18.0 - d, 0))), float(np.sum(np.maximum(d - 18.0, 0)))


def one(spec, rng, phys, wcache):
    import buildingsim as bs
    city = spec['weather_hint']
    if city not in wcache:
        wcache[city] = bs.read_epw(bs.epw_path(city))
    w, meta = wcache[city]
    w = w.copy()
    dT = float(rng.uniform(-3, 3))
    w['T_out'] = w['T_out'] + dT
    if 'dewpoint' in w:
        w['dewpoint'] = w['dewpoint'] + dT          # keep RH roughly fixed
    o = bs.simulate(spec, (w, meta), '2019-01-01', '2020-01-01', dt='1h', record_trend=False,
                    options={'physics': phys})
    T = o['ts']['T_out'].values if 'T_out' in o['ts'] else w['T_out'].values[:8760]
    hdd, cdd = degree_days(T)
    s = o['spec']
    km = o['summary']['kWh_by_meter']
    return {'eui': float(o['summary']['site_kWh'] / s['floor_area_m2']), 'hdd': hdd, 'cdd': cdd,
            'area_m2': float(s['floor_area_m2']), 'year_built': float(s.get('year_built', np.nan)),
            'system': str(s.get('system')), 'city': city, 'dT': dT, 'climate_zone': str(s.get('climate_zone')),
            'elec_kWh_m2': float(km.get('electricity', 0) / s['floor_area_m2']),
            'gas_kWh_m2': float(km.get('gas', 0) / s['floor_area_m2'])}


def build(ver, split, n, seed=1):
    import buildingsim as bs
    out = f'{ROOT}/banks/bldg_{ver}'
    os.makedirs(out, exist_ok=True)
    nsh = int(math.ceil(n / SHARD))
    wcache = {}
    for k in range(nsh):
        fn = f'{out}/{split}_{k:03d}.parquet'
        if os.path.exists(fn):
            continue
        t0 = time.time()
        s0 = int(np.random.SeedSequence([seed, 0 if split == 'train' else 1, k]).generate_state(1)[0] % (2**31))
        rng = np.random.default_rng(s0)
        specs = bs.generate_buildings(SHARD, seed=s0, primary_use='residential_single', mix='balanced',
                                      physics=PHYS[ver])
        rows = []
        for sp in specs:
            try:
                rows.append(one(sp, rng, PHYS[ver], wcache))
            except Exception as e:
                print('skip', repr(e)[:150], flush=True)
        pd.DataFrame(rows).to_parquet(fn + '.tmp'); os.replace(fn + '.tmp', fn)
        print(f'{ver} {split} shard {k+1}/{nsh} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
