"""Simulated fire-day bank from FireSim (LawFinder, law 2: the daily growth law).

One simulated fire = one 256 x 256 landscape at 90 m (a real dev LANDFIRE window, exam cells blanked by the guard,
or FireSim's parametric generator), real hourly HRRR weather from a random dev package (guarded) or the parametric
weather generator, a point ignition in the local afternoon, and one continuous FireSim run of 3-6 days
(Rothermel + Van Wagner crown, Nelson dead moisture, no spotting: the replay settings). Burned masks are read at the
local ~13:30 overpass each day (the FEDS PM convention), and each consecutive pair is one fire-day:

    inputs (what a real fire-day row also has):  current area A0 (km2) -> equivalent perimeter, daily mean 10 m
                                                 wind (m/s), mean 100-h dead fuel moisture (%), the fire's fuel group
    target:                                      new burned area in the next 24 h (km2)
    sim-only extras:                             mean slope, mean ROS factor, whether the fire touched the edge

Physics versions:
  f1  FireSim v1 as shipped (no ROS adjustment)
  f2  f1 with the dev-calibrated ROS factors by fuel group (shared/fire/checks/replay_calibration.json)

usage: python3 fire_bank.py <ver> <train|val> <n_fires> [seed]     (resumable shards of 20 fires)
"""
import os, sys, json, math, time
import numpy as np, pandas as pd
from scipy import ndimage

from _paths import ROOT  # the work folder (training/_paths.py)
N = 256; RES = 90.0; CELL_KM2 = (RES / 1000) ** 2
SHARD = 20
GROUP_NAMES = ['GR', 'GS', 'SH', 'TU', 'TL']
from _paths import REPO
CALIB = f'{REPO}/shared/fire/checks/replay_calibration.json'
PHYS = {'f1': {'note': 'FireSim v1 as shipped, replay settings (no spotting, no suppression, no ROS adjustment)'},
        'f2': {'note': 'f1 + dev-calibrated ROS factors by fuel group (replay_calibration.json: GR 0.5, GS 0.35, '
                       'SH 0.35, TU 0.35, TL 1.0, other 0.5)'}}


def fuel_group(fuel):
    """Dominant Scott & Burgan group of a fuel array (burnable cells only)."""
    f = np.asarray(fuel).ravel()
    cnt = {'GR': np.sum((f >= 101) & (f <= 109)), 'GS': np.sum((f >= 121) & (f <= 124)),
           'SH': np.sum((f >= 141) & (f <= 149)), 'TU': np.sum((f >= 161) & (f <= 165)),
           'TL': np.sum((f >= 181) & (f <= 189))}
    return max(cnt, key=cnt.get) if sum(cnt.values()) else None


def sim_fire(seed, ver):
    import firesim as fs
    rng = np.random.default_rng(seed)
    ndays = int(rng.integers(3, 7))
    if rng.random() < 0.6:
        ls = fs.generate_landscape((N, N), RES, kind='real', rng=rng)
    else:
        ls = fs.generate_landscape((N, N), RES, rng=rng, region=str(rng.choice(
            ['west_dry', 'mixed_conifer', 'chaparral', 'great_basin', 'southeast'])))
    fuel = np.asarray(ls['fuel'])
    burnable = (fuel < 91) | (fuel > 99)
    if burnable.mean() < 0.4:
        return []
    hours = ndays * 24 + 48
    w = fs.generate_weather(None, hours, kind='real', rng=rng)
    lon, lat = w.attrs['lonlat']
    t_first = w.index[0] + pd.Timedelta(hours=72)
    d0 = (t_first + pd.Timedelta(hours=lon / 15.0)).tz_convert(None).normalize() + pd.Timedelta(days=1)
    t_ig = (d0 + pd.Timedelta(hours=float(rng.uniform(11, 17)) - lon / 15.0)).tz_localize('UTC')
    # PM overpass times (13:30 local solar) after ignition + 6 h
    pm0 = (d0 + pd.Timedelta(hours=13.5 - lon / 15.0)).tz_localize('UTC')
    while pm0 < t_ig + pd.Timedelta(hours=6):
        pm0 += pd.Timedelta(days=1)
    pms = [pm0 + pd.Timedelta(days=k) for k in range(ndays)]
    pms = [t for t in pms if t <= w.index[-1]]
    if len(pms) < 2:
        return []
    for _ in range(100):
        r0 = int(rng.integers(N // 2 - 30, N // 2 + 30)); c0 = int(rng.integers(N // 2 - 30, N // 2 + 30))
        if burnable[r0, c0]:
            break
    else:
        return []
    params = dict(mherb=float(rng.uniform(60, 150)), mwoody=float(rng.uniform(70, 150)), fmc=float(rng.uniform(80, 130)))
    if ver == 'f2':
        from firesim.replay import calibrated_params
        params = calibrated_params(json.load(open(CALIB)), base=params)
    out = fs.simulate(ls, w, [{'rowcol': [(r0, c0)], 'time': t_ig}], t_ig, pms[-1], dt='1h', spread='rothermel',
                      crown='van_wagner', spotting=False, moisture='nelson', params=params, rng=rng,
                      return_perimeters=False)
    arr = out['arrival_time']
    moist = out['labels']['moisture']
    wx = w
    slope = np.asarray(ls['slope']) if 'slope' in ls else np.zeros_like(fuel, float)
    edge = np.zeros_like(burnable); edge[:4, :] = edge[-4:, :] = edge[:, :4] = edge[:, -4:] = True
    recs = []
    for k in range(len(pms) - 1):
        h0 = (pms[k] - t_ig).total_seconds() / 3600; h1 = (pms[k + 1] - t_ig).total_seconds() / 3600
        b0 = np.isfinite(arr) & (arr <= h0); b1 = np.isfinite(arr) & (arr <= h1)
        if b0.sum() == 0:
            continue
        if (b1 & edge).any():
            break                                      # the true fire left the grid: stop this fire
        near = ndimage.binary_dilation(b0, iterations=12)     # ~1 km around the current fire
        g = fuel_group(fuel[near & burnable])
        if g is None:
            continue
        win = (wx.index > pms[k]) & (wx.index <= pms[k + 1])
        mw = (moist.index > pms[k]) & (moist.index <= pms[k + 1])
        a0 = b0.sum() * CELL_KM2
        recs.append({'area0_km2': a0, 'new_km2': (b1 & ~b0).sum() * CELL_KM2,
                     'wind_ms': float(wx['wind_speed'][win].mean()), 'wind_max_ms': float(wx['wind_speed'][win].max()),
                     'fm100': float(moist['m100h'][mw].mean()), 'fm1': float(moist['m1h'][mw].mean()),
                     'rh_min': float(wx['RH'][win].min()), 'fuel_group': g,
                     'slope_deg': float(slope[near & burnable].mean()), 'day': k, 'n_days': len(pms) - 1,
                     'land': 'real' if 'source_fire' in ls.meta else 'parametric'})
    return recs


def build(ver, split, n_fires, seed=1):
    out = f'{ROOT}/banks/fire_{ver}'
    os.makedirs(out, exist_ok=True)
    json.dump({'version': ver, **PHYS[ver]}, open(f'{out}/phys.json', 'w'), indent=1)
    nsh = int(math.ceil(n_fires / SHARD))
    for k in range(nsh):
        fn = f'{out}/{split}_{k:03d}.parquet'
        if os.path.exists(fn):
            continue
        t0 = time.time(); recs = []
        for j in range(SHARD):
            s = [seed, 0 if split == 'train' else 1, k, j]
            try:
                rr = sim_fire(np.random.SeedSequence(s).generate_state(1)[0], ver)
            except Exception as e:
                print('skip', repr(e)[:160], flush=True); rr = []
            for r in rr:
                r['fire'] = f'{split}{k:03d}_{j:02d}'
            recs += rr
        pd.DataFrame(recs).to_parquet(fn + '.tmp')
        os.replace(fn + '.tmp', fn)
        print(f'{ver} {split} shard {k+1}/{nsh} {len(recs)} days {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ver, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    build(ver, split, n, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
