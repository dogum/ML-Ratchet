"""FireCast simulation bank: multi-day FireSim fires on real (dev, exam-guarded) and parametric landscapes under real
HRRR weather, seen through the VIIRS S-NPP emulator and the FEDS emulator, cut into PM-to-PM fire-days.

Per fire (physics version `phys/<ver>.json` holds the priors):
  * landscape 256 x 256 at 90 m: real LANDFIRE window from a dev fire package (firedata.guard; exam cells blanked)
    with probability p_real, else FireSim's parametric generator;
  * weather: real hourly HRRR from a random dev package and cell (guarded); the simulator runs on a PERTURBED copy
    (wind speed x lognormal per fire and per day, direction + normal per day, RH x lognormal per fire: the "forecast
    error"), while the network's weather features come from the unperturbed series through the dev-fitted
    HRRR -> gridMET operator with its residual noise;
  * point ignition at a random local afternoon hour; 12-h segments (08-20 local day, 20-08 night), each restarted
    from the burned mask, with
      - ROS adjustment: fire-level lognormal x per-group jitter (all groups),
      - night slow-down: night segments x r_night,
      - suppression/containment: a growing share of the perimeter (angular sectors, flanks and heel first) is
        lined at each 08:00 (a 2-cell non-burnable ring outside the burned edge),
      - spotting on for a share of fires, random ember rate,
      - random live moisture and foliar moisture;
  * VIIRS S-NPP detections at nominal overpasses (random cloud on some passes), FEDS emulator (out/calib_feds.json)
    -> 12-hourly cumulative perimeters and fire lines;
  * samples: consecutive PM steps (issue k -> verify k+1), skipped when the true fire reaches the grid edge.
Usage: python3 simbank.py <phys_ver> <split: train|val> <n_fires> [seed0]   (resumable: one shard per 25 fires)
"""
import os
import pickle
import time
import traceback
import zlib
from common import *  # noqa
import pandas as pd
import feds_emu as E
import features as FT
import wxfeat as WX

N = 256
SHARD = 25


def load_phys(ver):
    P = json.load(open(f'{PHYS}/{ver}.json'))
    return P


def local_h(t, lon):
    """local solar hour of UTC timestamp(s)"""
    return (t + pd.Timedelta(hours=lon / 15.0))


def contained_ring(burned, bins_on, centroid, width=2):
    """Non-burnable ring (width cells) just outside the burned edge, in the contained angular bins (72 x 5 deg)."""
    ring = ndimage.binary_dilation(burned, iterations=width) & ~burned
    rr, cc = np.nonzero(ring)
    if not len(rr):
        return np.zeros_like(burned)
    ang = np.degrees(np.arctan2(cc - centroid[1], -(rr - centroid[0]))) % 360.0     # from north, clockwise
    b = (ang // 5).astype(int) % 72
    keep = bins_on[b]
    m = np.zeros_like(burned)
    m[rr[keep], cc[keep]] = True
    return m


def grow_containment(bins_on, target_frac, head_deg, rng):
    """Add 5-degree bins until the contained fraction reaches target; heel/flanks are preferred over the head."""
    ang = np.arange(72) * 5.0 + 2.5
    w = 1.0 - 0.8 * np.cos(np.radians(ang - head_deg))         # head (downwind) hardest to line
    while bins_on.mean() < target_frac:
        cand = np.where(~bins_on)[0]
        if not len(cand):
            break
        # extend an existing arc with prob 0.7, else start a new one
        if bins_on.any() and rng.random() < 0.7:
            nb = cand[(bins_on[(cand - 1) % 72]) | (bins_on[(cand + 1) % 72])]
            if len(nb):
                cand = nb
        p = w[cand] / w[cand].sum()
        bins_on[rng.choice(cand, p=p)] = True
    return bins_on


def sim_fire(seed, P, cal_feds, cal_wx):
    import firesim as fs_
    from firesim import satellite as SAT
    rng = np.random.default_rng(seed)
    ndays = int(rng.integers(P['days'][0], P['days'][1] + 1))
    # ---- landscape
    kind = 'real' if rng.random() < P['p_real'] else 'parametric'
    if kind == 'real':
        ls = fs_.generate_landscape((N, N), 90.0, kind='real', rng=rng)
    else:
        ls = fs_.generate_landscape((N, N), 90.0, rng=rng, region=str(rng.choice(
            ['west_dry', 'mixed_conifer', 'chaparral', 'great_basin', 'southeast'])))
    fuel = np.asarray(ls['fuel'])
    burnable = (fuel < 91) | (fuel > 99)
    if burnable.mean() < 0.4:
        return None, 'unburnable'
    # ---- weather (real HRRR, guarded)
    hours = ndays * 24 + 48
    w = fs_.generate_weather(None, hours, kind='real', rng=rng)
    lon, lat = w.attrs['lonlat']
    t_first = w.index[0] + pd.Timedelta(hours=72)
    d0 = (t_first + pd.Timedelta(hours=lon / 15.0)).tz_convert(None).normalize() + pd.Timedelta(days=1)
    t_ig = (d0 + pd.Timedelta(hours=float(rng.uniform(*P['ign_hour'])) - lon / 15.0)).tz_localize('UTC')
    t_end_w = w.index[-1]
    # perturbed weather for the simulator
    ws_sim = w.copy()
    loc_day = (ws_sim.index + pd.Timedelta(hours=lon / 15.0)).tz_convert(None).normalize()
    days_u = np.unique(loc_day)
    f_ws_fire = float(np.exp(P['ws_sd_fire'] * rng.standard_normal()))
    f_day = dict(zip(days_u, np.exp(P['ws_sd_day'] * rng.standard_normal(len(days_u)))))
    d_day = dict(zip(days_u, P['wd_sd_day'] * rng.standard_normal(len(days_u))))
    fws = np.array([f_day[d] for d in loc_day]) * f_ws_fire
    ws_sim['wind_speed'] = ws_sim['wind_speed'].to_numpy() * fws
    ws_sim['wind_dir'] = (ws_sim['wind_dir'].to_numpy() + np.array([d_day[d] for d in loc_day])) % 360.0
    f_rh = float(np.exp(P['rh_sd_fire'] * rng.standard_normal()))
    ws_sim['RH'] = np.clip(ws_sim['RH'].to_numpy() * f_rh, 2.0, 100.0)
    # ---- fire-level parameters
    f_ros = float(np.clip(np.exp(np.log(P['ros_med']) + P['ros_sd'] * rng.standard_normal()), *P['ros_clip']))
    jit = {g: float(np.exp(P['ros_grp_sd'] * rng.standard_normal())) for g in sys.modules['firesim.simulate'].GROUPS}
    r_night = float(rng.uniform(*P['night']))
    f_first = float(rng.uniform(*P['first_boost'])) if rng.random() < P['p_first_boost'] else 1.0
    supp = rng.random() < P['p_supp']
    c_max = float(rng.uniform(*P['c_max'])) if supp else 0.0
    c_tau = float(rng.uniform(*P['c_tau_days']))
    spot = rng.random() < P['p_spot']
    ember = float(np.exp(rng.uniform(np.log(P['ember'][0]), np.log(P['ember'][1]))))
    base = dict(mherb=float(rng.uniform(*P['mherb'])), mwoody=float(rng.uniform(*P['mwoody'])),
                fmc=float(rng.uniform(*P['fmc'])), ember_rate=ember)
    # ---- ignition point: burnable, near the centre
    for _ in range(100):
        r0 = int(rng.integers(N // 2 - P['ign_box'], N // 2 + P['ign_box']))
        c0 = int(rng.integers(N // 2 - P['ign_box'], N // 2 + P['ign_box']))
        if burnable[r0, c0]:
            break
    else:
        return None, 'no ignition'
    # ---- segments
    bounds = []
    t = t_ig
    lh = lambda tt: ((tt + pd.Timedelta(hours=lon / 15.0)).hour + (tt + pd.Timedelta(hours=lon / 15.0)).minute / 60.0)
    t_stop = min(t_ig + pd.Timedelta(hours=ndays * 24 + 24), t_end_w - pd.Timedelta(hours=2))
    while t < t_stop:
        h = lh(t)
        nxt_h = 8.0 if (h >= 20.0 or h < 8.0) else 20.0
        dh = (nxt_h - h) % 24.0
        if dh < 1e-6:
            dh = 12.0
        t2 = min(t + pd.Timedelta(hours=dh), t_stop)
        bounds.append((t, t2, not (8.0 <= h < 20.0)))
        t = t2
    T0 = t_ig
    arr = np.full((N, N), np.nan)
    ros = np.zeros((N, N))
    hpa = np.zeros((N, N))
    ftype = np.zeros((N, N), np.int8)
    barrier = np.zeros((N, N), bool)
    bins_on = np.zeros(72, bool)
    day_i = 0
    labels = []
    for si, (s, e, night) in enumerate(bounds):
        sh = (s - T0) / pd.Timedelta(hours=1)
        burned = np.isfinite(arr)
        if si > 0 and not burned.any():
            break
        if si > 0 and not night and lh(s) < 8.5:          # 08:00 local: suppression update
            day_i += 1
            frac = c_max * (1.0 - np.exp(-day_i / c_tau))
            # head direction: toward of the daily mean simulated wind
            wd = ws_sim.loc[s:s + pd.Timedelta(hours=12), 'wind_dir'].to_numpy()
            head = (np.degrees(np.arctan2(np.mean(np.sin(np.radians(wd))), np.mean(np.cos(np.radians(wd))))) + 180) % 360
            bins_on = grow_containment(bins_on, frac, head, rng)
            rr, cc = np.nonzero(burned)
            barrier |= contained_ring(burned, bins_on, (rr.mean(), cc.mean()))
        if (s - T0) >= pd.Timedelta(hours=P.get('abort_after_h', 48)) and burned.sum() * CELL_KM2 < P.get('abort_km2', 0.0):
            return None, 'small'
        age = P.get('age_ros')
        f_age = age[min(day_i, len(age) - 1)] if age else 1.0
        f_seg = f_ros * (r_night if night else 1.0) * (f_first if day_i == 0 else 1.0) * f_age
        prm = dict(base, ros_adjust={g: f_seg * jit[g] for g in jit})
        ign = {'rowcol': [(r0, c0)]} if si == 0 else {'mask': burned}
        try:
            o = fs_.simulate(ls, ws_sim, ign, s, e, crown='van_wagner', spotting=spot, params=prm,
                             barriers=barrier if barrier.any() else None, return_perimeters=False,
                             rng=int(rng.integers(1 << 30)))
        except Exception as ex:
            return None, f'sim error {ex!r}'
        a = o['arrival_time']
        new = np.isfinite(a) & ~burned
        arr[new] = sh + a[new]
        R = o['rasters']
        ros[new] = np.nan_to_num(R['ros'][new], nan=1.0)
        hpa[new] = np.nan_to_num(R['heat_per_area'][new], nan=0.0)
        ftype[new] = R['fire_type'][new]
        labels.append(dict(seg=si, night=night, f=f_seg, new_cells=int(new.sum()), contained=float(bins_on.mean())))
    burned = np.isfinite(arr)
    if burned.sum() < 2:
        return None, 'no spread'
    # ---- observation: S-NPP nominal overpasses, FEDS 12-h steps
    T_end = bounds[-1][1]
    ops = [o_ for o_ in SAT.nominal_overpasses(T0 - pd.Timedelta(hours=1), T_end, lat, lon, 'viirs', rng)
           if o_[1] == 'S-NPP']
    meta = dict(ls.meta)
    meta.update(res_m=90.0, lat=lat, lon=lon)
    out_c = dict(arrival_time=arr, rasters=dict(ros=np.maximum(ros, 1e-3), heat_per_area=hpa), meta=meta,
                 start=T0, end=T_end, fuel_raster=fuel)
    # step labels: local day d, 'AM' (passes 22:00-05:00 -> labelled day d 00:00) or 'PM' (10:00-17:00)
    steps = {}
    for (tp, sat, scan) in ops:
        lt = (tp + pd.Timedelta(hours=lon / 15.0)).tz_convert(None)
        h = lt.hour + lt.minute / 60.0
        if 10.0 <= h <= 17.0:
            key = (lt.normalize(), 'PM')
        elif h >= 22.0 or h <= 5.0:
            key = ((lt + pd.Timedelta(hours=3)).normalize(), 'AM')
        else:
            continue
        steps.setdefault(key, []).append((tp, sat, scan))
    keys = sorted(steps, key=lambda k: (k[0], k[1] == 'PM'))
    det_rows = []
    for si, k in enumerate(keys):
        for (tp, sat, scan) in steps[k]:
            cf = float(rng.uniform(*P['cloud_frac'])) if rng.random() < P['p_cloud'] else 0.0
            ob = fs_.observe(out_c, 'viirs', overpasses=[(tp, sat, scan)], rng=int(rng.integers(1 << 30)),
                             lat=lat, lon=lon, cloud=cf)
            d = ob['detections']
            if len(d):
                d = d[['row', 'col', 'pix_km_scan', 'pix_km_track', 'frp_mw', 'true_fire']].copy()
                d['step'] = si
                det_rows.append(d)
    if not det_rows:
        return None, 'never detected'
    det = pd.concat(det_rows, ignore_index=True)
    pts = [E.points_raster(det.row[det.step == i], det.col[det.step == i], (N, N)) for i in range(len(keys))]
    masks, lines = E.sequence(pts, (N, N), 90.0, cal_feds)
    # step nominal times (label + 1.5 h local -> UTC), truth at those times
    t_step = [(k[0] + pd.Timedelta(hours=(13.5 if k[1] == 'PM' else 1.5)) - pd.Timedelta(hours=lon / 15.0)).tz_localize('UTC')
              for k in keys]
    th_step = [(tt - T0) / pd.Timedelta(hours=1) for tt in t_step]
    # ---- weather features per local day (unperturbed series -> gridMET-like)
    m100 = fs_.moisture.dead_moisture(w, classes=('100h',))['m100h'].to_numpy()
    hd = WX.hrrr_daily(w, lon, m100)
    gm = WX.to_gridmet_like(hd, cal_wx, rng=rng)
    pm_idx = [i for i, k in enumerate(keys) if k[1] == 'PM']
    samples = []
    first_pm = None
    for j, i in enumerate(pm_idx):
        if not masks[i].any():
            continue
        if first_pm is None:
            first_pm = i
        if j + 1 >= len(pm_idx):
            break
        i2 = pm_idx[j + 1]
        if (keys[i2][0] - keys[i][0]) != pd.Timedelta(days=1):
            continue
        truth2 = np.isfinite(arr) & (arr <= th_step[i2])
        edge = truth2[:3].any() or truth2[-3:].any() or truth2[:, :3].any() or truth2[:, -3:].any()
        if edge:
            break
        dd0, dd1 = keys[i][0], keys[i2][0]
        if dd0 not in gm.index or dd1 not in gm.index:
            continue
        wx = WX.feature_vector(gm.loc[dd0], gm.loc[dd1], 0.0)
        i_prev = pm_idx[j - 1] if j >= 1 and (keys[i][0] - keys[pm_idx[j - 1]][0]) == pd.Timedelta(days=1) else None
        samples.append(dict(k=i, k2=i2, kprev=i_prev, first=(i == first_pm), wx=wx,
                            truth_now=np.packbits(np.isfinite(arr) & (arr <= th_step[i])),
                            truth_next=np.packbits(truth2)))
    if not samples:
        return None, 'no fire-day'
    keep_steps = sorted(set([s['k'] for s in samples] + [s['k2'] for s in samples] +
                            [s['kprev'] for s in samples if s['kprev'] is not None] +
                            [s['k'] - 1 for s in samples if s['k'] >= 1]))
    rec = dict(seed=seed, kind=kind, source=ls.meta.get('source_fire'), lat=lat, lon=lon,
               land=FT.land_from_landscape(ls),
               masks={i: np.packbits(masks[i]) for i in keep_steps}, lines={i: np.packbits(lines[i]) for i in keep_steps},
               det=det[det.step.isin(keep_steps)].astype({'row': np.float32, 'col': np.float32}).reset_index(drop=True),
               keys=[(str(k[0].date()), k[1]) for k in keys], samples=samples,
               params=dict(f_ros=f_ros, r_night=r_night, f_first=f_first, supp=supp, c_max=c_max, c_tau=c_tau, spot=spot, ember=ember,
                           f_ws=f_ws_fire, f_rh=f_rh, ndays=ndays, **{k: v for k, v in base.items()}),
               seg=labels, area_final_km2=float(burned.sum() * CELL_KM2))
    return rec, 'ok'


def main(ver, split, n_fires, seed0=None):
    from scipy import ndimage as _nd
    globals()['ndimage'] = _nd
    P = load_phys(ver)
    cal_feds = json.load(open(shipped('calib_feds.json')))['best']
    cal_wx = json.load(open(shipped('calib_wx.json')))
    seed0 = seed0 if seed0 is not None else (1_000_000 if split == 'val' else 0) + P.get('seed_offset', 0)
    d = f'{BANK}/{ver}/{split}'
    os.makedirs(d, exist_ok=True)
    n_shards = int(np.ceil(n_fires / SHARD))
    t0 = time.time()
    stats = {}
    order = range(n_shards - 1, -1, -1) if os.environ.get('BANK_REVERSE') else range(n_shards)
    for sh in order:
        fn = f'{d}/shard_{sh:04d}.pkz'
        if os.path.exists(fn) or os.path.exists(fn + '.lock'):
            continue
        open(fn + '.lock', 'w').write(str(os.getpid()))
        recs = []
        for i in range(sh * SHARD, min((sh + 1) * SHARD, n_fires)):
            try:
                rec, why = sim_fire(seed0 + i, P, cal_feds, cal_wx)
            except Exception:
                traceback.print_exc()
                rec, why = None, 'exception'
            stats[why] = stats.get(why, 0) + 1
            if rec is not None:
                recs.append(rec)
        blob = pickle.dumps([zlib.compress(pickle.dumps(r, protocol=4), 4) for r in recs], protocol=4)
        with open(fn + '.tmp', 'wb') as f:
            f.write(blob)
        os.replace(fn + '.tmp', fn)
        os.remove(fn + '.lock')
        ns = sum(len(r['samples']) for r in recs)
        print(f'shard {sh} fires {len(recs)} samples {ns} {len(blob) / 1e6:.1f} MB  {time.time() - t0:.0f}s  {stats}  '
              f'mem {mem_mb():.0f}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]), int(a[4]) if len(a) > 4 else None)
