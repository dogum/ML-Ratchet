"""Twin tests on DEV fire-days: re-run each real fire-day in the simulator with the bank's priors (physics version
<ver>), starting from the real FEDS perimeter at issue, on the real LANDFIRE crop under the real HRRR series, and see
the result through the VIIRS + FEDS emulators. Compares the twin's next-day new area with the real one (prior
predictive check of the physics + priors) and stores member-0 twin targets so the network can be scored against twin
and real truth on identical inputs.
Usage: python3 twin.py <phys_ver> [members=3]  -> out/twin_<ver>.json, out/twin_<ver>_targets.pkz"""
import pickle
import time
import zlib
from common import *  # noqa
import pandas as pd
from scipy import ndimage
import feds_emu as E
import simbank as SB

SB.ndimage = ndimage


def run_member(ls, ws, now, T0, T1, lon, lat, P, rng, k_day, first, cal_feds):
    import firesim as fs_
    from firesim import satellite as SAT
    GROUPS = sys.modules['firesim.simulate'].GROUPS
    ws_sim = ws.copy()
    f_ws = float(np.exp(np.hypot(P['ws_sd_fire'], P['ws_sd_day']) * rng.standard_normal()))
    ws_sim['wind_speed'] = ws_sim['wind_speed'].to_numpy() * f_ws
    ws_sim['wind_dir'] = (ws_sim['wind_dir'].to_numpy() + P['wd_sd_day'] * rng.standard_normal()) % 360.0
    ws_sim['RH'] = np.clip(ws_sim['RH'].to_numpy() * np.exp(P['rh_sd_fire'] * rng.standard_normal()), 2, 100)
    f_ros = float(np.clip(np.exp(np.log(P['ros_med']) + P['ros_sd'] * rng.standard_normal()), *P['ros_clip']))
    jit = {g: float(np.exp(P['ros_grp_sd'] * rng.standard_normal())) for g in GROUPS}
    r_night = float(rng.uniform(*P['night']))
    f_first = float(rng.uniform(*P['first_boost'])) if (first and rng.random() < P['p_first_boost']) else 1.0
    supp = rng.random() < P['p_supp']
    c_max = float(rng.uniform(*P['c_max'])) if supp else 0.0
    c_tau = float(rng.uniform(*P['c_tau_days']))
    spot = rng.random() < P['p_spot']
    ember = float(np.exp(rng.uniform(np.log(P['ember'][0]), np.log(P['ember'][1]))))
    base = dict(mherb=float(rng.uniform(*P['mherb'])), mwoody=float(rng.uniform(*P['mwoody'])),
                fmc=float(rng.uniform(*P['fmc'])), ember_rate=ember)
    barrier = np.zeros(now.shape, bool)
    frac = c_max * (1 - np.exp(-k_day / c_tau)) if k_day > 0 else 0.0
    if frac > 0:
        wd = ws_sim.loc[T0:T0 + pd.Timedelta(hours=24), 'wind_dir'].to_numpy()
        head = (np.degrees(np.arctan2(np.mean(np.sin(np.radians(wd))), np.mean(np.cos(np.radians(wd))))) + 180) % 360
        bins = SB.grow_containment(np.zeros(72, bool), frac, head, rng)
        rr, cc = np.nonzero(now)
        barrier = SB.contained_ring(now, bins, (rr.mean(), cc.mean()))
    age = P.get('age_ros')
    f_age = age[min(k_day, len(age) - 1)] if age else 1.0
    lh = lambda tt: ((tt + pd.Timedelta(hours=lon / 15.0)).hour + (tt + pd.Timedelta(hours=lon / 15.0)).minute / 60.0)
    arr = np.where(now, -6.0, np.nan)
    ros = np.ones(now.shape)
    hpa = np.zeros(now.shape)
    t = T0
    while t < T1:
        h = lh(t)
        night = not (8.0 <= h < 20.0)
        dh = ((8.0 if night else 20.0) - h) % 24.0 or 12.0
        t2 = min(t + pd.Timedelta(hours=dh), T1)
        burned = np.isfinite(arr)
        f_seg = f_ros * (r_night if night else 1.0) * f_first * f_age
        prm = dict(base, ros_adjust={g: f_seg * jit[g] for g in jit})
        o = fs_.simulate(ls, ws_sim, {'mask': burned}, t, t2, crown='van_wagner', spotting=spot, params=prm,
                         barriers=barrier if barrier.any() else None, return_perimeters=False,
                         rng=int(rng.integers(1 << 30)))
        a = o['arrival_time']
        new = np.isfinite(a) & ~burned
        sh = (t - T0) / pd.Timedelta(hours=1)
        arr[new] = sh + a[new]
        ros[new] = np.nan_to_num(o['rasters']['ros'][new], nan=1.0)
        hpa[new] = np.nan_to_num(o['rasters']['heat_per_area'][new], nan=0.0)
        t = t2
    truth = np.isfinite(arr)
    # observation of the night (AM) and next-afternoon (PM) S-NPP passes -> FEDS continuation of the real perimeter
    meta = dict(ls.meta)
    meta.update(res_m=90.0, lat=lat, lon=lon)
    out_c = dict(arrival_time=arr, rasters=dict(ros=np.maximum(ros, 1e-3), heat_per_area=hpa), meta=meta,
                 start=T0, end=T1, fuel_raster=np.asarray(ls['fuel']))
    ops = [o_ for o_ in SAT.nominal_overpasses(T0 + pd.Timedelta(hours=3), T1 + pd.Timedelta(hours=2), lat, lon,
                                                'viirs', rng) if o_[1] == 'S-NPP']
    pts_am = np.zeros(now.shape, bool)
    pts_pm = np.zeros(now.shape, bool)
    for (tp, sat, scan) in ops:
        lt = (tp + pd.Timedelta(hours=lon / 15.0)).tz_convert(None)
        hh = lt.hour + lt.minute / 60.0
        if tp > T1 + pd.Timedelta(hours=2):
            continue
        cf = float(rng.uniform(*P['cloud_frac'])) if rng.random() < P['p_cloud'] else 0.0
        ob = fs_.observe(out_c, 'viirs', overpasses=[(tp, sat, scan)], rng=int(rng.integers(1 << 30)), lat=lat,
                         lon=lon, cloud=cf)
        d = ob['detections']
        if not len(d):
            continue
        p = E.points_raster(d.row.values, d.col.values, now.shape)
        if hh >= 22.0 or hh <= 5.0:
            pts_am |= p
        elif 10.0 <= hh <= 17.0:
            pts_pm |= p
    masks, _ = E.sequence([pts_am, pts_pm], now.shape, 90.0, cal_feds, init_mask=now)
    return masks[-1], truth, dict(f_ros=f_ros, r_night=r_night, c_frac=frac, spot=spot, f_ws=f_ws)


def main(ver, members=3):
    import firesim as fs_
    from firesim import realdata as RD
    import evaluate as EV
    fd_ = fd()
    P = SB.load_phys(ver)
    cal_feds = json.load(open(shipped('calib_feds.json')))['best']
    fn = f'{OUT}/twin_{ver}_rows.jsonl'
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)['i'] for l in open(fn)}
    tgt_fn = f'{OUT}/twin_{ver}_targets'
    os.makedirs(tgt_fn, exist_ok=True)
    cache = {}
    t0 = time.time()
    n = len(EV.dev_days())
    with open(fn, 'a') as fo:
        for i in range(n):
            if i in done:
                continue
            d = EV.dev_day(i)
            fid = d['fire_id']
            if fid not in cache:
                cache.clear()
                L = fd_.landscape(fid)
                W = fd_.weather(fid)
                r = fd_._row(fid)
                lon, lat = fd_._lonlat(r)
                ls_full = RD.landscape_from_firedata(L)
                cache[fid] = (ls_full, W, lon, lat)
            ls_full, W, lon, lat = cache[fid]
            y0, y1, x0, x1 = d['box']
            ls = ls_full.subset(y0, y1, x0, x1)
            gt = ls.meta['transform']
            rr, cc = np.nonzero(d['now'])
            x = gt[0] + (cc.mean() + 0.5) * gt[1]
            y = gt[3] + (rr.mean() + 0.5) * gt[5]
            ws = RD.weather_series(W, x, y)
            ws = RD.fill_precip_from_daily(ws, W.get('daily'), lon)
            T0 = (pd.Timestamp(d['t']) + pd.Timedelta(hours=1.5 - lon / 15.0)).tz_localize('UTC')
            T1 = T0 + pd.Timedelta(hours=24)
            if ws.index[0] > T0 - pd.Timedelta(hours=1) or ws.index[-1] < T1:
                fo.write(json.dumps(dict(i=i, fire_id=fid, k=d['k'], skip='weather period')) + '\n')
                continue
            rng = np.random.default_rng(10_000 + i)
            real_new = int((d['nxt'] & ~d['now']).sum())
            mem = []
            for m in range(members):
                try:
                    nxt_t, truth, info = run_member(ls, ws, d['now'], T0, T1, lon, lat, P, rng, d['k'], d['first'], cal_feds)
                except Exception as ex:
                    info = dict(error=repr(ex))
                    mem.append(info)
                    continue
                info.update(new_feds=int((nxt_t & ~d['now']).sum()), new_truth=int((truth & ~d['now']).sum()))
                mem.append(info)
                if m == 0:
                    open(f'{tgt_fn}/{i:04d}.bin', 'wb').write(zlib.compress(np.packbits(nxt_t).tobytes()))
            row = dict(i=i, fire_id=fid, k=d['k'], real_new=real_new, members=mem, area_km2=d['area_km2'])
            fo.write(json.dumps(row, default=float) + '\n')
            fo.flush()
            if i % 10 == 0:
                print(i, n, fid, real_new, [x.get('new_feds') for x in mem], f'{time.time() - t0:.0f}s', flush=True)
    rows = [json.loads(l) for l in open(fn)]
    rows = [r for r in rows if 'skip' not in r]
    lr = []
    for r in rows:
        for m in r['members']:
            if 'new_feds' in m:
                lr.append(np.log10((m['new_feds'] + 1) / (r['real_new'] + 1)))
    # rank of the real new area inside the member distribution (PIT)
    pit = []
    for r in rows:
        v = [m['new_feds'] for m in r['members'] if 'new_feds' in m]
        if v:
            pit.append((np.sum(np.array(v) < r['real_new']) + 0.5 * np.sum(np.array(v) == r['real_new'])) / len(v))
    out = dict(ver=ver, n_days=len(rows), members=members, log10_ratio_median=float(np.median(lr)),
               log10_ratio_mean=float(np.mean(lr)), log10_ratio_q25=float(np.quantile(lr, .25)),
               log10_ratio_q75=float(np.quantile(lr, .75)), pit_mean=float(np.mean(pit)),
               frac_real_below_all=float(np.mean(np.array(pit) == 0)), frac_real_above_all=float(np.mean(np.array(pit) == 1)))
    jdump(out, f'{OUT}/twin_{ver}.json')
    print(out)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
