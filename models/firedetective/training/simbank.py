"""FireDetective simulation bank: FireSim fires on real (dev, exam-blanked) or parametric landscapes at 180 m, driven
by REAL dev HRRR hourly weather; the model's weather input is the REAL gridMET daily of the same dates (as on real
fires); observed through the VIIRS emulator (S-NPP overpasses) and the dev-calibrated FEDS emulator.

Usage: python3 simbank.py <name> <seed0> <n_shards> [shard_size]
Writes data/bank/<name>/shard_XXX.npz (resumable: existing shards are skipped).
"""
from common import *
import time, functools, traceback, warnings
import firesim as fs
import firedata as fd
import importlib
import feats

warnings.filterwarnings('ignore')
G = importlib.import_module('firedata.grid')
RES = feats.RES
H_SIM = 150          # hours simulated after ignition
SPIN = 72
DEV = list(fd.fires('dev').Event_ID)
LABELS = ['crown', 'crown_active', 'log_ros_adj', 'log_mfac', 'm1_eff', 'lfmc', 'wind', 'spot', 'ia', 'contained',
          'wind_regime', 'log_wf', 'area_km2', 'edge', 'parametric', 'cruz']
PHYS = dict(v2=False, v3=False)


@functools.lru_cache(64)
def dev_landscape180(fid):
    L = dict(fd.landscape(fid)['layers'])
    L['fuel'] = np.where(L['fuel'] <= 0, 99, L['fuel'])
    g90 = fd.grid(fid)
    m = np.asarray(fd.guard.exam_mask(g90), bool)          # blank exam footprints (+2 km)
    L['fuel'] = np.where(m, 99, L['fuel'])
    for k in ('cc', 'ch', 'cbh', 'cbd'):
        L[k] = np.where(m, 0.0, L[k])
    C = feats.coarsen_layers(L, 2)
    g180 = G.coarsen(g90, 2)
    ny, nx = g180['shape']
    C = {k: v[:ny, :nx] for k, v in C.items()}
    return C, g180, int(m.sum())


@functools.lru_cache(64)
def dev_weather(fid):
    W = fd.weather(fid)
    daily = W['daily'].copy()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    hx, hy = fs.realdata._hrrr_xy(W)
    return W, daily, hx, hy


@functools.lru_cache(1)
def nelson_dref():
    """Precomputed stick diffusivities (same values dead_moisture computes on every call)."""
    M = fs.moisture
    P = dict(M.DEFAULT_NELSON)
    return {f'd_ref_{c}': M.diffusivity_for_timelag(M.STICK_RADIUS_CM[c], M.NOMINAL_TAU_H[c], P['film'], P)
            for c in ('1h', '10h', '100h')}


def sample_landscape(rng):
    if rng.random() < 0.25:
        region = str(rng.choice(['west_dry', 'mixed_conifer', 'chaparral', 'great_basin', 'southeast']))
        ls = fs.generate_landscape((300, 300), RES, rng=rng, kind='parametric', region=region)
        L = {k: np.asarray(ls[k]) for k in ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'elev', 'slope', 'aspect')}
        return L, dict(source='parametric:' + region, x0=0.0, y1=300 * RES, parametric=1)
    for _ in range(20):
        fid = DEV[int(rng.integers(len(DEV)))]
        C, g, nbl = dev_landscape180(fid)
        ny, nx = C['fuel'].shape
        sy, sx = min(ny, 400), min(nx, 400)
        r0 = int(rng.integers(0, ny - sy + 1)); c0 = int(rng.integers(0, nx - sx + 1))
        L = {k: v[r0:r0 + sy, c0:c0 + sx].copy() for k, v in C.items()}
        burnable = (L['fuel'] < 91) | (L['fuel'] > 99)
        if burnable.mean() < 0.4 or sy < 60 or sx < 60:
            continue
        x0 = g['bounds'][0] + c0 * RES
        y1 = g['bounds'][3] - r0 * RES
        return L, dict(source='dev:' + fid, x0=x0, y1=y1, parametric=0)
    return sample_landscape(np.random.default_rng(int(rng.integers(1 << 30))))


def sample_weather(rng, hours):
    need = hours + SPIN + 24
    for _ in range(50):
        fid = DEV[int(rng.integers(len(DEV)))]
        W, daily, hx, hy = dev_weather(fid)
        g = W['grid']
        nt = len(g['time'])
        if nt < need + 1:
            continue
        k0 = int(rng.integers(0, nt - need))
        iy = int(rng.integers(0, g['lat'].shape[0])); ix = int(rng.integers(0, g['lat'].shape[1]))
        x, y = float(hx[iy, ix]), float(hy[iy, ix])
        t = pd.DatetimeIndex(np.asarray(g['time']).astype('datetime64[ns]'))[k0:k0 + need]
        try:
            fd.guard.check_weather((x - 1500, y - 1500, x + 1500, y + 1500), t[0], t[-1])
        except fd.guard.ExamLeakError:
            continue
        df = pd.DataFrame({k: np.asarray(g[k])[k0:k0 + need, iy, ix] for k in
                           ('T', 'RH', 'wind_speed', 'wind_dir', 'precip', 'srad', 'cloud') if k in g}, index=t)
        if df[['T', 'RH', 'wind_speed', 'wind_dir']].isna().mean().max() > 0.05:
            continue
        gam = fs.realdata.grid_convergence_deg(x, y)
        w = fs.realdata._contract(df, gam)
        lon, lat = fs.realdata.lonlat_of(x, y)
        w = fs.realdata.fill_precip_from_daily(w, daily, lon)
        return w, daily, dict(wfire=fid, lon=float(lon), lat=float(lat), gamma=float(gam), hrrr=(iy, ix))
    raise RuntimeError('no weather')


def one(seed):
    rng = np.random.default_rng(seed)
    L, lmeta = sample_landscape(rng)
    w, daily, wmeta = sample_weather(rng, H_SIM)
    lon = wmeta['lon']
    ny, nx = L['fuel'].shape
    # parameters (the hidden "drivers")
    ros_adj = float(np.exp(rng.uniform(np.log(0.15), np.log(1.5))))
    mfac = float(np.exp(rng.uniform(np.log(0.7), np.log(1.45))))
    lfmc = float(rng.uniform(60, 200))
    mherb = float(np.clip(lfmc * rng.uniform(0.5, 1.5), 30, 250))
    fmc = float(np.clip(lfmc, 70, 150))
    wf = float(np.clip(np.exp(rng.normal(0, 0.3)), 0.5, 2.0))
    wdir_off = float(rng.normal(0, 15))
    spot = bool(rng.random() < 0.5)
    ember = float(np.exp(rng.uniform(np.log(0.5), np.log(8))))
    ia = bool(rng.random() < 0.25)
    cruz = bool(rng.random() < 0.5)
    cloud = float(rng.uniform(0, 0.3)) if rng.random() < 0.5 else 0.0
    # weather actually driving the fire: HRRR x wind factor, direction offset; dead moisture x mfac
    ww = w.copy()
    md = fs.moisture.dead_moisture(ww, method='nelson', params=nelson_dref())
    ww['m1h'] = md['m1h'] * mfac; ww['m10h'] = md['m10h'] * mfac; ww['m100h'] = md['m100h'] * mfac
    ww['mherb'] = mherb; ww['mwoody'] = lfmc
    ww['wind_speed'] = ww['wind_speed'] * wf
    ww['wind_dir'] = (ww['wind_dir'] + wdir_off) % 360
    rng2 = np.random.default_rng(seed + 7_777_777)     # v2 physics draws (keeps the v1 stream unchanged)
    f_night, contain = 1.0, None
    if PHYS['v2']:
        # night recovery of fine dead fuels (dev twin: real night FRP per pixel and night growth are lower than
        # simulated; FireSim's Nelson 100-h runs 2.4 points drier than NFDRS on dev sites)
        f_night = float(rng2.uniform(1.0, 1.8))
        lh = ((ww.index.tz_convert(None) + pd.Timedelta(hours=lon / 15.0)).hour.to_numpy() + 0.5)
        night = 0.5 * (1 - np.cos(2 * np.pi * (lh - 15.0) / 24.0))          # 0 at 15 LST, 1 at 03 LST
        fac = 1.0 + (f_night - 1.0) * night
        ww['m1h'] = ww['m1h'] * fac; ww['m10h'] = ww['m10h'] * np.sqrt(fac)
        if rng2.random() < 0.7:
            # anchor-and-flank suppression: a control line around the fire at t_h, open over the head sector
            # (centred downwind, width (1 - q) x 360 deg); optionally a later full ring (fire contained)
            contain = dict(t_h=float(rng2.uniform(12, 100)), q=float(rng2.uniform(0.5, 0.95)),
                           d=int(rng2.integers(1, 3)), t2=float(rng2.uniform(24, 96)) if rng2.random() < 0.4 else None)
    # ignition: local 10-18 h on the first day after spin-up, random burnable cell in the central half
    t_sp = ww.index[0] + pd.Timedelta(hours=SPIN)
    loc0 = local_day(t_sp, lon)
    ign_local = loc0 + pd.Timedelta(hours=float(rng.uniform(10, 18)))
    start = (ign_local - pd.Timedelta(hours=lon / 15.0)).tz_localize('UTC')
    if start < t_sp:
        start = start + pd.Timedelta(days=1)
    end = start + pd.Timedelta(hours=H_SIM)
    burnable = (L['fuel'] < 91) | (L['fuel'] > 99)
    cy0, cx0 = ny // 4, nx // 4
    cand = np.argwhere(burnable[cy0:ny - cy0, cx0:nx - cx0]) + [cy0, cx0]
    r, c = cand[int(rng.integers(len(cand)))]
    ls = fs.Landscape(meta=dict(crs='EPSG:5070', res_m=RES, transform=(lmeta['x0'], RES, 0.0, lmeta['y1'], 0.0, -RES),
                                nodata=-9999, lat=wmeta['lat'], lon=lon), res_m=RES, **L)
    params = dict(ros_adjust={'all': ros_adj}, fmc=fmc, crown_spread='cruz2005' if cruz else 'rothermel1991',
                  ember_rate=ember)
    waf_fixed = -1.0
    if PHYS['v3'] and rng2.random() < 0.5:
        # v3: wind sensitivity. Half the sims use one unsheltered-style WAF for every cell, U(0.25, 0.6), instead of
        # the canopy-sheltered per-cell WAF (dev: real growth follows the wind more than simulated growth does)
        waf_fixed = float(rng2.uniform(0.25, 0.6))
        params['waf'] = waf_fixed
    sup = {'initial_attack': dict(start_h=float(rng.uniform(1, 6)),
                                  line_rate_m_per_h=float(np.exp(rng.uniform(np.log(200), np.log(2000)))))} if ia else None
    sim_seed = int(rng.integers(1 << 30))
    out = fs.simulate(ls, ww, {'rowcol': [(int(r), int(c))]}, start, end, dt='12h', crown='van_wagner',
                      spotting=spot, moisture='given', suppression=sup, params=params, rng=sim_seed,
                      return_perimeters=False)
    contained_line = 0.0
    if contain is not None:
        a0 = out['arrival_time']
        B = np.isfinite(a0) & (a0 <= contain['t_h'])
        if B.sum() >= 4 and np.isfinite(a0).sum() > B.sum():
            from scipy import ndimage as ndi
            D = ndi.distance_transform_edt(~B)
            ring = (D > contain['d']) & (D <= contain['d'] + 2)
            rr_, cc_ = np.nonzero(B)
            ang = np.arctan2(np.arange(ny)[:, None] - rr_.mean(), np.arange(nx)[None, :] - cc_.mean())
            tw = start + pd.Timedelta(hours=contain['t_h'])
            seg = ww[(ww.index >= tw) & (ww.index <= tw + pd.Timedelta(hours=24))]
            th_from = np.radians(seg['wind_dir'].to_numpy()); sp_ = seg['wind_speed'].to_numpy()
            th_to = np.arctan2(-np.sum(sp_ * np.sin(th_from)), -np.sum(sp_ * np.cos(th_from)))   # downwind azimuth
            ang_dw = np.arctan2(-np.cos(th_to), np.sin(th_to))                                    # raster frame
            dang = np.abs((ang - ang_dw + np.pi) % (2 * np.pi) - np.pi)
            ring &= dang > (1 - contain['q']) * np.pi
            sup2 = dict(sup or {})
            sup2['lines'] = [dict(mask=ring, time=contain['t_h'])]
            if contain['t2'] is not None:
                t2 = contain['t_h'] + contain['t2']
                B2 = np.isfinite(a0) & (a0 <= t2)     # approximate: the unsuppressed run's area at t2
                D2 = ndi.distance_transform_edt(~B2)
                sup2['lines'].append(dict(mask=(D2 > 1) & (D2 <= 3), time=t2))
            out = fs.simulate(ls, ww, {'rowcol': [(int(r), int(c))]}, start, end, dt='12h', crown='van_wagner',
                              spotting=spot, moisture='given', suppression=sup2, params=params, rng=sim_seed,
                              return_perimeters=False)
            contained_line = contain['q']
    arr = out['arrival_time']
    if np.isfinite(arr).sum() * (RES / 1000) ** 2 < 2.0:      # MTBS-size fires only (>= ~500 ac)
        return None
    # S-NPP overpasses -> half-day bins (local 06-18 = day bin)
    ops = [o for o in fs.satellite.nominal_overpasses(start, end, wmeta['lat'], lon, 'viirs', rng=rng) if o[1] == 'S-NPP']
    ob = fs.observe(out, 'viirs', overpasses=ops, rng=rng, lat=wmeta['lat'], lon=lon, cloud=cloud)
    det = ob['detections']
    if len(det):
        det = det[det.true_fire]
    bins = {}
    for t, _, _ in ops:
        lt = pd.Timestamp(t).tz_convert(None) + pd.Timedelta(hours=lon / 15.0)
        isday = 6 <= lt.hour < 18
        key = (lt.normalize() if isday or lt.hour >= 18 else lt.normalize() - pd.Timedelta(days=1), isday)
        # night bin belongs to the night starting that evening; FEDS AM label = next calendar day 00:00
        bins.setdefault(key, []).append(t)
    keys = sorted(bins, key=lambda k: (k[0], not k[1]))
    steps = []
    for k in keys:
        ts = bins[k]
        d = det[det.time.isin(ts)] if len(det) else det
        steps.append(dict(key=k, t=max(ts), det=d))
    first = next((i for i, s in enumerate(steps) if len(s['det'])), None)
    if first is None:
        return None
    steps = steps[first:first + feats.T_STEPS]
    if len(steps) < 2:
        return None
    # real FEDS records are often short: keep a random window length 2..len (uniform)
    steps = steps[:int(rng.integers(2, len(steps) + 1))]
    em_in = [dict(x=s['det'].x.values if len(s['det']) else np.zeros(0), y=s['det'].y.values if len(s['det']) else np.zeros(0),
                  ds=s['det'].pix_km_scan.values if len(s['det']) else np.zeros(0),
                  dt=s['det'].pix_km_track.values if len(s['det']) else np.zeros(0)) for s in steps]
    masks = feats.feds_emulate((ny, nx), lmeta['x0'], lmeta['y1'], RES, em_in)
    dets = [dict(col=(e['x'] - lmeta['x0']) / RES - 0.5, row=(lmeta['y1'] - e['y']) / RES - 0.5,
                 frp=s['det'].frp_mw.values if len(s['det']) else np.zeros(0)) for e, s in zip(em_in, steps)]
    day = np.array([s['key'][1] for s in steps])
    # weather day for each step (as on real fires: PM -> same day; AM -> previous evening = the bin's day)
    days = [s['key'][0] for s in steps]
    from real import wx_rows
    wx = wx_rows(daily, days)
    seq = feats.seq_features(masks, dets, day, wx, gamma_deg=wmeta['gamma'])
    seq_h = feats.seq_features(masks, dets, day, wx, wind_hidden=True, gamma_deg=wmeta['gamma'])
    am = feats.area_mask_from(masks)
    st = feats.static_features(L, am, pd.Timestamp(days[0]).dayofyear, len(steps))
    # labels over the true burned area at the last step
    t_last = (steps[-1]['t'] - start) / pd.Timedelta(hours=1)
    b = np.isfinite(arr) & (arr <= t_last)
    nb = int(b.sum())
    ft = out['rasters']['fire_type']
    crown = float((ft[b] >= 2).mean()) if nb else 0.0
    crown_a = float((ft[b] == 3).mean()) if nb else 0.0
    pw, ps = out['rasters']['phi_w'], out['rasters']['phi_s']
    regime = float((pw[b] > ps[b]).mean()) if nb else 0.0
    m = out['labels']['moisture']
    lh = (m.index.tz_convert(None) + pd.Timedelta(hours=lon / 15.0)).hour
    sel = (lh >= 12) & (lh <= 18) & (m.index <= steps[-1]['t'])
    m1_eff = float(m['m1h'][sel].mean()) if sel.any() else float(m['m1h'].mean())
    uniq = sorted(set(days))
    wind = float(np.mean([daily.loc[d, 'vs'] for d in uniq if d in daily.index])) if any(d in daily.index for d in uniq) else np.nan
    edge = bool(b[:3].any() or b[-3:].any() or b[:, :3].any() or b[:, -3:].any())
    lab = dict(crown=crown, crown_active=crown_a, log_ros_adj=np.log(ros_adj), log_mfac=np.log(mfac), m1_eff=m1_eff,
               lfmc=lfmc, wind=wind, spot=float(spot), ia=float(ia), contained=float(out['labels']['contained_h'] is not None),
               wind_regime=regime, log_wf=np.log(wf), area_km2=nb * (RES / 1000) ** 2, edge=float(edge),
               parametric=float(lmeta['parametric']), cruz=float(cruz), f_night=f_night, contain_q=contained_line,
               contain_t=(contain['t_h'] if contain is not None and contained_line > 0 else -1.0), waf_fixed=waf_fixed)
    return dict(seq=seq, seq_h=seq_h, static=st, lab=np.array([lab[k] for k in LABELS], np.float32),
                meta=dict(seed=int(seed), lsrc=lmeta['source'], **{k: v for k, v in wmeta.items() if k != 'hrrr'},
                          n_spots=out['labels']['n_spot_ignitions'], t0=str(start)))


def run(name, seed0, n_shards, shard=100):
    global LABELS
    PHYS['v3'] = name.endswith('_v3')
    PHYS['v2'] = name.endswith('_v2') or PHYS['v3']
    if PHYS['v2']:
        LABELS = LABELS + ['f_night', 'contain_q', 'contain_t']
    if PHYS['v3']:
        LABELS = LABELS + ['waf_fixed']
    d = f'{BANK}/{name}'
    os.makedirs(d, exist_ok=True)
    for s in range(n_shards):
        fn = f'{d}/shard_{s:03d}.npz'
        if os.path.exists(fn):
            continue
        t = time.time()
        S, Sh, St, Lb, M = [], [], [], [], []
        nfail = 0
        for i in range(shard):
            seed = seed0 + s * shard + i
            try:
                r = one(seed)
            except Exception as e:
                nfail += 1
                print('fail', seed, repr(e)[:200], flush=True)
                continue
            if r is None:
                continue
            S.append(r['seq']); Sh.append(r['seq_h']); St.append(r['static']); Lb.append(r['lab']); M.append(json.dumps(r['meta']))
        np.savez_compressed(fn + '.tmp.npz', seq=np.array(S, np.float16), seq_h=np.array(Sh, np.float16),
                            static=np.array(St, np.float32), lab=np.array(Lb, np.float32), meta=np.array(M),
                            labels=np.array(LABELS))
        os.replace(fn + '.tmp.npz', fn)
        print(f'shard {s} n={len(S)} fail={nfail} {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    name, seed0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    shard = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    run(name, seed0, n, shard)
