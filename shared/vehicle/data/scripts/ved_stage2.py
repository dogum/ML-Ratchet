"""Stage 2: VED stage-1 parquet -> per-trip 1 Hz npz + json (shared trip contract).
Resumable: one done-marker per bucket. usage: python3 -I ved_stage2.py [bucket ...]
"""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, glob, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, _VEH)
from realdata.geo import Dem, utm_forward, elevation_profile, heading_from_xy
from realdata.weather import Weather
from realdata.vedspec import ved_vehicle_table

ROOT = _VEH + '/data'
OUT = f'{ROOT}/proc/ved/trips'
SPLIT = json.load(open(f'{ROOT}/proc/splits_ved_part.json'))['VED']['vehicles']
VT = ved_vehicle_table().set_index('VehId')
DENS = {'gasoline': 745.0, 'diesel': 832.0}
AFR = {'gasoline': 14.08, 'diesel': 14.5}   # 14.08 = E10 stoichiometric (VED paper, Alg. 1)
RHO_STD = 1.184                              # g/L, SAE J1979 absolute-load reference density
SPLIT_GAP_S = 60.0                           # gaps longer than this split a VED trip into segments
FORCE = os.environ.get('FORCE') == '1'
GAP_S = 5.0                                  # seconds farther than this from any logged row are 'gap'

dem = Dem(); wx = Weather()


def held_interp(t, v, grid, plateau=2.5):
    """Logger repeats the last value between updates. Interpolate through update rows (value changes),
    keeping the end of long plateaus so a held value is not ramped. NaN-only -> None."""
    ok = np.isfinite(v)
    if ok.sum() == 0: return None
    t = t[ok]; v = v[ok]
    if len(v) == 1: return np.full(len(grid), v[0], np.float64)
    ch = np.r_[True, v[1:] != v[:-1]]
    keep = ch.copy()
    idx = np.where(ch)[0]
    # plateau ends: row before a change if it is > plateau s after the change row that started it
    nxt = np.r_[idx[1:], len(v)]
    for a, b in zip(idx, nxt):
        if t[b - 1] - t[a] > plateau: keep[b - 1] = True
    keep[-1] = True
    return np.interp(grid, t[keep], v[keep])


def build_segment(veh, trip, seg_k, nseg, d, vinfo):
    t = d.ts_ms.values / 1000.0
    t = t - t[0]
    T = int(np.floor(t[-1])) + 1
    grid = np.arange(T, dtype=np.float64)
    # validity: grid second within GAP_S of a logged row
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
    # The logger writes a row only when some value changes: a stopped car (esp. a hybrid with the engine off)
    # produces no rows. A gap with the car stationary at both ends is a quiet period (values held), not a dropout.
    srow = pd.Series(d.speed_kmh.values).ffill().bfill().values
    still = (srow[j - 1] < 1.0) & (srow[j] < 1.0)
    valid = (near <= GAP_S) | still
    quiet_s = int(np.sum((near > GAP_S) & still))
    I = lambda c: held_interp(t, d[c].values.astype(np.float64), grid)
    spd = I('speed_kmh')
    if spd is None or I('lat') is None:
        return None
    v = np.clip(spd / 3.6, 0, None)
    lat = I('lat'); lon = I('lon')
    x, y = utm_forward(lat, lon)
    # distance by speed (trapezoid)
    s = np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]))]
    dist = float(s[-1])
    gps_len = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    moving = v > 0.5
    # elevation / grade along travelled distance
    mv = np.r_[True, np.diff(s) > 0.01]
    sx, xx, yy = s[mv], x[mv], y[mv]
    if len(sx) < 2: sx, xx, yy = np.array([0, 1.0]), np.r_[x[0], x[0]], np.r_[y[0], y[0]]
    sg, zr, zs, g, filled, nodem = elevation_profile(dem, lambda q: np.interp(q, sx, xx), lambda q: np.interp(q, sx, yy), dist)
    elev = np.interp(s, sg, zs) if np.isfinite(zs).any() else np.full(T, np.nan)
    grade = np.interp(s, sg, g) if np.isfinite(g).any() else np.full(T, np.nan)
    heading = heading_from_xy(x, y, moving)
    # time and weather
    t0_utc = (d.DayNum.values[0] - 1.0) * 86400.0 + d.ts_ms.values[0] / 1000.0
    W = wx.at(t0_utc + grid, elev)
    # engine / fuel
    fuel_type = vinfo.fuel
    rpm = I('rpm')
    fuel = None; fsrc = 'none'
    if fuel_type != 'none':
        fl = I('fuel_Lph')
        if fl is not None:
            fuel = np.clip(fl, 0, None) * DENS[fuel_type] / 3600.0; fsrc = 'obd_fuel_rate_Lph'
        else:
            trims = []
            for a, b in (('stft1', 'stft2'), ('ltft1', 'ltft2')):
                xa, xb = I(a), I(b)
                arr = [q for q in (xa, xb) if q is not None]
                trims.append(np.nanmean(np.vstack(arr), axis=0) if arr else np.zeros(T))
            corr = (1 + (np.nan_to_num(trims[0]) + np.nan_to_num(trims[1])) / 100.0) / AFR[fuel_type]
            maf = I('maf')
            if maf is not None:
                fuel = np.clip(maf, 0, None) * corr; fsrc = 'maf_x_trims_over_afr'
            else:
                al = I('absload')
                if al is not None and rpm is not None and np.isfinite(vinfo.displacement_L):
                    fuel = np.clip(al, 0, None) / 100 * RHO_STD * vinfo.displacement_L * np.clip(rpm, 0, None) / 120.0 * corr
                    fsrc = 'absload_rpm_displacement'
            if fuel is not None and rpm is not None:
                fuel = np.where(rpm < 100, 0.0, fuel)   # engine off: MAF PID holds a stale value
    nanT = np.full(T, np.nan)
    hvA, hvV, soc = I('hv_A'), I('hv_V'), I('hv_soc')
    batt_I = -hvA if hvA is not None else nanT   # VED: negative = discharge; contract: + = discharge
    batt_V = hvV if hvV is not None else nanT
    soc = soc / 100.0 if soc is not None else nanT
    ackw, acw, htw = I('ac_kW'), I('ac_W'), I('heater_W')
    aux = None
    if ackw is not None: aux = ackw * 1000.0
    elif acw is not None: aux = acw.copy()
    if htw is not None: aux = htw if aux is None else aux + htw
    oat = I('oat')
    fuel = fuel if fuel is not None else nanT
    # invalidate measured outputs inside gaps
    fuel_v = np.where(valid, fuel, np.nan)
    bI = np.where(valid, batt_I, np.nan)
    # QC
    acc = np.gradient(v)
    gps_step = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    gps_spd3 = np.convolve(gps_step, np.ones(3) / 3, mode='same')
    dropout = int(np.sum((v < 0.3) & (gps_spd3 > 8.0)))
    fuel_cov = float(np.isfinite(fuel_v).mean()) if fsrc != 'none' else 0.0
    fuel_g = float(np.nansum(fuel_v)) if fsrc != 'none' else np.nan
    pw = bI * batt_V
    elec_cov = float(np.isfinite(pw).mean())
    elec_kWh = float(np.nansum(pw) / 3.6e6) if elec_cov > 0 else np.nan
    trip_id = f'VED_{veh}_{trip}' + (f'_{seg_k}' if nseg > 1 else '')
    arrays = dict(t=grid, v=v, grade=grade, elev=elev, lat=lat, lon=lon, heading=heading,
                  T_amb=W['T_amb'], wind_speed=W['wind_speed'], wind_dir=W['wind_dir'], rho_air=W['rho_air'],
                  surface=W['surface'], fuel_rate=fuel_v, engine_rpm=rpm if rpm is not None else nanT,
                  batt_current=bI, batt_voltage=batt_V, soc=soc, aux_power=aux if aux is not None else nanT,
                  oat=oat if oat is not None else nanT, valid=valid.astype(np.uint8))
    arrays = {k: (a.astype(np.float32) if k not in ('surface', 'valid') else a) for k, a in arrays.items()}
    t_start = pd.Timestamp('2017-11-01') + pd.Timedelta(seconds=float(t0_utc))
    dur = float(T)
    meta = dict(trip_id=trip_id, vehicle_id=f'VED_{veh}', source='VED', date=t_start.isoformat() + 'Z',
                distance_m=round(dist, 1), duration_s=dur,
                T_amb_mean=float(np.nanmean(W['T_amb'])) if np.isfinite(W['T_amb']).any() else None,
                split=SPLIT[f'VED_{veh}'], ved_trip=int(trip), segment=int(seg_k), n_segments=int(nseg),
                architecture=vinfo.architecture, ved_type=vinfo.ved_type, fuel_type=fuel_type,
                fuel_density_g_per_L=DENS.get(fuel_type), fuel_source=fsrc, fuel_g=fuel_g, fuel_coverage=fuel_cov,
                elec_kWh_net_discharge=elec_kWh, elec_coverage=elec_cov,
                v_start=float(v[0]), v_end=float(v[-1]), v_mean=dist / dur if dur > 0 else 0.0,
                idle_frac=float(np.mean(v < 0.3)), gap_frac=float(1 - valid.mean()), quiet_held_s=quiet_s,
                max_gap_s=float(np.max(np.diff(t))) if len(t) > 1 else 0.0,
                speed_dropout_s=dropout, accel_spikes=int(np.sum(np.abs(acc) > 6.0)),
                gps_len_over_odo=gps_len / dist if dist > 50 else None,
                elev_closing_frac=filled, elev_nodem_frac=float(nodem),
                elev_gain_m=float(np.sum(np.clip(np.diff(elev), 0, None))) if np.isfinite(elev).all() else None,
                oat_mean=float(np.nanmean(oat)) if oat is not None else None,
                wx_station_T=pd.Series(W['src_T']).mode().iloc[0] if len(W['src_T']) else '',
                wet_frac=float(np.mean(W['surface'] == 2)), snow_frac=float(np.mean(W['surface'] == 4)),
                aux_logged=aux is not None, hour_local=int(t_start.tz_localize('UTC').tz_convert('America/Detroit').hour),
                notes='VED trips are privacy-trimmed fragments: they start/end at major intersections, often mid-drive and with a warm engine.')
    # quality flags
    q_len = dist >= 1000 and dur >= 120
    q_gaps = meta['gap_frac'] <= 0.05 and dropout <= 5
    ff = fuel_v[np.isfinite(fuel_v)]
    q_fuel = (fsrc != 'none' and fuel_cov >= 0.95 and len(ff) > 0 and ff.max() < 25 and
              (fuel_g > 0 if vinfo.ved_type == 'ICE' else fuel_g >= 0))
    if q_fuel and vinfo.ved_type in ('ICE',) and dist > 0:
        L100 = fuel_g / DENS[fuel_type] / (dist / 1e5)
        meta['L_per_100km'] = L100
        q_fuel = 2.0 < L100 < 60.0
    elif q_fuel and dist > 0:
        meta['L_per_100km'] = fuel_g / DENS[fuel_type] / (dist / 1e5)
    q_elec = elec_cov >= 0.95 and np.isfinite(elec_kWh)
    q_elev = np.isfinite(elev).all() and nodem < 0.05
    meta.update(q_len=bool(q_len), q_gaps=bool(q_gaps), q_fuel=bool(q_fuel), q_elec=bool(q_elec), q_elev=bool(q_elev))
    if vinfo.ved_type in ('ICE', 'HEV'): lab = q_fuel
    elif vinfo.ved_type == 'PHEV': lab = q_fuel and q_elec
    else: lab = q_elec
    meta['q_label_ok'] = bool(q_len and q_gaps and lab and q_elev)
    meta['q_route_ok'] = bool(q_len and q_gaps and q_elev)
    return trip_id, arrays, meta


def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o


def run_bucket(b):
    done = f'{ROOT}/proc/ved/stage2_b{b:02d}.done'
    if os.path.exists(done) and not FORCE: return
    fs = sorted(glob.glob(f'{ROOT}/proc/ved_stage1/b{b:02d}/*.parquet'))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d.sort_values(['VehId', 'Trip', 'ts_ms']).drop_duplicates(['VehId', 'Trip', 'ts_ms'], keep='last')
    n = 0; t0 = time.time()
    for (veh, trip), g in d.groupby(['VehId', 'Trip'], sort=False):
        vdir = f'{OUT}/VED_{veh}'; os.makedirs(vdir, exist_ok=True)
        ts = g.ts_ms.values / 1000.0
        cuts = np.where(np.diff(ts) > SPLIT_GAP_S)[0] + 1
        segs = np.split(np.arange(len(g)), cuts)
        segs = [s for s in segs if len(s) >= 5 and ts[s[-1]] - ts[s[0]] >= 30]
        for k, sidx in enumerate(segs):
            tid = f'VED_{veh}_{trip}' + (f'_{k}' if len(segs) > 1 else '')
            if os.path.exists(f'{vdir}/{tid}.json') and not FORCE: continue
            try:
                res = build_segment(veh, trip, k, len(segs), g.iloc[sidx], VT.loc[veh])
                if res is None:
                    print('SKIP no speed/gps', veh, trip, k, flush=True); continue
                tid, arr, meta = res
            except Exception as e:
                print('FAIL', veh, trip, k, repr(e), flush=True); continue
            np.savez_compressed(f'{vdir}/{tid}.npz', **arr)
            json.dump(jsonable(meta), open(f'{vdir}/{tid}.json', 'w'))
            n += 1
    open(done, 'w').write(f'{n}\n')
    print(f'bucket {b}: {n} trips in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    bs = [int(x) for x in sys.argv[1:]] or list(range(16))
    for b in bs: run_bucket(b)
