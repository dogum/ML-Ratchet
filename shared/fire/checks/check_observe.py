"""Satellite emulator vs real VIIRS on DEV fires (exam fires never touched).

Real: FEDS VIIRS S-NPP 375 m fire pixels (fd.firepix) per 12-hourly step (00 = night overpass ~01:30 local solar time,
12 = day ~13:30). A step can merge two consecutive S-NPP orbits; pixels are split into overpasses by their pixel size
(DS, DT), which is fixed by the scan angle.

Emulated: to test the sensor model rather than the spread model, the emulator is driven by the OBSERVED progression:
  * arrival time of each 90 m cell from the FEDS 12-hourly perimeters: a cell new in step k sits at the fraction
    d_prev / (d_prev + d_next) of the way from the previous to the current perimeter, and that fraction is mapped to a
    time in (t_{k-1}, t_k] through a diurnal spread-rate weight exp(cos(2 pi (LST - 15 h) / 24)) (afternoon peak);
  * front-normal ROS = 1 / |grad arrival| (0.05-200 m/min);
  * heat per unit area: Rothermel HPA of the LANDFIRE fuel model at dry summer moisture (1/10/100-h 6/7/8 %, live
    90/120 %); canopy fuel is not added (crown fire FRP is underestimated);
  * firesim.observe(sensor='viirs') at each real overpass time with the real pixel size, no cloud (the real data
    include cloud and smoke omission, so the uncalibrated emulator should over-detect on cloudy days).
Calibration on dev (in-sample and 2-fold by fire): multipliers of the day and night detection thresholds (FRP50) so
that the median log ratio of emulated to real counts per overpass is ~0; FRP per detected pixel compared before/after.
Output: checks/observe_viirs.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from _common import write, OUT, _clean
import firesim as fs
from firesim import realdata as RD
from firesim import satellite as SAT

fd = RD._fd()
CACHE = os.path.join(OUT, 'observe_cache')
os.makedirs(CACHE, exist_ok=True)
GRID_F = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]          # FRP50 multipliers tried (day and night separately)
E_MULT = [1.0, 2.0, 4.0]                           # energy multipliers (missing heavy-fuel / canopy energy)
FRP50_D0, FRP50_N0 = 1.2, 0.5                      # uncalibrated (literature-style) VIIRS thresholds, MW at nadir
KAPPA = 1.0                                        # diurnal spread weighting (peak/trough = e^2 ~ 7)


def obs_arrival(fid, ls_full, pr):
    """Observed arrival times (hours since t_ref) on a crop of the fire grid, from all FEDS half-day perimeters."""
    from scipy.ndimage import distance_transform_edt
    g = fd.grid(fid)
    M = np.stack([fd._grid.burn([geom], g) for geom in pr.geometry])
    M = np.maximum.accumulate(M, axis=0)
    fin = M[-1]
    r, c = np.nonzero(fin)
    pad = int(3000 / g['res_m'])
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad + 1, fin.shape[0])
    c0, c1 = max(c.min() - pad, 0), min(c.max() + pad + 1, fin.shape[1])
    M = M[:, r0:r1, c0:c1]
    tt = pd.DatetimeIndex(pr.t_utc).tz_localize('UTC')
    t_ref = tt[0] - pd.Timedelta(hours=12)
    th = ((tt - t_ref) / pd.Timedelta(hours=1)).to_numpy()
    arr = np.full(M.shape[1:], np.nan)
    # first perimeter: from its centre outward over the 12 h before t0
    m0 = M[0]
    if m0.any():
        d_in = distance_transform_edt(m0)
        arr[m0] = th[0] - 12.0 * d_in[m0] / max(d_in.max(), 1.0)
    # diurnal timing inside a step: spread rate weighted by w(LST) = exp(KAPPA cos(2 pi (LST - 15) / 24))
    lon = RD.lonlat_of(float(pr.geometry.iloc[-1].centroid.x), float(pr.geometry.iloc[-1].centroid.y))[0]
    tf = np.arange(th[0] - 12.0, th[-1] + 0.01, 1.0 / 6.0)
    utc_h = ((t_ref + pd.to_timedelta(tf, unit='h')).hour + (t_ref + pd.to_timedelta(tf, unit='h')).minute / 60.0)
    lst = (np.asarray(utc_h) + lon / 15.0) % 24.0
    wcum = np.concatenate([[0.0], np.cumsum(np.exp(KAPPA * np.cos(2 * np.pi * (lst[:-1] - 15.0) / 24.0)))]) / 6.0
    for k in range(1, len(M)):
        new = M[k] & ~M[k - 1]
        if not new.any():
            continue
        d_prev = distance_transform_edt(~M[k - 1])[new]
        d_next = distance_transform_edt(M[k])[new]
        frac = d_prev / np.maximum(d_prev + d_next, 1e-9)
        w0, w1 = np.interp([th[k - 1], th[k]], tf, wcum)
        arr[new] = np.interp(w0 + frac * (w1 - w0), wcum, tf)
    ls = ls_full.subset(r0, r1, c0, c1)
    return ls, arr, th, tt, t_ref, (r0, r1, c0, c1)


def hpa_raster(ls):
    fuel = np.asarray(ls['fuel'])
    hpa = np.zeros(fuel.shape)
    for num in np.unique(fuel):
        if 91 <= num <= 99:
            continue
        b = fs.behave(int(num), m1=6, m10=7, m100=8, mherb=90, mwoody=120)
        hpa[fuel == num] = b.get('hpa_kjm2', 0.0)
    return hpa


def ros_from_arrival(arr, res):
    gy, gx = np.gradient(np.where(np.isfinite(arr), arr * 60.0, np.nan), res)
    gm = np.hypot(gx, gy)
    ros = np.where(np.isfinite(gm) & (gm > 0), 1.0 / gm, 1.0)
    return np.clip(ros, 0.05, 200.0)


def overpasses_for_step(px):
    """Split one step's pixels into overpasses by pixel size; returns list of (ds, dt, n, frp list)."""
    if not len(px):
        return []
    out = []
    # merge near-identical sizes: cluster on DS (scan-angle proxy) with 0.03 km tolerance
    ds = np.sort(px.DS.unique())
    groups = []
    for v in ds:
        if groups and v - groups[-1][-1] <= 0.03:
            groups[-1].append(v)
        else:
            groups.append([v])
    for gv in groups:
        sel = px[px.DS.isin(gv)]
        out.append((float(sel.DS.median()), float(sel.DT.median()), int(len(sel)), sel.FRP.tolist()))
    return out


def build_fire(fid):
    fn = os.path.join(CACHE, fid + '.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    assert not fd.guard.is_exam(fid)
    L = fd.landscape(fid)
    ls_full = RD.landscape_from_firedata(L)
    pr = fd.perimeters(fid)
    px_all = fd.firepix(fid)
    ls, arr, th, tt, t_ref, box = obs_arrival(fid, ls_full, pr)
    hpa = hpa_raster(ls)
    ros = ros_from_arrival(arr, ls.res)
    t = ls.meta['transform']
    cx, cy = t[0] + ls.shape[1] / 2 * t[1], t[3] + ls.shape[0] / 2 * t[5]
    lon, lat = RD.lonlat_of(cx, cy)
    out = dict(rasters=dict(ros=ros, heat_per_area=hpa), arrival_time=arr, fuel_raster=np.asarray(ls['fuel']),
               meta=dict(res_m=ls.res, transform=ls.meta['transform'], lat=lat, lon=lon),
               start=t_ref, end=tt[-1] + pd.Timedelta(hours=1))
    rows = []
    for k in range(len(pr)):
        label = pr.t.iloc[k]
        px = px_all[px_all.t == label]
        ops = overpasses_for_step(px)
        dn = 'N' if label.hour == 0 else 'D'
        new_cells = int(np.sum(np.isfinite(arr) & (arr > (th[k - 1] if k else -1e9)) & (arr <= th[k])))
        if not ops:
            ops = [(None, None, 0, [])]
        for j, (ds, dtk, n, frps) in enumerate(ops):
            t_op = tt[k] + pd.Timedelta(minutes=100 * j * (1 if j else 0))
            rows.append(dict(k=k, label=str(label), t_utc=str(t_op), daynight=dn, ds=ds, dt=dtk, n_real=n,
                             frp_real=frps, n_overpasses=len(ops), new_cells_step=new_cells,
                             burned_cells=int(np.sum(np.isfinite(arr) & (arr <= th[k])))))
    res = dict(fire_id=fid, lat=lat, lon=lon, rows=rows)
    # emulate for each threshold multiplier (deterministic expected counts + one random draw at f=1)
    for r in rows:
        tq = pd.Timestamp(r['t_utc'])
        pix = (r['ds'], r['dt']) if r['ds'] else (0.53, 0.45)     # dev median pixel when no detection
        th_op = (tq - t_ref) / pd.Timedelta(hours=1)
        frp = SAT.cell_frp(out, th_op)
        r['active_cells'] = int((frp > 0.01).sum())
        r['total_frp_cells_mw'] = float(frp.sum())
        r['emu'] = {}
        for e in E_MULT:
            out_e = dict(out, rasters=dict(ros=ros, heat_per_area=hpa * e))
            for f in GRID_F:
                P = dict(frp50_day=FRP50_D0 * f,
                         frp50_night=FRP50_N0 * f, false_rate=0.0, energy_mult=1.0)
                ob = fs.observe(out_e, 'viirs', overpasses=[(tq, 'S-NPP', 0.0, pix)], rng=r['k'] * 7 + 1, params=P)
                o = ob['overpasses'].iloc[0]
                det = ob['detections']
                tp = det[det.true_fire] if len(det) else det
                r['emu'][f'{e}_{f}'] = dict(expected=float(o.expected_det), n=int(o.n_det),
                                            frp=[round(float(v), 3) for v in tp.frp_mw] if len(tp) else [])
    with open(fn, 'w') as fh:
        json.dump(_clean(res), fh)
    return res


def lr(a, b):
    return np.log10((np.asarray(a, float) + 1.0) / (np.asarray(b, float) + 1.0))


def fit(rows):
    """Step 1: energy multiplier e matching the median FRP of detected pixels (day and night pooled, at f = 1).
    Step 2: FRP50 multipliers (day, night) minimising |median log10((expected + 1) / (real + 1))|."""
    frp_r = np.concatenate([r['frp_real'] for r in rows if r['frp_real']])
    se = {}
    for e in E_MULT:
        fe = np.concatenate([r['emu'][f'{e}_1.0']['frp'] for r in rows if r['emu'][f'{e}_1.0']['frp']] or [np.array([1e-9])])
        se[e] = abs(np.log10(np.median(fe) / np.median(frp_r)))
    e = min(se, key=se.get)
    best = dict(e=e)
    for dn in ('D', 'N'):
        sub = [r for r in rows if r['daynight'] == dn]
        sc = {f: abs(np.median(lr([r['emu'][f'{e}_{f}']['expected'] for r in sub], [r['n_real'] for r in sub])))
              for f in GRID_F}
        best[dn] = min(sc, key=sc.get)
    return best


def stats(rows, fsel):
    out = {}
    for dn in ('D', 'N', 'all'):
        sub = [r for r in rows if dn == 'all' or r['daynight'] == dn]
        if not sub:
            continue
        key = lambda r: f"{fsel['e']}_{fsel[r['daynight']]}"
        e = np.array([r['emu'][key(r)]['expected'] for r in sub])
        n = np.array([r['emu'][key(r)]['n'] for r in sub])
        real = np.array([r['n_real'] for r in sub])
        frp_r = np.concatenate([r['frp_real'] for r in sub if r['frp_real']] or [np.array([])])
        frp_e = np.concatenate([r['emu'][key(r)]['frp'] for r in sub if r['emu'][key(r)]['frp']] or [np.array([])])
        ok = real + n > 0
        out[dn] = dict(
            n_overpasses=int(len(sub)), real_total=int(real.sum()), emu_total=int(n.sum()),
            median_log10_ratio_expected=float(np.median(lr(e, real))),
            mean_abs_log10_ratio=float(np.mean(np.abs(lr(n, real)))),
            spearman_counts=float(pd.Series(n).corr(pd.Series(real), method='spearman')),
            pearson_log_counts=float(np.corrcoef(np.log10(n + 1), np.log10(real + 1))[0, 1]) if ok.sum() > 2 else None,
            frac_zero_real=float(np.mean(real == 0)), frac_zero_emu=float(np.mean(n == 0)),
            within_factor2=float(np.mean(np.abs(lr(n, real)) <= np.log10(2))),
            frp_real_median=float(np.median(frp_r)) if len(frp_r) else None,
            frp_emu_median=float(np.median(frp_e)) if len(frp_e) else None,
            frp_real_p90=float(np.percentile(frp_r, 90)) if len(frp_r) else None,
            frp_emu_p90=float(np.percentile(frp_e, 90)) if len(frp_e) else None)
    return out


def main():
    ids = list(fd.fires('dev').sort_values('feds_area_km2').Event_ID)
    allrows = []
    t0 = time.time()
    for fid in ids:
        try:
            res = build_fire(fid)
        except FileNotFoundError as e:
            print(fid, 'not ready', e, flush=True)
            continue
        for r in res['rows']:
            r['fire_id'] = fid
        allrows += res['rows']
        print(fid, len(res['rows']), 'overpasses', '%.0fs' % (time.time() - t0), flush=True)
    # only overpasses while the fire is burning (some active cells or real detections)
    rows = [r for r in allrows if r['active_cells'] > 0 or r['n_real'] > 0]
    fires = sorted(set(r['fire_id'] for r in rows))
    f1 = {'e': 1.0, 'D': 1.0, 'N': 1.0}
    cal = fit(rows)
    # 2-fold by fire (seeded)
    rng = np.random.default_rng(20261118)
    perm = rng.permutation(fires)
    folds = [set(perm[::2]), set(perm[1::2])]
    cv_rows = []
    cv_fits = []
    for a in range(2):
        tr = [r for r in rows if r['fire_id'] not in folds[a]]
        te = [r for r in rows if r['fire_id'] in folds[a]]
        fa = fit(tr)
        cv_fits.append(fa)
        cv_rows.append(stats(te, fa))
    res = dict(
        what='VIIRS S-NPP emulator driven by the observed FEDS progression vs real FEDS VIIRS pixels, dev fires only',
        n_fires=len(fires), n_overpasses=len(rows),
        uncalibrated=stats(rows, f1),
        calibration=dict(multipliers=cal, energy_multiplier=cal['e'],
                         frp50_day_mw=FRP50_D0 * cal['D'],
                         frp50_night_mw=FRP50_N0 * cal['N'],
                         adopted_as_defaults='firesim.satellite: SENSORS viirs frp50_night=0.25, ENERGY_MULT=2.0',
                         rule='energy multiplier from %s matching the median FRP of detected pixels; then FRP50 '
                              'multipliers from %s minimising |median log10((expected+1)/(real+1))| per day/night'
                              % (E_MULT, GRID_F)),
        calibrated_in_sample=stats(rows, cal),
        calibrated_2fold=dict(fits=cv_fits, fold_stats=cv_rows),
        notes=['real counts include cloud and smoke omission; the emulator here has no cloud',
               'commission (false alarms) cannot be measured with FEDS pixels: FEDS keeps only pixels it assigned to '
               'a fire', 'canopy fuel energy is not included in the emulator HPA here (crown fires: FRP low)',
               'overpass times are approximate (FEDS half-day label + 1.5 h - lon/15; second orbit +100 min)'])
    write('observe_viirs.json', res)
    print(json.dumps(_clean({k: res[k] for k in ('uncalibrated', 'calibration', 'calibrated_in_sample')}), indent=1))


if __name__ == '__main__':
    main()
