"""observe(): what a polar-orbiting or geostationary active-fire product would report for a simulated fire.

Physical chain (documented defaults, every one overridable through `params`):
1. Fire radiative power of each model cell at the overpass time:
   * flaming front: a cell releases its flaming heat (HPA x area, plus CFB x canopy load x 18 MJ/kg) while the
     front crosses it, i.e. for dt_cross = res / ROS + residence time (384/sigma min, Anderson 1969);
   * post-frontal (smouldering and large-fuel) combustion: an extra energy `post_ratio` x flaming energy by fuel
     group (grass 0.2, shrub 0.6, timber/litter 2.0, slash/blowdown 4.0), released with exponential decay
     (time constant `post_tau_h`, default 3 h);
   * FRP = radiative fraction (0.15; Freeborn et al. 2008, Wooster et al. 2005) x energy release rate.
2. Sensor pixels: the model grid is aggregated into pixels of the sensor footprint at the overpass scan angle
   (VIIRS I-band 375 m at nadir, ~2x at the swath edge; MODIS 1 km, ~2x along track and ~4.8x along scan at the
   edge; GOES ABI 2 km at the sub-satellite point, grown by the view zenith angle), with a random sub-pixel offset.
3. Detection probability: logistic in log FRP, P = 1 / (1 + (FRP50 / FRP)^k), FRP50 proportional to pixel area
   (the detection is a brightness-temperature anomaly over the pixel). Day and night thresholds differ.
   Defaults (nadir pixel): VIIRS 375 m 1.2 MW day / 0.5 MW night; MODIS 10 / 5 MW; GOES 40 / 20 MW; k = 2.5.
   These are order-of-magnitude values from Schroeder et al. (2014), Giglio et al. (2016), Li et al. (2020). VIIRS
   was calibrated on 40 dev fires (checks/check_observe.py -> observe_viirs.json): night FRP50 0.5 -> 0.25 MW and an
   energy multiplier ENERGY_MULT = 2 on the heat per unit area; MODIS and GOES keep the literature thresholds.
4. Omission under cloud: a cloud mask (given, or random spatially correlated with fraction `cloud_frac`) removes
   detections; `smoke_omission` (default 0) removes a share of pixels with very high FRP (thick plumes).
5. Commission: false alarms as a Poisson process with `false_rate` per pixel per overpass (default VIIRS 2e-6).
Overpasses: list of (UTC time, satellite, scan angle deg) or 'nominal' (a circular sun-synchronous orbit model with
the real equator-crossing times: S-NPP and NOAA-20/21 13:25 ascending, Aqua 13:30, Terra 10:30 descending; the
orbit phase is random, so the day-to-day scan-angle pattern is plausible but not the real one) or 'goes' (every
10 min).
"""
import math

import numpy as np
import pandas as pd

R_E = 6371.0
SENSORS = {
    # VIIRS night threshold: dev-calibrated 0.25 MW (checks/observe_viirs.json; literature-style default was 0.5)
    'viirs': dict(nadir_km=0.375, h_km=833.0, max_scan=56.28, frp50_day=1.2, frp50_night=0.25, false_rate=2e-6,
                  sats={'S-NPP': (13.42, 101.4, 98.7, 0.0), 'NOAA-20': (13.42, 101.4, 98.7, 0.5),
                        'NOAA-21': (13.42, 101.4, 98.7, 0.25)}),
    'modis': dict(nadir_km=1.0, h_km=705.0, max_scan=55.0, frp50_day=10.0, frp50_night=5.0, false_rate=5e-6,
                  sats={'Aqua': (13.5, 98.8, 98.2, 0.0), 'Terra': (22.5, 98.9, 98.2, 0.3)}),
    'goes': dict(nadir_km=2.0, frp50_day=40.0, frp50_night=20.0, false_rate=1e-5, sub_lon={'GOES-East': -75.2,
                                                                                          'GOES-West': -137.2}),
}
# energy multiplier on the simulated heat per unit area (fuel energy Rothermel's fine-fuel HPA leaves out: large
# woody fuels, duff, canopy). Dev-calibrated 2.0 on 40 dev fires (checks/observe_viirs.json); 1.0 = uncalibrated.
ENERGY_MULT = 2.0
POST_RATIO = {'GR': 0.2, 'GS': 0.4, 'SH': 0.6, 'TU': 1.5, 'TL': 2.0, 'SB': 4.0, 'A_grass': 0.2, 'A_shrub': 0.6,
              'A_timber': 2.0, 'A_slash': 4.0}


def scan_geometry(sensor, scan_deg):
    """Pixel size (along-scan, along-track) km and pixel area for a scan angle (spherical Earth)."""
    s = SENSORS[sensor]
    h = s['h_km']
    th = math.radians(abs(scan_deg))
    rr = (R_E + h) / R_E
    # Earth central angle and slant range
    zen = math.asin(min(rr * math.sin(th), 1.0))
    alpha = zen - th
    slant = R_E * math.sin(alpha) / max(math.sin(th), 1e-9) if th > 1e-6 else h
    ifov = s['nadir_km'] / h
    along_track = ifov * slant
    along_scan = ifov * slant / max(math.cos(zen), 0.05)
    if sensor == 'viirs':
        # VIIRS aggregates 3:1 -> 2:1 -> 1:1 samples across the scan, holding growth to ~2x at the edge
        agg = 1.0 / 3.0 if abs(scan_deg) < 31.72 else (0.5 if abs(scan_deg) < 44.86 else 1.0)
        along_scan = along_scan * (3 * agg)
        along_scan = min(along_scan, s['nadir_km'] * 2.1)
        along_track = min(along_track, s['nadir_km'] * 2.0)
    return along_scan, along_track


def nominal_overpasses(start, end, lat, lon, sensor='viirs', rng=None):
    """Overpasses from a circular sun-synchronous orbit model; returns list of (time, satellite, scan_deg)."""
    rng = np.random.default_rng(rng)
    s = SENSORS[sensor]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    out = []
    la = math.radians(lat)
    for sat, (ltan, per_min, inc_deg, phase) in s['sats'].items():
        inc = math.radians(inc_deg)
        P = per_min * 60.0
        t0 = start - pd.Timedelta(hours=2) + pd.Timedelta(seconds=float((phase + rng.random()) * P))
        n = int((end - t0).total_seconds() / P) + 3
        for k in range(-1, n):
            tn = t0 + pd.Timedelta(seconds=k * P)
            utc_h = tn.hour + tn.minute / 60 + tn.second / 3600
            lon_node = 15.0 * (ltan - utc_h)
            # time along this orbit when the satellite is at the target latitude (ascending and descending)
            sv = math.sin(la) / math.sin(inc)
            if abs(sv) > 1:
                continue
            for u in (math.asin(sv), math.pi - math.asin(sv)):
                tau = (u % (2 * math.pi)) / (2 * math.pi) * P
                lon_sat = lon_node + math.degrees(math.atan2(math.cos(inc) * math.sin(u), math.cos(u))) \
                    - 360.0 * tau / 86164.0
                dlon = (lon - lon_sat + 540.0) % 360.0 - 180.0
                d_km = abs(dlon) * 111.32 * math.cos(la) * abs(math.sin(inc))
                alpha = d_km / R_E
                rr = (R_E + s['h_km']) / R_E
                if alpha >= math.acos(1.0 / rr):        # below the horizon
                    continue
                scan = math.degrees(math.atan2(math.sin(alpha), rr - math.cos(alpha)))
                if scan > s['max_scan']:
                    continue
                t = tn + pd.Timedelta(seconds=tau)
                if start <= t <= end:
                    out.append((t, sat, scan * np.sign(dlon)))
    return sorted(out, key=lambda x: x[0])


def _goes_geometry(lat, lon, sat):
    sub = SENSORS['goes']['sub_lon'][sat]
    la = math.radians(lat)
    dl = math.radians(lon - sub)
    cosb = math.cos(la) * math.cos(dl)
    b = math.acos(cosb)
    rs = 42164.0
    d = math.sqrt(rs ** 2 + R_E ** 2 - 2 * rs * R_E * cosb)
    vza = math.asin(rs / d * math.sin(b))
    g = 1.0 / max(math.cos(vza), 0.1)
    return 2.0 * g ** 0.5, 2.0 * g, math.degrees(vza)


def cell_frp(out, t_h, params=None):
    """FRP (MW) per model cell at time t_h (hours since start)."""
    P = dict(chi_r=0.15, post_tau_h=3.0, energy_mult=ENERGY_MULT)
    P.update(params or {})
    r = out['rasters']
    arr = out['arrival_time']
    res = out['meta']['res_m']
    area = res * res
    burned = np.isfinite(arr) & (arr <= t_h)
    frp = np.zeros(arr.shape)
    if not burned.any():
        return frp
    ros = np.where(burned, np.maximum(r['ros'], 1e-3), 1.0)
    e_flame = np.where(burned, r['heat_per_area'], 0.0) * area * P['energy_mult']     # kJ
    dt_cross = res / ros + 0.5                                            # minutes (crossing + ~residence)
    age = (t_h - arr) * 60.0                                              # minutes since arrival
    flaming = burned & (age >= 0) & (age <= dt_cross)
    frp[flaming] += e_flame[flaming] / (dt_cross[flaming] * 60.0) / 1000.0   # MW
    # post-frontal
    from .simulate import fuel_group
    fuel = out.get('fuel_raster')
    pr = P.get('post_ratio_raster')
    if pr is None:
        pr = np.full(arr.shape, 0.5)
        if fuel is not None:
            for num in np.unique(fuel[burned]):
                pr[fuel == num] = P.get('post_ratio', POST_RATIO).get(fuel_group(num), 0.5)
    pr = pr * float(P.get('post_scale', 1.0))
    tau = P['post_tau_h'] * 3600.0
    ages = np.maximum(age * 60.0, 0)
    post = burned & (ages < 8 * tau)
    frp[post] += (pr[post] * e_flame[post] / tau * np.exp(-ages[post] / tau)) / 1000.0
    return P['chi_r'] * frp


def observe(out, sensor='viirs', overpasses='nominal', rng=None, lat=None, lon=None, cloud=None, params=None,
            return_masks=False):
    """Emulate active-fire detections for a simulate() output.

    overpasses: 'nominal' | 'goes' | list of (time, satellite, scan_deg[, (pix_km_scan, pix_km_track)]) | list of times.
    cloud: None (random with params['cloud_frac'], default 0), a float fraction, or {time: bool mask (model grid)}.
    Returns dict(detections=DataFrame, overpasses=DataFrame summary, masks=list (optional)).
    """
    rng = np.random.default_rng(rng)
    P = dict(k=2.5, cloud_frac=0.0, cloud_scale_km=20.0, smoke_omission=0.0, smoke_frp=500.0)
    P.update(SENSORS[sensor] if sensor in SENSORS else {})
    P.update(params or {})
    ls_meta = out['meta']
    res = ls_meta['res_m']
    arr = out['arrival_time']
    ny, nx = arr.shape
    if lat is None or lon is None:
        lat = ls_meta.get('lat', 40.0) if lat is None else lat
        lon = ls_meta.get('lon', -115.0) if lon is None else lon
    start = out['start']
    end = out['end']
    if isinstance(overpasses, str):
        if overpasses == 'goes' or sensor == 'goes':
            sat = 'GOES-West' if lon < -105 else 'GOES-East'
            ops = [(t, sat, None) for t in pd.date_range(start, end, freq=P.get('goes_step', '10min'))]
        else:
            ops = nominal_overpasses(start, end, lat, lon, sensor, rng)
    else:
        ops = []
        for o in overpasses:
            if isinstance(o, tuple):
                ops.append((pd.Timestamp(o[0]), o[1] if len(o) > 1 else sensor, o[2] if len(o) > 2 else 0.0)
                           + ((tuple(o[3]),) if len(o) > 3 and o[3] is not None else ()))
            else:
                ops.append((pd.Timestamp(o), sensor, 0.0))
    from .weather import solar_elevation
    dets, summ, masks = [], [], []
    for op in ops:
        t, sat, scan = op[:3]
        pix_given = op[3] if len(op) > 3 else None
        if t.tz is None:
            t = t.tz_localize('UTC')
        th = (t - start) / pd.Timedelta(hours=1)
        frp = cell_frp(out, th, P)
        if sensor == 'goes':
            ps, pt, vza = _goes_geometry(lat, lon, sat)
            scan = vza
        else:
            ps, pt = scan_geometry(sensor, scan or 0.0)
        if pix_given is not None:
            ps, pt = float(pix_given[0]), float(pix_given[1])
        day = solar_elevation(pd.DatetimeIndex([t]), lat, lon)[0] > math.radians(-5)
        f50_nadir = P['frp50_day'] if day else P['frp50_night']
        nadir_area = P['nadir_km'] ** 2
        # aggregate cells into pixels (along-scan ~ east-west, along-track ~ north-south)
        px = max(ps * 1000.0 / res, 1e-6)
        py = max(pt * 1000.0 / res, 1e-6)
        ox, oy = rng.random() * px, rng.random() * py
        cols = np.floor((np.arange(nx) + 0.5 + ox) / px).astype(int)
        rows = np.floor((np.arange(ny) + 0.5 + oy) / py).astype(int)
        npx, npy = cols.max() + 1, rows.max() + 1
        pix_frp = np.zeros((npy, npx))
        np.add.at(pix_frp, (rows[:, None].repeat(nx, 1), cols[None, :].repeat(ny, 0)), frp)
        area_km2 = ps * pt
        f50 = f50_nadir * area_km2 / nadir_area
        with np.errstate(divide='ignore'):
            pdet = np.where(pix_frp > 0, 1.0 / (1.0 + (f50 / np.maximum(pix_frp, 1e-12)) ** P['k']), 0.0)
        det = rng.random(pdet.shape) < pdet
        # cloud
        if isinstance(cloud, dict):
            cm = cloud.get(t, None)
            cmask_pix = _to_pix(cm, rows, cols, npy, npx) if cm is not None else np.zeros_like(det)
        else:
            cf = float(cloud) if isinstance(cloud, (int, float)) else P['cloud_frac']
            cmask_pix = _random_cloud(npy, npx, cf, P['cloud_scale_km'] / max(ps, pt), rng) if cf > 0 else \
                np.zeros_like(det)
        n_cloud = int((det & cmask_pix).sum())
        det &= ~cmask_pix
        if P['smoke_omission'] > 0:
            sm = (pix_frp > P['smoke_frp']) & (rng.random(det.shape) < P['smoke_omission'])
            det &= ~sm
        # commission
        nfa = rng.poisson(P['false_rate'] * npx * npy)
        fa = np.zeros_like(det)
        if nfa:
            fa[rng.integers(0, npy, nfa), rng.integers(0, npx, nfa)] = True
            fa &= ~cmask_pix & ~det
        iy, ix = np.nonzero(det | fa)
        # pixel centres back to model row/col
        r_c = (iy + 0.5) * py - oy - 0.5
        c_c = (ix + 0.5) * px - ox - 0.5
        tr = ls_meta.get('transform')
        for j in range(len(iy)):
            d = dict(time=t, satellite=sat, sensor=sensor, scan_deg=scan, daynight='D' if day else 'N',
                     row=float(r_c[j]), col=float(c_c[j]), pix_km_scan=ps, pix_km_track=pt,
                     frp_mw=float(pix_frp[iy[j], ix[j]] if det[iy[j], ix[j]] else rng.lognormal(0.0, 0.7)),
                     true_fire=bool(det[iy[j], ix[j]]))
            if tr is not None and len(tr) == 6:
                d['x'] = tr[0] + (c_c[j] + 0.5) * tr[1]
                d['y'] = tr[3] + (r_c[j] + 0.5) * tr[5]
            dets.append(d)
        active_cells = int((frp > 0).sum())
        flaming_cells = int((np.isfinite(arr) & (arr <= th) & ((th - arr) * 60.0 <= res / np.maximum(
            np.nan_to_num(out['rasters']['ros'], nan=1e-3), 1e-3) + 0.5)).sum())
        summ.append(dict(time=t, satellite=sat, scan_deg=scan, daynight='D' if day else 'N', pix_km2=area_km2,
                         n_det=int(det.sum()), n_false=int(fa.sum()), n_cloud_omitted=n_cloud,
                         expected_det=float(pdet.sum()),
                         fire_pixels=int((pix_frp > 0).sum()), active_cells=active_cells,
                         flaming_cells=flaming_cells, flaming_area_ha=flaming_cells * res * res / 1e4,
                         total_frp_mw=float(frp.sum()), cloud_frac=float(cmask_pix.mean()),
                         burned_ha=float((np.isfinite(arr) & (arr <= th)).sum() * res * res / 1e4)))
        if return_masks:
            m = (det | fa)[rows[:, None], cols[None, :]]
            masks.append(m)
    dd = pd.DataFrame(dets)
    so = pd.DataFrame(summ)
    res_d = dict(detections=dd, overpasses=so)
    if return_masks:
        res_d['masks'] = masks
    return res_d


def _to_pix(mask, rows, cols, npy, npx):
    m = np.zeros((npy, npx))
    c = np.zeros((npy, npx))
    np.add.at(m, (rows[:, None].repeat(len(cols), 1), cols[None, :].repeat(len(rows), 0)), mask.astype(float))
    np.add.at(c, (rows[:, None].repeat(len(cols), 1), cols[None, :].repeat(len(rows), 0)), 1.0)
    return (m / np.maximum(c, 1)) > 0.5


def _random_cloud(npy, npx, frac, scale_pix, rng):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((npy, npx)), max(scale_pix, 0.5), mode='wrap')
    thr = np.quantile(f, 1 - frac)
    return f > thr
