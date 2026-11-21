"""Per-ignition input features, computed the same way for simulated (training) and real (dev / exam) ignitions.

Landscape (coarse summary, documented): the 250 m national LF2016 grids (nearest-neighbour samples of the 30 m
LANDFIRE rasters, data/nat). Fuel class at the ignition pixel (9 classes: GR GS SH TU TL SB + non-burnable urban 91,
agriculture 93, other 92/98/99/nodata), class fractions in a 5 x 5 (1.25 km) and 13 x 13 (3.25 km) box, mean canopy
cover, canopy height and slope in the 5 x 5 box, slope at the pixel.
Weather: gridMET daily on the discovery day and the day before (erc, bi, fm100, fm1000, vs, rmin, rmax, tmmx, tmmn,
pr, vpd).
Record: NWCG cause class (3), federal reporting agency (1), discovery hour (sin, cos, missing flag), day of year
(sin, cos), region (9).
"""
from common import *

X0, Y1, R = -2362425.0, 3177435.0, 250.0
FCLS = ['GR', 'GS', 'SH', 'TU', 'TL', 'SB', 'NBU', 'NBA', 'NBO']
WXF = ['erc', 'bi', 'fm100', 'fm1000', 'vs', 'rmin', 'rmax', 'tmmx', 'tmmn', 'pr', 'vpd']


def fuel_class_lut():
    lut = np.full(256, 8, np.int8)
    for c in range(256):
        if 101 <= c <= 109: lut[c] = 0
        elif 121 <= c <= 124: lut[c] = 1
        elif 141 <= c <= 149: lut[c] = 2
        elif 161 <= c <= 165: lut[c] = 3
        elif 181 <= c <= 189: lut[c] = 4
        elif 201 <= c <= 204: lut[c] = 5
        elif c == 91: lut[c] = 6
        elif c == 93: lut[c] = 7
    return lut


def land_features(x, y, chunk=50000):
    """x, y: EPSG:5070 metres (the point itself). Returns DataFrame of landscape features."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    r = np.floor((Y1 - y) / R).astype(np.int64)
    c = np.floor((x - X0) / R).astype(np.int64)
    out = {}
    lut = fuel_class_lut()
    d13 = np.arange(-6, 7)
    for layer in ['fuel', 'cc', 'ch', 'slope']:
        a = np.load(f'{DATA}/nat/{layer}.npz')['a']
        NY, NX = a.shape
        res = {}
        for s in range(0, n, chunk):
            rr = np.clip(r[s:s + chunk, None, None] + d13[None, :, None], 0, NY - 1)
            cc = np.clip(c[s:s + chunk, None, None] + d13[None, None, :], 0, NX - 1)
            p = a[rr, cc]                                   # m x 13 x 13
            if layer == 'fuel':
                q = lut[p]
                cen = q[:, 6, 6]
                p5 = q[:, 4:9, 4:9].reshape(len(q), -1)
                p13 = q.reshape(len(q), -1)
                for k, nm in enumerate(FCLS):
                    res.setdefault(f'pt_{nm}', []).append((cen == k).astype(np.float32))
                    res.setdefault(f'f5_{nm}', []).append((p5 == k).mean(1).astype(np.float32))
                    res.setdefault(f'f13_{nm}', []).append((p13 == k).mean(1).astype(np.float32))
            else:
                p5 = p[:, 4:9, 4:9].reshape(len(p), -1).astype(np.float32)
                if layer == 'cc':
                    p5 = np.where(p5 > 100, 0, p5)
                res.setdefault(f'{layer}5', []).append(p5.mean(1))
                if layer == 'slope':
                    res.setdefault('slope_pt', []).append(p[:, 6, 6].astype(np.float32))
        for k, v in res.items():
            out[k] = np.concatenate(v)
        del a
    return pd.DataFrame(out)


def record_features(cause, fed, hour, doy, region):
    hour = np.asarray(hour, float)
    miss = ~np.isfinite(hour)
    h = np.where(miss, 12.0, hour)
    f = dict(hr_sin=np.sin(2 * np.pi * h / 24) * ~miss, hr_cos=np.cos(2 * np.pi * h / 24) * ~miss,
             hr_miss=miss.astype(float), doy_sin=np.sin(2 * np.pi * np.asarray(doy) / 365.25),
             doy_cos=np.cos(2 * np.pi * np.asarray(doy) / 365.25), fed=np.asarray(fed, float))
    cause = np.asarray(cause).astype(str)
    for k, nm in zip(CAUSES, ['c_hum', 'c_nat', 'c_miss']):
        f[nm] = (cause == k).astype(float)
    region = np.asarray(region).astype(str)
    for rg in REGIONS:
        f[f'r_{rg}'] = (region == rg).astype(float)
    return pd.DataFrame(f)


def wx_features(d0, dm1):
    """d0, dm1: DataFrames with WXF columns for the discovery day and the day before."""
    f = {}
    for v in WXF:
        f[v] = np.asarray(d0[v], float)
        f[v + '_m1'] = np.asarray(dm1[v], float)
    return pd.DataFrame(f)


def fod_features(d):
    """Features for real FPA-FOD records d (from common.fod_records, with gridMET merged)."""
    import glob
    L = land_features(d.x.values, d.y.values)
    W = wx_features(d[WXF], d[[v + '_m1' for v in WXF]].rename(columns=lambda s: s[:-3]))
    Rr = record_features(d.cause.values, d.fed.values, d.hour.values, d.DISCOVERY_DOY.values, d.region.values)
    return pd.concat([L.reset_index(drop=True), W.reset_index(drop=True), Rr.reset_index(drop=True)], axis=1)


def fod_gridmet(d):
    import glob
    wx = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{FP}/weather/gridmet/fod_*.parquet'))])
    return d.merge(wx, on='FOD_ID', how='left')
