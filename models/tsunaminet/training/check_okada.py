"""Check okada.py against (1) GeoClaw's independent Okada implementation (clawpack/geoclaw dtopotools.py, BSD-3;
its two kernel functions re-typed here as the reference, unchanged in substance) and (2) NOAA MOST's own static
deformation grid for SIFT unit source acsza1, served by the NCTR compressed propagation server."""
import os, json, numpy as np, scipy.io
from _paths import DATA
from okada import okada_uz
POISSON = 0.25
def gc_ss(y1, y2, d, q):
    sn, cs = np.sin(d), np.cos(d); db = y2 * sn - q * cs; r = np.sqrt(y1 ** 2 + y2 ** 2 + q ** 2)
    a4 = 2.0 * POISSON / cs * (np.log(r + db) - sn * np.log(r + y2))
    return -(db * q / r / (r + y2) + q * sn / (r + y2) + a4 * sn) / (2.0 * np.pi)
def gc_ds(y1, y2, d, q):
    sn, cs = np.sin(d), np.cos(d); db = y2 * sn - q * cs; r = np.sqrt(y1 ** 2 + y2 ** 2 + q ** 2); xx = np.sqrt(y1 ** 2 + q ** 2)
    a5 = 4. * POISSON / cs * np.arctan((y2 * (xx + q * cs) + xx * (r + xx) * sn) / y1 / (r + xx) / cs)
    return -(db * q / r / (r + y1) + sn * np.arctan(y1 * y2 / q / r) - a5 * sn * cs) / (2.0 * np.pi)
def geoclaw_uz(X, Y, s, slip=1.0):
    D = np.pi / 180; L2M = 111.19492664455873e3   # use the same metres-per-degree as okada.py so only the kernels differ
    dip, rake, st = s['dip'] * D, s['rake'] * D, s['strike'] * D
    L, W = s['length'] * 1e3, s['width'] * 1e3; db = s['depth'] * 1e3 + W * np.sin(dip)
    xx = L2M * np.cos(D * Y) * (X - s['lon']); yy = L2M * (Y - s['lat'])
    x1 = xx * np.sin(st) + yy * np.cos(st); x2 = -(xx * np.cos(st) - yy * np.sin(st))
    p = x2 * np.cos(dip) + db * np.sin(dip); q = x2 * np.sin(dip) - db * np.cos(dip); h = L / 2
    f = gc_ss(x1 + h, p, dip, q) - gc_ss(x1 + h, p - W, dip, q) - gc_ss(x1 - h, p, dip, q) + gc_ss(x1 - h, p - W, dip, q)
    g = gc_ds(x1 + h, p, dip, q) - gc_ds(x1 + h, p - W, dip, q) - gc_ds(x1 - h, p, dip, q) + gc_ds(x1 - h, p - W, dip, q)
    return slip * (np.cos(rake) * f + np.sin(rake) * g)

out = {}
U = {u['name']: u for u in json.load(open(os.path.join(DATA, 'unit_sources_pacific.json')))}
rng = np.random.default_rng(0)
errs = []
for name in ['acsza1', 'acszb1', 'kisza22', 'kiszb22', 'cssza60', 'ntsza30']:
    if name not in U: continue
    s = U[name]; X, Y = np.meshgrid(np.linspace(s['lon'] - 2.5, s['lon'] + 2.5, 151), np.linspace(s['lat'] - 2, s['lat'] + 2, 121))
    a = okada_uz(X, Y, s); b = geoclaw_uz(X, Y, s)
    for rake in (0.0, 45.0):  # strike-slip part too
        s2 = dict(s, rake=rake); errs.append(np.abs(okada_uz(X, Y, s2) - geoclaw_uz(X, Y, s2)).max() / np.abs(geoclaw_uz(X, Y, s2)).max())
    errs.append(np.abs(a - b).max() / np.abs(b).max())
    out[name] = dict(max_uplift=float(a.max()), max_subsidence=float(a.min()), rel_maxdiff_vs_geoclaw=float(np.abs(a - b).max() / np.abs(b).max()))
out['geoclaw_worst_rel_diff_all_rakes'] = float(max(errs))
# NOAA MOST deformation for acsza1 (4' grid, compressed server crop)
f = scipy.io.netcdf_file(os.path.join(DATA, 'nctr_check', 'ac001a_src.nc'), 'r', mmap=False)
gl, gn = f.variables['grid_lat'][:].astype(float), f.variables['grid_lon'][:].astype(float)
dm = -f.variables['deformation'][:].astype(float); f.close()   # MOST stores the deformation positive DOWN (checked: r = -0.995 unflipped)
X, Y = np.meshgrid(gn, gl); mine = okada_uz(X, Y, U['acsza1'])
ok = np.isfinite(dm) & (np.abs(dm) < 1e3)
r = np.corrcoef(mine[ok], dm[ok])[0, 1]
out['noaa_most_acsza1'] = dict(corr=float(r), most_max=float(dm[ok].max()), most_min=float(dm[ok].min()), mine_max=float(mine[ok].max()),
                               mine_min=float(mine[ok].min()), rms_diff=float(np.sqrt(np.mean((mine[ok] - dm[ok]) ** 2))),
                               rel_rms=float(np.sqrt(np.mean((mine[ok] - dm[ok]) ** 2)) / np.abs(dm[ok]).max()),
                               volume_ratio=float(mine[ok].sum() / dm[ok].sum()))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(DATA, 'checks', 'okada_check.json'), 'w'), indent=1)
