"""Training landscapes.

generate_landscape(kind='parametric') builds a random but plausible landscape:
  * terrain: fractal (spectral synthesis, power-law spectrum slope `beta`) scaled to a relief, plus an optional
    large-scale tilt; slope and aspect by Horn's method;
  * vegetation: a site-wetness index (elevation, northness, topographic position, noise) sets the life form
    (grass, grass-shrub, shrub, timber-understory, timber-litter) through thresholds that shift with `region`;
    patches (smoothed random fields) choose specific Scott & Burgan models from class-conditional weights;
  * canopy for timber (and tall shrub) cells: correlated cc, ch, cbh, cbd drawn from LANDFIRE-like ranges;
  * non-burnable: bare/rock on high peaks (99), lakes in low basins (98), rivers along flow paths (98), roads (91),
    agriculture (93) and urban (91) patches; lines are 4-connected so fire cannot pass diagonally.
generate_landscape(kind='real') cuts a random window from B's real LANDFIRE tiles that is not an exam window
(through fire/firedata when it exists; raises if the data package is missing).
"""
import numpy as np

from .landscape import Landscape, slope_aspect, rasterize_lines
from .fuels import number_of

CLASS_MODELS = {
    'grass': {'GR1': .25, 'GR2': .4, 'GR3': .05, 'GR4': .1, 'GR5': .05, 'GR6': .05, 'GR7': .07, 'GR8': .02,
              'GR9': .01},
    'grass_shrub': {'GS1': .35, 'GS2': .45, 'GS3': .1, 'GS4': .1},
    'shrub': {'SH1': .12, 'SH2': .25, 'SH3': .05, 'SH4': .08, 'SH5': .15, 'SH6': .05, 'SH7': .18, 'SH8': .06,
              'SH9': .06},
    'timber_understory': {'TU1': .3, 'TU2': .15, 'TU3': .05, 'TU4': .05, 'TU5': .45},
    'timber_litter': {'TL1': .1, 'TL2': .08, 'TL3': .2, 'TL4': .1, 'TL5': .15, 'TL6': .07, 'TL7': .04, 'TL8': .16,
                      'TL9': .05, 'SB1': .03, 'SB2': .02},
}
# cumulative wetness thresholds for the five classes (dry -> wet) by region
REGIONS = {
    'west_dry': [0.30, 0.45, 0.70, 0.88],
    'mixed_conifer': [0.12, 0.22, 0.40, 0.70],
    'chaparral': [0.15, 0.25, 0.80, 0.92],
    'great_basin': [0.35, 0.65, 0.92, 0.97],
    'southeast': [0.20, 0.30, 0.45, 0.75],
}
CLASSES = ['grass', 'grass_shrub', 'shrub', 'timber_understory', 'timber_litter']


def fractal_field(shape, beta, rng):
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    ph = rng.standard_normal((ny, nx // 2 + 1)) + 1j * rng.standard_normal((ny, nx // 2 + 1))
    f = np.fft.irfft2(amp * ph, s=(ny, nx))
    f -= f.mean()
    return f / (f.std() + 1e-12)


def _smooth_noise(shape, scale_cells, rng):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal(shape), max(scale_cells, 0.5), mode='reflect')
    return (f - f.mean()) / (f.std() + 1e-12)


def _flow_lines(elev, n, rng, min_len=20):
    """Steepest-descent paths from random high points (D8), as pixel polylines."""
    ny, nx = elev.shape
    lines = []
    for _ in range(n):
        r, c = int(rng.integers(0, ny)), int(rng.integers(0, nx))
        path = [(c + 0.5, r + 0.5)]
        for _s in range(4 * (ny + nx)):
            best = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ny and 0 <= cc < nx and elev[rr, cc] < elev[r, c] and (
                            best is None or elev[rr, cc] < elev[best]):
                        best = (rr, cc)
            if best is None:
                break
            r, c = best
            path.append((c + 0.5, r + 0.5))
        if len(path) >= min_len:
            lines.append(np.array(path))
    return lines


def generate_landscape(shape=(256, 256), res_m=60.0, rng=None, kind='parametric', region='west_dry', relief_m=None,
                       beta=None, base_elev_m=None, roads=None, rivers=None, lakes=True, agriculture=True, urban=True,
                       patch_scale_m=600.0, **kw):
    """Random training landscape (contract layout). Returns a Landscape; .meta['labels'] holds the latent fields
    and generator settings. kind='real' delegates to realdata.random_landscape (B's non-exam LANDFIRE windows)."""
    if kind == 'real':
        from . import realdata
        return realdata.random_landscape(shape=shape, res_m=res_m, rng=rng, **kw)
    rng = np.random.default_rng(rng)
    ny, nx = shape
    relief = relief_m if relief_m is not None else float(rng.choice([50, 200, 500, 900, 1500]) * rng.uniform(0.7, 1.3))
    beta = beta if beta is not None else float(rng.uniform(2.6, 3.4))
    z = fractal_field(shape, beta, rng)
    tilt = rng.normal(0, 0.3, 2)
    yy, xx = np.mgrid[0:ny, 0:nx]
    z = z + tilt[0] * (yy / ny - 0.5) * 2 + tilt[1] * (xx / nx - 0.5) * 2
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    base = base_elev_m if base_elev_m is not None else float(rng.uniform(300, 2000))
    elev = base + relief * z
    slope, aspect = slope_aspect(elev, res_m)
    # site wetness: higher, north-facing, valley bottoms wetter; plus noise
    from scipy.ndimage import uniform_filter
    tpi = elev - uniform_filter(elev, size=max(3, int(1500 / res_m)) | 1)
    north = np.cos(np.radians(aspect)) * np.sin(np.radians(np.minimum(slope, 45)))
    wet = (0.45 * z + 0.25 * north - 0.2 * tpi / (np.std(tpi) + 1e-9) * 0.3 + 0.35 * _smooth_noise(shape, patch_scale_m / res_m, rng)
           + 0.15 * _smooth_noise(shape, 3 * patch_scale_m / res_m, rng))
    q = np.argsort(np.argsort(wet.ravel())).reshape(shape) / wet.size     # uniform ranks
    thr = REGIONS[region] if isinstance(region, str) else region
    cls = np.digitize(q, thr)
    fuel = np.zeros(shape, np.int64)
    patch = _smooth_noise(shape, patch_scale_m / res_m / 2, rng)
    pq = np.argsort(np.argsort(patch.ravel())).reshape(shape) / patch.size
    for ci, cname in enumerate(CLASSES):
        sel = cls == ci
        if not sel.any():
            continue
        models = list(CLASS_MODELS[cname].items())
        rng.shuffle(models)
        w = np.array([m[1] for m in models])
        # perturb class weights per landscape (variety)
        w = w * rng.lognormal(0, 0.5, len(w))
        cw = np.cumsum(w / w.sum())
        idx = np.minimum(np.searchsorted(cw, pq[sel]), len(models) - 1)
        fuel[sel] = np.array([number_of(m[0]) for m in models])[idx]
    # canopy
    cc = np.zeros(shape)
    ch = np.zeros(shape)
    cbh = np.zeros(shape)
    cbd = np.zeros(shape)
    timber = cls >= 3
    n1 = _smooth_noise(shape, patch_scale_m / res_m, rng)
    n2 = _smooth_noise(shape, patch_scale_m / res_m, rng)
    cc[timber] = np.clip(55 + 18 * n1[timber] + 10 * (q[timber] - 0.8) * 5, 15, 95)
    ch[timber] = np.clip(22 + 7 * n2[timber], 8, 45)
    cbh[timber] = np.clip(np.exp(np.log(3.0) + 0.6 * _smooth_noise(shape, patch_scale_m / res_m, rng)[timber]),
                          0.3, 0.6 * ch[timber])
    cbd[timber] = np.clip(0.04 + 0.0022 * (cc[timber] - 15) * np.exp(0.3 * n1[timber]), 0.02, 0.35)
    # some shrub cells with low canopy (woodland)
    wood = (cls == 2) & (n1 > 1.0)
    cc[wood] = np.clip(25 + 10 * n1[wood], 10, 50)
    ch[wood] = np.clip(8 + 3 * n2[wood], 4, 15)
    cbh[wood] = 1.0
    cbd[wood] = 0.06
    # non-burnable features
    nb_labels = {}
    if z.max() > 0 and relief > 600:
        rock = z > rng.uniform(0.93, 0.98)
        fuel[rock] = 99
        nb_labels['rock_cells'] = int(rock.sum())
    if lakes and rng.random() < 0.6:
        lake = (z < np.quantile(z, rng.uniform(0.005, 0.03))) & (slope < 10)
        fuel[lake] = 98
        nb_labels['lake_cells'] = int(lake.sum())
    rivers = rivers if rivers is not None else int(rng.integers(0, 3))
    if rivers:
        rl = _flow_lines(elev, int(rivers) * 6, rng, min_len=max(20, int(0.3 * min(ny, nx))))[:int(rivers)]
        if rl:
            m = rasterize_lines(shape, rl, res_m, None, 1 + int(res_m < 45))
            fuel[m] = 98
            nb_labels['river_cells'] = int(m.sum())
    roads = roads if roads is not None else int(rng.integers(0, 4))
    if roads:
        rl = []
        for _ in range(int(roads)):
            pts = [(rng.uniform(0, nx), rng.uniform(0, ny))]
            ang = rng.uniform(0, 2 * np.pi)
            for _k in range(int(rng.integers(3, 8))):
                ang += rng.normal(0, 0.5)
                L = rng.uniform(0.1, 0.4) * max(nx, ny)
                pts.append((pts[-1][0] + L * np.cos(ang), pts[-1][1] + L * np.sin(ang)))
            rl.append(np.array(pts))
        m = rasterize_lines(shape, rl, res_m, None, 1)
        fuel[m] = 91
        nb_labels['road_cells'] = int(m.sum())
    if agriculture and rng.random() < 0.35:
        ag = (_smooth_noise(shape, 2 * patch_scale_m / res_m, rng) > 1.4) & (slope < 6)
        fuel[ag] = 93
        nb_labels['ag_cells'] = int(ag.sum())
    if urban and rng.random() < 0.2:
        ur = (_smooth_noise(shape, 2 * patch_scale_m / res_m, rng) > 1.9) & (slope < 10)
        fuel[ur] = 91
        nb_labels['urban_cells'] = int(ur.sum())
    nb = (fuel >= 91) & (fuel <= 99)
    for a in (cc, ch, cbh, cbd):
        a[nb] = 0.0
    meta = dict(res_m=float(res_m), crs=None, transform=(0.0, float(res_m), 0.0, float(ny * res_m), 0.0, -float(res_m)),
                nodata=-9999, generator='parametric',
                labels=dict(region=region, relief_m=relief, beta=beta, base_elev_m=base, class_thresholds=list(thr),
                            **nb_labels))
    ls = Landscape(meta=meta, fuel=fuel, cc=cc, ch=ch, cbh=cbh, cbd=cbd, elev=elev, slope=slope, aspect=aspect)
    ls.meta['labels']['life_form'] = cls.astype(np.int8)
    return ls


def fuel_treatment(ls, mask, kind='thin', **kw):
    """Edit a landscape inside `mask`: 'thin' (cbd x 0.5, cbh + 3 m, cc x 0.6), 'prescribed_burn' (surface fuel ->
    TL1 or GR1, cbh + 2 m), 'masticate' (shrub -> TU1/SB1-like low load), 'fuelbreak' (fuel -> GR1, no canopy),
    'nonburnable' (99). Returns a new Landscape."""
    out = ls.copy()
    m = np.asarray(mask, bool)
    if kind == 'thin':
        out['cbd'][m] *= kw.get('cbd_factor', 0.5)
        out['cbh'][m] += kw.get('cbh_add', 3.0)
        out['cc'][m] *= kw.get('cc_factor', 0.6)
    elif kind == 'prescribed_burn':
        f = out['fuel']
        timber = m & (f >= 161) & (f < 200)
        f[timber] = number_of('TL1')
        f[m & ~timber & ((f < 91) | (f > 99))] = number_of('GR1')
        out['cbh'][m] += kw.get('cbh_add', 2.0)
    elif kind == 'masticate':
        f = out['fuel']
        f[m & (f >= 141) & (f <= 149)] = number_of('TU1')
    elif kind == 'fuelbreak':
        out['fuel'][m] = number_of('GR1')
        for k in ('cc', 'ch', 'cbh', 'cbd'):
            out[k][m] = 0.0
    elif kind == 'nonburnable':
        out['fuel'][m] = 99
    else:
        raise ValueError(kind)
    return out
