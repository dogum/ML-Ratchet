"""Landscape container (the shared contract): raster layers on a projected grid plus meta.

Layers (LANDFIRE conventions): fuel (FBFM40/FBFM13 numbers, 91-99 non-burnable), cc (%), ch (m), cbh (m), cbd (kg/m3),
elev (m), slope (degrees), aspect (degrees from N, the direction the slope faces). Meta: crs, transform (GDAL 6-tuple
or affine-like), res_m, nodata. Row 0 is the northern edge (north-up rasters).
"""
import json
import os

import numpy as np

LAYERS = ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'elev', 'slope', 'aspect')


class Landscape(dict):
    """dict of layers with a .meta dict. Construct from arrays: Landscape(fuel=..., elev=..., res_m=30)."""

    def __init__(self, meta=None, res_m=None, **layers):
        super().__init__()
        self.meta = dict(meta or {})
        if res_m is not None:
            self.meta['res_m'] = float(res_m)
        for k, v in layers.items():
            self[k] = np.asarray(v)
        self.meta.setdefault('nodata', -9999)
        self.meta.setdefault('crs', None)
        self.meta.setdefault('transform', None)
        if 'res_m' not in self.meta:
            raise ValueError('res_m (cell size in metres) is required')
        self.complete()

    @property
    def shape(self):
        return self['fuel'].shape

    @property
    def res(self):
        return float(self.meta['res_m'])

    def complete(self):
        """Fill missing layers with documented defaults; derive slope/aspect from elev when absent."""
        if 'fuel' not in self:
            raise ValueError('fuel layer is required')
        sh = self['fuel'].shape
        nod = self.meta.get('nodata', -9999)
        f = np.asarray(self['fuel']).astype(np.int64)
        f = np.where((f == nod) | (f < 0), 99, f)
        self['fuel'] = f
        if 'elev' not in self:
            self['elev'] = np.zeros(sh)
        if 'slope' not in self or 'aspect' not in self:
            s, a = slope_aspect(self['elev'], self.res)
            self.setdefault('slope', s)
            self.setdefault('aspect', a)
        for k, d in (('cc', 0.0), ('ch', 0.0), ('cbh', 0.0), ('cbd', 0.0)):
            if k not in self:
                self[k] = np.full(sh, d)
        for k in ('cc', 'ch', 'cbh', 'cbd', 'elev', 'slope', 'aspect'):
            a = np.asarray(self[k], float)
            a = np.where((a == nod) | ~np.isfinite(a), 0.0, a)
            self[k] = a
        # LANDFIRE integer scalings are the caller's job (see firedata); sanity checks here
        if np.nanmax(self['cbd']) > 2.0:
            raise ValueError('cbd looks scaled (kg/m3 expected; LANDFIRE stores 100*kg/m3)')
        if np.nanmax(self['ch']) > 150 or np.nanmax(self['cbh']) > 100:
            raise ValueError('ch/cbh look scaled (m expected; LANDFIRE stores 10*m)')
        return self

    def copy(self):
        return Landscape(meta=dict(self.meta), **{k: np.array(v) for k, v in self.items()})

    def subset(self, r0, r1, c0, c1):
        m = dict(self.meta)
        t = m.get('transform')
        if t is not None and len(t) == 6:
            # GDAL order (x0, dx, 0, y0, 0, dy)
            m['transform'] = (t[0] + c0 * t[1], t[1], t[2], t[3] + r0 * t[5], t[4], t[5])
        return Landscape(meta=m, **{k: np.array(v[r0:r1, c0:c1]) for k, v in self.items()})

    def save(self, path):
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.items()})
        with open(os.path.splitext(path)[0] + '.json', 'w') as f:
            json.dump(self.meta, f, default=str)

    @classmethod
    def load(cls, path, meta=None):
        """Load an .npz (+ sidecar .json meta) in the contract layout."""
        z = np.load(path, allow_pickle=False)
        js = os.path.splitext(path)[0] + '.json'
        m = dict(meta or {})
        if os.path.exists(js) and not meta:
            m = json.load(open(js))
        layers = {k: z[k] for k in z.files if k in LAYERS}
        return cls(meta=m, **layers)

    def xy(self, row, col):
        """Projected x, y of cell centres (needs a GDAL-order transform)."""
        t = self.meta['transform']
        return t[0] + (np.asarray(col) + 0.5) * t[1], t[3] + (np.asarray(row) + 0.5) * t[5]

    def rowcol(self, x, y):
        t = self.meta['transform']
        return (np.floor((np.asarray(y) - t[3]) / t[5]).astype(int),
                np.floor((np.asarray(x) - t[0]) / t[1]).astype(int))


def as_landscape(obj):
    if isinstance(obj, Landscape):
        return obj
    if isinstance(obj, str):
        return Landscape.load(obj)
    if isinstance(obj, tuple) and len(obj) == 2:
        layers, meta = obj
        return Landscape(meta=meta, **{k: v for k, v in layers.items() if k in LAYERS})
    if isinstance(obj, dict):
        meta = obj.get('meta', {})
        res = obj.get('res_m', meta.get('res_m'))
        m = dict(meta)
        if res is not None:
            m['res_m'] = res
        return Landscape(meta=m, **{k: v for k, v in obj.items() if k in LAYERS})
    raise TypeError('landscape must be a Landscape, dict of layers (+ meta/res_m), (layers, meta) or a .npz path')


def slope_aspect(elev, res):
    """Horn (1981) slope (degrees) and aspect (degrees from N, direction the slope faces; flat = 0)."""
    z = np.pad(np.asarray(elev, float), 1, mode='edge')
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)      # east positive
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)      # south positive (rows go south)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # downslope direction vector = -grad; north component = +dzdy (since grad north = -dzdy)
    asp = np.degrees(np.arctan2(-dzdx, dzdy)) % 360.0
    asp = np.where(slope < 1e-6, 0.0, asp)
    return slope, asp


def rasterize_lines(shape, lines, res, transform=None, width_cells=1):
    """4-connected rasterisation of polylines (list of (N,2) arrays in pixel (col,row) coordinates, or projected
    x,y when `transform` is given) into a bool mask. 4-connected so fire cannot leak through diagonal gaps."""
    m = np.zeros(shape, bool)
    for ln in lines:
        p = np.asarray(ln, float)
        if transform is not None:
            col = (p[:, 0] - transform[0]) / transform[1]
            row = (p[:, 1] - transform[3]) / transform[5]
            p = np.column_stack([col, row])
        for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
            n = int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)) * 4)) + 1
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            ci = np.floor(xs).astype(int)
            ri = np.floor(ys).astype(int)
            prev = None
            for r, c in zip(ri, ci):
                if prev is not None and r != prev[0] and c != prev[1]:
                    _set(m, prev[0], c)        # fill the corner to keep 4-connectivity
                _set(m, r, c)
                prev = (r, c)
    if width_cells > 1:
        from scipy.ndimage import binary_dilation
        m = binary_dilation(m, iterations=width_cells - 1)
    return m


def _set(m, r, c):
    if 0 <= r < m.shape[0] and 0 <= c < m.shape[1]:
        m[r, c] = True
