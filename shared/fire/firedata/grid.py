"""Grids and rasterisation. All fire products live on EPSG:5070 (NAD83 / CONUS Albers, metres)."""
import numpy as np, shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin

CRS = 'EPSG:5070'


def grid_from_window(x0, y0, x1, y1, res=90.0):
    nx = int(round((x1 - x0) / res)); ny = int(round((y1 - y0) / res))
    return dict(crs=CRS, res_m=float(res), transform=tuple(from_origin(x0, y1, res, res))[:6], shape=(ny, nx),
                bounds=(x0, y0, x0 + nx * res, y1))


def coarsen(g, k):
    x0, y0, x1, y1 = g['bounds']
    return grid_from_window(x0, y1 - (g['shape'][0] // k) * k * g['res_m'], x0 + (g['shape'][1] // k) * k * g['res_m'],
                            y1, g['res_m'] * k)


def burn(geoms, g, all_touched=False):
    """Rasterise polygons (shapely, EPSG:5070) onto grid g -> bool mask. Cell centre rule unless all_touched."""
    geoms = [x for x in np.atleast_1d(geoms) if x is not None and not shapely.is_empty(x)]
    if not geoms: return np.zeros(g['shape'], bool)
    from affine import Affine
    return rasterize(geoms, out_shape=g['shape'], transform=Affine(*g['transform']), all_touched=all_touched,
                     dtype='uint8').astype(bool)


def cell_xy(g):
    ny, nx = g['shape']; a, _, c, _, e, f = g['transform']
    x = c + a * (np.arange(nx) + 0.5); y = f + e * (np.arange(ny) + 0.5)
    return np.meshgrid(x, y)
