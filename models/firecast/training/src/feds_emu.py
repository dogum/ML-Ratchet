"""FEDS emulator: turn VIIRS fire pixels (real or simulated) into FEDS-like cumulative perimeters and active fire lines
on a 90 m raster. FEDS (Chen et al. 2022) builds each 12-hourly perimeter as a concave hull of all active-fire
pixels so far, buffered by about half a VIIRS pixel, and the active fire line as the perimeter segments near the
newest pixels. Here (raster version):
    mask_k  = fill_holes(closing(disks(points <= k, radius b), radius a)) | mask_{k-1}
    line_k  = boundary(mask_k) within d of the pixels of step k
b, a, d are fitted on DEV fires against the real FEDS products (calib_feds.py -> out/calib_feds.json).
"""
import numpy as np
from scipy import ndimage

DEFAULT = dict(b_m=190.0, a_m=600.0, d_m=750.0)


def disk(r_cells):
    r = int(np.ceil(r_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r_cells * r_cells + 1e-9


def points_raster(rows, cols, shape):
    m = np.zeros(shape, bool)
    r = np.round(np.asarray(rows)).astype(int)
    c = np.round(np.asarray(cols)).astype(int)
    k = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    m[r[k], c[k]] = True
    return m


def footprint(pts, res, b_m):
    """Cells within b_m of any point cell."""
    if not pts.any():
        return pts.copy()
    d = ndimage.distance_transform_edt(~pts, sampling=res)
    return d <= b_m


def close_fill(m, res, a_m):
    if not m.any():
        return m
    if a_m > 0:
        p = int(np.ceil(a_m / res)) + 2
        mp = np.pad(m, p)
        dil = ndimage.distance_transform_edt(~mp, sampling=res) <= a_m
        clo = ndimage.distance_transform_edt(dil, sampling=res) > a_m
        m = clo[p:-p, p:-p] | m
    return ndimage.binary_fill_holes(m)


def boundary(m):
    if not m.any():
        return m
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def fire_line(mask, step_pts, res, d_m):
    """Perimeter cells within d_m of the pixels of the current step."""
    if not step_pts.any() or not mask.any():
        return np.zeros_like(mask)
    d = ndimage.distance_transform_edt(~step_pts, sampling=res)
    return boundary(mask) & (d <= d_m)


def sequence(step_points, shape, res=90.0, prm=None, init_mask=None):
    """step_points: list of bool point rasters (one per 12-hourly step, in time order).
    Returns lists of cumulative masks and fire lines."""
    P = dict(DEFAULT)
    P.update(prm or {})
    cum = np.zeros(shape, bool)
    prev = np.zeros(shape, bool) if init_mask is None else init_mask.copy()
    masks, lines = [], []
    for pts in step_points:
        cum |= pts
        if init_mask is None:
            m = close_fill(footprint(cum, res, P['b_m']), res, P['a_m']) | prev
        else:      # continue from an observed perimeter: hull over the old perimeter and the new pixels together
            m = close_fill(footprint(cum, res, P['b_m']) | prev, res, P['a_m']) | prev
        masks.append(m)
        lines.append(fire_line(m, pts, res, P['d_m']))
        prev = m
    return masks, lines
