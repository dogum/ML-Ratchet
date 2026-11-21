"""Exam guard: refuses exam fires' landscapes and fire-day weather for simulation inputs.

Rule (also in splits.json['rules']['simulation_inputs']):
  * landscape windows used to build simulated training data must not intersect any exam fire footprint
    (MTBS final perimeter + 2 km, any year; plus WildfireSpreadTS-only exam fires' VIIRS hull + 2 km, or their
    whole tile while the hull is not yet computed);
  * weather must not come from within 50 km of an exam footprint during that fire's active window
    [ignition - 3 d, last FEDS step + 3 d].
Dev fires, aux_rx fires and places/times that touch no exam fire are free.
"""
import os, json
import numpy as np, pandas as pd, shapely
P = os.path.join(os.environ.get('ML_RATCHET_FIRE') or os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'proc').replace(os.sep, '/')
LAND_BUF, WX_BUF, WX_DAYS = 2000.0, 50000.0, 3


class ExamLeakError(RuntimeError):
    pass


_cache = {}


def _load():
    if 'fp' in _cache: return _cache['fp']
    d = pd.read_parquet(f'{P}/guard/exam_footprints.parquet')
    g = shapely.from_wkb(d.wkb.values)
    _cache['fp'] = (d, g, shapely.STRtree(g))
    return _cache['fp']


def _box(bbox, crs=5070):
    x0, y0, x1, y1 = bbox
    if crs != 5070:
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, 5070, always_xy=True)
        X, Y = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
        px, py = tr.transform(X.ravel(), Y.ravel())
        return shapely.convex_hull(shapely.multipoints(np.column_stack([px, py])))
    return shapely.box(x0, y0, x1, y1)


def exam_hits_landscape(bbox, crs=5070):
    d, g, tree = _load()
    b = _box(bbox, crs)
    idx = tree.query(shapely.buffer(b, LAND_BUF))
    idx = [i for i in idx if shapely.intersects(g[i], shapely.buffer(b, LAND_BUF))]
    return d.iloc[idx]


def check_landscape(bbox, crs=5070):
    """Raise ExamLeakError if a landscape window (bbox in `crs`, default EPSG:5070 metres) touches an exam fire."""
    h = exam_hits_landscape(bbox, crs)
    if len(h):
        raise ExamLeakError(f'landscape window touches {len(h)} exam fire footprint(s): {list(h.fire_id[:5])}')
    return True


def check_weather(bbox, t0, t1, crs=5070):
    """Raise ExamLeakError if weather over bbox in [t0, t1] (UTC) is exam fire-day weather."""
    d, g, tree = _load()
    b = shapely.buffer(_box(bbox, crs), WX_BUF)
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    idx = [i for i in tree.query(b) if shapely.intersects(g[i], b)]
    hit = d.iloc[idx]
    hit = hit[(hit.t0 - pd.Timedelta(days=WX_DAYS) <= t1) & (hit.t1 + pd.Timedelta(days=WX_DAYS) >= t0)]
    if len(hit):
        raise ExamLeakError(f'weather box/time overlaps {len(hit)} exam fire-day window(s): {list(hit.fire_id[:5])}')
    return True


def exam_mask(grid):
    """Bool mask of cells of `grid` (firedata.grid dict) inside exam footprints (+2 km): blank these to reuse a window."""
    from .grid import burn
    x0, y0, x1, y1 = grid['bounds']
    h = exam_hits_landscape((x0, y0, x1, y1))
    if not len(h): return np.zeros(grid['shape'], bool)
    return burn(shapely.buffer(shapely.from_wkb(h.wkb.values), LAND_BUF), grid)


def is_exam(fire_id):
    from . import splits
    s = splits()
    return fire_id in set(s['fires']['exam']) or (fire_id in s['wsts'] and s['wsts'][fire_id]['split'] != 'dev') or \
        (fire_id in s['gofer'] and s['gofer'][fire_id]['split'] != 'dev')
