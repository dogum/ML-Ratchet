"""Build release/firecast.py: vendored feature code + numpy network + scoring (numpy/scipy only), from the same
source the training used (inspect.getsource, so the release cannot drift from training)."""
import inspect
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # training/ (for _paths)
from _paths import WORK  # noqa: E402  (puts src/ and shared/fire on sys.path)
import features as FT
import wxfeat as WX
import firecast_np as NP
import firedata.metrics as M

HEAD = '''"""FireCast v1.0: next-day wildfire spread probability on a 90 m grid (numpy + scipy only).

    import firecast as fc
    net = fc.FireCastNP('firecast_weights.npz')
    prob = fc.predict(net, day)          # P(cell burned by the next afternoon VIIRS overpass), calibrated on dev
    fc.nextday(day['now'], day['nxt'], prob, fc.CELL_KM2, thr=net.thr, window=fc.score_window(day['now']))

`day` is a dict (see README.md): now (bool HxW burned area at the issue overpass, 90 m, north up), prev (bool or None,
24 h earlier), fl (bool or None: FEDS active fire line), pix_pm / pix_am (VIIRS detections: row, col, ps, pt [km],
frp [MW] at the issue and the previous night overpass), land (optional LANDFIRE dict: fg, slope, aspect/2, cc),
wx (16 floats from feature_vector(gridMET day0, gridMET day1, grid_convergence_deg)), has_prev, first.
Feature code is generated from the training source (src/features.py, src/wxfeat.py); metrics from firedata.metrics.
"""
import numpy as np
from scipy import ndimage

CELL_KM2 = 90.0 * 90.0 / 1e6
'''


def src(objs):
    return '\n\n'.join(inspect.getsource(o) for o in objs)


parts = [HEAD,
         "RES = 90.0\nSPATIAL = %r\nN_SPATIAL = len(SPATIAL)\nN_FIRE = 6\nN_WX = 16\nVEC_PAIRS = %r\nWX_VEC_IDX = %r\n"
         % (FT.SPATIAL, FT.VEC_PAIRS, FT.WX_VEC_IDX),
         '# ---- features (src/features.py) ----',
         src([FT.fuel_group_code, FT.land_from_landscape, FT.pixel_raster, FT.full_fields, FT.dihedral, FT.stack,
              FT.fire_scalars, FT.req_m, FT.make_inputs]),
         '# ---- weather features (src/wxfeat.py) ----',
         "DAY_KEYS = %r\n" % WX.DAY_KEYS,
         src([WX.feature_vector, WX._nz]),
         '# ---- network (src/firecast_np.py; BatchNorm folded) ----',
         "LAYERS = %r\n" % NP.LAYERS,
         src([NP.conv3x3, NP.relu, NP.pool2, NP.up2, NP.FireCastNP]),
         '# ---- scoring (firedata.metrics) ----',
         src([M.average_precision, M.f1, M.dice, M.nextday, M.score_window]),
         '''

def predict(net, day, land=True):
    """Calibrated next-day burn probability for every cell of the day's grid."""
    X, s = make_inputs(day['now'], day.get('prev'), day.get('fl'), day.get('pix_pm'), day.get('pix_am'),
                       day.get('land') if land else None, day['wx'], bool(day.get('has_prev', False)),
                       bool(day.get('first', False)))
    return net.prob(X, s)
''']
open(f'{WORK}/release/firecast.py', 'w').write('\n\n'.join(parts))
print('ok')
