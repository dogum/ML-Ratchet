"""Result (48 h): no line 3972 cells outside; 4-ring time-blocked 3551 (leaks); 8-ring time-blocked 0; 4-ring non-burnable 0.
Upstream candidate U1: time-blocked suppression lines leak diagonally in FireSim.
Uniform grass (GR2=102), flat, steady wind. A closed ring line around a point ignition, built at t=0:
 (a) 4-connected ring as a suppression line (time-blocked), (b) the same ring as permanent non-burnable fuel (91),
 (c) an 8-connected-thick ("8-boundary") ring as a suppression line. Count cells burned outside the ring at 48 h."""
import os, sys, json
import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import firesim as fs
n = 121
ls = fs.generate_landscape((n, n), 90.0, kind='parametric', region='great_basin', rng=1)
fuel = np.asarray(ls['fuel']).copy(); fuel[:] = 102
for k in ('slope', 'aspect'):
    if k in ls.layers if hasattr(ls, 'layers') else False: pass
ls['fuel'][:] = 102
try:
    ls['slope'][:] = 0
except Exception: pass
start = pd.Timestamp('2020-08-01 18:00', tz='UTC')
w = fs.generate_weather(start - pd.Timedelta(hours=72), hours=120, climate='west_dry_summer', lon=-110, rng=3)
yy, xx = np.mgrid[:n, :n]; r = np.hypot(yy - n // 2, xx - n // 2)
inside = r < 7
ring4 = ndi.binary_dilation(inside) & ~inside            # 4-connected boundary: diagonal corners touch only
ring8 = ndi.binary_dilation(inside, structure=np.ones((3, 3), bool)) & ~inside
out_mask = ~(inside | ndi.binary_dilation(inside, structure=np.ones((3, 3), bool)))
ign = {'rowcol': [(n // 2, n // 2)]}
end = start + pd.Timedelta(hours=48)
res = {}
def burned_out(o):
    a = o['arrival_time']; b = np.isfinite(a) & (a <= 48)
    return int((b & out_mask).sum()), int(b.sum())
o = fs.simulate(ls, w, ign, start, end, spotting=False, return_perimeters=False, rng=0); res['no_line'] = burned_out(o)
o = fs.simulate(ls, w, ign, start, end, spotting=False, return_perimeters=False, rng=0,
                suppression={'lines': [{'mask': ring4, 'time': 0.0}]}); res['ring4_timeblocked'] = burned_out(o)
o = fs.simulate(ls, w, ign, start, end, spotting=False, return_perimeters=False, rng=0,
                suppression={'lines': [{'mask': ring8, 'time': 0.0}]}); res['ring8_timeblocked'] = burned_out(o)
ls['fuel'][ring4] = 91
o = fs.simulate(ls, w, ign, start, end, spotting=False, return_perimeters=False, rng=0); res['ring4_nonburnable'] = burned_out(o)
res = {k: dict(cells_outside=v[0], cells_burned=v[1]) for k, v in res.items()}
res['ring4_cells'] = int(ring4.sum()); res['ring8_cells'] = int(ring8.sum())
print(json.dumps(res, indent=1))
json.dump(res, open(_paths.WORK + '/results/upstream_U1_leak.json', 'w'), indent=1)
