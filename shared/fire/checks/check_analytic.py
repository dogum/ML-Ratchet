"""Analytic checks of the spread solver.

1. Ellipse: uniform fuel, wind and moisture, point ignition. The burned area after time T must be the BehavePlus
   ellipse (head + back = L, L/W from 1 + 0.25 UE): area ratio, head and back ROS (from arrival times interpolated
   along the spread axis), L/W from the second moments of the burned cells. Several resolutions (30/60/90 m) and
   wind directions 0-45 deg (grid-orientation bias), four solver settings.
2. Two-fuel strip: a line fire crossing from fuel A into fuel B; arrival = x/R_A, then X1/R_A + (x-X1)/R_B.
3. Fuel break: a 2-cell non-burnable strip must stop the fire (no spotting); with a gap, arrival behind the break
   must equal the shortest path through the gap (no-wind circle metric).
4. Refraction: no wind, point ignition in fuel A, targets in fuel B: arrival = Fermat minimum over the interface.
5. Time-varying rate: no wind, moisture changes at t = 2 h; radius = R1 t then R1 2 + R2 (t - 2).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar

from _common import write
import firesim as fs

T0 = pd.Timestamp('2021-08-01', tz='UTC')
IDX = pd.date_range(T0, periods=3000, freq='h')


def wx(ws=0.0, wd=270.0, m=(4.0, 5.0, 6.0), steps=None):
    w = pd.DataFrame(dict(T=30.0, RH=15.0, wind_speed=ws, wind_dir=wd, precip=0.0, m1h=m[0], m10h=m[1], m100h=m[2],
                          mherb=60.0, mwoody=90.0), index=IDX)
    if steps:
        for t, mm in steps:
            w.loc[w.index >= T0 + pd.Timedelta(hours=t), ['m1h', 'm10h', 'm100h']] = mm
    return w


METHODS = {'mtt16+simplex (default)': dict(method='mtt'), 'mtt32+simplex (level_set)': dict(method='level_set'),
           'mtt16 plain (Finney-style stencil)': dict(method='mtt', params=dict(simplex=False)),
           'ca8 (8-neighbour)': dict(method='ca')}


def ellipse_case(fuel, ws, wd, res, cfg, length_cells=180):
    b1 = fs.behave(fuel, m1=4, m10=5, m100=6, mherb=60, mwoody=90, wind_speed=ws)
    T = length_cells * res / ((b1['ros_mmin'] + b1['ros_back_mmin']) * 60)
    b = fs.behave(fuel, m1=4, m10=5, m100=6, mherb=60, mwoody=90, wind_speed=ws, elapsed_h=T)
    n = int(length_cells * 1.3) + 20
    ls = fs.Landscape(fuel=np.full((n, n), fuel), res_m=res)
    az = np.radians((wd + 180) % 360)
    off = (b['ros_mmin'] - b['ros_back_mmin']) * T * 60 / 2 / res
    r0 = int(round(n / 2 + off * np.cos(az)))
    c0 = int(round(n / 2 - off * np.sin(az)))
    p = dict(cfg.get('params', {}))
    out = fs.simulate(ls, wx(ws, wd), {'rowcol': [(r0, c0)]}, T0, T0 + pd.Timedelta(hours=T), moisture='given',
                      method=cfg['method'], params=p, return_perimeters=False)
    a = out['arrival_time']
    burned = np.isfinite(a)
    area = burned.sum() * res * res / 1e4
    # head/back ROS from interpolated arrival along the axis at 80 % of the expected distance
    def ros_at(dist_m, ang):
        rr = r0 - dist_m / res * np.cos(ang)
        cc = c0 + dist_m / res * np.sin(ang)
        t = map_coordinates(np.nan_to_num(a, nan=1e9), [[rr], [cc]], order=1)[0]
        return dist_m / (t * 60) if t < 1e8 else np.nan
    dh = 0.8 * b['ros_mmin'] * T * 60
    db = 0.8 * b['ros_back_mmin'] * T * 60
    rh = ros_at(dh, az)
    rb = ros_at(db, az + np.pi) if db > 3 * res else np.nan
    # L/W from second moments of burned cells
    yy, xx = np.nonzero(burned)
    cov = np.cov(np.vstack([xx, yy]))
    ev = np.sort(np.linalg.eigvalsh(cov))
    lw = float(np.sqrt(ev[1] / ev[0]))
    return dict(fuel=fs.code_of(fuel), wind_ms=ws, wind_from=wd, res_m=res, lw_theory=b['lw_ratio'], lw_sim=lw,
                area_theory_ha=b['area_ha'], area_sim_ha=area, area_ratio=area / b['area_ha'],
                head_ros_theory=b['ros_mmin'], head_ros_sim=rh, head_ratio=rh / b['ros_mmin'],
                back_ros_theory=b['ros_back_mmin'], back_ratio=(rb / b['ros_back_mmin']) if np.isfinite(rb) else None,
                hours=T, wall_s=out['wall_s'])


ell = []
cases = [(102, 0.0), (102, 3.0), (165, 6.0), (102, 8.0), (102, 13.0)]
for name, cfg in METHODS.items():
    for fuel, ws in cases:
        for res in (30.0, 60.0, 90.0):
            for wd in (270.0, 262.5, 255.0, 247.5, 240.0, 232.5, 225.0):
                if ws == 0 and wd != 270.0:
                    continue
                if res != 60.0 and wd not in (270.0, 247.5, 225.0):
                    continue
                r = ellipse_case(fuel, ws, wd, res, cfg, length_cells={30.0: 180, 60.0: 90, 90.0: 60}[res])
                r['solver'] = name
                ell.append(r)
        print(name, fuel, ws, 'done', flush=True)
df = pd.DataFrame(ell)
summ = {}
for name, g in df.groupby('solver'):
    s = dict(n=len(g), area_ratio_mean=float(g.area_ratio.mean()), area_ratio_min=float(g.area_ratio.min()),
             area_ratio_max=float(g.area_ratio.max()), head_ratio_mean=float(g.head_ratio.mean()),
             head_ratio_min=float(g.head_ratio.min()), head_ratio_max=float(g.head_ratio.max()),
             lw_rel_err_max=float(np.max(np.abs(g.lw_sim / g.lw_theory - 1))))
    bias = []
    for (fu, ws, res), gg in g.groupby(['fuel', 'wind_ms', 'res_m']):
        if len(gg) > 1:
            bias.append(float(gg.area_ratio.max() - gg.area_ratio.min()))
    s['orientation_bias_area_ratio_range_max'] = max(bias) if bias else 0.0
    by_lw = g.groupby(g.lw_theory.round(1)).area_ratio.agg(['min', 'mean', 'max'])
    s['area_ratio_by_lw'] = {str(k): dict(v) for k, v in by_lw.iterrows()}
    summ[name] = s
    print(name, {k: v for k, v in s.items() if k != 'area_ratio_by_lw'})

# ---- 2. two-fuel strip (line ignition, wind from west)
res = 30.0
ny, nx = 40, 300
fuel = np.full((ny, nx), 102)
X1 = 120
fuel[:, X1:] = 165
ls = fs.Landscape(fuel=fuel, res_m=res)
ws = 4.0
bA = fs.behave(102, m1=4, m10=5, m100=6, mherb=60, wind_speed=ws)
bB = fs.behave(165, m1=4, m10=5, m100=6, mherb=60, mwoody=90, wind_speed=ws)
T = (X1 * res / bA['ros_mmin'] + (nx - X1 - 5) * res / bB['ros_mmin']) / 60.0
out = fs.simulate(ls, wx(ws, 270.0), {'rowcol': [(r, 0) for r in range(ny)]}, T0, T0 + pd.Timedelta(hours=T + 0.5),
                  moisture='given', return_perimeters=False)
a = out['arrival_time'][ny // 2]
x = np.arange(nx) * res
hand = np.where(np.arange(nx) < X1, x / bA['ros_mmin'], X1 * res / bA['ros_mmin'] + (x - X1 * res) / bB['ros_mmin']) / 60
# MTT samples the fuel per cell: the interface sits at the cell boundary x = (X1 - 0.5) res
hand2 = np.where(np.arange(nx) < X1, x / bA['ros_mmin'],
                 (X1 - 0.5) * res / bA['ros_mmin'] + (x - (X1 - 0.5) * res) / bB['ros_mmin']) / 60
ok = np.isfinite(a)
strip = dict(R_A_mmin=bA['ros_mmin'], R_B_mmin=bB['ros_mmin'], max_abs_err_h=float(np.nanmax(np.abs(a - hand2)[ok])),
             max_rel_err=float(np.nanmax((np.abs(a - hand2) / np.maximum(hand2, 1e-9))[ok][5:])),
             arrival_end_sim_h=float(a[nx - 6]), arrival_end_hand_h=float(hand2[nx - 6]),
             note='interface at the cell edge between columns X1-1 and X1')
print('strip', strip)

# ---- 3. fuel break with and without a gap (no wind -> circular metric, R = R0)
n = 160
res = 30.0
b0 = fs.behave(102, m1=4, m10=5, m100=6, mherb=60, wind_speed=0.0)
R = b0['ros_mmin']
fuel = np.full((n, n), 102)
fuel[:, 80:82] = 99
ls = fs.Landscape(fuel=fuel, res_m=res)
T = 1.2 * n * res / R / 60
o1 = fs.simulate(ls, wx(0.0), {'rowcol': [(80, 40)]}, T0, T0 + pd.Timedelta(hours=T), moisture='given',
                 return_perimeters=False)
crossed = int(np.isfinite(o1['arrival_time'][:, 82:]).sum())
fuel2 = fuel.copy()
fuel2[100:104, 80:82] = 102            # a 4-cell gap (rows 100-103)
ls2 = fs.Landscape(fuel=fuel2, res_m=res)
T2 = 3 * n * res / R / 60
o2 = fs.simulate(ls2, wx(0.0), {'rowcol': [(80, 40)]}, T0, T0 + pd.Timedelta(hours=T2), moisture='given',
                 return_perimeters=False)
a2 = o2['arrival_time']
errs = []
ign = np.array([80.0, 40.0])
for tr in range(10, 150, 7):
    for tc in range(90, 155, 8):
        tgt = np.array([tr, tc], float)
        # shortest path through the gap opening (rows 99.5..103.5 at the break column edges)
        f = lambda y: (np.hypot(y - ign[0], 79.5 - ign[1]) + 2.0 + np.hypot(tgt[0] - y, tgt[1] - 81.5)) * res / R / 60
        y = minimize_scalar(f, bounds=(99.5 + 0.5, 103.5 - 0.5), method='bounded').x
        hand = f(y)
        if np.isfinite(a2[tr, tc]):
            errs.append((a2[tr, tc] - hand) / hand)
errs = np.array(errs)
fb = dict(cells_burned_beyond_solid_break=crossed, gap_targets=len(errs), gap_rel_err_median=float(np.median(errs)),
          gap_rel_err_max_abs=float(np.max(np.abs(errs))), gap_rel_err_p90_abs=float(np.quantile(np.abs(errs), 0.9)))
print('fuel break', fb)

# ---- 4. refraction (no wind): fuel A (GR2) left, fuel B (TL3, slower) right
n = 200
fuel = np.full((n, n), 102)
fuel[:, 100:] = 183
ls = fs.Landscape(fuel=fuel, res_m=res)
RA = fs.behave(102, m1=4, m10=5, m100=6, mherb=60)['ros_mmin']
RB = fs.behave(183, m1=4, m10=5, m100=6)['ros_mmin']
T = (100 * res / RA + 100 * res / RB) / 60 * 1.2
o = fs.simulate(ls, wx(0.0), {'rowcol': [(100, 60)]}, T0, T0 + pd.Timedelta(hours=T), moisture='given',
                return_perimeters=False)
a = o['arrival_time']
errs = []
for tr in range(10, 191, 12):
    for tc in range(110, 195, 12):
        f = lambda y: (np.hypot(y - 100, 99.5 - 60) / RA + np.hypot(tr - y, tc - 99.5) / RB) * res / 60
        y = minimize_scalar(f, bounds=(-200, 400), method='bounded').x
        h = f(y)
        if np.isfinite(a[tr, tc]):
            errs.append((a[tr, tc] - h) / h)
errs = np.array(errs)
refr = dict(R_A=RA, R_B=RB, n=len(errs), rel_err_median=float(np.median(errs)), rel_err_max_abs=float(np.max(np.abs(errs))))
print('refraction', refr)

# ---- 5. time-varying rate: dry (4/5/6 %) for 2 h then moister (8/9/10 %)
R1 = fs.behave(102, m1=4, m10=5, m100=6, mherb=60)['ros_mmin']
R2 = fs.behave(102, m1=8, m10=9, m100=10, mherb=60)['ros_mmin']
n = 160
ls = fs.Landscape(fuel=np.full((n, n), 102), res_m=res)
o = fs.simulate(ls, wx(0.0, steps=[(2, (8.0, 9.0, 10.0))]), {'rowcol': [(80, 80)]}, T0, T0 + pd.Timedelta(hours=7),
                moisture='given', return_perimeters=False)
a = o['arrival_time']
yy, xx = np.mgrid[0:n, 0:n]
r = np.hypot(yy - 80, xx - 80) * res
r_switch = R1 * 120
hand = np.where(r <= r_switch, r / R1, 120 + (r - r_switch) / R2) / 60
ok = np.isfinite(a) & (r > 3 * res)
tv = dict(R1=R1, R2=R2, rel_err_median=float(np.median(((a - hand) / hand)[ok])),
          rel_err_max_abs=float(np.max(np.abs((a - hand) / hand)[ok])))
print('time-varying', tv)

write('analytic.json', dict(
    what='Analytic tests of the arrival-time solver (uniform ellipse at 3 resolutions and 7 wind directions, '
         'heterogeneous hand solutions, time-varying rate)', ellipse_summary=summ, ellipse_cases=ell,
    two_fuel_strip=strip, fuel_break=fb, refraction=refr, time_varying=tv))
