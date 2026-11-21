"""FireSim demo: one generated landscape, two days of fire weather, a 36-hour fire, and what VIIRS would see.

    cd shared/fire/firesim && OMP_NUM_THREADS=1 python3 demo.py      # ~10 s (first run: + numba compile)

Writes demo_out/demo.png and prints point fire behaviour (BehavePlus-style), the run summary and the ledger.
"""
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import firesim as fs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'demo_out')
os.makedirs(OUT, exist_ok=True)

# 1. point fire behaviour: chaparral-like shrub (SH5) on a 20-degree slope, 6 m/s 10-m wind blowing upslope
b = fs.behave('SH5', m1=4, m10=5, m100=7, mherb=60, mwoody=80, wind_speed=6.0, slope=20)
print('behave SH5: head ROS %.1f m/min, flank %.1f, back %.1f, L/W %.2f, FLI %.0f kW/m, flame %.1f m, '
      'max spot %.0f m' % (b['ros_mmin'], b['ros_flank_mmin'], b['ros_back_mmin'], b['lw_ratio'], b['fli_kwm'],
                           b['flame_length_m'], b['spot_max_surface_m']))
b = fs.behave('TU5', m1=4, m10=5, m100=7, wind_speed=12.0, cc=60, ch=20, cbh=2, cbd=0.15)
print('behave TU5 under canopy, 12 m/s: fire type %d (3 = active crown), ROS %.1f m/min, torching index %.1f m/s, '
      'crowning index %.1f m/s' % (b['fire_type'], b['ros_final_mmin'], b['torching_index_ms'], b['crowning_index_ms']))

# 2. landscape (30 x 24 km at 60 m) and weather with a dry wind event
t0 = time.perf_counter()
ls = fs.generate_landscape((400, 500), 60.0, rng=7, region='mixed_conifer', relief_m=800)
w = fs.generate_weather('2021-08-01', hours=24 * 5, climate='west_dry_summer', rng=3, wind=6.0, rh_min=14.0,
                        p_wet=0.0, lat=40.0, lon=-121.0)
start, end = '2021-08-03 18:00', '2021-08-05 06:00'     # 11:00 local -> 23:00 local next day
out = fs.simulate(ls, w, {'rowcol': [(260, 120)]}, start, end, crown='van_wagner', spotting=True, rng=1,
                  perimeter_times=[6, 12, 24, 36])
print('simulate: %.0f ha in 36 h (%.0f ha active crown, %d spot fires), %.2f s wall' % (
    out['area'].iloc[-1], out['ledger']['area_ha_active_crown'], out['labels']['n_spot_ignitions'], out['wall_s']))
gc = out['ledger'].get('growth_check', {})
print('ledger: energy %.0f TJ, growth check (perimeter x ROS / raster area) %.3f' % (
    out['ledger']['energy_total_GJ'] / 1e3, gc.get('ratio_total', np.nan)))

# 3. what VIIRS would report (nominal S-NPP / NOAA-20 / NOAA-21 overpasses, 20 % random cloud)
ob = fs.observe(out, 'viirs', rng=2, lat=40.0, lon=-121.0, params=dict(cloud_frac=0.2))
ops = ob['overpasses']
print('VIIRS: %d overpasses, %d detections (%d false alarms)' % (len(ops), len(ob['detections']),
                                                                  int((~ob['detections']['true_fire']).sum()) if len(ob['detections']) else 0))
print(ops[['time', 'satellite', 'scan_deg', 'daynight', 'n_det', 'expected_det', 'burned_ha']].assign(
    time=ops['time'].dt.strftime('%m-%d %H:%M')).round(1).to_string())
print('total demo time %.1f s' % (time.perf_counter() - t0))

# 4. figure
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, LightSource  # noqa: E402

ny, nx = ls.shape
ext = [0, nx * ls.res / 1000, 0, ny * ls.res / 1000]
hs = LightSource(315, 40).hillshade(ls['elev'], dx=ls.res, dy=ls.res)
f = ls['fuel']
grp = np.select([f < 91, (f >= 91) & (f <= 99), f < 121, f < 141, f < 161, f < 181, f < 201],
                [5, 0, 1, 2, 3, 4, 5], 5)
fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
cm = ListedColormap(['#9aa3ad', '#e8d77a', '#c4c56b', '#b5793b', '#4f7d3a', '#2e5a2a'])
ax[0, 0].imshow(grp, cmap=cm, vmin=-0.5, vmax=5.5, extent=ext)
ax[0, 0].imshow(hs, cmap='gray', alpha=0.35, extent=ext)
ax[0, 0].set_title('Fuels: grey none, yellow grass, olive grass-shrub,\nbrown shrub, greens timber', fontsize=10)
a = out['arrival_time']
ax[0, 1].imshow(hs, cmap='gray', extent=ext)
im = ax[0, 1].imshow(np.ma.masked_invalid(a), cmap='inferno_r', extent=ext, alpha=0.85)
yy = (np.arange(ny)[::-1] + 0.5) * ls.res / 1000
xx = (np.arange(nx) + 0.5) * ls.res / 1000
ax[0, 1].contour(xx, yy, np.nan_to_num(a, nan=1e3), levels=[6, 12, 24, 36], colors='k', linewidths=0.8)
plt.colorbar(im, ax=ax[0, 1], label='arrival time (h)')
ax[0, 1].set_title('Arrival time, isochrones at 6, 12, 24, 36 h')
ft = out['rasters']['fire_type']
ax[1, 0].imshow(hs, cmap='gray', extent=ext)
ax[1, 0].imshow(np.ma.masked_equal(ft, 0), cmap=ListedColormap(['#f2b134', '#e5562b', '#8b1a1a']), vmin=0.5,
                vmax=3.5, extent=ext)
d = ob['detections']
if len(d):
    tp = d[d['true_fire']]
    ax[1, 0].scatter((tp['col'] + 0.5) * ls.res / 1000, (ny - tp['row'] - 0.5) * ls.res / 1000, s=4, c='#1f6fd1',
                     label='VIIRS detections (simulated)')
    fa = d[~d['true_fire']]
    ax[1, 0].scatter((fa['col'] + 0.5) * ls.res / 1000, (ny - fa['row'] - 0.5) * ls.res / 1000, s=20, marker='x',
                     c='m', label='false alarms')
    ax[1, 0].legend(loc='lower right', fontsize=8)
ax[1, 0].set_title('Fire type (surface / passive crown / active crown) and VIIRS')
ax2 = ax[1, 1]
ar = out['area']
hrs = (ar.index - ar.index[0]) / np.timedelta64(1, 'h')
ax2.plot(hrs, ar.values, 'k-', label='burned area (ha)')
ax2.set_xlabel('hours since ignition (11:00 local)')
ax2.set_ylabel('burned area (ha)')
ax3 = ax2.twinx()
m = out['labels']['moisture']
mh = (m.index - ar.index[0]) / np.timedelta64(1, 'h')
ax3.plot(mh, m['m1h'], 'tab:blue', lw=1, label='1-h dead fuel moisture (%)')
ax3.plot(mh, out['labels']['weather']['wind_speed'], 'tab:green', lw=1, label='10-m wind (m/s)')
ax3.set_ylim(0, None)
ax2.legend(loc='upper left', fontsize=8)
ax3.legend(loc='center left', fontsize=8)
ax2.set_xlim(0, hrs.max())
ax2.set_title('Burned area, 1-h dead fuel moisture and wind')
rr, cc = np.nonzero(np.isfinite(a))
x0, x1 = cc.min() * ls.res / 1000 - 3, (cc.max() + 1) * ls.res / 1000 + 3
y0, y1 = (ny - rr.max() - 1) * ls.res / 1000 - 3, (ny - rr.min()) * ls.res / 1000 + 3
for a_ in (ax[0, 1], ax[1, 0]):
    a_.set_xlim(max(x0, 0), min(x1, ext[1]))
    a_.set_ylim(max(y0, 0), min(y1, ext[3]))
ax[0, 0].add_patch(plt.Rectangle((max(x0, 0), max(y0, 0)), min(x1, ext[1]) - max(x0, 0), min(y1, ext[3]) - max(y0, 0),
                                 fill=False, ec='r', lw=1.5))
for a_ in ax.ravel()[:3]:
    a_.set_xlabel('km')
    a_.set_ylabel('km')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'demo.png'), dpi=90)
print('wrote', os.path.join(OUT, 'demo.png'))
