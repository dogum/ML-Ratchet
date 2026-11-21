"""Crown fire checks: Van Wagner (1977) initiation and active thresholds vs hand calculations; Cruz et al. (2005)
and Rothermel (1991) crown ROS vs hand calculations; behave() fire-type transitions; the simulator's fire-type
map vs behave() on a uniform forest; torching/crowning indices bracket the transitions."""
import math

import numpy as np
import pandas as pd

from _common import write, rel
import firesim as fs

out = {}
# 1. I0 = (0.010 CBH (460 + 25.9 FMC))^1.5 by hand (Van Wagner 1977, eq. 4 with C = 0.010)
hand = []
for cbh in (0.5, 1, 2, 4, 6, 8, 12):
    for fmc in (70, 100, 130):
        h = (0.010 * cbh * (460 + 25.9 * fmc)) ** 1.5
        m = fs.crown.van_wagner_i0(cbh, fmc)
        hand.append(dict(cbh=cbh, fmc=fmc, hand_kwm=h, firesim=m, rel_err=rel(m, h)))
out['i0_hand'] = dict(n=len(hand), max_rel_err=max(x['rel_err'] for x in hand),
                      example_cbh6_fmc100_kwm=fs.crown.van_wagner_i0(6.0, 100.0), cases=hand)
# Van Wagner (1977) quotes ~2500 kW/m for CBH 6 m and FMC 100 %
out['i0_vw1977_example'] = dict(cbh=6.0, fmc=100.0, firesim=fs.crown.van_wagner_i0(6.0, 100.0), quoted=2500.0,
                                rel_err=rel(fs.crown.van_wagner_i0(6.0, 100.0), 2500.0))
# 2. R_active = 3/CBD; Cruz 2005 hand values
cr = []
for u10, cbd, effm in ((10, 0.1, 6), (30, 0.2, 6), (50, 0.15, 4), (20, 0.3, 10), (60, 0.05, 8)):
    h = 11.02 * u10 ** 0.9 * cbd ** 0.19 * math.exp(-0.17 * effm)
    m = fs.crown.cruz2005_active(u10, cbd, effm)
    cac = h / (3 / cbd)
    hp = h * math.exp(-cac)
    mp = fs.crown.cruz2005_passive(u10, cbd, effm)
    cr.append(dict(u10_kmh=u10, cbd=cbd, effm=effm, active_hand=h, active=m, cac=cac, passive_hand=hp, passive=mp,
                   rel_err=max(rel(m, h), rel(mp, hp))))
out['cruz2005_hand'] = dict(max_rel_err=max(c['rel_err'] for c in cr), cases=cr)
# 3. Rothermel 1991: 3.34 x FM10 ROS at WAF 0.4 with the same moisture
b10 = fs.behave(10, m1=6, m10=7, m100=8, mherb=60, mwoody=90, wind_speed=8.0, waf=0.4)
r10 = b10['ros_mmin']
b = fs.behave('TU5', m1=6, m10=7, m100=8, mherb=60, mwoody=90, wind_speed=8.0, cbh=1.0, cbd=0.2, ch=20, cc=60)
out['rothermel1991'] = dict(r10_mmin=r10, crown_active_mmin=b['crown']['cros_active_mmin'],
                            ratio=b['crown']['cros_active_mmin'] / r10, expected_ratio=3.34)
# 4. behave transitions vs wind for a pine stand (TU5, CBH 2 m, CBD 0.15, CH 20 m)
trans = []
for u in np.arange(0, 25.1, 1.0):
    bb = fs.behave('TU5', m1=5, m10=6, m100=7, mherb=60, mwoody=90, wind_speed=u, cbh=2.0, cbd=0.15, ch=20.0, cc=60)
    trans.append(dict(u10_ms=u, fire_type=bb['fire_type'], surface_fli=bb['fli_kwm'], i0=bb['crown']['i0_kwm'],
                      ros_final=bb['ros_final_mmin'], cfb=bb['crown']['cfb'], cros=bb['crown']['cros_active_mmin'],
                      rac=bb['crown']['r_active_crit_mmin']))
bb = fs.behave('TU5', m1=5, m10=6, m100=7, mherb=60, mwoody=90, wind_speed=10.0, cbh=2.0, cbd=0.15, ch=20.0, cc=60)
ti, ci = bb['torching_index_ms'], bb['crowning_index_ms']
first_passive = next((t['u10_ms'] for t in trans if t['fire_type'] >= 2), None)
first_active = next((t['u10_ms'] for t in trans if t['fire_type'] == 3), None)
# indices are 20-ft winds; behave input is 10-m (x 1/1.15)
out['transitions'] = dict(cases=trans, torching_index_20ft_ms=ti, crowning_index_20ft_ms=ci,
                          first_passive_u10=first_passive, first_active_u10=first_active,
                          ti_as_u10=ti * 1.15, ci_as_u10=ci * 1.15,
                          consistent=bool(first_passive is not None and abs(first_passive - ti * 1.15) <= 1.0 + 1e-9
                                          and (first_active is None or abs(first_active - max(ci, ti) * 1.15) <= 1.0)))
print('transitions TI/CI (as 10 m) ', ti * 1.15, ci * 1.15, 'first passive/active', first_passive, first_active)
# 5. simulator fire type on a uniform stand vs behave head-fire type
T0 = pd.Timestamp('2021-08-01', tz='UTC')
idx = pd.date_range(T0, periods=30, freq='h')
sim = []
for u in (2.0, 6.0, 10.0, 16.0):
    n = 120
    ls = fs.Landscape(fuel=np.full((n, n), 165), cc=np.full((n, n), 60.0), ch=np.full((n, n), 20.0),
                      cbh=np.full((n, n), 2.0), cbd=np.full((n, n), 0.15), res_m=30)
    w = pd.DataFrame(dict(T=30.0, RH=15.0, wind_speed=u, wind_dir=270.0, precip=0.0, m1h=5.0, m10h=6.0, m100h=7.0,
                          mherb=60.0, mwoody=90.0), index=idx)
    bh0 = fs.behave('TU5', m1=5, m10=6, m100=7, mherb=60, mwoody=90, wind_speed=u, cbh=2.0, cbd=0.15, ch=20.0,
                    cc=60)
    hrs = min(24.0, 60 * 30.0 / bh0['ros_final_mmin'] / 60.0)
    o = fs.simulate(ls, w, {'rowcol': [(60, 10)]}, T0, T0 + pd.Timedelta(hours=hrs), moisture='given',
                    return_perimeters=False)
    ft = o['rasters']['fire_type']
    head = ft[60, 11:][np.isfinite(o['arrival_time'][60, 11:])]
    bh = fs.behave('TU5', m1=5, m10=6, m100=7, mherb=60, mwoody=90, wind_speed=u, cbh=2.0, cbd=0.15, ch=20.0, cc=60)
    sim.append(dict(u10_ms=u, behave_fire_type=bh['fire_type'],
                    sim_head_fire_type_mode=int(np.bincount(head.astype(int)).argmax()) if len(head) else 0,
                    area_frac_by_type={int(k): float((ft == k).sum() / max((ft > 0).sum(), 1)) for k in (1, 2, 3)},
                    behave_ros_final=bh['ros_final_mmin'],
                    sim_head_ros=float(np.nanmedian(o['rasters']['ros_dir'][60, 20:][np.isfinite(o['arrival_time'][60, 20:])]))
                    if np.isfinite(o['arrival_time'][60, 20:]).any() else None))
out['simulator_vs_behave'] = sim
out['simulator_head_type_agrees'] = all(s['behave_fire_type'] == s['sim_head_fire_type_mode'] for s in sim)
print(sim)
write('crown.json', dict(what='Crown fire initiation and spread vs hand calculations; behave and simulator consistency',
                         **out))
