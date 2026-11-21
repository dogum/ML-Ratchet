"""Run pyretechnics (isolated venv firesim/.refenv) on a grid of cases; writes out/pyre_values.json.
Run with: $ML_RATCHET_FIRE/firesim/.refenv/bin/python pyre_run.py [out_dir]   (default out_dir: ../out)"""
import json, math, os, sys, itertools
import pyretechnics.fuel_models as fm
import pyretechnics.surface_fire as sf
import pyretechnics.crown_fire as cf
import pyretechnics

MPH_TO_MMIN = 26.8224
scen_dead = {'D1': (0.03, 0.04, 0.05), 'D2': (0.06, 0.07, 0.08), 'D3': (0.09, 0.10, 0.11), 'D4': (0.12, 0.13, 0.14)}
scen_live = {'L1': (0.30, 0.60), 'L2': (0.60, 0.90), 'L3': (0.90, 1.20), 'L4': (1.20, 1.50)}
nums = [n for n in fm.list_fuel_model_numbers() if not (91 <= n <= 99)]
rows = []
for num, (dk, d), (lk, l), wind, slope_pct, wdir_rel in itertools.product(
        nums, scen_dead.items(), scen_live.items(), (0.0, 2.0, 5.0, 10.0), (0.0, 30.0, 70.0), (0.0, 90.0)):
    if slope_pct == 0.0 and wdir_rel != 0.0:
        continue
    M = (d[0], d[1], d[2], d[0], l[0], l[1])
    m = fm.moisturize(fm.get_fuel_model(num), M)
    s0 = sf.calc_surface_fire_behavior_no_wind_no_slope(m, 1.0)
    # aspect 180 (south-facing) -> upslope is north (0 deg); wind blowing toward wdir_rel => from wdir_rel + 180
    upwind = (wdir_rel + 180.0) % 360.0
    x = sf.calc_surface_fire_behavior_max(s0, wind * MPH_TO_MMIN, upwind, slope_pct / 100.0, 180.0, False, 'rothermel')
    d3 = x['max_spread_direction']
    rows.append(dict(num=num, dead=dk, live=lk, wind_mph=wind, slope_pct=slope_pct, wind_rel_deg=wdir_rel,
                     r0_mmin=s0['base_spread_rate'], fli0_kwm=s0['base_fireline_intensity'],
                     ros_mmin=x['max_spread_rate'], fli_kwm=x['max_fireline_intensity'], fl_m=x['max_flame_length'],
                     lw=x['length_to_width_ratio'], dir_deg=math.degrees(math.atan2(d3[0], d3[1])) % 360.0,
                     dir_z=d3[2]))
crown = []
for cbh, fmc, u10, cbd, effm in itertools.product((1.0, 3.0, 6.0, 10.0), (0.8, 1.0, 1.3), (10.0, 30.0, 60.0),
                                                  (0.05, 0.15, 0.3), (0.04, 0.08, 0.12)):
    ci = cf.calc_crown_fire_behavior_max(20.0, cbh, cbd, 18000.0, effm, u10, 180.0, 0.0, 0.0)
    # I0 recovered from the initiation test by bisection (the I0 function itself is C-only)
    lo, hi = 0.0, 1e6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cf.van_wagner_crown_fire_initiation(mid, 0.9, cbh, fmc):
            hi = mid
        else:
            lo = mid
    crown.append(dict(cbh=cbh, fmc=fmc, u10_kmh=u10, cbd=cbd, effm=effm, i0=hi,
                      cros=ci['max_spread_rate'], ftype=ci['max_fire_type'], crit=ci['critical_spread_rate'],
                      lw=ci['length_to_width_ratio'], fli=ci['max_fireline_intensity']))
waf = []
for depth_ft, ch_ft, cc in itertools.product((0.2, 1.0, 2.5, 6.0), (0.0, 20.0, 60.0, 120.0), (0.0, 0.1, 0.3, 0.6, 0.9)):
    waf.append(dict(depth_ft=depth_ft, ch_ft=ch_ft, cc=cc, waf=sf.calc_midflame_wind_speed(1.0, depth_ft, ch_ft, cc)))
tables = {n: dict(fm.get_fuel_model(n)) for n in nums}
tables = {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in t.items()} for n, t in tables.items()}
out = dict(version=getattr(pyretechnics, '__version__', '2026.10.20'), surface=rows, crown=crown, waf=waf, tables=tables)
p = os.path.join(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'),
                 'pyre_values.json')
json.dump(out, open(p, 'w'))
print('rows', len(rows), 'crown', len(crown), 'waf', len(waf))
