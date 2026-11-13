"""Dev experiment: hybrid control variants (patched, not edited) vs real VED dev HEV fuel."""
import sys, json
from common import *
import realdata as rd, numpy as np
import roadsim.strategies as S
import wrappers, vehfit

def patch(min_on, min_off, start_mult, v_on=None):
    S.HybridStrategy.min_on_s = min_on; S.HybridStrategy.min_off_s = min_off
    orig = S.HybridStrategy.__dict__.get('_orig_setup') or S.HybridStrategy.setup
    S.HybridStrategy._orig_setup = orig
    def setup(self, veh, pt, trip, st):
        orig(self, veh, pt, trip, st); self.start_W *= start_mult
    S.HybridStrategy.setup = setup
    of = S.HybridStrategy.__dict__.get('_orig_forced') or S.HybridStrategy.forced_on
    S.HybridStrategy._orig_forced = of
    def forced_on(self, st, ctx):
        if v_on is not None and ctx.get('moving') and st.get('v', 0) > v_on: return True
        return of(self, st, ctx)
    S.HybridStrategy.forced_on = forced_on

v = rd.ved_vehicles()
variants = {'base': (4, 2, 1.0), 'on10': (10, 5, 1.0), 'on10s4': (10, 5, 4.0), 'on20s8': (20, 8, 8.0)}
for name, (a, b, c) in variants.items():
    patch(a, b, c)
    specs = {}
    for vid in ['VED_241', 'VED_375', 'VED_435']:
        r = v.loc[vid]; specs[vid], info = vehfit.label_match(complete_spec(rd.load_vehicle(vid)), r.unadj_city_mpg, r.unadj_hwy_mpg)
    specs['VED_258'] = specs['VED_241']; specs['VED_440'] = specs['VED_435']
    lr = []; on = []
    for meta, arr in rd.load_trips('dev', arch='hev_powersplit', label_ok=True, limit=80, seed=3):
        trip = {k: arr[k] for k in ('t', 'v', 'grade', 'elev', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface')}
        o = wrappers.run(specs[meta['vehicle_id']], trip, 'maf', init_temps=warm_temps(float(arr['T_amb'].mean())))
        ok = arr['valid'] == 1
        lr.append(np.log(np.sum(o['fuel_rate_maf'][ok]) / max(np.nansum(arr['fuel_rate'][ok]), 1e-3)))
        on.append((np.mean(o['eng_on']), np.mean(arr['engine_rpm'][ok] > 100)))
    lr = np.array(lr); on = np.array(on)
    print(name, 'median lr %.3f MdAPE %.3f  on sim %.2f real %.2f' % (np.median(lr), np.median(np.abs(np.exp(lr) - 1)), on[:, 0].mean(), on[:, 1].mean()), flush=True)
