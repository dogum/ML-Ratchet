"""Check (15 Nov 2026): the numpy bus reader with the buscal map (tripdetective.TripDetectiveBus.read_mission, from the
raw logs) against buscal.py's bank-based numbers on a few dev gate missions (dev only)."""
from common import *
import _paths as _P
import importlib.util, pandas as pd
import realdata as rd, busbank, buscal, evaluate as E
spec = importlib.util.spec_from_file_location('tdrel', os.path.join(_P.MODEL, 'tripdetective.py')); TD = importlib.util.module_from_spec(spec); spec.loader.exec_module(TD)
cal = jload(f'{ROOT}/buscal_pb1.json')
bus = TD.TripDetectiveBus(); bus.cal = cal
ctx = buscal.leg_table(E.load_net(f'{ROOT}/ckpt/pb1.pt'), busbank.load_real('dev', all_legs=True))
ctx['pred'] = buscal.apply(ctx, cal['rung'], np.array(cal['coef']))
zl = rd.ztbus_full()
dmax = 0
for mis in sorted(ctx.mission.unique())[:6]:
    legs = zl[zl.mission == mis].sort_values('leg'); L_ = []
    for t in legs.trip_id:
        m, a = rd.load_trip(t)
        T = np.where(np.isfinite(a['T_amb']), a['T_amb'], np.nanmean(a['T_amb']) if np.isfinite(a['T_amb']).any() else 12.0)
        rho = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > 0.5), a['rho_air'], 1.2)
        L_.append(dict(v=np.nan_to_num(a['v']), power_W=np.nan_to_num(a['elec_power']), grade=np.nan_to_num(a['grade']), T_amb=T, rho_air=rho, valid=a['valid']))
    R = bus.read_mission(L_)
    for t, r in zip(legs.trip_id, R):
        p = ctx.set_index('trip_id').loc[t, 'pred']; dmax = max(dmax, abs(p - r['mass_kg']))
print('max |numpy reader - buscal| over', 6, 'missions:', round(dmax, 2), 'kg')
