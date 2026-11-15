"""Probe (15 Nov 2026): force-balance mass estimates per ZTBus dev gate leg from the logged traction force and from the
logged power, on acceleration / grade segments; how each follows passengers overall and within a mission. Dev only."""
from common import *
import pandas as pd
import realdata as rd, busbank
zl = rd.ztbus_full(); zl = zl[(zl.split == 'dev') & zl.q_label_ok]
zl = zl[zl.mission.map(busbank.mission_half) == 'gate']
sm = lambda x, k=3: np.convolve(x, np.ones(k) / k, mode='same')
rows = []
for tid, row in zl.set_index('trip_id').iterrows():
    m, a = rd.load_trip(tid)
    v = np.nan_to_num(a['v']); g = np.nan_to_num(a['grade']); F = np.asarray(a['traction_force'], float); P = np.asarray(a['elec_power'], float)
    rho = np.where(np.isfinite(a['rho_air']) & (a['rho_air'] > .5), a['rho_air'], 1.2)
    acc = np.zeros_like(v); acc[1:-1] = (v[2:] - v[:-2]) / 2
    vs, as_, gs = sm(v), sm(acc), sm(g)
    out = dict(trip_id=tid, mission=row.mission, bus=row.vehicle_id, pax=row.passengers_mean, pax_frac=row.passengers_frac,
               truth=busbank.CURB_KG + busbank.PAX_KG * row.passengers_mean, dur=row.duration_s, T=row.T_amb_mean, season=row.season,
               F_ok=float(np.isfinite(F).mean()))
    for nm, y, extra in [('F', sm(np.nan_to_num(F)), None), ('P', sm(np.nan_to_num(P)), vs)]:
        for seg, msk in [('all', (vs > 2) & (as_ > -0.3)), ('acc', (vs > 1) & (as_ > 0.3))]:
            k = msk & (y > 0) & np.isfinite(F)
            if k.sum() < 200: out[f'{nm}_{seg}'] = np.nan; continue
            if nm == 'F': X = np.c_[1.03 * as_ + 9.81 * gs, np.ones(len(v)), 0.5 * rho * vs ** 2][k]
            else: X = np.c_[vs * (1.03 * as_ + 9.81 * gs), vs, 0.5 * rho * vs ** 3, np.ones(len(v))][k]
            b = np.linalg.lstsq(X, y[k], rcond=None)[0]; out[f'{nm}_{seg}'] = b[0]
    rows.append(out)
d = pd.DataFrame(rows); d.to_csv(f'{ROOT}/probe/bus_dev_force.csv', index=False)
print(len(d), 'legs; pax_frac median', d.pax_frac.median().round(3), 'traction force finite', d.F_ok.median().round(3), 'duration median h', (d.dur.median() / 3600).round(2))
g = d.groupby('mission')
for k in ['F_all', 'F_acc', 'P_all', 'P_acc']:
    x = d[k]; ok = np.isfinite(x)
    xx = x - g[k].transform('mean'); tt = d.truth - g.truth.transform('mean'); w = ok & (g.truth.transform('size') > 1)
    print(f'{k:6s} level truth/x {np.median(d.truth[ok] / x[ok]):.3f}  CV {np.std(x[ok] / d.truth[ok]) / np.mean(x[ok] / d.truth[ok]):.3f}  corr {np.corrcoef(x[ok], d.truth[ok])[0,1]:.3f}'
          f'  within-mission corr {np.corrcoef(xx[w], tt[w])[0,1]:.3f} slope {np.polyfit(tt[w], xx[w], 1)[0]:.2f}  per bus x/truth {(d[ok].groupby("bus")[k].mean() / d[ok].groupby("bus").truth.mean()).round(3).to_dict()}')
