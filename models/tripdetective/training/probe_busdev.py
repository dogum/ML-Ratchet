"""Probe (15 Nov 2026): on the ZTBus dev gate legs, how the bus network (pb1) and the physics fit follow passengers,
overall and within a mission (same bus, same day). Dev only."""
from common import *
import pandas as pd
import evaluate as E, busbank, busladder as BL, busexam as BX
CK = sys.argv[1] if len(sys.argv) > 1 else 'pb1'
net = E.load_net(f'{ROOT}/ckpt/{CK}.pt')
rb = busbank.load_real('dev')
legs, m, s = BL.real_predict(net, rb); meta = {x['trip_id']: x for x in rb['meta']}
d = pd.DataFrame([meta[t] for t in legs]); d['net'] = m; d['lsq'] = BX.lsq_mass(rb, legs)
d['truth'] = d.mass_kg; d = d[np.isfinite(d.truth)]
print(len(d), 'legs', d.mission.nunique(), 'missions', d.bus.nunique(), 'buses; legs/mission', d.groupby('mission').size().describe()[['mean', '50%', 'max']].tolist())
print('truth sd', d.truth.std().round(0), 'pax mean/sd', d.pax.mean().round(1), d.pax.std().round(1), 'dumb MAE', (d.truth - d.truth.median()).abs().mean().round(0))
for k in ['net', 'lsq']:
    x = d[k]; print(k, 'level truth/x', np.median(d.truth / x).round(3), 'corr with truth', np.corrcoef(x, d.truth)[0, 1].round(3),
                    'CV of x/truth', (x / d.truth).std().round(3))
    dm = d.groupby('mission')[[k, 'truth']].transform('mean'); xx = x - dm[k]; tt = d.truth - dm.truth
    ok = d.groupby('mission').mission.transform('size') > 1
    print('   within-mission: corr', np.corrcoef(xx[ok], tt[ok])[0, 1].round(3), 'slope', np.polyfit(tt[ok], xx[ok], 1)[0].round(2),
          'sd of truth deviations', tt[ok].std().round(0), 'sd of x deviations', xx[ok].std().round(0))
    bm = d.groupby('mission')[[k, 'truth']].mean()
    print('   between missions: corr', np.corrcoef(bm[k], bm.truth)[0, 1].round(3), 'n', len(bm))
    bb = d.groupby('bus')[[k, 'truth']].mean(); print('   per bus mean x/truth', (bb[k] / bb.truth).round(3).to_dict())
print(d.groupby('season').agg(n=('truth', 'size'), pax=('pax', 'mean'), net=('net', 'mean'), lsq=('lsq', 'mean')).round(1))
d.to_csv(f'{ROOT}/probe/bus_dev_legs_{CK}.csv', index=False)
