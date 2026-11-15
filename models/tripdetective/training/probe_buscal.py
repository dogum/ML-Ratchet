"""Probe (15 Nov 2026): dev-only calibration maps for the bus mass, judged by 5-fold cross-validation over dev gate
missions (fit on 4/5 of the missions, score the other 1/5). Reads work/probe/bus_dev_legs.csv (probe_busdev.py)."""
from common import *
import pandas as pd, hashlib
d = pd.read_csv(f'{ROOT}/probe/bus_dev_legs_{sys.argv[1] if len(sys.argv) > 1 else "pb1"}.csv')
d['fold'] = d.mission.map(lambda m: int(hashlib.sha256(m.encode()).hexdigest(), 16) % 5)
g = d.groupby('mission')
d['n_m'] = g.truth.transform('size')
for k in ['net', 'lsq']:
    d[k + '_mm'] = g[k].transform('mean'); d[k + '_dev'] = d[k] - d[k + '_mm']
d['b183'] = (d.bus == 'ZTBus_B183').astype(float)
d['logT'] = d['T']

def design(x, name):
    one = np.ones(len(x))
    return {'dumb': np.c_[one],
            'level x net': None,
            'a + b net': np.c_[one, x.net],
            'a + b (net - mission mean)': np.c_[one, x.net_dev],
            'a + b dev + c mission mean': np.c_[one, x.net_dev, x.net_mm],
            '+ bus': np.c_[one, x.net_dev, x.net_mm, x.b183],
            'dev(net) + dev(lsq)': np.c_[one, x.net_dev, x.lsq_dev],
            'a + b dev + bus': np.c_[one, x.net_dev, x.b183],
            }[name]

names = ['dumb', 'level x net', 'a + b net', 'a + b (net - mission mean)', 'a + b dev + c mission mean', '+ bus', 'dev(net) + dev(lsq)', 'a + b dev + bus']
for nm in names:
    pred = np.zeros(len(d)); coefs = []
    for f in range(5):
        tr, te = d.fold != f, d.fold == f
        if nm == 'dumb': pred[te] = d.truth[tr].median(); continue
        if nm == 'level x net': pred[te] = d.net[te] * np.median(d.truth[tr] / d.net[tr]); continue
        A = design(d[tr], nm); c = np.linalg.lstsq(A, d.truth[tr], rcond=None)[0]; coefs.append(c)
        pred[te] = design(d[te], nm) @ c
    e = pred - d.truth
    single = d.n_m == 1
    print(f'{nm:30s} CV MAE {np.abs(e).mean():6.0f}  median AE {np.median(np.abs(e)):5.0f}  slope {np.polyfit(d.truth, pred, 1)[0]:.2f}'
          f'  r {np.corrcoef(pred, d.truth)[0,1] if np.std(pred) > 0 else 0:.2f}  (legs alone in their mission: {single.sum()})',
          '' if not coefs else np.round(np.mean(coefs, 0), 3))
