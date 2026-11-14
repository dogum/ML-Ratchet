"""Dev-year (2019, 2021, cross-year fits) report for a checkpoint: model vs baselines per BA. Dev only."""
import sys, numpy as np
from common import *
import evaluate as E
from exam import block
ck = sys.argv[1]
nets = E.load(ck); B = dict(np.load(f'{DATA}/banks/real_dev.npz')); pred = E.predict(nets, B)
sc = {k: E.sigma_scale(B, pred, k) for k in ('d', 'w', 's')}
print('sigma scales', {k: round(v, 2) for k, v in sc.items()})
for ba in BAS + ['ALL']:
    r = (B['meta_ba'] == ba) if ba != 'ALL' else np.ones(len(B['meta_ba']), bool)
    o = block(B, pred, {'d': 1, 'w': 1, 's': 1}, r)
    d, w, s = o['demand'], o['wind'], o['solar']
    print(f"{ba:5s} demand {d['model']:.2f} (DA {d['ba_day_ahead']:.2f} pers {d['persistence_24h']:.2f} prior {d['physics_prior']:.2f}) c68 {d['cov68']:.2f}"
          f" | wind {w['model']:.1f} (prior {w['physics_prior']:.1f} pers {w['persistence_24h']:.1f} clim {w['cf_climatology']:.1f}) c68 {w['cov68']:.2f}"
          f" | solar {s['model']:.1f} (prior {s['physics_prior']:.1f} pers {s['persistence_24h']:.1f} clim {s['cf_climatology']:.1f}) c68 {s['cov68']:.2f}")
