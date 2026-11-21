"""Figure: one dev fire-day per row (fixed seeded draw among growth days of 3 size strata): observed burned at the
issue time, observed new burn, uncalibrated and dev-calibrated replay probability. -> firesim/demo_out/replay_examples.png"""
import json
import os

import numpy as np
import pandas as pd

from _common import OUT, FIRE_ROOT
import replay_dev as RV
from firesim import replay as RP

rows = RV.load_rows()
g = pd.DataFrame([dict(fire_id=r['fire_id'], i=r['i'], n_new=r['n_new_obs'], n_now=r['n_now']) for r in rows if r['n_new_obs'] >= 20])
g['stratum'] = pd.qcut(g.n_now, 3, labels=['small', 'medium', 'large'])
pick = g.groupby('stratum', observed=True).sample(1, random_state=20261118)
cal = json.load(open(RV.CALIB_FILE))
params = RP.calibrated_params(cal)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(len(pick), 3, figsize=(12, 4 * len(pick)))
for row, (_, p) in enumerate(pick.iterrows()):
    for info, d in RV.fire_days(p.fire_id):
        if info['i'] != p.i or d is None:
            continue
        now_c = d['now'][d['box'][0]:d['box'][1], d['box'][2]:d['box'][3]]
        nxt_c = d['nxt'][d['box'][0]:d['box'][1], d['box'][2]:d['box'][3]]
        r_u = RP.replay_day(d['ls'], d['w'], now_c, d['start'], d['end'], ensemble=True, rng=info['i'])
        r_c = RP.replay_day(d['ls'], d['w'], now_c, d['start'], d['end'], params=params, ensemble=True, rng=1000 + info['i'])
        rr, cc = np.nonzero(nxt_c | (r_u['prob'] > 0.05))
        pad = 15
        sl = (slice(max(rr.min() - pad, 0), rr.max() + pad), slice(max(cc.min() - pad, 0), cc.max() + pad))
        km = d['res'] / 1000
        for j, (title, img) in enumerate([('observed: burned (grey), new next day (red)', None),
                                          ('uncalibrated replay P(burn by next day)', r_u['prob']),
                                          ('dev-calibrated replay P(burn by next day)', r_c['prob'])]):
            a = ax[row, j]
            base = np.zeros(now_c.shape + (3,)) + 1.0
            base[nxt_c & ~now_c] = (0.85, 0.2, 0.15)
            base[now_c] = (0.55, 0.55, 0.55)
            if img is not None:
                base = np.ones(now_c.shape + (3,))
                pm = np.clip(img, 0, 1)
                base[..., 0] = 1 - 0.1 * pm
                base[..., 1] = 1 - 0.6 * pm
                base[..., 2] = 1 - 0.95 * pm
                base[now_c] = (0.55, 0.55, 0.55)
            a.imshow(base[sl], extent=[0, base[sl].shape[1] * km, 0, base[sl].shape[0] * km], interpolation='nearest')
            if img is not None:
                a.contour(np.arange(base[sl].shape[1]) * km + km / 2, (base[sl].shape[0] - np.arange(base[sl].shape[0])) * km - km / 2,
                          (nxt_c & ~now_c)[sl].astype(float), [0.5], colors='r', linewidths=0.8)
            a.set_title(f"dev fire {p.fire_id}, fire-day {info['i']}\n{title}" if j == 0 else '\n' + title, fontsize=9)
            a.set_xlabel('km')
fig.tight_layout()
os.makedirs(os.path.join(FIRE_ROOT, 'firesim', 'demo_out'), exist_ok=True)
fig.savefig(os.path.join(FIRE_ROOT, 'firesim', 'demo_out', 'replay_examples.png'), dpi=80)
print(pick)
