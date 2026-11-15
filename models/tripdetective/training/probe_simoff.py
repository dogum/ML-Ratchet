"""Probe (15 Nov 2026): in a simulated bank, the rule's signal (engine-off share while moving, rpm < 100 at v > 5 m/s)
per car vs architecture and engine size, and the network's HEV probability. usage: probe_simoff.py <bank> [ckpt]"""
from common import *
import simbank, evaluate as E
b = simbank.load_bank(sys.argv[1]); o = b['offs']; meta = {m['vid']: m for m in b['meta']}
cars = sorted(set(b['veh'].tolist())); rows = []
for c in cars:
    js = np.where(b['veh'] == c)[0]; offs = []
    for j in js:
        r = b['raw'][o[j]:o[j + 1]].T.astype(np.float32); f = b['flags'][j]
        mv = (r[0] > 5) & (r[10] > .5)
        if f[2] > .5 and mv.sum() >= 10: offs.append(np.mean(r[6][mv] < 0.1))
    m = meta[c]; rows.append((c, m['arch'], m.get('disp'), np.mean(offs) if offs else np.nan, len(offs), m['mass']))
if len(sys.argv) > 2:
    net = E.load_net(f'{ROOT}/ckpt/{sys.argv[2]}')
    inp = E.bank_inputs(b, np.arange(len(b['veh']))); cmap = {c: i for i, c in enumerate(cars)}
    P = E.predict(net, inp, np.array([cmap[v] for v in b['veh']]))['arch']
else: P = None
import collections
print('arch            n  off-share pct 5/25/50/75/95 (cars with rpm logged)')
for a in ARCHS:
    x = np.array([r[3] for r in rows if r[1] == a and np.isfinite(r[3])])
    if len(x): print(f'{a:16s} {len(x):4d} ' + ' '.join(f'{v:.3f}' for v in np.percentile(x, [5, 25, 50, 75, 95])))
hev = [i for i, r in enumerate(rows) if r[1].startswith('hev') and np.isfinite(r[3])]
print('HEVs by displacement (<=1.6 L vs larger): off-share median',
      np.median([rows[i][3] for i in hev if rows[i][2] <= 1.6]) if any(rows[i][2] <= 1.6 for i in hev) else None,
      sum(rows[i][2] <= 1.6 for i in hev), np.median([rows[i][3] for i in hev if rows[i][2] > 1.6]))
if P is not None:
    pv = np.array([P[i, 1:4].sum() for i in range(len(rows))])
    for lo, hi in [(0, .03), (.03, .08), (.08, .15), (.15, .25), (.25, 1.01)]:
        sel = [i for i in hev if lo <= rows[i][3] < hi]
        cv = [i for i, r in enumerate(rows) if r[1] == 'conv' and np.isfinite(r[3]) and lo <= r[3] < hi]
        print(f'off {lo:.2f}-{hi:.2f}: HEV n {len(sel):3d} called HEV {np.mean(pv[sel] > .5) if sel else float("nan"):.2f} | conv n {len(cv):3d} called HEV {np.mean(pv[cv] > .5) if cv else float("nan"):.2f}')
