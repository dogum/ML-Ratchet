"""FlowCast demo: three real EXAM basins (never used for training or for choosing anything), water years
1995-1996. The ungauged network sees only forcing and attributes; the observed flow is used only to score it.
The second part runs the optional GAUGED nowcast weights, which also read the observed flow up to `lead` days
before each day, against persistence at the same lead.

Run: python demo.py            (numpy + pandas)
     python demo.py --plot     also writes work/demo_flowcast.png (needs matplotlib)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import flowcast as fc


def nse(p, o):
    return 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)


m = fc.FlowCast()
now = fc.FlowCast(os.path.join(HERE, 'flowcast_nowcast_weights.npz'))
with open(os.path.join(HERE, 'examples', 'examples.json'), encoding='utf-8') as fh:
    ex = json.load(fh)
rows = []
print('Ungauged (forcing + attributes only), WY1995-1996:')
for e in ex:
    f = pd.read_csv(os.path.join(HERE, 'examples', e['forcing_csv']), index_col=0, parse_dates=True)
    obs = pd.read_csv(os.path.join(HERE, 'examples', e['obs_csv']), index_col=0, parse_dates=True)['Q_mm']
    out = m.predict(f, e['attrs'], e['start'], e['end'])
    o = obs.reindex(out.index)
    v = o.notna()
    cov = np.mean((o[v] >= out.q05[v]) & (o[v] <= out.q95[v]))
    print(f"{e['id']}  {e['name'][:44]:44s}  NSE {nse(out.Q[v], o[v]):5.2f}  90% band holds {100*cov:4.0f}% of days  "
          f"mean obs {o[v].mean():.2f} / pred {out.Q[v].mean():.2f} mm/d (x{out.Q[v].mean() / o[v].mean():.2f})")
    rows.append((e, f, obs, out, o))

print('\nGauged nowcast (also reads the observed flow up to `lead` days earlier), NSE vs persistence at the same lead.')
print('The gauge record here starts on the first predicted day, so the first warm-up runs with the gauge flag off.')
for e, f, obs, out, o in rows:
    cells = []
    for lead in (1, 3, 7):
        q = now.predict(f, e['attrs'], e['start'], e['end'], qobs=obs, lead=lead).Q
        pers = obs.shift(lead, freq='D').reindex(out.index)
        v = o.notna() & pers.notna()
        cells.append(f"lead {lead} d {nse(q[v], o[v]):5.2f} / {nse(pers[v], o[v]):5.2f}")
    print(f"{e['id']}  " + '   '.join(cells))

if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work = os.path.join(os.environ['ML_RATCHET_WORK'], 'flowcast') if os.environ.get('ML_RATCHET_WORK') \
        else os.path.join(HERE, 'work')
    os.makedirs(work, exist_ok=True)
    fig, ax = plt.subplots(len(rows), 1, figsize=(10, 2.6 * len(rows)))
    for a, (e, f, obs, out, o) in zip(np.atleast_1d(ax), rows):
        a.fill_between(out.index, out.q05, out.q95, color='#9cc3d5', alpha=0.6, label='90 % band')
        a.plot(out.index, out.Q, color='#0b4f6c', lw=1, label='FlowCast (ungauged)')
        a.plot(o.index, o, color='k', lw=0.8, label='observed (USGS)')
        a.set_title(e['name'], fontsize=9)
        a.set_ylabel('mm/day')
    np.atleast_1d(ax)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(work, 'demo_flowcast.png'), dpi=110)
    print('wrote', os.path.join(work, 'demo_flowcast.png'))
