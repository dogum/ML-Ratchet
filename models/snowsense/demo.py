"""SnowSense demo: three real exam stations (a fixed random draw, one per region), water years 2016-2025.
Runs in a few seconds with numpy and pandas only:  python3 demo.py
SnowSense never saw these stations or any SWE measurement; the observed SWE is used only to score it."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from snowsense import SnowSense
from pipeline import run, nse

model = SnowSense(os.path.join(HERE, 'snowsense_weights.npz'))
with open(os.path.join(HERE, 'examples', 'stations.json'), encoding='utf-8') as fh:
    st = json.load(fh)
for sid, s in st.items():
    daily, annual = run(os.path.join(HERE, 'examples', f'{sid}.csv'), s['lat'], s['elev_m'], model)
    wy = daily.index.year + (daily.index.month >= 10)
    print(f"\n{s['name']}, {s['state']} ({sid}, {s['elev_m']:.0f} m)  NSE WY2016-2025: {nse(daily['obs_eod'], daily['swe']):.3f}")
    print(' WY   peak SWE mm [68%]      obs   peak date  melt-out  [68%]                obs melt-out')
    for y, r in annual.iterrows():
        o = daily['obs_eod'][wy == y]
        if o.notna().sum() < 300:
            continue
        omo = None
        if o.max() >= 25:
            after = o.loc[o.idxmax():]
            b = after[after < 5]
            omo = b.index[0].strftime('%d %b') if len(b) else '-'
        f = lambda d: d.strftime('%d %b') if d == d else '-'
        print(f" {y} {r.peak_swe_mm:6.0f} [{r.peak_lo68:4.0f}-{r.peak_hi68:4.0f}]  {o.max():6.0f}   {f(r.peak_date)}     "
              f"{f(r.melt_out_date)}  [{f(r.melt_lo68)} - {f(r.melt_hi68)}]   {omo}")
