"""DemandCast demo: 16 real exam days (a fixed random draw, two per grid) with precomputed inputs.
Prints DemandCast vs the grid's own day-ahead forecast for demand, and vs physics alone for wind and solar.
    python3 demo.py
"""
import os, numpy as np
from demandcast import DemandCast

HERE = os.path.dirname(os.path.abspath(__file__))
E = dict(np.load(os.path.join(HERE, 'examples', 'exam_days.npz')))
m = DemandCast()
d = m.predict_demand(E['xd'], E['pd_total'])
w = m.predict_cf('w', E['xw'], E['pw'])
s = m.predict_cf('s', E['xs'], E['ps'], night=E['night'])


def mape(p, y):
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return 100 * np.mean(np.abs(p - y)[ok] / y[ok])


def nmae(p, y):
    ok = np.isfinite(y)
    return 100 * np.sum(np.abs(p - y)[ok]) / max(np.sum(y[ok]), 1e-9) if ok.any() else np.nan


print(f"{'grid':5s} {'day':10s} | demand MAPE %: ours  grid-DA  | in 95% band | wind err %: ours physics | solar err %: ours physics")
for i in range(len(E['ba'])):
    y = E['d_true'][i]
    inb = np.mean((y >= d['lo95'][i]) & (y <= d['hi95'][i]))
    ws = f"{nmae(w['cf'][i], E['yw'][i]):6.1f} {nmae(E['pw'][i], E['yw'][i]):6.1f}" if E['hasw'][i] else '     -      -'
    ss = f"{nmae(s['cf'][i], E['ys'][i]):6.1f} {nmae(E['ps'][i], E['ys'][i]):6.1f}" if E['hass'][i] else '     -      -'
    print(f"{E['ba'][i]:5s} {E['day'][i]:10s} |        {mape(d['mw'][i], y):5.2f}  {mape(E['df'][i], y):6.2f}  |    {inb:4.2f}     |  {ws}  |  {ss}")
print(f"\nall 16 days: demand MAPE ours {mape(d['mw'], E['d_true']):.2f} %, grid's own forecast {mape(E['df'], E['d_true']):.2f} %")
