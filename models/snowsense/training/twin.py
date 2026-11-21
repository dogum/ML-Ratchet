"""Twin test on the dev stations (WY2011-2025, the dev-gate years).

For each dev station: its calibrated twins (degree-day and energy balance, calibrated on WY <= 2010 by
calib_prior.py, driven by the same station forcing) and the network's reading of the real forcing.
Since the network input is the same for twin and real, the comparison asks: does the network reproduce what
a calibrated physics model of this station gives (net vs twin), and how far is that from the pillow (net vs real)?
Biases (peak, melt-out, accumulation-season and melt-season errors) of net-real against net-twin show where the
simulator, not the network, misses reality.
python3 twin.py <run>
"""
import json
import sys
import numpy as np
import pandas as pd
from common import *
from evalset import station_set, physics
from evaluate import load, predict, band, melt_out_idx

run = sys.argv[1]
cal = json.load(open(f'{ROOT}/results/calib_dev.json'))
net = load(run)


def nse(o, s):
    m = o.notna() & s.notna()
    o, s = o[m], s[m]
    return float(1 - ((o - s) ** 2).sum() / ((o - o.mean()) ** 2).sum())


rows = []
for sid, lat, elev, ok in station_set('dev'):
    df = load_station(sid)
    X, D, Y, wys = station_windows(sid, list(range(2011, 2026)), lat, elev, df)
    Q = predict(net, X)
    s_net = stitch(band(Q, 2), D, wys)
    obs = df['SWE_eod'].reindex(s_net.index).where(np.isin(wy_of(s_net.index), ok))
    r = {'station': sid, 'elev': elev, 'lat': lat}
    series = {'net': s_net, 'real': obs}
    for snow in ('degree_day', 'energy_balance'):
        c = cal.get(sid, {}).get(snow)
        if c:
            series[f'twin_{snow[:2]}'] = physics(sid, lat, elev, snow, params=c['params'], df=df).reindex(s_net.index)
    series['default_eb'] = physics(sid, lat, elev, 'energy_balance', df=df).reindex(s_net.index)
    for a, b in (('net', 'real'), ('twin_en', 'real'), ('twin_de', 'real'), ('net', 'twin_en'), ('net', 'twin_de'),
                 ('default_eb', 'real')):
        if a in series and b in series:
            ob = series[b].where(obs.notna())
            r[f'nse_{a}_vs_{b}'] = nse(ob, series[a])
    # annual biases relative to real and to the EB twin
    pk, mo, acc, mel = {k: [] for k in ('real', 'twin')}, {k: [] for k in ('real', 'twin')}, [], []
    for y in ok:
        sl = wy_of(s_net.index) == y
        o = obs[sl]
        if o.notna().sum() < 300 or o.max() < 25:
            continue
        n = s_net[sl]
        refs = [('real', o)] + ([('twin', series['twin_en'][sl])] if 'twin_en' in series else [])
        for k, ref in refs:
            pk[k].append((n.max() - ref.max()) / max(ref.max(), 1) * 100)
            a, b = melt_out_idx(n.values), melt_out_idx(np.nan_to_num(ref.values))
            if a is not None and b is not None:
                mo[k].append(a - b)
        # errors by season: Dec-Feb mean (accumulation) and Apr-Jun mean (melt)
        mm = n.index.month
        acc.append(float((n - o)[np.isin(mm, [12, 1, 2])].mean()))
        mel.append(float((n - o)[np.isin(mm, [4, 5, 6])].mean()))
    for k in ('real', 'twin'):
        r[f'peak_bias_pct_vs_{k}'] = float(np.median(pk[k])) if pk[k] else np.nan
        r[f'melt_bias_d_vs_{k}'] = float(np.median(mo[k])) if mo[k] else np.nan
    r['err_DJF_mm'] = float(np.nanmedian(acc)) if acc else np.nan
    r['err_AMJ_mm'] = float(np.nanmedian(mel)) if mel else np.nan
    if sid in cal and cal[sid].get('energy_balance'):
        r.update({f'eb_{k}': v for k, v in cal[sid]['energy_balance']['params'].items()})
    rows.append(r)
t = pd.DataFrame(rows)
t.to_parquet(f'{ROOT}/results/twin_{run}.parquet')
med = t.drop(columns=['station']).median(numeric_only=True).to_dict()
json.dump({'run': run, 'median': med, 'n': len(t)}, open(f'{ROOT}/results/twin_{run}.json', 'w'), indent=1)
print(pd.Series(med).round(3).to_string())
