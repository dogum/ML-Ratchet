"""Dead fuel moisture on DEV fire sites vs an independent operational model.

Real 10-h fuel-stick observations (NFMD) were not obtainable (NFMD left wfas.net; see fire/firedata/STATUS.md), so the
check uses gridMET's daily 100-h and 1000-h dead fuel moisture (fm100, fm1000: the NFDRS 1978 equations driven by
gridMET daily weather; Abatzoglou 2013) at each dev fire centroid. FireSim's Nelson-type stick model is driven by the
hourly HRRR series at the same place (precipitation from HRRR, or gridMET daily / 24 where HRRR has none).
Compared: FireSim daily mean 100-h / 1000-h vs gridMET fm100 / fm1000 (local calendar days, after a 10-day spin-up),
bias, RMSE, correlation; plus the 1-h and 10-h diurnal ranges on the same days (no gridMET counterpart).
Output: checks/moisture_real.json
"""
import numpy as np
import pandas as pd

from _common import write
import firesim as fs
from firesim import realdata as RD

fd = RD._fd()
rows = []
per_fire = []
for fid in fd.fires('dev').sort_values('Event_ID').Event_ID:
    assert not fd.guard.is_exam(fid)
    W = fd.weather(fid)
    if 'grid' not in W or 'daily' not in W:
        continue
    r = fd.fires('dev').set_index('Event_ID').loc[fid]
    w = RD.weather_series(W, float(r.cx), float(r.cy))
    lon = w.attrs['lonlat'][0]
    w = RD.fill_precip_from_daily(w, W['daily'], lon)
    if w[['T', 'RH', 'wind_speed']].isna().any().any():
        w = w.interpolate(limit_direction='both')
    m = fs.dead_moisture(w, classes=('1h', '10h', '100h', '1000h'))
    loc = m.copy()
    loc['day'] = (loc.index + pd.to_timedelta(lon / 15.0, unit='h')).tz_convert(None).normalize()
    full = loc.groupby('day').size() == 24
    dm = loc.groupby('day').agg(m100=('m100h', 'mean'), m1000=('m1000h', 'mean'), r1=('m1h', lambda s: s.max() - s.min()),
                                r10=('m10h', lambda s: s.max() - s.min()), m1min=('m1h', 'min'))
    dm = dm[full.reindex(dm.index).fillna(False).to_numpy(bool)]
    dm = dm.iloc[10:]                       # spin-up
    g = W['daily'].copy()
    g.index = pd.DatetimeIndex(g.index).normalize()
    j = dm.join(g[['fm100', 'fm1000', 'pr']], how='inner').dropna(subset=['fm100'])
    if len(j) < 5:
        continue
    j['fire_id'] = fid
    j['precip_source'] = 'gridmet' if w['precip_from_daily'].mean() > 0.5 else 'hrrr'
    rows.append(j)
    per_fire.append(dict(fire_id=fid, n_days=len(j), bias100=float((j.m100 - j.fm100).mean()),
                         r100=float(j[['m100', 'fm100']].corr().iloc[0, 1]) if len(j) > 3 else None,
                         bias1000=float((j.m1000 - j.fm1000).mean()), precip_source=j.precip_source.iloc[0]))
    print(fid, len(j), 'bias100 %.1f' % per_fire[-1]['bias100'], flush=True)

D = pd.concat(rows)


def st(a, b):
    d = a - b
    return dict(n=int(len(d)), bias=float(d.mean()), mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())),
                pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(pd.Series(a).corr(pd.Series(b), method='spearman')),
                mean_firesim=float(a.mean()), mean_gridmet=float(b.mean()))


res = dict(
    what='FireSim Nelson-type dead fuel moisture (hourly HRRR) vs gridMET NFDRS fm100/fm1000 (daily) at dev fire centroids',
    note='NFMD 10-h stick observations were not obtainable; gridMET fm100/fm1000 are model values (NFDRS 1978), an '
         'independent operational reference, not observations',
    n_fires=int(D.fire_id.nunique()), n_days=int(len(D)),
    m100_vs_fm100=st(D.m100.to_numpy(), D.fm100.to_numpy()),
    m1000_vs_fm1000=st(D.m1000.to_numpy(), D.fm1000.to_numpy()),
    m100_vs_fm100_hrrr_precip_only=st(D[D.precip_source == 'hrrr'].m100.to_numpy(), D[D.precip_source == 'hrrr'].fm100.to_numpy()) if (D.precip_source == 'hrrr').sum() > 10 else None,
    diurnal=dict(m1h_daily_range_median=float(D.r1.median()), m10h_daily_range_median=float(D.r10.median()),
                 m1h_daily_min_median=float(D.m1min.median())),
    per_fire=per_fire,
)
write('moisture_real.json', res)
print({k: v for k, v in res.items() if k != 'per_fire'})
