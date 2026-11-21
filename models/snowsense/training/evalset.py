"""Real evaluation sets and the uncalibrated physics baselines (same protocol as B's snow baselines).

Scoring window WY2011-2025, complete water years only (snotel_meta.wy_complete), obs = SWE(D+1).
  dev  : the 39 dev stations with >= 5 complete scoring years (the dev gate; any period of dev stations is free)
  exam : the 621 exam stations B scored (snow_scores.parquet), for row-by-row comparison with B's climatology and
         per-station calibrated degree-day.
Uncalibrated physics: WaterSim snow-only (model='none'), default parameters, station P/T, Hargreaves radiation,
1-year spin-up from WY2010.
"""
import json
import sys
import numpy as np
import pandas as pd
from common import *

SCORE_WYS = range(2011, 2026)


def station_set(split):
    m = wd().snotel_meta(split=split).set_index('station_id')
    if split == 'exam':
        b = pd.read_parquet(SNOW_SCORES)
        ids = sorted(set(b[(b.split == 'exam')].station_id))
    else:
        ids = sorted(m.index)
    rows = []
    for sid in ids:
        ok = [y for y in m.loc[sid, 'wy_complete'] if y in SCORE_WYS]
        if len(ok) >= 5:
            rows.append((sid, float(m.loc[sid, 'lat']), float(m.loc[sid, 'elev_m']), ok))
    return rows


def physics(sid, lat, elev, snow, params=None, df=None, start='2010-10-01', end='2025-09-30'):
    import watersim as ws
    if df is None:
        df = load_station(sid)
    f = df.loc['2009-10-01':end, ['P', 'T_mean', 'T_min', 'T_max']]
    spec = {'id': sid, 'area_km2': 1.0, 'lat': lat, 'lon': -110.0, 'elev_mean_m': elev}
    o = ws.simulate(spec, f, start, end, model='none', snow=snow, routing='none', pet='hargreaves', params=params,
                    labels=False)
    return o['ts']['SWE']


def b_scores():
    b = pd.read_parquet(SNOW_SCORES)
    return b


if __name__ == '__main__':
    split = sys.argv[1]
    rows = []
    for k, (sid, lat, elev, ok) in enumerate(station_set(split)):
        df = load_station(sid)
        for snow, tag in (('degree_day', 'dd_uncal'), ('energy_balance', 'eb_uncal')):
            s = physics(sid, lat, elev, snow, df=df)
            r = score(df['SWE_eod'], s, ok)
            r.update(station_id=sid, method=tag)
            rows.append(r)
        if k % 50 == 0:
            print(k, sid, round(rows[-1]['nse'], 3), flush=True)
    sc = pd.DataFrame(rows)
    sc.to_parquet(f'{ROOT}/results/physics_{split}.parquet')
    print(sc.groupby('method')[['nse', 'rmse_mm', 'peak_swe_abs_err_pct', 'peak_swe_bias_pct', 'meltout_mae_days']].median())
