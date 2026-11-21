"""Aggregate hourly NWM v3.0 CHANOBS streamflow to daily means -> proc/nwm/daily_local.parquet, daily_utc.parquet.

daily_local: mean of the 24 hourly values whose valid time falls in the gauge's local-STANDARD-time calendar day
(NWIS tz_cd: EST -5, CST -6, MST -7, PST -8 h), the convention of USGS daily values. Days with < 20 hourly values -> NaN.
daily_utc: UTC calendar day.
"""
import glob, os
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
H = f'{ROOT}/raw/nwm/chanobs_hourly'
OFF = {'EST': 5, 'EDT': 5, 'CST': 6, 'CDT': 6, 'MST': 7, 'MDT': 7, 'PST': 8, 'PDT': 8, 'AKST': 9}


def main():
    a = pd.read_parquet(f'{ROOT}/proc/attributes/camels.parquet').set_index('gauge_id')
    fs = sorted(glob.glob(f'{H}/*.parquet'))
    months = [os.path.basename(f)[:7] for f in fs]
    loc, utc = [], []
    for i, f in enumerate(fs):
        h = pd.read_parquet(f)
        cols = list(h.columns)
        offs = np.array([OFF.get(a.loc['camels_' + c, 'nwis_tz'], 6) for c in cols])
        nxt = pd.Period(months[i], 'M') + 1
        nf = f'{H}/{nxt}.parquet'
        hh = pd.concat([h, pd.read_parquet(nf).iloc[:12]]) if os.path.exists(nf) else h
        g = h.groupby(h.index.floor('D'))
        du = g.mean().where(g.count() >= 20)
        utc.append(du.astype('float32'))
        days = du.index
        out = np.full((len(days), len(cols)), np.nan, dtype=np.float32)
        arr = hh.values; t = hh.index
        for o in np.unique(offs):
            sel = offs == o
            tl = t - pd.Timedelta(hours=int(o))
            gl = pd.DataFrame(arr[:, sel], index=tl.floor('D'))
            m = gl.groupby(level=0).mean().where(gl.groupby(level=0).count() >= 20).reindex(days)
            out[:, sel] = m.values
        loc.append(pd.DataFrame(out, index=days, columns=cols))
    for name, parts in [('daily_local', loc), ('daily_utc', utc)]:
        d = pd.concat(parts).sort_index()
        d = d[~d.index.duplicated()]
        d.index.name = 'date'
        d.to_parquet(f'{ROOT}/proc/nwm/{name}.parquet', compression='zstd')
        print(name, d.shape, d.index.min(), d.index.max())


if __name__ == '__main__':
    main()
