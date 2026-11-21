"""Parse NWIS daily values and peaks for the CAMELS-US gauges -> proc/nwis.

daily/<camels_id>.parquet : Q_cfs, Q_cms, Q_mm (over spec area_km2 = NWIS drainage area), src='nwis', flag (USGS
                            qualifier string: A approved, P provisional, e estimated, Ice, Eqp, Bkw, Dis, Rat, ...).
                            If a site has several daily-mean series (ts ids), the one with most values is primary and
                            the others fill gaps (flag gets '+alt').
peaks.parquet            : annual peak file (peak_dt, peak_cfs, peak_cms, peak_cd codes, gage_ht) for all gauges.
coverage.parquet         : per gauge counts per period and flag fractions.
"""
import gzip, glob, io, json, os, sys
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
CFS = 0.028316846592
sys.path.insert(0, _CODE)


def read_rdb(path):
    t = gzip.open(path, 'rt').read()
    lines = [l for l in t.splitlines() if l and not l.startswith('#')]
    if len(lines) < 3:
        return None
    return pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[2:])), sep='\t', dtype=str)


def main():
    import waterdata as wd
    os.makedirs(f'{ROOT}/proc/nwis/daily', exist_ok=True)
    cov = []
    for p in sorted(glob.glob(f'{ROOT}/raw/nwis/dv/*.rdb.gz')):
        g = os.path.basename(p)[:8]
        bid = f'camels_{g}'
        df = read_rdb(p)
        if df is None:
            cov.append({'gauge_id': bid, 'n': 0}); continue
        vcols = [c for c in df.columns if c.endswith('_00060_00003')]
        idx = pd.to_datetime(df.datetime)
        series = []
        for c in vcols:
            v = pd.to_numeric(df[c], errors='coerce')
            f = df[c + '_cd'].fillna('') if c + '_cd' in df else pd.Series('', index=df.index)
            txt = df[c].where(v.isna() & df[c].notna(), '')            # e.g. 'Ice', 'Eqp', 'Ssn' in the value field
            f = (f + np.where(txt != '', ':' + txt, '')).str.strip(':')
            s = pd.DataFrame({'Q_cfs': v.values, 'flag': f.values}, index=idx)
            series.append(s[s.Q_cfs.notna() | (s.flag != '')])
        series.sort(key=lambda s: -s.Q_cfs.notna().sum())
        out = series[0]
        for alt in series[1:]:
            alt = alt[alt.Q_cfs.notna()].copy(); alt['flag'] = alt.flag + '+alt'
            out = out.combine_first(alt) if len(out) else alt
        out = out[~out.index.duplicated()].sort_index()
        area = wd.spec(bid)['area_km2']
        out['Q_cms'] = out.Q_cfs * CFS
        out['Q_mm'] = out.Q_cms * 86.4 / area
        out['src'] = 'nwis'
        out.loc[out.Q_cfs < 0, ['Q_cfs', 'Q_cms', 'Q_mm']] = np.nan
        out = out[['Q_mm', 'Q_cms', 'Q_cfs', 'src', 'flag']]
        out[['Q_mm', 'Q_cms', 'Q_cfs']] = out[['Q_mm', 'Q_cms', 'Q_cfs']].astype('float32')
        out.index.name = 'date'
        out.to_parquet(f'{ROOT}/proc/nwis/daily/{bid}.parquet', compression='zstd')
        q = out.Q_mm
        rec = {'gauge_id': bid, 'n': int(q.notna().sum()), 'start': str(q.dropna().index.min().date()) if q.notna().any() else None,
               'end': str(q.dropna().index.max().date()) if q.notna().any() else None, 'n_series': len(vcols)}
        for name, (a, b) in {'cal': ('1999-10-01', '2008-09-30'), 'val': ('1989-10-01', '1999-09-30'),
                             'recent': ('2015-10-01', '2023-09-30'), 'wy2024_26': ('2023-10-01', '2026-09-30')}.items():
            n = (b and (pd.Timestamp(b) - pd.Timestamp(a)).days + 1)
            rec[f'cov_{name}'] = float(q.loc[a:b].notna().sum() / n)
        fl = out.flag.fillna('')
        rec['frac_ice'] = float(fl.str.contains('Ice').mean())
        rec['frac_est'] = float(fl.str.contains(r'(^|:)e(:|$)|e$', regex=True).mean())
        rec['frac_prov'] = float(fl.str.contains('P').mean())
        cov.append(rec)
    c = pd.DataFrame(cov)
    c.to_parquet(f'{ROOT}/proc/nwis/coverage.parquet')
    print(len(c), 'gauges'); print(c.describe().T[['mean', 'min', '50%', 'max']])
    # peaks
    rows = []
    for p in sorted(glob.glob(f'{ROOT}/raw/nwis/peak/*.rdb.gz')):
        g = os.path.basename(p)[:8]
        df = read_rdb(p)
        if df is None or 'peak_va' not in df:
            continue
        rows.append(pd.DataFrame({'gauge_id': f'camels_{g}', 'peak_dt': df.peak_dt, 'peak_tm': df.get('peak_tm'),
                                  'peak_cfs': pd.to_numeric(df.peak_va, errors='coerce'),
                                  'peak_cd': df.peak_cd.fillna(''), 'gage_ht_ft': pd.to_numeric(df.gage_ht, errors='coerce')}))
    if rows:
        pk = pd.concat(rows, ignore_index=True)
        pk['peak_cms'] = pk.peak_cfs * CFS
        pk['date'] = pd.to_datetime(pk.peak_dt.str.replace('-00', '-01'), errors='coerce')   # month/day unknown -> 01
        pk['date_partial'] = pk.peak_dt.str.contains('-00')
        pk['wy'] = pk.date.dt.year + (pk.date.dt.month >= 10).astype(int)
        pk.to_parquet(f'{ROOT}/proc/nwis/peaks.parquet')
        print(len(pk), 'peaks at', pk.gauge_id.nunique(), 'gauges')


if __name__ == '__main__':
    main()
