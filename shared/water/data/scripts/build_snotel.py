"""Parse SNOTEL daily CSVs -> proc/snotel (rerunnable; processes whatever has been downloaded).

daily/<id>_<state>_SNTL.parquet: SWE mm (start of day, i.e. 00:00 local standard time), SWE_flag, P_accum mm (water-year
accumulation, start of day), P mm/day (PRCP if published, else the next-day difference of P_accum; negative -> NaN),
T_mean/T_max/T_min degC (that day), snow_depth_mm. Index = local calendar day.
Cleaning: SWE < 0 -> NaN; SWE jumps > +250 mm/day or > -150 mm/day -> NaN; T outside [-50, 45] degC -> NaN;
P_accum diffs < 0 (water-year resets, corrections) -> NaN.
stations.parquet: metadata + record stats (n complete water years, peak SWE median).
"""
import glob, gzip, io, os
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
IN2MM = 25.4


def f2c(x):
    return (x - 32) * 5 / 9


def parse(p):
    t = gzip.open(p, 'rt').read()
    lines = [l for l in t.splitlines() if l and not l.startswith('#')]
    if len(lines) < 2:
        return None
    d = pd.read_csv(io.StringIO('\n'.join(lines)))
    cols = {}
    for c in d.columns:
        lc = c.lower()
        if lc.startswith('snow water equivalent') and 'qc flag' in lc: cols[c] = 'SWE_flag'
        elif lc.startswith('snow water equivalent'): cols[c] = 'SWE_in'
        elif lc.startswith('precipitation accumulation') and 'qc flag' in lc: cols[c] = 'Pacc_flag'
        elif lc.startswith('precipitation accumulation'): cols[c] = 'Pacc_in'
        elif lc.startswith('precipitation increment'): cols[c] = 'PRCP_in'
        elif lc.startswith('air temperature average'): cols[c] = 'TAVG_F'
        elif lc.startswith('air temperature maximum'): cols[c] = 'TMAX_F'
        elif lc.startswith('air temperature minimum'): cols[c] = 'TMIN_F'
        elif lc.startswith('snow depth'): cols[c] = 'SNWD_in'
        elif lc == 'date': cols[c] = 'date'
    d = d.rename(columns=cols)
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['date']).set_index('date').sort_index()
    out = pd.DataFrame(index=d.index)
    num = lambda c: pd.to_numeric(d[c], errors='coerce') if c in d else pd.Series(np.nan, index=d.index)
    swe = num('SWE_in') * IN2MM
    swe[swe < 0] = np.nan
    dj = swe.diff()
    swe[(dj > 250) | (dj < -150)] = np.nan
    out['SWE'] = swe
    out['SWE_flag'] = d['SWE_flag'].astype(str) if 'SWE_flag' in d else ''
    pacc = num('Pacc_in') * IN2MM
    out['P_accum'] = pacc
    pd_ = pacc.shift(-1) - pacc              # precip during day D = acc(start of D+1) - acc(start of D)
    pd_[pd_ < 0] = np.nan
    pr = num('PRCP_in') * IN2MM
    out['P'] = pr.where(pr.notna(), pd_)
    out['P_src'] = np.where(pr.notna(), 'PRCP', np.where(pd_.notna(), 'PREC_diff', ''))
    for a, b in [('TAVG_F', 'T_mean'), ('TMAX_F', 'T_max'), ('TMIN_F', 'T_min')]:
        x = f2c(num(a)); x[(x < -50) | (x > 45)] = np.nan
        out[b] = x
    out['snow_depth_mm'] = num('SNWD_in') * IN2MM
    for c in ['SWE', 'P_accum', 'P', 'T_mean', 'T_max', 'T_min', 'snow_depth_mm']:
        out[c] = out[c].astype('float32')
    return out


def main():
    os.makedirs(f'{ROOT}/proc/snotel/daily', exist_ok=True)
    st = pd.read_csv(f'{ROOT}/raw/snotel/stations.csv', comment='#', dtype=str)
    st.columns = ['sid', 'name', 'state', 'network', 'elev_ft', 'lat', 'lon', 'huc12', 'county', 'start', 'end'][:len(st.columns)]
    rows = []
    for r in st.itertuples(index=False):
        sid = f'{r.sid}_{r.state}_SNTL'
        p = f'{ROOT}/raw/snotel/daily/{sid}.csv.gz'
        q = f'{ROOT}/proc/snotel/daily/{sid}.parquet'
        if not os.path.exists(p):
            continue
        if not os.path.exists(q) or os.path.getmtime(q) < os.path.getmtime(p):
            d = parse(p)
            if d is None or d.SWE.notna().sum() == 0:
                continue
            d.to_parquet(q, compression='zstd')
        d = pd.read_parquet(q, columns=['SWE', 'P', 'T_mean'])
        wy = d.index.year + (d.index.month >= 10).astype(int)
        g = d.groupby(wy)
        ok = (g.SWE.count() >= 330) & (g.T_mean.count() >= 300) & (g.P.count() >= 300)
        okwy = ok[ok].index
        pk = g.SWE.max().loc[okwy]
        rows.append({'station_id': sid, 'triplet': f'{r.sid}:{r.state}:SNTL', 'name': r.name, 'state': r.state,
                     'lat': float(r.lat), 'lon': float(r.lon), 'elev_m': float(r.elev_ft) * 0.3048, 'huc12': r.huc12,
                     'start': str(d.SWE.dropna().index.min().date()), 'end': str(d.SWE.dropna().index.max().date()),
                     'n_wy_complete': int(len(okwy)), 'wy_complete': [int(x) for x in okwy],
                     'peak_swe_median_mm': float(pk.median()) if len(pk) else np.nan,
                     'n_wy_complete_2001_2010': int(sum(2001 <= y <= 2010 for y in okwy)),
                     'n_wy_complete_2011_2025': int(sum(2011 <= y <= 2025 for y in okwy))})
    m = pd.DataFrame(rows)
    m.to_parquet(f'{ROOT}/proc/snotel/stations.parquet')
    print(len(m), 'stations;', (m.n_wy_complete >= 10).sum(), 'with >=10 complete WY')


if __name__ == '__main__':
    main()
