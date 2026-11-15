"""Download EIA-930 six-month bulk CSVs (BALANCE, INTERCHANGE, SUBREGION), 2018 onward, convert each to a
compact parquet part, delete the CSV. Resumable: skips parts already written.
Source: https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_<TYPE>_<YYYY>_<Jan_Jun|Jul_Dec>.csv"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, re, time, subprocess
import numpy as np, pandas as pd
RAW = f'{GRID_ROOT}/data/raw/eia930'; PARTS = f'{GRID_ROOT}/data/proc/eia930/parts'
os.makedirs(RAW, exist_ok=True); os.makedirs(PARTS, exist_ok=True)
URL = 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_{t}_{y}_{h}.csv'
FUELMAP = {'Coal': 'COL', 'Natural Gas': 'NG', 'Nuclear': 'NUC', 'All Petroleum Products': 'OIL',
           'Hydropower and Pumped Storage': 'WAT', 'Solar': 'SUN', 'Wind': 'WND', 'Other Fuel Sources': 'OTH',
           'Unknown Fuel Sources': 'UNK'}

def parse_time(s):
    s = s.astype(str).str.strip()
    for fmt in ('%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S'):
        t = pd.to_datetime(s, format=fmt, errors='coerce')
        if t.notna().mean() > 0.99: return t
    return pd.to_datetime(s, format='mixed', errors='coerce')

def short(c):
    """Map an EIA column name to a short snake name."""
    base = c; kind = 'rep'
    m = re.match(r'^(.*?) \((Imputed|Adjusted)\)$', c.replace(' (MW)', '', 1) if False else c)
    if c.endswith('(Imputed)'): kind = 'imp'; base = c[:-len(' (Imputed)')]
    elif c.endswith('(Adjusted)'): kind = 'adj'; base = c[:-len(' (Adjusted)')]
    base = base.replace(' (MW)', '')
    if base.startswith('Net Generation from '):
        f = base[len('Net Generation from '):]
        code = FUELMAP.get(f, re.sub(r'[^A-Za-z]+', '_', f).strip('_').upper())
        name = 'ng_' + code
    else:
        name = {'Demand Forecast': 'df', 'Demand': 'd', 'Net Generation': 'ng', 'Total Interchange': 'ti',
                'Sum(Valid DIBAs)': 'sum_diba'}.get(base, re.sub(r'[^A-Za-z]+', '_', base).strip('_').lower())
    return f'{name}__{kind}'

def num(s):
    if s.dtype == object: s = s.astype(str).str.replace(',', '', regex=False).replace({'nan': None, '': None})
    return pd.to_numeric(s, errors='coerce').astype('float32')

def convert(t, csv, out):
    df = pd.read_csv(csv, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    o = pd.DataFrame({'ba': df['Balancing Authority'].astype('category')})
    o['time_utc'] = parse_time(df['UTC Time at End of Hour']) - pd.Timedelta(hours=1)   # hour-BEGINNING UTC
    o['time_utc'] = o['time_utc'].dt.tz_localize('UTC')
    if t == 'BALANCE':
        o['region'] = df.get('Region', pd.Series([None]*len(df))).astype('category')
        for c in df.columns:
            if '(MW)' in c or c.startswith('Net Generation') or c.startswith('Demand') or c.startswith('Total Interchange') or c.startswith('Sum('):
                o[short(c)] = num(df[c])
    elif t == 'INTERCHANGE':
        o['diba'] = df['Directly Interconnected Balancing Authority'].astype('category')
        o['interchange_MW'] = num(df['Interchange (MW)'])
    else:
        o['subregion'] = df['Sub-Region'].astype('category')
        o['demand_MW'] = num(df['Demand (MW)'])
    o = o.dropna(subset=['time_utc'])
    o.to_parquet(out + '.tmp', index=False, compression='zstd'); os.replace(out + '.tmp', out)
    return len(o)

jobs = []
for y in range(2018, 2027):
    for h in ('Jan_Jun', 'Jul_Dec'):
        for t in ('BALANCE', 'SUBREGION', 'INTERCHANGE'):
            jobs.append((t, y, h))
for t, y, h in jobs:
    out = f'{PARTS}/{t}_{y}_{h}.parquet'
    if os.path.exists(out): continue
    csv = f'{RAW}/{t}_{y}_{h}.csv'
    url = URL.format(t=t, y=y, h=h)
    ok = False
    for attempt in range(4):
        r = subprocess.run(['curl', '-sS', '-f', '-L', '--retry', '3', '-o', csv, url])
        if r.returncode == 0: ok = True; break
        if r.returncode == 22: break   # HTTP error (e.g. 404: not published)
        time.sleep(30)
    if not ok:
        print(f'MISSING {t} {y} {h} rc={r.returncode}', flush=True); continue
    try:
        n = convert(t, csv, out); print(f'ok {t} {y} {h} rows={n} csvMB={os.path.getsize(csv)/1e6:.0f}', flush=True)
    except Exception as e:
        print(f'FAIL convert {t} {y} {h}: {e!r}', flush=True)
    finally:
        if os.path.exists(csv): os.remove(csv)
print('DONE', flush=True)
