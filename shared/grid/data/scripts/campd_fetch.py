"""EPA CAMPD hourly unit emissions for Texas, 2018-2025, from the CAMPD bulk files (no API key needed for the files:
https://api.epa.gov/easey/bulk-files/emissions/hourly/state/emissions-hourly-<YYYY>-tx.csv ; the file *listing*
endpoint camd-services/bulk-files needs an api.data.gov key and was not used). Streams each year to parquet,
deletes the CSV. Times in CAMPD are local STANDARD time (no DST); converted to UTC with the plant's EIA timezone
(standard offset). Resumable."""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, subprocess, time
import numpy as np, pandas as pd
RAW = f'{GRID_ROOT}/data/raw/campd'; OUT = f'{GRID_ROOT}/data/proc/campd/hourly'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
URL = 'https://api.epa.gov/easey/bulk-files/emissions/hourly/state/emissions-hourly-{y}-tx.csv'
years = [int(a) for a in sys.argv[1:]] or list(range(2018, 2026))
# plant timezone from EIA (PUDL out_eia__yearly_plants); default America/Chicago
pl = pd.read_parquet(f'{GRID_ROOT}/data/raw/pudl/out_eia__yearly_plants.parquet', columns=['plant_id_eia', 'timezone'])
tz = pl.dropna().drop_duplicates('plant_id_eia', keep='last').set_index('plant_id_eia').timezone
def std_offset_h(z):
    return pd.Timestamp('2021-01-15 12:00', tz=z).utcoffset().total_seconds() / 3600
off = {p: std_offset_h(z) for p, z in tz.items()}
cols = {'Facility ID': 'plant_id_epa', 'Unit ID': 'unit_id_epa', 'Date': 'date', 'Hour': 'hour',
        'Operating Time': 'op_time', 'Gross Load (MW)': 'gross_load_MW', 'Steam Load (1000 lb/hr)': 'steam_klb_h',
        'Heat Input (mmBtu)': 'heat_input_MMBtu', 'Heat Input Measure Indicator': 'hi_meas',
        'CO2 Mass (short tons)': 'co2_short_tons', 'CO2 Mass Measure Indicator': 'co2_meas',
        'NOx Mass (lbs)': 'nox_lbs', 'SO2 Mass (lbs)': 'so2_lbs'}
static_cols = ['Facility ID', 'Facility Name', 'Unit ID', 'Associated Stacks', 'Primary Fuel Type', 'Secondary Fuel Type',
               'Unit Type', 'SO2 Controls', 'NOx Controls', 'Program Code']
for y in years:
    out = f'{OUT}/tx_{y}.parquet'
    if os.path.exists(out): continue
    csv = f'{RAW}/tx_{y}.csv'
    for a in range(4):
        r = subprocess.run(['curl', '-sS', '-f', '-L', '--retry', '3', '-o', csv, URL.format(y=y)])
        if r.returncode == 0: break
        time.sleep(60)
    if r.returncode != 0: print('MISSING', y, flush=True); continue
    parts = []; statics = []
    for ch in pd.read_csv(csv, usecols=list(cols) + static_cols, dtype=str, chunksize=1_000_000):
        statics.append(ch[static_cols].drop_duplicates())
        c = ch[list(cols)].rename(columns=cols)
        o = pd.DataFrame({'plant_id_epa': c.plant_id_epa.astype('int32'), 'unit_id_epa': c.unit_id_epa.astype('category')})
        loc = pd.to_datetime(c.date, format='%Y-%m-%d') + pd.to_timedelta(c.hour.astype(int), unit='h')
        o_h = o.plant_id_epa.map(off).fillna(-6.0)   # EPA facility id == EIA plant id for nearly all units (checked in crosswalk)
        o['time_utc'] = (loc - pd.to_timedelta(o_h, unit='h')).dt.tz_localize('UTC')
        for k in ['op_time', 'gross_load_MW', 'steam_klb_h', 'heat_input_MMBtu', 'co2_short_tons', 'nox_lbs', 'so2_lbs']:
            o[k] = pd.to_numeric(c[k], errors='coerce').astype('float32')
        o['hi_meas'] = c.hi_meas.astype('category'); o['co2_meas'] = c.co2_meas.astype('category')
        parts.append(o)
    d = pd.concat(parts, ignore_index=True)
    for k in ['unit_id_epa', 'hi_meas', 'co2_meas']: d[k] = d[k].astype(str).replace('nan', None).astype('category')
    d = d.sort_values(['plant_id_epa', 'unit_id_epa', 'time_utc'])
    d.to_parquet(out + '.tmp', index=False, compression='zstd', row_group_size=500_000); os.replace(out + '.tmp', out)
    s = pd.concat(statics).drop_duplicates(); s['year'] = y
    s.to_parquet(f'{OUT}/units_tx_{y}.parquet', index=False)
    print(f'ok {y} rows={len(d)} units={d.groupby(["plant_id_epa","unit_id_epa"],observed=True).ngroups} csvMB={os.path.getsize(csv)/1e6:.0f}', flush=True)
    os.remove(csv)
print('DONE', flush=True)
