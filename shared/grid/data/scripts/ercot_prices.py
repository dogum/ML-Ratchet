"""ERCOT historical settlement point prices at load zones and hubs (public ERCOT MIS reports, no login):
 DAM: report type 13060 (DAMLZHBSPP_<year>.zip), hourly; RTM: report type 13061 (RTMLZHBSPP_<year>.zip), 15-min.
List: https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=<id>
File: https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId=<DocID>
Times are Central Prevailing Time, hour-ending, with a repeated-hour flag at the DST fall-back; converted to UTC,
interval-BEGINNING. Prices in $/MWh."""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, io, json, zipfile, requests, time
import numpy as np, pandas as pd
RAW = f'{GRID_ROOT}/data/raw/ercot'; OUT = f'{GRID_ROOT}/data/proc/prices'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
def docs(rt):
    cache = f'{RAW}/list_{rt}.json'
    if os.path.exists(cache): return json.load(open(cache))
    for a in range(6):
        r = requests.get(f'https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId={rt}', timeout=60)
        if r.status_code == 200 and r.text.startswith('{'): break
        print('list retry', rt, r.status_code, flush=True); time.sleep(120 * (a + 1))
    L = [x['Document'] for x in r.json()['ListDocsByRptTypeRes']['DocumentList']]
    json.dump(L, open(cache, 'w')); return L
def to_utc(date, he, interval, rep):
    loc = pd.to_datetime(date, format='%m/%d/%Y') + pd.to_timedelta(he - 1, unit='h') + pd.to_timedelta((interval - 1) * 15, unit='min')
    amb = ~(rep.astype(str).str.upper().str.startswith('Y'))   # first occurrence = DST (True)
    t = loc.dt.tz_localize('America/Chicago', ambiguous=amb.values, nonexistent='shift_forward')
    return t.dt.tz_convert('UTC')
out = {}
for rt, kind in [(13060, 'dam'), (13061, 'rtm')]:
    parts = []
    for d in docs(rt):
        y = int(d['FriendlyName'][-4:])
        if y < 2018: continue
        z = f'{RAW}/{d["ConstructedName"]}'
        if not os.path.exists(z):
            for a in range(6):
                time.sleep(15 + 60 * a)   # ERCOT throttles rapid downloads (HTTP 403); be gentle
                r = requests.get(f'https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={d["DocID"]}', timeout=300)
                if r.status_code == 200: break
                print('retry', d['FriendlyName'], r.status_code, flush=True)
            r.raise_for_status(); open(z, 'wb').write(r.content)
        with zipfile.ZipFile(z) as zf:
            name = zf.namelist()[0]
            xl = pd.read_excel(io.BytesIO(zf.read(name)), sheet_name=None)
        df = pd.concat(xl.values(), ignore_index=True)
        key = ['Delivery Date', 'Hour Ending'] if kind == 'dam' else ['Delivery Date', 'Delivery Hour', 'Delivery Interval']
        nbad = int(df[key].isna().any(axis=1).sum())
        if nbad: print('dropping rows with missing time keys:', nbad, flush=True)
        df = df.dropna(subset=key)
        if kind == 'dam':
            t = to_utc(df['Delivery Date'], df['Hour Ending'].astype(str).str.split(':').str[0].astype(int), pd.Series(1, index=df.index), df['Repeated Hour Flag'])
            p = pd.DataFrame({'time_utc': t, 'node': df['Settlement Point'].astype(str), 'price': df['Settlement Point Price'].astype('float32')})
        else:
            t = to_utc(df['Delivery Date'], df['Delivery Hour'].astype(int), df['Delivery Interval'].astype(int), df['Repeated Hour Flag'])
            p = pd.DataFrame({'time_utc': t, 'node': df['Settlement Point Name'].astype(str), 'price': df['Settlement Point Price'].astype('float32')})
        parts.append(p); print(kind, y, len(p), p.node.nunique(), flush=True)
    a = pd.concat(parts, ignore_index=True).drop_duplicates(['time_utc', 'node']).sort_values(['node', 'time_utc'])
    a['node'] = a.node.astype('category')
    a.to_parquet(f'{OUT}/ercot_{kind}.parquet', index=False, compression='zstd')
    if kind == 'rtm':   # hourly mean of the four 15-min SPPs (hour-beginning UTC)
        h = a.assign(time_utc=a.time_utc.dt.floor('h')).groupby(['node', 'time_utc'], observed=True).price.agg(['mean', 'count']).reset_index()
        h = h.rename(columns={'mean': 'price', 'count': 'n_intervals'}); h['price'] = h.price.astype('float32')
        h.to_parquet(f'{OUT}/ercot_rtm_hourly.parquet', index=False, compression='zstd')
print('DONE')
