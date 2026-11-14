"""NESO open data via the documented CKAN datastore SQL API (api.neso.energy). The direct CSV downloads redirect to a
Cloudflare R2 storage host that this sandbox's proxy blocks (403), so only the datastore API is used, gently
(one query at a time, pauses between queries). Resumable: each piece is written once.

Stages (argv): inertia | genmix | scan [YYYY-MM ...]
  inertia -> raw/neso/inertia.parquet          outturn + market-provided inertia per settlement period (GVA.s)
  genmix  -> raw/neso/genmix.parquet           historic GB generation mix (half-hourly, MW), 2018+
  scan    -> raw/neso/scan/<YYYY-MM>_<h>.parquet  per-minute summary of 1 s frequency: min, max, mean,
             largest 10 s fall (d10min) and rise (d10max), sample count, tz suffix
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, json, re, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neso_sql import sql
RAW = f'{GRID_ROOT}/data/raw/neso'; os.makedirs(f'{RAW}/scan', exist_ok=True)
pkg = json.load(open(f'{RAW}/pkg.json'))['result']
MONTHS = {m: i for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                       'September', 'October', 'November', 'December'], 1)}
FREQ_RES = {}
for r in pkg['resources']:
    m = re.match(r'(\w+) (\d{4})', r['name'])
    if m and m.group(1) in MONTHS and r.get('datastore_active'):
        FREQ_RES[f'{m.group(2)}-{MONTHS[m.group(1)]:02d}'] = r['id']
json.dump(FREQ_RES, open(f'{RAW}/freq_resources.json', 'w'), indent=0)


def paged(rid, where='', order='"_id"', page=30000):
    out = []; last = 0
    while True:
        w = f'WHERE "_id" > {last}' + (f' AND {where}' if where else '')
        r = sql(f'SELECT * FROM "{rid}" {w} ORDER BY {order} LIMIT {page}')
        if not r: break
        out += r; last = max(int(x['_id']) for x in r)
        print(' page', rid[:8], len(out), flush=True)
        if len(r) < page: break
    return pd.DataFrame(out).drop(columns=['_full_text'], errors='ignore')


def stage_inertia():
    f = f'{RAW}/inertia.parquet'
    if os.path.exists(f): return
    p = json.load(open(f'{RAW}/system-inertia.json'))['result']
    parts = []
    for r in p['resources']:
        d = paged(r['id']); d['resource'] = r['name']; parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    for c in a.columns: a[c] = a[c].astype(str)
    a.to_parquet(f, index=False)


def stage_genmix():
    f = f'{RAW}/genmix.parquet'
    if os.path.exists(f): return
    d = paged('f93d1835-75bc-43e5-84ad-12472b180a98', where='"DATETIME" >= \'2018-01-01\'')
    d.to_parquet(f, index=False)


def stage_scan(months):
    for ym in months:
        rid = FREQ_RES.get(ym)
        if rid is None: print('no resource', ym); continue
        y, m = map(int, ym.split('-'))
        nxt = f'{y + (m == 12)}-{(m % 12) + 1:02d}-01'
        for h, (a, b) in enumerate([(f'{ym}-01', f'{ym}-16'), (f'{ym}-16', nxt)]):
            f = f'{RAW}/scan/{ym}_{h}.parquet'
            if os.path.exists(f): continue
            q = f'''SELECT substring(dtm::text,1,16) AS m, min(ff) AS fmin, max(ff) AS fmax, avg(ff) AS fmean,
                    min(d10) AS d10min, max(d10) AS d10max, count(*) AS n, max(substring(dtm::text,21,5)) AS tz
                    FROM (SELECT dtm, f::float AS ff, f::float - lag(f::float,10) OVER (ORDER BY dtm) AS d10
                          FROM "{rid}" WHERE dtm >= '{a}' AND dtm < '{b}') x GROUP BY 1 ORDER BY 1'''
            t0 = time.time()
            try:
                r = sql(q, timeout=900)
            except Exception as e:
                print('FAIL', ym, h, e, flush=True); continue
            d = pd.DataFrame(r)
            if d.empty:
                print('EMPTY', ym, h, flush=True); continue
            for c in ['fmin', 'fmax', 'fmean', 'd10min', 'd10max']: d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
            d['n'] = pd.to_numeric(d['n']).astype('int16')
            d.to_parquet(f, index=False)
            print('scan', ym, h, len(d), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'inertia': stage_inertia()
    elif st == 'genmix': stage_genmix()
    elif st == 'scan':
        months = sys.argv[2:] or sorted(k for k in FREQ_RES if k >= '2018-01')
        # 9 Aug 2019 month first, then most recent first (exam years), then the rest
        pri = ['2019-08'] + sorted([k for k in months if k != '2019-08'], reverse=True)
        stage_scan(pri)
    print('DONE', st, flush=True)
