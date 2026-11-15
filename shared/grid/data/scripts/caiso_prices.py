"""CAISO OASIS LMPs (documented public API, oasis.caiso.com/oasisapi/SingleZip), hubs and DLAPs:
 DAM hourly (PRC_LMP, market_run_id=DAM) and real-time 15-min (PRC_RTPD_LMP, market_run_id=RTPD).
One request per node-group per month, >= 6 s apart (OASIS rate limit). OASIS only serves roughly the last
~3 years; months with 'no data' are logged and skipped. Resumable (one parquet per market-month)."""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, io, time, zipfile, requests, sys
import pandas as pd
RAW = f'{GRID_ROOT}/data/raw/caiso'; os.makedirs(RAW, exist_ok=True)
NODES = ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND', 'DLAP_PGAE-APND', 'DLAP_SCE-APND', 'DLAP_SDGE-APND']
Q = {'dam': ('PRC_LMP', 'DAM', 12), 'rtpd': ('PRC_RTPD_LMP', 'RTPD', 3)}
def get(url):
    for a in range(5):
        time.sleep(6 + 30 * a)
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200: return r.content
            print('http', r.status_code, flush=True)
        except Exception as e: print('err', e, flush=True)
    return None
months = pd.date_range('2018-01-01', '2026-11-01', freq='MS')
for kind, (qn, mr, ver) in Q.items():
    for m in months[::-1]:
        f = f'{RAW}/{kind}_{m:%Y-%m}.parquet'; nf = f + '.nodata'
        if os.path.exists(f) or os.path.exists(nf): continue
        a = m.tz_localize('America/Los_Angeles').tz_convert('UTC'); b = (m + pd.offsets.MonthBegin(1)).tz_localize('America/Los_Angeles').tz_convert('UTC')
        parts = []
        # RTPD is 15-min: split month in halves to stay under row limits
        spans = [(a, b)] if kind == 'dam' else [(a, a + (b - a) / 2), (a + (b - a) / 2, b)]
        for s, e in spans:
            url = (f'https://oasis.caiso.com/oasisapi/SingleZip?queryname={qn}&market_run_id={mr}&startdatetime={s:%Y%m%dT%H:%M}-0000'
                   f'&enddatetime={e:%Y%m%dT%H:%M}-0000&version={ver}&node={",".join(NODES)}&resultformat=6')
            c = get(url)
            if c is None: continue
            with zipfile.ZipFile(io.BytesIO(c)) as z:
                for n in z.namelist():
                    if n.endswith('.csv'): parts.append(pd.read_csv(z.open(n)))
        if not parts:
            open(nf, 'w').write('no data'); print(kind, f'{m:%Y-%m}', 'no data', flush=True); continue
        d = pd.concat(parts, ignore_index=True)
        tcol = 'INTERVALSTARTTIME_GMT'; vcol = 'MW' if 'MW' in d else ('PRC' if 'PRC' in d else 'VALUE'); typ = 'LMP_TYPE' if 'LMP_TYPE' in d else 'XML_DATA_ITEM'
        d = d[d[typ].isin(['LMP', 'MCC', 'MCL', 'MCE', 'LMP_PRC', 'LMP_CONG_PRC', 'LMP_LOSS_PRC', 'LMP_ENE_PRC'])]
        o = pd.DataFrame({'time_utc': pd.to_datetime(d[tcol], utc=True), 'node': d['NODE'].astype(str),
                          'comp': d[typ].astype(str).replace({'LMP_PRC': 'LMP', 'LMP_CONG_PRC': 'MCC', 'LMP_LOSS_PRC': 'MCL', 'LMP_ENE_PRC': 'MCE'}),
                          'price': d[vcol].astype('float32')})
        o = o.pivot_table(index=['time_utc', 'node'], columns='comp', values='price', aggfunc='first').reset_index()
        o.columns.name = None
        o.to_parquet(f, index=False); print(kind, f'{m:%Y-%m}', len(o), flush=True)
print('DONE', flush=True)
