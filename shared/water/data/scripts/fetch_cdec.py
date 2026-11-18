"""California CDEC daily reservoir data (storage AF sensor 15, inflow cfs sensor 76, outflow cfs sensor 23), 1990..2025,
for major reservoirs that are also in ResOpsUS (crosswalk by name, below). Extends those records to WY2025 for a
recent-years reservoir exam. Endpoint: cdec.water.ca.gov/dynamicapp/req/CSVDataServlet (documented CSV service).
Out: raw/cdec/<STA>.csv.gz ; proc/reservoirs/cdec/<STA>.parquet (storage_hm3, inflow_cms, release_cms, UTC-naive local day)
"""
import gzip, io, os, time
import numpy as np, pandas as pd, requests
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
XW = {'SHA': 'grand_132', 'ORO': 'grand_148', 'NML': 'grand_198', 'CLE': 'grand_131', 'DNP': 'grand_210',
      'BER': 'grand_185', 'ISB': 'grand_612', 'EXC': 'grand_214', 'PNF': 'grand_600', 'FOL': 'grand_182',
      'BUL': 'grand_158', 'MIL': 'grand_595', 'WRS': 'grand_180', 'CMN': 'grand_191', 'NHG': 'grand_193',
      'WHI': 'grand_133', 'CCH': 'grand_620', 'PAR': 'grand_190'}
AF = 1233.48184 / 1e6   # hm3 per acre-foot
CFS = 0.028316846592


def main():
    os.makedirs(f'{ROOT}/raw/cdec', exist_ok=True); os.makedirs(f'{ROOT}/proc/reservoirs/cdec', exist_ok=True)
    rows = []
    for sta, gid in XW.items():
        p = f'{ROOT}/raw/cdec/{sta}.csv.gz'
        if not os.path.exists(p):
            for i in range(5):
                try:
                    r = requests.get('https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet',
                                     params={'Stations': sta, 'SensorNums': '15,76,23', 'dur_code': 'D',
                                             'Start': '1990-01-01', 'End': '2025-12-31'}, timeout=(30, 600))
                    r.raise_for_status(); break
                except Exception as e:
                    print('retry', sta, e, flush=True); time.sleep(15)
            with gzip.open(p + '.tmp', 'wt') as f:
                f.write(r.text)
            os.replace(p + '.tmp', p)
            time.sleep(1)
        d = pd.read_csv(p, dtype=str)
        d['date'] = pd.to_datetime(d['OBS DATE'].str[:8], format='%Y%m%d', errors='coerce')
        d['v'] = pd.to_numeric(d.VALUE, errors='coerce')
        d.loc[d.v < -9000, 'v'] = np.nan
        w = d.pivot_table(index='date', columns='SENSOR_TYPE', values='v', aggfunc='first')
        out = pd.DataFrame(index=w.index)
        out['storage_hm3'] = w.get('STORAGE') * AF
        out['inflow_cms'] = w.get('INFLOW') * CFS
        out['release_cms'] = w.get('OUTFLOW') * CFS
        out = out.astype('float32'); out.index.name = 'date'
        out.to_parquet(f'{ROOT}/proc/reservoirs/cdec/{sta}.parquet', compression='zstd')
        rows.append({'cdec_id': sta, 'res_id': gid, 'start': str(out.dropna(how='all').index.min().date()),
                     'end': str(out.dropna(how='all').index.max().date()), 'n_all3': int(out.dropna().shape[0])})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{ROOT}/proc/reservoirs/cdec/crosswalk.csv', index=False)


if __name__ == '__main__':
    main()
