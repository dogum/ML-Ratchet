"""USGS NWIS for the 671 CAMELS-US gauges (resumable, polite).

  site info  : waterservices.usgs.gov/nwis/site (expanded RDB)       -> raw/nwis/site_expanded.rdb
  daily Q    : waterservices.usgs.gov/nwis/dv  00060 mean, 1980-01-01..today -> raw/nwis/dv/<site>.rdb.gz
  peaks      : nwis.waterdata.usgs.gov/nwis/peak (annual peak file)   -> raw/nwis/peak/<site>.rdb.gz
Each file is written atomically; existing files are skipped on restart.
"""
import gzip, os, sys, time, json
import pandas as pd, requests
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
RAW = f'{ROOT}/raw/nwis'
S = requests.Session()
S.headers['User-Agent'] = 'mlratchet-data-rebuild/1.0 (research data rebuild)'   # generic; no personal details
STOP = (401, 403, 429, 451)   # blocked or throttled: stop, do not work around it


def gauges():
    a = pd.read_csv(f'{ROOT}/raw/caravan/attributes/camels/attributes_other_camels.csv', dtype=str)
    return [g.split('_')[1] for g in a.gauge_id]


def get(url, params, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=(30, 180))
            if r.status_code == 404:
                return None
            if r.status_code in STOP:
                raise SystemExit(f'HTTP {r.status_code} from USGS: stopping (blocked or throttled); re-run later')
            r.raise_for_status()
            return r.text
        except Exception as e:
            print('retry', url, params.get('sites') or params.get('site_no'), e, flush=True)
            time.sleep(10 * (i + 1))
    raise IOError(f'failed {url} {params}')


def save(path, text):
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        f.write(text if text is not None else '#NOT FOUND\n')
    os.replace(tmp, path)


def main():
    os.makedirs(f'{RAW}/dv', exist_ok=True); os.makedirs(f'{RAW}/peak', exist_ok=True)
    gs = gauges()
    print(len(gs), 'gauges', flush=True)
    sp = f'{RAW}/site_expanded.rdb'
    if not os.path.exists(sp):
        parts = []
        for i in range(0, len(gs), 100):
            t = get('https://waterservices.usgs.gov/nwis/site/', {'format': 'rdb', 'sites': ','.join(gs[i:i + 100]),
                                                                  'siteOutput': 'expanded', 'siteStatus': 'all'})
            parts.append(t)
            time.sleep(1)
        open(sp + '.tmp', 'w').write('\n'.join(parts)); os.replace(sp + '.tmp', sp)
    end = time.strftime('%Y-%m-%d')
    for k, g in enumerate(gs):
        p = f'{RAW}/dv/{g}.rdb.gz'
        if not os.path.exists(p):
            t = get('https://waterservices.usgs.gov/nwis/dv/', {'format': 'rdb', 'sites': g, 'parameterCd': '00060',
                                                                'statCd': '00003', 'startDT': '1980-01-01', 'endDT': end,
                                                                'siteStatus': 'all'})
            save(p, t); time.sleep(1.0)   # sequential and polite
        if k % 25 == 0:
            print('dv', k, g, flush=True)
    for k, g in enumerate(gs):
        p = f'{RAW}/peak/{g}.rdb.gz'
        if not os.path.exists(p):
            try:
                t = get('https://nwis.waterdata.usgs.gov/nwis/peak', {'site_no': g, 'agency_cd': 'USGS', 'format': 'rdb'}, tries=4)
                save(p, t)
            except IOError as e:
                print('peak failed (will retry on next run)', g, flush=True)
            time.sleep(1.0)
        if k % 25 == 0:
            print('peak', k, g, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
