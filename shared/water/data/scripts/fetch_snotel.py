"""NRCS SNOTEL daily data for every SNOTEL station (resumable), via the documented NRCS report generator.

  station list : customMultipleStationReport (network SNTL)      -> raw/snotel/stations.csv
  daily        : customSingleStationReport/daily/start_of_period  -> raw/snotel/daily/<id>_<state>_SNTL.csv.gz
Elements: WTEQ (SWE, in, start of day) + QC flag, PREC (water-year accumulated precip, in, start of day) + QC flag,
PRCP (daily precip increment, in), TAVG/TMAX/TMIN (degF, day), SNWD (snow depth, in, start of day).
"""
import gzip, os, time
import pandas as pd, requests
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
RAW = f'{ROOT}/raw/snotel'
RG = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv'
S = requests.Session()
S.headers['User-Agent'] = 'mlratchet-data-rebuild/1.0 (research data rebuild)'   # generic; no personal details
STOP = (401, 403, 429, 451)   # blocked or throttled: stop, do not work around it


def get(url, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, timeout=(30, 300))
            if r.status_code in STOP:
                raise SystemExit(f'HTTP {r.status_code} from NRCS: stopping (blocked or throttled); re-run later')
            r.raise_for_status()
            return r.text
        except Exception as e:
            print('retry', url[-80:], e, flush=True)
            time.sleep(10 * (i + 1))
    raise IOError(url)


def main():
    os.makedirs(f'{RAW}/daily', exist_ok=True)
    sp = f'{RAW}/stations.csv'
    if not os.path.exists(sp):
        t = get(f'{RG}/customMultipleStationReport/daily/start_of_period/network=%22SNTL%22%7Cname/0,0/'
                'stationId,name,state.code,network.code,elevation,latitude,longitude,huc12.huc12Code,county.name,'
                'inServiceDate,outServiceDate')
        open(sp + '.tmp', 'w').write(t); os.replace(sp + '.tmp', sp)
    st = pd.read_csv(sp, comment='#', dtype=str)
    print(len(st), 'stations', flush=True)
    for k, r in enumerate(st.itertuples(index=False)):
        sid, state = r[0], r[2]
        trip = f'{sid}:{state}:SNTL'
        p = f'{RAW}/daily/{sid}_{state}_SNTL.csv.gz'
        if os.path.exists(p):
            continue
        t = get(f'{RG}/customSingleStationReport/daily/start_of_period/{trip}%7Cid=%22%22%7Cname/POR_BEGIN,POR_END/'
                'WTEQ::value,WTEQ::qcFlag,PREC::value,PREC::qcFlag,PRCP::value,TAVG::value,TMAX::value,TMIN::value,SNWD::value')
        with gzip.open(p + '.tmp', 'wt') as f:
            f.write(t)
        os.replace(p + '.tmp', p)
        time.sleep(1.0)   # sequential and polite
        if k % 50 == 0:
            print(k, trip, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
