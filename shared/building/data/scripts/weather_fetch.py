"""Weather downloads for real sites (resumable).

  screen : identify BDG2 sites with no published coordinates by matching their (ISD-derived) air temperature against
           candidate ASOS airports in the site's time zone (2016 only, tmpf). Output tmp/site_screen.csv
  asos   : for every BDG2 site, download 2016-2017 ASOS (IEM) for the 6 candidate stations nearest the site point and keep
           all; the best match is chosen in build_bdg2.py
  power  : NASA POWER hourly (UTC) 2016-01-01..2017-12-31 at every BDG2 site point
Usage: python3 -I weather_fetch.py screen|asos|power
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import sys, os, io, json, time
import numpy as np, pandas as pd, requests

D = f'{BLDG_ROOT}/data'
RA = f'{D}/raw/asos'; RP = f'{D}/raw/nasa_power'
os.makedirs(RA, exist_ok=True); os.makedirs(RP, exist_ok=True)
IEM = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'

CAND = {
    'US/Eastern': 'MCO TPA MIA JAX ATL CLT RDU DCA IAD BWI PHL EWR JFK LGA BOS PIT CLE CMH DTW IND CVG BDL PVD ALB SYR BUF '
                  'RIC ORF CHS SAV TLH GSO SDF LEX BTV PWM MHT ITH ROC HPN ACY TTN ILG ABE AVP MDT BGM ORH FLL PBI RSW '
                  'GNV DAB SFB ORL ISM MLB VRB SRQ PIE TIX LAL OCF'.split(),
    'US/Mountain': 'SLC DEN ABQ BOI COS TUS ELP BIL CYS PUB GJT SAF BZN MSO OGD PVU FCA HLN BJC APA FNL LMO RAP IDA PIH'.split(),
    'US/Pacific': 'SFO OAK SJC SMF LAX BUR SAN SEA PDX LAS RNO FAT SBA EUG SNA LGB ONT PSP BFL SCK MRY SLE GEG OLM BLI '
                  'ACV RDD SBP OXR HWD PAO NUQ CCR LVK SQL SAC MHR MCC VNY SMO TOA CRQ MYF BFI PAE RNT TIW'.split(),
}


def asos(st, y1, y2, extra=False):
    data = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'mslp', 'alti', 'skyc1', 'skyc2', 'skyc3', 'p01i'] if extra else ['tmpf']
    q = '&'.join(f'data={d}' for d in data)
    u = (f'{IEM}?station={st}&{q}&year1={y1}&month1=1&day1=1&year2={y2 + 1}&month2=1&day2=2&tz=Etc/UTC'
         '&format=onlycomma&latlon=yes&elev=yes&missing=empty&trace=0.0001&report_type=3&report_type=4')
    for a in range(5):
        try:
            r = requests.get(u, timeout=300)
            if r.status_code == 200 and r.text.startswith('station'):
                return r.text
            print('status', st, r.status_code, flush=True)
        except Exception as e:
            print('err', st, e, flush=True)
        time.sleep(10 * (a + 1))
    return None


def hourly_T(txt):
    a = pd.read_csv(io.StringIO(txt), parse_dates=['valid'])
    if a.empty: return None, None
    a['T'] = (a.tmpf - 32) / 1.8
    a['t'] = (a.valid + pd.Timedelta(minutes=10)).dt.floor('h')     # :51-:59 METARs belong to the next hour stamp
    a = a.sort_values('valid').groupby('t').T.last()
    a.index = a.index.tz_localize('UTC')
    return a, None


STD_OFF = {'US/Eastern': -5, 'US/Central': -6, 'US/Mountain': -7, 'US/Pacific': -8, 'Europe/London': 0, 'Europe/Dublin': 0}


def bdg2_weather_utc(site, tz):
    """BDG2 weather stamps are local STANDARD time (checked vs ASOS, DATA.md): UTC = stamp - standard offset."""
    w = pd.read_csv(f'{D}/raw/bdg2/weather.csv', parse_dates=['timestamp'])
    g = w[w.site_id == site].set_index('timestamp').airTemperature
    g.index = (g.index - pd.Timedelta(hours=STD_OFF[tz])).tz_localize('UTC')
    return g


def screen():
    m = pd.read_csv(f'{D}/raw/bdg2/metadata.csv')
    miss = m[m.lat.isna()].groupby('site_id').timezone.first()
    rows = []
    out = f'{D}/tmp/site_screen.csv'
    done = set(pd.read_csv(out).apply(lambda r: (r.site, r.station), axis=1)) if os.path.exists(out) else set()
    for site, tz in miss.items():
        g = bdg2_weather_utc(site, tz)
        g = g[g.index.year == 2016]
        for st in CAND[tz]:
            if (site, st) in done: continue
            f = f'{RA}/screen_{st}_2016.csv'
            if not os.path.exists(f):
                txt = asos(st, 2016, 2016)
                if txt is None: continue
                open(f, 'w').write(txt); time.sleep(0.5)
            a, _ = hourly_T(open(f).read())
            if a is None: continue
            j = pd.concat([g, a], axis=1, join='inner').dropna()
            if len(j) < 1000: continue
            d = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
            rows.append(dict(site=site, station=st, n=len(j), mae=float(d.mean()), exact=float((d < 0.06).mean())))
            pd.DataFrame(rows).to_csv(out + '.part', index=False)
            print(site, st, rows[-1]['mae'], rows[-1]['exact'], flush=True)
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)]).to_csv(out, index=False)


def site_points():
    """Site points: median building coordinate; Wolf longitude sign fixed; unlocated sites from screen results."""
    m = pd.read_csv(f'{D}/raw/bdg2/metadata.csv')
    g = m.groupby('site_id').agg(lat=('lat', 'median'), lon=('lng', 'median'), tz=('timezone', 'first'))
    return g


def asos_all():
    pts = json.load(open(f'{D}/proc/bdg2/site_points.json'))
    for site, p in pts.items():
        for st in p['asos_candidates']:
            f = f'{RA}/{st}_2016_2017.csv'
            if os.path.exists(f): continue
            txt = asos(st, 2016, 2017, extra=True)
            if txt is None: continue
            open(f + '.tmp', 'w').write(txt); os.replace(f + '.tmp', f)
            print('asos', site, st, len(txt), flush=True); time.sleep(1)


def power():
    pts = json.load(open(f'{D}/proc/bdg2/site_points.json'))
    for site, p in pts.items():
        f = f'{RP}/{site}.parquet'
        if os.path.exists(f): continue
        url = ('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW,RH2M,QV2M,WS10M,WD10M,PS,'
               'ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,CLRSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN,CLOUD_AMT'
               f'&community=RE&longitude={p["lon"]:.4f}&latitude={p["lat"]:.4f}&start=20151231&end=20180101'
               '&format=CSV&time-standard=UTC')
        r = None
        for a in range(4):
            try:
                r = requests.get(url, timeout=600)
                if r.status_code == 200 and 'YEAR,MO' in r.text: break
                print('status', r.status_code, r.text[:300], flush=True)
            except Exception as e:
                print('err', e, flush=True)
            time.sleep(30 * (a + 1))
        txt = r.text; body = txt[txt.index('YEAR,MO'):]
        x = pd.read_csv(io.StringIO(body)).replace(-999.0, np.nan)
        x['time_utc'] = pd.to_datetime(dict(year=x.YEAR, month=x.MO, day=x.DY, hour=x.HR), utc=True)
        x = x.drop(columns=['YEAR', 'MO', 'DY', 'HR'])
        x = x.astype({c: 'float32' for c in x.columns if c != 'time_utc'})
        x.to_parquet(f, index=False)
        open(f'{RP}/{site}_header.txt', 'w').write(txt[:txt.index('YEAR,MO')])
        print('power', site, len(x), flush=True); time.sleep(2)


if __name__ == '__main__':
    {'screen': screen, 'asos': asos_all, 'power': power}[sys.argv[1]]()
