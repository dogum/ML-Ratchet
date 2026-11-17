"""TMYx typical-year weather (climate.onebuilding.org, EPW) for every BDG2 site -> proc/tmy/<site>.parquet + .json.

EPW rows are hour-ENDING local standard time; stored hour-beginning as `time_lst` in a dummy non-leap year 2001
(TMY months come from different years). Columns follow the weather contract.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os, re, io, json, zipfile, time
import numpy as np, pandas as pd, requests

D = f'{BLDG_ROOT}/data'; R = f'{D}/raw/tmy'; O = f'{D}/proc/tmy'
os.makedirs(R, exist_ok=True); os.makedirs(O, exist_ok=True)
OB = 'https://climate.onebuilding.org'
US = 'WMO_Region_4_North_and_Central_America/USA_United_States_of_America'
SITES = {'Bear': (f'{US}/CA_California', r'Oakland\.Intl\.AP\.724930'), 'Swan': (f'{US}/CA_California', r'Moffett'),
         'Bobcat': (f'{US}/UT_Utah', r'Ogden-Hinckley|Ogden'), 'Bull': (f'{US}/TX_Texas', r'Austin-Camp\.Mabry|Austin-Mabry'),
         'Cockatoo': (f'{US}/NY_New_York', r'Ithaca'), 'Eagle': (f'{US}/NJ_New_Jersey', r'Trenton'), 'Peacock': (f'{US}/NJ_New_Jersey', r'Trenton'),
         'Fox': (f'{US}/AZ_Arizona', r'Phoenix-Sky\.Harbor'), 'Gator': (f'{US}/FL_Florida', r'Orlando\.Intl'), 'Panther': (f'{US}/FL_Florida', r'Orlando\.Intl'),
         'Hog': (f'{US}/MN_Minnesota', r'Minneapolis-St\.Paul'), 'Rat': (f'{US}/VA_Virginia', r'Washington-Reagan|Washington\.Reagan|Reagan'),
         'Crow': ('WMO_Region_4_North_and_Central_America/CAN_Canada/ON_Ontario', r'Ottawa'),
         'Moose': ('WMO_Region_4_North_and_Central_America/CAN_Canada/ON_Ontario', r'Ottawa'),
         'Lamb': ('WMO_Region_6_Europe/GBR_United_Kingdom/WAL_Wales', r'Cardiff'),
         'Mouse': ('WMO_Region_6_Europe/GBR_United_Kingdom/ENG_England', r'London\.Heathrow|Heathrow'),
         'Robin': ('WMO_Region_6_Europe/GBR_United_Kingdom/ENG_England', r'London\.Heathrow|Heathrow'),
         'Shrew': ('WMO_Region_6_Europe/GBR_United_Kingdom/ENG_England', r'London\.Heathrow|Heathrow'),
         'Wolf': ('WMO_Region_6_Europe/IRL_Ireland', r'Dublin')}


def listing(path):
    for u in (f'{OB}/{path}/', f'{OB}/{path}/index.html'):
        try:
            r = requests.get(u, timeout=60)
            if r.status_code == 200:
                h = re.findall(r'href="([^"]+\.zip)"', r.text)
                if h: return u.rsplit('/', 1)[0] + '/', h
        except Exception as e:
            print('err', u, e)
    return None, []


def parse_epw(txt):
    lines = txt.splitlines()
    loc = lines[0].split(',')
    rows = [l.split(',') for l in lines[8:] if l.strip()]
    a = pd.DataFrame(rows)
    num = lambda i: pd.to_numeric(a[i], errors='coerce').to_numpy()
    mo, dy, hr = (pd.Series(num(i)).astype(int) for i in (1, 2, 3))
    t = pd.DatetimeIndex(pd.to_datetime(dict(year=2001, month=mo, day=dy)) + pd.to_timedelta(hr - 1, unit='h'))
    w = pd.DataFrame({'T_out': num(6), 'dewpoint': num(7), 'RH': num(8), 'pressure': num(9), 'GHI': num(13), 'DNI': num(14),
                      'DHI': num(15), 'wind_dir': num(20), 'wind_speed': num(21), 'cloud': num(22) / 10.0}, index=t)
    w.index.name = 'time_lst'
    w = w.replace({999.9: np.nan, 9999.0: np.nan, 999999.0: np.nan})
    meta = dict(city=loc[1], region=loc[2], country=loc[3], source_epw=loc[4], wmo=loc[5], lat=float(loc[6]), lon=float(loc[7]),
                tz_offset_h=float(loc[8]), elev=float(loc[9]))
    return w.astype('float32'), meta


def main():
    cache = {}
    for site, (path, pat) in SITES.items():
        if os.path.exists(f'{O}/{site}.json'): continue
        if path not in cache: cache[path] = listing(path); time.sleep(1)
        base, files = cache[path]
        cand = [f for f in files if re.search(pat, f) and 'TMYx.2011-2025' in f] or [f for f in files if re.search(pat, f) and 'TMYx' in f]
        if not cand:
            print('NO TMYx for', site, path, pat); continue
        f = sorted(cand)[0]
        z = f'{R}/{os.path.basename(f)}'
        if not os.path.exists(z):
            r = requests.get(base + f, timeout=120); r.raise_for_status(); open(z, 'wb').write(r.content); time.sleep(1)
        zz = zipfile.ZipFile(z); epw = [n for n in zz.namelist() if n.endswith('.epw')][0]
        w, meta = parse_epw(zz.read(epw).decode('latin-1'))
        meta.update(site_id=site, file=f, url=base + f, period='TMYx 2011-2025' if '2011-2025' in f else 'TMYx',
                    licence='climate.onebuilding.org TMYx: free to use with attribution (Lawrie & Crawley)')
        w.to_parquet(f'{O}/{site}.parquet'); json.dump(meta, open(f'{O}/{site}.json', 'w'), indent=1)
        print(site, f, len(w), round(float(w.T_out.mean()), 1), flush=True)


if __name__ == '__main__':
    main()
