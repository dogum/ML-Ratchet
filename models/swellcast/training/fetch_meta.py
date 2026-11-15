"""Download the small catalogue files the other scripts read (added 15 Nov 2026 to rebuild work/ from scratch):
  data/ndbc/station_table.txt                  NDBC station table (positions)
  data/ndbc/list_{swden,swdir,swr1}.txt        which historical spectral files exist (one '<id><w|d|j><year>' per line)
  data/ibtracs/ibtracs.{EP,NA,WP,SP}.csv       IBTrACS v04r01 per-basin CSV (NOAA NCEI)
  data/geo/land-{50m,110m}.json                Natural Earth land via world-atlas 2.0.2 (topojson)
All public documented endpoints; everything goes to work/ (see _paths.py)."""
import os, re, subprocess
import _paths as _P
D = os.path.join(_P.WORK, 'data')
NDBC = 'https://www.ndbc.noaa.gov/data'
IBT = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.{}.list.v04r01.csv'
ATLAS = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/{}'

def curl(url, fn):
    if os.path.exists(fn): return 'have'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    r = subprocess.run(['curl', '-sS', '-f', '-L', '-m', '600', '-o', fn + '.part', url], capture_output=True, text=True)
    if r.returncode != 0: return 'FAIL ' + r.stderr.strip()
    os.replace(fn + '.part', fn); return 'ok'

if __name__ == '__main__':
    print('stations', curl(f'{NDBC}/stations/station_table.txt', os.path.join(D, 'ndbc/station_table.txt')))
    for kind in ('swden', 'swdir', 'swr1'):
        fn = os.path.join(D, f'ndbc/list_{kind}.txt')
        if not os.path.exists(fn):
            html = os.path.join(D, f'ndbc/{kind}.html'); print(kind, curl(f'{NDBC}/historical/{kind}/', html))
            names = sorted(set(re.findall(r'href="([0-9a-z]+[wdj]\d{4})\.txt\.gz"', open(html, errors='ignore').read())))
            open(fn, 'w').write('\n'.join(names) + '\n'); os.remove(html); print(kind, len(names), 'files listed')
    for b in ('EP', 'NA', 'WP', 'SP'): print(b, curl(IBT.format(b), os.path.join(D, f'ibtracs/ibtracs.{b}.csv')))
    for f in ('land-50m.json', 'land-110m.json'): print(f, curl(ATLAS.format(f), os.path.join(D, 'geo', f)))
