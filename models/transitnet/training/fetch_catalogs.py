"""Download the two catalogues the other scripts read (added 15 Nov 2026 to rebuild work/ from scratch; the original
copies lived in the build sandbox and were not kept):
  data/pscomppars.csv   NASA Exoplanet Archive, Planetary Systems Composite Parameters (TAP, every column)
  data/toi.csv          ExoFOP TESS Objects of Interest (CSV export)
Both catalogues change week to week, so a rebuild reads a newer snapshot than v1.0 did. The file dates are recorded in
data/catalog_snapshot.json."""
import _paths as _P  # portable paths (see _paths.py)
import os, json, time, subprocess
D = _P.WORK + '/data/'
SRC = {'pscomppars.csv': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv',
       'toi.csv': 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'}

if __name__ == '__main__':
    os.makedirs(D, exist_ok=True); snap = {}
    for fn, url in SRC.items():
        out = D + fn
        if not os.path.exists(out):
            r = subprocess.run(['curl', '-sS', '-f', '-L', '-m', '900', '-o', out + '.part', url], capture_output=True, text=True)
            if r.returncode != 0: print(fn, 'FAIL', r.stderr.strip()); continue
            os.replace(out + '.part', out)
        n = sum(1 for _ in open(out, encoding='utf-8', errors='ignore')) - 1; snap[fn] = dict(url=url, rows=n, fetched=time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(out))))
        print(fn, n, 'rows')
    json.dump(snap, open(D + 'catalog_snapshot.json', 'w'), indent=1)
