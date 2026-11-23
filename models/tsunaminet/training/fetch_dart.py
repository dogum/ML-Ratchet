"""Download NDBC historical DART files (yearly, water-column height; T=1 15-min, T=2 1-min, T=3 15-s) for Pacific
stations 2006-2025 from https://www.ndbc.noaa.gov/data/historical/dart/. Resumable."""
import os, time, urllib.request
from _paths import DATA
OUT = os.path.join(DATA, 'dart'); os.makedirs(OUT, exist_ok=True)
L = [l.strip() for l in open(os.path.join(DATA, 'dart_hist_list.txt'))]
P = [l for l in L if l[:2] in ('21', '32', '43', '46', '51', '52', '54', '55') and 2006 <= int(l[6:10]) <= 2025]
for k, fn in enumerate(P):
    dst = os.path.join(OUT, fn)
    if os.path.exists(dst): continue
    for t in range(4):
        try:
            data = urllib.request.urlopen(urllib.request.Request('https://www.ndbc.noaa.gov/data/historical/dart/' + fn,
                                          headers={'User-Agent': 'tsunaminet-research-script'}), timeout=120).read()
            open(dst + '.part', 'wb').write(data); os.replace(dst + '.part', dst); break
        except Exception as e:
            print('retry', fn, e, flush=True); time.sleep(5 * (t + 1))
    if k % 50 == 0: print(k, len(P), fn, flush=True)
    time.sleep(0.2)
print('done', len(P))
