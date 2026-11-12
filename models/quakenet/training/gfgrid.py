"""Green's functions for QuakeNet's physics: full-wavefield synthetic seismograms of the ak135f Earth model (AxiSEM, periods
down to 2.1 s) from the EarthScope Syngine service, on a grid of source depths and epicentral distances. Ten fundamental
Green's functions per point (Z/R/T responses to the basic moment-tensor parts); any moment tensor and azimuth is built from
them locally. Requested one at a time."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, time, io, json, numpy as np, urllib.request
from obspy import read
D = _P.WORK + '/data/gf/'; os.makedirs(D, exist_ok=True)
URL = ('https://service.earthscope.org/irisws/syngine/1/query?model=ak135f_2s&greensfunction=1&sourcedistanceindegrees={d}'
       '&sourcedepthinmeters={z}&format=miniseed&units=velocity&dt=0.5&starttime=P-60&endtime=P%2B2400')
DEPTHS = [2, 5, 10, 15, 20, 25, 33, 40, 50, 60, 70, 85, 100, 120, 140, 165, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
DISTS = list(range(25, 96))
COMP = ['ZSS', 'ZDS', 'ZDD', 'ZEP', 'RSS', 'RDS', 'RDD', 'REP', 'TSS', 'TDS']
if __name__ == '__main__':
    t0 = time.time(); n = 0
    for z in DEPTHS:
        for d in DISTS:
            out = f'{D}z{z:03d}_d{d:03d}.npy'
            if os.path.exists(out): continue
            for a in range(4):
                try:
                    raw = urllib.request.urlopen(URL.format(d=d, z=z * 1000), timeout=120).read(); st = read(io.BytesIO(raw)); break
                except Exception as e:
                    print('retry', z, d, e, flush=True); time.sleep(10 * (a + 1))
            else:
                continue
            tr = {t.stats.channel: t for t in st}; L = min(len(t.data) for t in st)
            np.save(out, np.stack([tr[c].data[:L] for c in COMP]).astype(np.float32))
            if n == 0: json.dump(dict(t_start_after_origin=float(st[0].stats.starttime - st[0].stats.starttime.__class__('1900-01-01')), dt=0.5), open(D + f'meta_z{z:03d}_d{d:03d}.json', 'w'))
            n += 1
            if n % 50 == 0: print(n, 'done', '%.0fs' % (time.time() - t0), flush=True)
    print('finished', n, flush=True)
