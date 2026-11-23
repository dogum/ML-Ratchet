"""Download ETOPO 2022 (60 arc-sec surface elevation, NCEI THREDDS OPeNDAP) over the Pacific at stride 2 (2'),
then block-average 2x2 to a 4' grid. Domain: lon 119E..292E, lat 65S..65N. Resumable (one file per chunk)."""
import os, sys, struct, time, urllib.request, numpy as np
from _paths import DATA
URL = 'https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc.dods'
OUT = os.path.join(DATA, 'etopo'); os.makedirs(OUT, exist_ok=True)
I0, I1 = 1500, 9300          # lat rows (-65..65)
JW = (17940, 21600)          # 119E .. 180
JE = (0, 6720)               # -180 .. -68 (=292E)
CH = 300

def get(i0, i1, j0, j1):
    q = f'?z[{i0}:2:{i1-1}][{j0}:2:{j1-1}]'
    for k in range(5):
        try:
            raw = urllib.request.urlopen(URL + q, timeout=300).read(); break
        except Exception as e:
            print('retry', e, flush=True); time.sleep(10 * (k + 1))
    p = raw.index(b'Data:\n') + 6
    n = struct.unpack('>i', raw[p:p + 4])[0]
    a = np.frombuffer(raw[p + 8:p + 8 + 4 * n], '>f4').astype(np.float32)
    return a.reshape((i1 - i0) // 2, (j1 - j0) // 2)

rows = []
for i0 in range(I0, I1, CH):
    fn = os.path.join(OUT, f'chunk_{i0}.npy')
    if not os.path.exists(fn):
        a = np.concatenate([get(i0, i0 + CH, *JW), get(i0, i0 + CH, *JE)], 1)
        np.save(fn, a); print('chunk', i0, a.shape, flush=True)
    rows.append(np.load(fn))
z2 = np.concatenate(rows, 0)                         # 2' grid
z4 = z2.reshape(z2.shape[0] // 2, 2, z2.shape[1] // 2, 2).mean((1, 3))
lat = -65 + (4 * np.arange(z4.shape[0]) + 1.5) / 60.0   # centre of the 2x2 block of 2' samples
lon = 119 + (4 * np.arange(z4.shape[1]) + 1.5) / 60.0
np.savez_compressed(os.path.join(DATA, 'etopo_pacific_4m.npz'), z=z4.astype(np.float32), lat=lat, lon=lon)
print('done', z4.shape, lat[[0, -1]], lon[[0, -1]])
