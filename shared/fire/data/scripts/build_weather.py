"""Hourly HRRR (hrrrzarr) per fire window -> proc/weather/hrrr/<Event_ID>.npz.
Usage: python3 -I build_weather.py dev|replay|<Event_ID> ...   (resumable)
Period: first FEDS step - 2 d .. min(last FEDS step, first + 40 d) + 1 d (UTC hours). Fires before 2016-08-25 have no
HRRR-Zarr: skipped. Grid = HRRR 3 km cells covering the window (+1 cell)."""
import os, sys, json, time
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
sys.path.insert(0, _CODE)
from firedata import hrrr
P = f'{_DATA}/data/proc'
OUT = f'{P}/weather/hrrr'; os.makedirs(OUT, exist_ok=True)
KEYS = ['T', 'RH', 'wind_speed', 'wind_dir', 'gust', 'precip', 'srad', 'cloud']


def build(r):
    fn = f'{OUT}/{r.Event_ID}.npz'
    if os.path.exists(fn):
        if not REPAIR: return 'skip'
        m = json.loads(str(np.load(fn)['meta']))
        if m['frac_hours_missing'] <= 0.0: return 'skip'
        os.remove(fn)  # rebuild once (proxy drops leave NaN hours)
    t0 = pd.Timestamp(r.feds_t0) - pd.Timedelta(days=2)
    t1 = min(pd.Timestamp(r.feds_t1), pd.Timestamp(r.feds_t0) + pd.Timedelta(days=40)) + pd.Timedelta(days=1)
    if t1 < pd.Timestamp('2016-08-25'): return 'no HRRR-Zarr before 2016-08-25'
    o = hrrr.extract(r.wx0, r.wy0, r.wx1, r.wy1, np.datetime64(t0, 'h'), np.datetime64(t1, 'h'))
    lat, lon = o['lat'], o['lon']
    from pyproj import Transformer
    clon, clat = Transformer.from_crs(5070, 4326, always_xy=True).transform(r.cx, r.cy)
    d = (lat - clat) ** 2 + ((lon - clon) * np.cos(np.deg2rad(clat))) ** 2
    iy, ix = np.unravel_index(np.argmin(d), d.shape)
    miss = float(np.isnan(o['T']).all(axis=(1, 2)).mean())
    meta = dict(fire_id=r.Event_ID, source='HRRR analyses (hrrrzarr, sfc *_anl.zarr), hourly UTC', vars=KEYS,
                units=dict(T='degC 2 m', RH='% 2 m', wind_speed='m/s 10 m (earth-relative)', wind_dir='deg from (true N)',
                           gust='m/s surface', precip='mm/h (PRATE x 3600, instantaneous rate)', srad='W/m2 DSWRF',
                           cloud='% total cloud'),
                wind_20ft_note='20-ft (6.1 m) wind ~ 10 m wind / 1.15 (Lawson & Armitage 2008; BehavePlus convention)',
                centre_index=[int(iy), int(ix)], hrrr_rows=o['rows'], hrrr_cols=o['cols'], frac_hours_missing=miss,
                period_utc=[str(o['time'][0]), str(o['time'][-1])])
    np.savez_compressed(fn, meta=json.dumps(meta), time=o['time'].astype('datetime64[h]').astype(np.int64),
                        lat=lat, lon=lon, **{k: o[k].astype(np.float16) for k in KEYS})
    return f'ok {len(o["time"])} h, {lat.shape} cells, missing {miss:.3f}'


REPAIR = '--repair' in sys.argv

if __name__ == '__main__':
    sys.argv = [a for a in sys.argv if a != '--repair']
    f = pd.read_parquet(f'{P}/fires.parquet')
    ids = []
    for a in sys.argv[1:]:
        if a == 'dev': ids += list(f.Event_ID[f.split == 'dev'])
        elif a == 'replay': ids += json.load(open(f'{P}/baselines/replay_subset.json'))['fires']
        else: ids.append(a)
    for e in ids:
        t = time.time()
        try: s = build(f[f.Event_ID == e].iloc[0])
        except Exception as ex: s = f'FAIL {ex!r}'
        print(e, s, f'{time.time() - t:.0f}s', flush=True)
    print('DONE')
