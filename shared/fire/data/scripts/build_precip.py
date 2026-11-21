"""Patch hourly precip into proc/weather/hrrr/<Event_ID>.npz from HRRR forecasts (firedata.hrrr.precip; the analysis
PRATE is absent in hrrrzarr). Usage: python3 -I build_precip.py dev|replay|<Event_ID> ...  Resumable (meta flag)."""
import os, sys, json, time
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
sys.path.insert(0, _CODE)
from firedata import hrrr
P = f'{_DATA}/data/proc'


def patch(r):
    fn = f'{P}/weather/hrrr/{r.Event_ID}.npz'
    if not os.path.exists(fn): return 'no hrrr file'
    z = dict(np.load(fn)); meta = json.loads(str(z['meta']))
    if meta.get('precip_source', '').startswith('HRRR fcst'):
        fm = meta.get('frac_precip_missing', 0)
        if not (REPAIR and 0 < fm < 1): return 'skip'  # 1.0 = no hrrrzarr forecasts (before mid-2018)
    t = z['time'].astype('datetime64[h]')
    o = hrrr.precip(r.wx0, r.wy0, r.wx1, r.wy1, t[0], t[-1])
    assert o['precip'].shape == z['T'].shape, (o['precip'].shape, z['T'].shape)
    z['precip'] = o['precip'].astype(np.float16)
    meta['precip_source'] = 'HRRR fcst APCP_1hr_acc_fcst, 00z/12z cycles, leads f01-f12 (mm in the hour ending at t)'
    meta['units']['precip'] = 'mm per hour ending at t (HRRR 1 h accumulation, short-lead forecast)'
    meta['frac_precip_missing'] = float(np.isnan(o['precip']).all(axis=(1, 2)).mean())
    z['meta'] = json.dumps(meta)
    np.savez_compressed(fn + '.tmp.npz', **z); os.replace(fn + '.tmp.npz', fn)
    return f"ok missing {meta['frac_precip_missing']:.3f} total {np.nansum(o['precip'][:, 0, 0]):.1f} mm"


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
        try: s = patch(f[f.Event_ID == e].iloc[0])
        except Exception as ex: s = f'FAIL {ex!r}'
        print(e, s, f'{time.time() - t:.0f}s', flush=True)
    print('DONE')
