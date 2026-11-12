import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); import twin, simbank as SB, calib as CB
sp = json.load(open(twin.D + 'split.json')); pool = SB.load_pool('noise_train'); out = {}
for name, cal in (('v1', None), ('v2', CB.make('v2'))):
    R = []
    for kic in sp['dev']:
        Y, real, tw = twin.twins(kic, pool, calib=cal); br = twin.bands(real[0], Y['numax']); bt = np.mean([twin.bands(t[0], Y['numax']) for t in tw], 0); R.append((br - bt).tolist())
    R = np.array(R); out[name] = dict(per_star=R.tolist(), median_below_hump_above=np.nanmedian(R, 0).tolist()); print(name, np.nanmedian(R, 0).round(3), flush=True)
json.dump(out, open(_P.WORK + '/twin_test.json', 'w'), indent=1)
