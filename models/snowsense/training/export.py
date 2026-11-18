"""Export a checkpoint to the numpy release (release/snowsense_weights.npz, with the dev widen factors) and the
browser (web/weights.json); verify numpy == torch on real dev windows.
python3 export.py <run>
"""
import json
import sys
import numpy as np
import torch
from common import *
from evaluate import load, predict
from evalset import station_set

run = sys.argv[1]
net = load(run)
sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
k = json.load(open(f'{ROOT}/results/k_{run}.json'))
np.savez(f'{ROOT}/release/snowsense_weights.npz', **sd, k=np.array([k['daily'], k['peak'], k['melt']]))
json.dump({'weights': {n: np.round(v.astype(np.float64), 7).tolist() for n, v in sd.items()},
           'k': {'daily': k['daily'], 'peak': k['peak'], 'melt': k['melt']}, 'run': run},
          open(f'{ROOT}/web/weights.json', 'w'))

sys.path.insert(0, MODEL)   # the snowsense.py module; the weights just written are under WORK/release
import snowsense
m = snowsense.SnowSense(f'{ROOT}/release/snowsense_weights.npz')
sid, lat, elev, ok = station_set('dev')[0]
X, D, Y, wys = station_windows(sid, ok[:3], lat, elev)
Qt = predict(net, X)
Qn = m.forward(X.astype(np.float64))
sw = lambda Q: 100 * np.maximum(Q, 0) ** 2
d = {'station': sid, 'n_windows': len(X), 'max_abs_z': float(np.abs(Qt - Qn).max()),
     'max_abs_swe_mm': float(np.abs(sw(Qt) - sw(Qn)).max())}
json.dump(d, open(f'{ROOT}/results/export_check.json', 'w'), indent=1)
print(d)
