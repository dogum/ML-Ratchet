"""Write release/weights.npz: ensemble members + dev-fitted sigma factors + metadata."""
from common import *
import json, glob
import numpy as np
model, fac = sys.argv[1], sys.argv[2]
version = sys.argv[3] if len(sys.argv) > 3 else '1.0'
fs = sorted(glob.glob(f'{DATA}/models/{model}_s[0-9].npz'))
out = dict(n_members=np.array(len(fs)), factors=np.array(json.dumps(json.load(open(fac)))),
           meta=np.array(json.dumps(dict(model=model, version=version, features=170, members=len(fs)))))
for i, f in enumerate(fs):
    for k, v in np.load(f).items():
        out[f'm{i}_{k}'] = v
os.makedirs(f'{ROOT}/release', exist_ok=True)
np.savez_compressed(f'{ROOT}/release/weights.npz', **out)
print('saved', len(fs), 'members', sum(v.size for k, v in out.items() if k.startswith('m')), 'numbers')
