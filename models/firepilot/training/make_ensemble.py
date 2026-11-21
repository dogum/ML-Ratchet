"""Stack trained single nets (same feature standardisation) into one ensemble file.
  python3 make_ensemble.py <out_rung> <member1> <member2> ..."""
import sys, os, json
import numpy as np
import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
ROOT = _paths.WORK + '/models'
out, mem = sys.argv[1], sys.argv[2:]
Z = [np.load(os.path.join(ROOT, m + '.npz')) for m in mem]
for z in Z[1:]:
    assert np.allclose(z['mu'], Z[0]['mu']) and np.allclose(z['sd'], Z[0]['sd'])
d = dict(mu=Z[0]['mu'], sd=Z[0]['sd'], sig_scale=1.0, n_members=len(Z))
for k, z in enumerate(Z):
    for i in range(3):
        d[f'W{i}_{k}'] = z[f'W{i}']
        d[f'b{i}_{k}'] = z[f'b{i}']
np.savez(os.path.join(ROOT, out + '.npz'), **d)
json.dump(dict(rung=out, members=mem), open(os.path.join(ROOT, out + '.json'), 'w'))
print('ok', out, len(Z))
