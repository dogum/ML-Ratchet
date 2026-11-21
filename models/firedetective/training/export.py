"""Export a rung to release/weights (numpy npz + calib.json) and verify numpy == torch on dev records.
python3 export.py <rung>"""
from common import *
import shutil, torch
from ratchet import load_net
from train import predict, real_arrays
from real import load_records

sys.path.insert(0, f'{ROOT}/release')
REL = f'{ROOT}/release'
rung = sys.argv[1]
net, ns = load_net(rung)
W = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
if any(k.startswith('members.') for k in W):     # ensemble: members.<i>.<name> -> m<i>.<name>
    W = {'m' + k[len('members.'):]: v for k, v in W.items()}
else:
    W = {'m0.' + k: v for k, v in W.items()}
W.update({k: np.asarray(v, np.float32) for k, v in ns.items()})
os.makedirs(f'{REL}/weights', exist_ok=True)
np.savez(f'{REL}/weights/firedetective_v1.npz', **W)
shutil.copy(f'{ROOT}/models/calib_{rung}.json', f'{REL}/weights/calib.json')
shutil.copy(f'{SRC}/feats.py', f'{REL}/feats.py')   # import: was {ROOT}/src/feats.py
if not os.path.exists(f'{REL}/firedetective.py'):   # import: the sandbox release/ already held it
    shutil.copy(f'{MODEL}/firedetective.py', f'{REL}/firedetective.py')
import importlib, firedetective as fdet
importlib.reload(fdet)
m = fdet.load(REL)
recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
S, Sh, St = real_arrays(recs)
mu_t, sd_t, pb_t = predict(net, ns, S, St)
mu_n, sd_n, pb_n = m.raw(S, St)
d = dict(mu=float(np.abs(mu_t - mu_n).max()), sd=float(np.abs(np.log(sd_t) - np.log(sd_n)).max()),
         p=float(np.abs(pb_t - pb_n).max()), n=len(S), params=int(sum(v.size for k, v in W.items() if '.' in k)), members=len({k.split('.')[0] for k in W if '.' in k}))
jdump(dict(rung=rung, numpy_vs_torch_max_abs=d), f'{REL}/weights/export_check.json')
print(d)
