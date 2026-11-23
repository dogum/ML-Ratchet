"""Export a trained rung to numpy weights (release/tsunaminet_weights.npz) and verify numpy == torch."""
import os, sys, json, numpy as np, torch
from _paths import ROOT, BANK
import evallib
sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model folder: tsunaminet.py
rung = sys.argv[1]
net = evallib.load_net(rung)
W = {}
convs = [m for m in net.cnn if isinstance(m, torch.nn.Conv1d)]
for i, c in enumerate(convs):
    W[f'c{i}w'] = c.weight.detach().numpy(); W[f'c{i}b'] = c.bias.detach().numpy()
for name, seq in (('st', net.st), ('ctx', net.ctx)):
    lin = [m for m in seq if isinstance(m, torch.nn.Linear)]
    for i, l in enumerate(lin): W[f'{name}{i}w'] = l.weight.detach().numpy(); W[f'{name}{i}b'] = l.bias.detach().numpy()
for i, L in enumerate(net.flow.layers):
    W[f'f{i}m'] = L.mask.numpy(); lin = [m for m in L.net if isinstance(m, torch.nn.Linear)]
    for j, l in enumerate(lin): W[f'f{i}w{j}'] = l.weight.detach().numpy(); W[f'f{i}b{j}'] = l.bias.detach().numpy()
from model import MU, SD
cal = json.load(open(os.path.join(ROOT, 'calibration.json'))) if os.path.exists(os.path.join(ROOT, 'calibration.json')) else {}
meta = dict(rung=rung, extra=int(net.extra), conv=[[c.stride[0], c.padding[0]] for c in convs], n_flow=len(net.flow.layers), mu=MU.tolist(), sd=SD.tolist(), calibration=cal)
np.savez(os.path.join(ROOT, 'release', 'tsunaminet_weights.npz'), meta=json.dumps(meta), **{k: v.astype(np.float32) for k, v in W.items()})
# verify
import importlib, tsunaminet; importlib.reload(tsunaminet)
npn = tsunaminet.TsunamiNet(os.path.join(ROOT, 'release', 'tsunaminet_weights.npz'))
z = np.load(os.path.join(BANK, 'bank_val_p1.npz')); X, S, M = z['X'][:16].astype(np.float32), z['S'][:16], z['M'][:16]
zz = np.random.default_rng(0).standard_normal((64, 4)).astype(np.float32)
err = 0
for i in range(16):
    import features as F
    e = torch.from_numpy(F.nnls_features(X[i], M[i])[None]) if net.extra else None
    a = net.sample(torch.from_numpy(X[i:i + 1]), torch.from_numpy(S[i:i + 1]), torch.from_numpy(M[i:i + 1]), 64, torch.from_numpy(zz), e).numpy()
    b = npn.sample(X[i], S[i], M[i], z=zz, raw=True)
    err = max(err, float(np.abs(a - b).max()))
print('numpy vs torch max abs diff', err)
json.dump(dict(rung=rung, numpy_vs_torch_max_abs=err), open(os.path.join(ROOT, 'release', 'export_check.json'), 'w'))
