"""Check: numpy rollout (exam path) == torch rollout (training path) == task.run_policy (B's loop), and time it.
usage: python3 check_equiv.py [weights.npz]   (random SOP-like init if no weights)"""
import sys, time, pickle, json
import numpy as np, torch
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from policy import torch_policy, rollout_torch, rollout_np, NumpyPolicy, export_weights, task_arrays, FEATS, load_policy
from task import run_policy, cost
torch.set_num_threads(1)
ROOT = _paths.WORK   # the old workspace root (data/, runs/, exam/, web/, tmp/, ratchet.json)
tasks = pickle.load(open(f'{ROOT}/data/val_g1.pkl', 'rb'))[:16]
T = 1000
for t in tasks:
    for k in ('I', 'D'):
        t[k] = t[k][:T]
    t['dates'] = t['dates'][:T]
res = {}
for fset in ('base', 'clim'):
    if len(sys.argv) > 1:
        pol = load_policy(sys.argv[1]); fset = pol.fset; w = pol.w
    else:
        torch.manual_seed(1); net = torch_policy(FEATS[fset]).double()
        with torch.no_grad():
            for m in net:
                if hasattr(m, 'weight'):
                    m.weight.normal_(0, 0.5)
        w = export_weights(net, torch.zeros(FEATS[fset]), torch.ones(FEATS[fset])); pol = NumpyPolicy(w, fset)
    net = torch_policy(FEATS[fset], w['W0'].shape[1], sum(1 for k in w if k.startswith('W')) - 1).double()
    lin = [m for m in net if hasattr(m, 'weight')]
    with torch.no_grad():
        for i, m in enumerate(lin):
            m.weight.copy_(torch.tensor(w[f'W{i}'].T)); m.bias.copy_(torch.tensor(w[f'b{i}']))
    R, S = rollout_np(pol, tasks)
    Jn = np.array([np.mean(cost(r, t['D'], t['Qmax'])) for r, t in zip(R, tasks)])
    Jl = []
    for t in tasks:
        r, s = run_policy(pol, t); Jl.append(np.mean(cost(r, t['D'], t['Qmax'])))
    cal = [task_arrays(t, T) for t in tasks]
    b = {k: torch.tensor(np.stack([c[k] for c in cal])) for k in cal[0]}
    b.update({'I': torch.tensor(np.stack([t['I'] for t in tasks])), 'q': torch.tensor([t['Qmax'] for t in tasks]),
              'cap': torch.tensor([t['cap'] for t in tasks]), 'S0': torch.tensor([t['S0'] for t in tasks]),
              'ema0': torch.tensor([pol.init_ema(t) for t in tasks])})
    t0 = time.time()
    Jt, obj = rollout_torch(net, torch.tensor(w['mu']), torch.tensor(w['sd']), b, fset)
    torch.log(obj + 0.02).mean().backward(); dt = time.time() - t0
    res[fset] = {'max|J_np - J_torch|': float(np.max(np.abs(Jn - Jt.detach().numpy()))),
                 'max|J_np - J_loop|': float(np.max(np.abs(Jn - np.array(Jl)))),
                 'J_median': float(np.median(Jn)), 'torch_fwd_bwd_s_per_1000_steps_b16': dt}
    print(fset, res[fset], flush=True)
    if len(sys.argv) > 1:
        break
json.dump(res, open(f'{ROOT}/checks_equiv.json', 'w'), indent=1)
