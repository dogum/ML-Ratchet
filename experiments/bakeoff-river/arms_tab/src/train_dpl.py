"""Differentiable parameter learning (Tsai et al. 2021 style): an MLP maps the 18-value static vector to the 17
parameters of WaterSim's HBV + degree-day snow + channel UH (+ a precipitation multiplier); it is trained THROUGH
the torch re-implementation of WaterSim (hbv_torch.py, verified to 7e-14 mm/day) on the sim TRAIN bank.

usage: python3 train_dpl.py RUN_DIR CONFIG_JSON      (resumable: RUN_DIR/ckpt.pt every 20 iterations)
Each iteration: `batch` random train runs (index < 11,400; 11,400+ are held out for monitoring and the error bands),
a random window of 365 warm-up rows (no gradient) + `target_days` rows, loss on qo over the target rows.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/home/claude/bakeoff/river/harness')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bake  # noqa: E402
import hbv_torch as H  # noqa: E402

torch.set_num_threads(1)
run_dir, cfg_path = sys.argv[1], sys.argv[2]
cfg = {'batch': 64, 'target_days': 730, 'iters': 1500, 'lr': 3e-3, 'lr_end': 1e-4, 'hidden': [64, 64],
       'loss': 'nse', 'seed': 0, 'warm': 365, 'max_cpu_h': 1.3}
cfg.update(json.load(open(cfg_path)))
os.makedirs(run_dir, exist_ok=True)
NHOLD = 11400
B = bake.load_bank('train', ['F', 'qo', 'attrs', 'st'])
st_all = np.asarray(B['st'][:NHOLD], np.float64)
mu, sd = st_all.mean(0), st_all.std(0) + 1e-6


class Net(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        dims = [18] + list(hidden)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = torch.nn.Linear(dims[-1], len(H.PARAMS))
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.copy_(torch.tensor(H.from_params(H.DEFAULTS)))

    def forward(self, x):
        for L in self.layers:
            x = torch.nn.functional.silu(L(x))
        return self.out(x)


torch.manual_seed(cfg['seed'])
if cfg.get('model', 'mlp') == 'formula':        # tune the constants of fixed symbolic formulas (formula arm)
    import formula_torch as FT
    net = FT.FormulaNet(json.load(open(cfg['formulas'])), H.NAMES, mu, sd)
else:
    net = Net(cfg['hidden']).double()
opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
ck = os.path.join(run_dir, 'ckpt.pt')
state = {'it': 0, 'cpu_s': 0.0, 'log': []}
if cfg.get('init_from') and not os.path.exists(ck):      # warm start (weights only) from another rung's checkpoint
    net.load_state_dict(torch.load(cfg['init_from'], weights_only=False)['net'])
    print('init from', cfg['init_from'], flush=True)
if os.path.exists(ck):
    s = torch.load(ck, weights_only=False)
    net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); state = s['state']
    print('resumed at', state['it'], flush=True)
rng = np.random.default_rng([cfg['seed'], state['it']])
T = cfg['warm'] + cfg['target_days']


def get_batch(idx, offs):
    F = np.stack([np.asarray(B['F'][i, o:o + T], np.float64) for i, o in zip(idx, offs)])
    y = np.stack([np.asarray(B['qo'][i, o + cfg['warm']:o + T], np.float64) for i, o in zip(idx, offs)])
    at = np.asarray(B['attrs'][idx], np.float64)
    x = (np.asarray(B['st'][idx], np.float64) - mu) / sd
    pm = np.exp(np.asarray(B['st'][idx, 11], np.float64))      # forcing mean P (an input, not the target)
    return [torch.tensor(a) for a in (F, y, at, x, pm)]


def forward(F, at, x):
    p = H.to_params(net(x))
    frac, dz = H.bands(at[:, 2])
    return H.run(torch.clamp(F[..., 0], min=0.0), F[..., 1], F[..., 2], p, frac, dz, at[:, 0], at[:, 9],
                 cfg['warm']), p


def loss_fn(q, y, pm=None):
    if cfg['loss'] == 'nse':
        s = y.std(1) + 0.1
        return (((q - y) ** 2).mean(1) / s ** 2).mean()
    if cfg['loss'] == 'lognse':            # robust: log(1 + NSE*) per run, so a few hopeless runs do not dominate
        s = y.std(1) + 0.1
        return torch.log1p(((q - y) ** 2).mean(1) / s ** 2).mean()
    if cfg['loss'] == 'bnse':              # bounded: x / (1 + x), x = NSE* per run (C2M-like, Mathevet et al. 2006)
        s = y.std(1) + 0.1
        x = ((q - y) ** 2).mean(1) / s ** 2
        return (x / (1.0 + x)).mean()
    if cfg['loss'] == 'mse_pm':            # MSE of flow / p_mean (input scale): no 1/variance weighting, no shrink
        return (((q - y) / pm[:, None]) ** 2).mean()
    if cfg['loss'] == 'mse_norm':          # MSE on flow / mean flow of the window
        m = y.mean(1, keepdim=True) + 0.05
        return (((q - y) / m) ** 2).mean()
    raise ValueError(cfg['loss'])


def save():
    tmp = ck + '.tmp'
    torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'state': state, 'cfg': cfg, 'mu': mu, 'sd': sd}, tmp)
    os.replace(tmp, ck)


# fixed monitoring set: 48 held-out train runs, window at offset 0
mon_idx = np.arange(NHOLD, NHOLD + 48)
mon = get_batch(mon_idx, np.zeros(48, int))
t_last = time.process_time()
while state['it'] < cfg['iters'] and state['cpu_s'] < cfg['max_cpu_h'] * 3600:
    it = state['it']
    lr = cfg['lr_end'] + 0.5 * (cfg['lr'] - cfg['lr_end']) * (1 + math.cos(math.pi * it / cfg['iters']))
    for g in opt.param_groups:
        g['lr'] = lr
    opt.zero_grad()
    ok = True
    for _a in range(cfg.get('accum', 1)):                 # gradient accumulation over `accum` batches
        idx = np.sort(rng.choice(NHOLD, cfg['batch'], replace=False))
        offs = rng.integers(0, 1462 - T - 1 + 1, cfg['batch'])
        F, y, at, x, pm = get_batch(idx, offs)
        q, _ = forward(F, at, x)
        loss = loss_fn(q, y, pm) / cfg.get('accum', 1)
        if torch.isfinite(loss):
            loss.backward()
        else:
            ok = False
    if ok:
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
    state['it'] = it + 1
    now = time.process_time(); state['cpu_s'] += now - t_last; t_last = now
    rec = {'it': it + 1, 'loss': float(loss.detach()), 'lr': lr, 'cpu_s': round(state['cpu_s'], 1)}
    if (it + 1) % 20 == 0 or it + 1 == cfg['iters']:
        with torch.no_grad():
            qm, _ = forward(mon[0], mon[2], mon[3])
            ym = mon[1]
            nse = 1 - ((qm - ym) ** 2).sum(1) / ((ym - ym.mean(1, keepdim=True)) ** 2).sum(1)
            rec['mon_nse_median'] = float(nse.median())
        state['log'].append(rec)
        save()
        print(rec, flush=True)
save()
print('done', state['it'], 'cpu_h', state['cpu_s'] / 3600, flush=True)
