"""Train BatteryPilot by direct policy search through the battery physics, on simulated price paths only.

    python3 train.py <rung_name> <bank_glob> [--steps N] [--init other_rung] [--hidden 128] [--window 336]

Each step samples B windows (14 days) from simulated path-years and a battery (1, 2 or 4 h), computes the value
curves for every hour (they depend only on causal features), then runs the offer-curve battery physics hour by hour
in torch and maximises revenue net of the linear degradation cost, plus the value of the energy left at the end
minus the value of the energy at the start (both at the window's mean DA price).  Each window's revenue is divided
by its 2 h perfect-foresight value (precomputed per day in the bank), so every window counts like a capture rate.
Checkpoints every 100 steps (resumable); exports ckpt/<rung>.npz for the numpy model.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from battery import Battery, SOC_KNOTS
from features import build, N_FEAT

torch.set_num_threads(1)
ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
DUR = (1.0, 2.0, 4.0)


def load_bank(pattern):
    paths = []
    files = sorted(sum([glob.glob(p) for p in pattern.split(',')], []))
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        paths.append({k: z[k] for k in z.files})
    return paths


class Net(nn.Module):
    def __init__(self, n_in=N_FEAT, hidden=128, depth=2, K=len(SOC_KNOTS)):
        super().__init__()
        layers = []; d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]; d = hidden
        layers += [nn.Linear(d, K)]
        self.f = nn.Sequential(*layers)
        self.register_buffer('mu', torch.zeros(n_in)); self.register_buffer('sd', torch.ones(n_in))
        with torch.no_grad():
            self.f[-1].bias.copy_(torch.tensor([1.5] + [-0.5] * (K - 1)))

    def forward(self, X):
        h = self.f((X - self.mu) / self.sd)
        z0 = h[..., :1]
        steps = nn.functional.softplus(h[..., 1:]) + 0.01
        return torch.clamp(torch.cat([z0, z0 - torch.cumsum(steps, -1)], -1), -12.0, 12.0)


def s_star_t(v, y):
    a, b = v[:, :-1], v[:, 1:]
    frac = torch.clamp((a - y[:, None]) / torch.clamp(a - b, min=1e-6), 0.0, 1.0)
    return (frac * torch.as_tensor(np.diff(SOC_KNOTS), dtype=v.dtype)).sum(-1)


def rollout(V, rt, cl, E, P, eta_c, eta_d, soc0):
    """Differentiable offer-curve execution.  V (B,W,K) $/MWh, rt/cl (B,W), E/P/eta (B,), soc0 (B,).
    Returns revenue (B,), final soc (B,), throughput (B,)."""
    B, W, _ = V.shape
    soc = soc0
    rev = torch.zeros(B, dtype=V.dtype); thr = torch.zeros(B, dtype=V.dtype)
    dn = P / (eta_d * E); up = P * eta_c / E
    for t in range(W):
        v = V[:, t]; p = rt[:, t]
        s_hi = s_star_t(v, eta_d * (p - cl[:, t]))
        s_lo = s_star_t(v, p / eta_c)
        tgt = torch.minimum(torch.maximum(soc, s_lo), torch.maximum(s_hi, s_lo))
        tgt = torch.minimum(torch.maximum(tgt, soc - dn), soc + up)
        tgt = torch.clamp(tgt, 0.0, 1.0)
        d = tgt - soc
        ch = torch.relu(d) * E / eta_c
        dis = torch.relu(-d) * E * eta_d
        rev = rev + p * (dis - ch) - cl[:, t] * dis
        thr = thr + dis
        soc = tgt
    return rev, soc, thr


class Sampler:
    def __init__(self, paths, W=336, seed=0):
        self.paths = paths; self.W = W
        self.rng = np.random.default_rng(seed)
        self.bats = {d: Battery(1.0, d) for d in DUR}

    def batch(self, B):
        Xs, rts, cls, Es, norms, ps_all, vals = [], [], [], [], [], [], []
        durs = []
        for _ in range(B):
            p = self.paths[self.rng.integers(len(self.paths))]
            n = len(p['rt'])
            t0 = int(self.rng.integers(168, n - self.W))
            d = float(self.rng.choice(DUR)); bat = self.bats[d]
            t = np.arange(t0, t0 + self.W)
            cl = bat.c_lin(p['T'][t])
            X, ps = build(p, t, d, cl, ps=p['ps'][t])
            Xs.append(X); rts.append(p['rt'][t]); cls.append(cl); durs.append(d)
            # normaliser: perfect-foresight value of these days for this battery (precomputed per day, 1/2/4 h)
            dd = p['day'][t]
            pfd = p[f'pf{int(d)}'] if f'pf{int(d)}' in p else p['pf2']
            days = np.unique(dd)
            norm = float(pfd[np.searchsorted(p['day_list'], days)].sum())
            norms.append(max(norm, 0.05 * float(np.mean(ps)) * d * len(days)))
            vals.append(float(np.mean(p['da'][t])))
        f = lambda a: torch.as_tensor(np.array(a), dtype=torch.float32)  # noqa
        return dict(X=f(Xs), rt=f(rts), cl=f(cls), d=f(durs), norm=f(norms), val=f(vals))


def step_loss(net, b, rng, eta_rt=0.85):
    B = b['X'].shape[0]
    z = net(b['X'])
    ps = None
    # value curves in $: the last-but-two feature is log(ps/30)
    ps = 30.0 * torch.exp(b['X'][:, :, -1])
    V = torch.sinh(z) * ps[..., None]
    E = b['d'] * 1.0; P = torch.ones(B)
    eta = torch.full((B,), float(np.sqrt(eta_rt)))
    soc0 = torch.as_tensor(rng.uniform(0, 1, B), dtype=torch.float32)
    rev, socT, thr = rollout(V, b['rt'], b['cl'], E, P, eta, eta, soc0)
    term = (socT - soc0) * E * eta * b['val']
    obj = (rev + term) / b['norm']
    return -obj.mean(), dict(capture=float(obj.mean()), cycles=float((thr / (E * eta)).mean()))


def standardise(net, sampler, n=48):
    b = sampler.batch(n)
    X = b['X'].reshape(-1, b['X'].shape[-1])
    net.mu.copy_(X.mean(0)); net.sd.copy_(X.std(0).clamp(min=0.05))


def export(net, fn):
    w = {'x_mu': net.mu.numpy(), 'x_sd': net.sd.numpy()}
    lin = [m for m in net.f if isinstance(m, nn.Linear)]
    for i, m in enumerate(lin):
        w[f'W{i}'] = m.weight.detach().numpy().astype(np.float64)
        w[f'b{i}'] = m.bias.detach().numpy().astype(np.float64)
    np.savez(fn, **w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rung'); ap.add_argument('bank')
    ap.add_argument('--steps', type=int, default=1500); ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--window', type=int, default=336); ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--init', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(f'{ROOT}/ckpt', exist_ok=True)
    paths = load_bank(a.bank)
    print('paths', len(paths), flush=True)
    torch.manual_seed(a.seed)
    net = Net(hidden=a.hidden)
    sampler = Sampler(paths, W=a.window, seed=a.seed)
    rng = np.random.default_rng(a.seed + 1)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    ck = f'{ROOT}/ckpt/{a.rung}.pt'
    step0 = 0; hist = []
    if os.path.exists(ck):
        s = torch.load(ck, weights_only=False)
        net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        step0 = s['step']; hist = s['hist']; sampler.rng = s['rng']; rng = s['rng2']
        print('resumed at', step0, flush=True)
    else:
        if a.init:
            s = torch.load(f'{ROOT}/ckpt/{a.init}.pt', weights_only=False)
            net.load_state_dict(s['net'])
        else:
            standardise(net, sampler)
    t0 = time.time(); run = []
    for step in range(step0, a.steps):
        b = sampler.batch(a.batch)
        loss, info = step_loss(net, b, rng)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(info['capture'])
        if (step + 1) % 25 == 0:
            hist.append((step + 1, float(np.mean(run)))); print(step + 1, 'capture(window) %.3f' % np.mean(run),
                                                                 'cycles/window %.1f' % info['cycles'],
                                                                 '%.1fs' % (time.time() - t0), flush=True)
            run = []
        if (step + 1) % 100 == 0 or step + 1 == a.steps:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1,
                            hist=hist, rng=sampler.rng, rng2=rng), ck + '.tmp')
            os.replace(ck + '.tmp', ck)
            export(net, f'{ROOT}/ckpt/{a.rung}.npz')
    json.dump(dict(rung=a.rung, bank=a.bank, steps=a.steps, hist=hist, args=vars(a)),
              open(f'{ROOT}/ckpt/{a.rung}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
