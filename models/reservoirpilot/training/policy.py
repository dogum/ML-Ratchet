"""ReservoirPilot policy: features (causal only), the small MLP, and rollouts in numpy (exam) and torch (training).

Inputs on day t (all known before today's release is chosen):
  storage S_t, capacity, today's inflow I_t, exponential means of past inflow (5, 20, 60 days), today's demand D_t
  and the demand calendar for the next 30 / 90 days (B's D(doy) is a fixed day-of-year curve), Qmax.
  Feature set 'clim' adds the calibration-half inflow climatology Ic(doy) for the next 30 / 90 days (the same kind
  of calibration-half statistic B's rule curve uses for its storage target).
Never used: future inflow, observed future storage/release.
Output: two sigmoids h, e:  R_req = h D + 1.25 e max(Qmax - D, 0)  (h < 1 = hedging, e > 0 = release above demand
to make flood space). B's mass balance then caps the release by the water stored and spills above capacity.
"""
import numpy as np

K = 0.0864
EMA_TAUS = (5.0, 20.0, 60.0)
FEATS = {'base': 11, 'clim': 15}


def fwd_mean(curve, doy, w):
    """Mean of a 366-day curve over days doy..doy+w-1 (wrapping), for each doy in the array."""
    c = np.concatenate([curve, curve[:w]])
    cs = np.concatenate([[0.0], np.cumsum(c)])
    i = np.asarray(doy) - 1
    return (cs[i + w] - cs[i]) / w


def task_arrays(tk, n=None):
    """Per-day calendar arrays for a task (numpy, float64)."""
    doy = np.asarray(tk['dates'].dayofyear) if 'dates' in tk else None
    if n is not None:
        doy = doy[:n]
    Dc, Ic = np.asarray(tk['Dcurve'], float), np.asarray(tk['Icurve'], float)
    return {'D': Dc[doy - 1], 'Dn30': fwd_mean(Dc, doy, 30), 'Dn90': fwd_mean(Dc, doy, 90),
            'Ic30': fwd_mean(Ic, doy, 30), 'Ic90': fwd_mean(Ic, doy, 90)}


def features(xp, S, I, emas, D, Dn30, Dn90, Ic30, Ic90, q, cap, fset):
    """xp = numpy or torch namespace (needs asinh, log, stack). All arguments broadcast (batch,)."""
    qK = q * K
    f = [S / cap, xp.asinh(I / q)] + [xp.asinh(e / q) for e in emas] + [
        D / q, Dn30 / q, xp.log(cap / qK) / 3.0, xp.asinh((cap - S) / qK) / 3.0,
        xp.asinh((S + I * K - D * K - cap) / qK), xp.asinh(S / (Dn30 * K + 1e-3 * qK) / 10.0)]
    if fset == 'clim':
        f += [xp.asinh(Ic30 / q), xp.asinh(Ic90 / q),
              xp.asinh((S + 90 * K * (Ic90 - Dn90) - cap) / qK) / 3.0,
              xp.asinh((S + 30 * K * (Ic30 - Dn30)) / qK) / 3.0]
    return f


RULE0 = (0.75, 3.0, 10.0)      # TunedSOP optimum on simulated g1 tasks (classical.py): flood pool f, tau, H


def head(xp, o, S, I, D, q, cap, kind, base=RULE0):
    """Map network outputs to a requested release.
    'hd'  : R = h D + 1.25 e max(Qmax - D, 0), h, e sigmoids (free form).
    'rule': the network sets, every day, the three parameters of the classical flood-pool + hedging rule:
            flood-pool level f, drawdown time tau, hedging horizon H (initialised at the tuned constants)."""
    if xp is np:
        sig = lambda z: 1 / (1 + np.exp(-z)); clamp = np.clip; mn, mx, wh, ex = np.minimum, np.maximum, np.where, np.exp
    else:
        sig = xp.sigmoid; clamp = xp.clamp; mn, mx, wh, ex = xp.minimum, xp.maximum, xp.where, xp.exp
    if kind == 'hd':
        h = sig(o[..., 0]); e = sig(o[..., 1])
        return h * D + 1.25 * e * mx(q - D, 0.0 * q)
    f0, t0, H0 = base
    cf = float(np.log((f0 - 0.3) / 0.75 / (1 - (f0 - 0.3) / 0.75)))
    f = 0.3 + 0.75 * sig(o[..., 0] + cf)
    tau = t0 * ex(clamp(o[..., 1], -3.0, 3.0)); H = H0 * ex(clamp(o[..., 2], -3.0, 3.0))
    rh = D * mn(1.0 + 0.0 * D, (S + I * K) / mx(H * D * K, 1e-12 + 0.0 * D))
    extra = mn(q, D + (S - f * cap) / (tau * K))
    return mx(rh, wh(D < q, extra, rh))


class NumpyPolicy:
    """Pure-numpy MLP policy (tanh hidden layers). weights: dict W0,b0,W1,b1,...; fset; mu/sd feature scaling."""

    def __init__(self, w, fset='base'):
        self.kind = str(w.get('head', 'hd'))
        self.base = tuple(np.asarray(w['base'], float)) if 'base' in w else RULE0
        self.w = {k: np.asarray(v, np.float64) for k, v in w.items() if k not in ('fset', 'head', 'base')}
        self.fset = fset
        self.nl = sum(1 for k in self.w if k.startswith('W'))

    def mlp(self, x):
        x = (x - self.w['mu']) / self.w['sd']
        for i in range(self.nl):
            x = x @ self.w[f'W{i}'] + self.w[f'b{i}']
            if i < self.nl - 1:
                x = np.tanh(x)
        return x

    def request(self, S, I, emas, cal, t, q, cap):
        q = np.maximum(q, 1e-9)           # guard: a record whose Qmax is 0 would make every feature infinite
        f = features(np, S, I, emas, cal['D'][t], cal['Dn30'][t], cal['Dn90'][t], cal['Ic30'][t], cal['Ic90'][t],
                     q, cap, self.fset)
        x = np.stack(f, -1)
        o = self.mlp(x)
        return head(np, o, S, I, cal['D'][t], q, cap, self.kind, self.base)

    # single-task interface for task.run_policy
    def reset(self, tk):
        self.tk = tk; self.cal = task_arrays(tk, len(tk['I']))
        self.emas = [float(self.init_ema(tk))] * len(EMA_TAUS)

    def init_ema(self, tk):
        if self.fset == 'clim':
            return task_arrays(tk, 1)['Ic30'][0]
        return tk['I'][0]

    def act(self, t, S):
        I = self.tk['I'][t]
        self.emas = [e + (I - e) / tau for e, tau in zip(self.emas, EMA_TAUS)]
        return float(self.request(S, I, self.emas, self.cal, t, self.tk['Qmax'], self.tk['cap']))


def rollout_np(pol, tasks):
    """Batch rollout of a NumpyPolicy over many tasks (variable length, padded) with B's mass balance.
    Returns lists of release and storage arrays."""
    B = len(tasks); n = np.array([len(t['I']) for t in tasks]); T = n.max()
    I = np.zeros((B, T)); cal = {k: np.ones((B, T)) for k in ('D', 'Dn30', 'Dn90', 'Ic30', 'Ic90')}
    for i, tk in enumerate(tasks):
        I[i, :n[i]] = tk['I']; ca = task_arrays(tk, n[i])
        for k in cal:
            cal[k][i, :n[i]] = ca[k]
    calT = {k: v.T for k, v in cal.items()}
    q = np.array([t['Qmax'] for t in tasks]); cap = np.array([t['cap'] for t in tasks])
    S = np.array([t['S0'] for t in tasks], float)
    e0 = np.array([pol.init_ema(t) for t in tasks], float); emas = [e0.copy() for _ in EMA_TAUS]
    R = np.zeros((B, T)); SS = np.zeros((B, T))
    for t in range(T):
        It = I[:, t]
        emas = [e + (It - e) / tau for e, tau in zip(emas, EMA_TAUS)]
        r = pol.request(S, It, emas, calT, t, q, cap)
        avail = S + It * K
        r = np.minimum(np.maximum(r, 0.0), np.maximum(avail, 0) / K)
        s1 = avail - r * K
        sp = np.maximum(s1 - cap, 0.0); r = r + sp / K; s1 = s1 - sp
        S = np.maximum(s1, 0.0); R[:, t] = r; SS[:, t] = S
    return [R[i, :n[i]] for i in range(B)], [SS[i, :n[i]] for i in range(B)]


# ------------------------------------------------------------------------------------------------ torch side
def torch_policy(nin, hidden=64, nlayers=2, kind='hd'):
    import torch
    layers = []; d = nin
    for _ in range(nlayers):
        layers += [torch.nn.Linear(d, hidden), torch.nn.Tanh()]; d = hidden
    last = torch.nn.Linear(d, 2 if kind == 'hd' else 3)
    with torch.no_grad():
        last.weight.mul_(0.01)
        last.bias.copy_(torch.tensor([4.5, -5.5]) if kind == 'hd' else torch.zeros(3))   # start at SOP / tuned rule
    return torch.nn.Sequential(*layers, last)


def rollout_torch(net, mu, sd, batch, fset, terminal=0.0, tail=0, kind='hd', base=RULE0):
    """Differentiable rollout. batch: dict of torch tensors I, D, Dn30, Dn90, Ic30, Ic90 (B, T); q, cap, S0, ema0 (B,).
    The policy acts for the first T - tail days; for the last `tail` days SOP takes over (still differentiable), so
    the storage the policy leaves behind is valued by what it does for the following year, not by a hard rule.
    Returns per-task J over the policy days and the objective (mean daily cost over all T days + optional terminal)."""
    import torch
    I = batch['I']; Bn, T = I.shape; q, cap = batch['q'], batch['cap']
    S = batch['S0']; emas = [batch['ema0'] for _ in EMA_TAUS]
    S_start = S; Tp = T - tail
    tot = torch.zeros(Bn, dtype=I.dtype); totp = None
    for t in range(T):
        It = I[:, t]; D = batch['D'][:, t]
        if t < Tp:
            emas = [e + (It - e) / tau for e, tau in zip(emas, EMA_TAUS)]
            f = features(torch, S, It, emas, D, batch['Dn30'][:, t], batch['Dn90'][:, t], batch['Ic30'][:, t],
                         batch['Ic90'][:, t], q, cap, fset)
            o = net((torch.stack(f, -1) - mu) / sd)
            r = head(torch, o, S, It, D, q, cap, kind, base)
        else:
            if totp is None:
                totp = tot
            r = D
        avail = S + It * K
        r = torch.minimum(r, torch.relu(avail) / K)
        s1 = avail - r * K
        sp = torch.relu(s1 - cap); r = r + sp / K; s1 = s1 - sp
        S = torch.relu(s1)
        tot = tot + (torch.relu(D - r) / D.clamp_min(1e-6)) ** 2 + 10 * (torch.relu(r - q) / q) ** 2
    J = (tot if totp is None else totp) / Tp
    obj = (tot + terminal * torch.relu(S_start - S) / cap) / T
    return J, obj


def export_weights(net, mu, sd):
    w = {'mu': mu.detach().numpy().astype(np.float64), 'sd': sd.detach().numpy().astype(np.float64)}
    lin = [m for m in net if hasattr(m, 'weight')]
    for i, m in enumerate(lin):
        w[f'W{i}'] = m.weight.detach().numpy().T.astype(np.float64); w[f'b{i}'] = m.bias.detach().numpy().astype(np.float64)
    return w


class TunedSOP:
    """Classical comparator: SOP with a flood pool and linear hedging (3 parameters, tuned on simulated tasks).
    R = D * min(1, (S + I K) / (H D K))                      hedge when less than H days of demand are available
    R = max(R, min(Qmax, D + (S - f cap) / (tau K)))         draw down toward f * capacity, never above Qmax"""

    fset = 'tuned_sop'

    def __init__(self, f=0.85, tau=10.0, H=30.0):
        self.f, self.tau, self.H = f, tau, H

    def init_ema(self, tk):
        return 0.0

    def request(self, S, I, emas, cal, t, q, cap):
        D = cal['D'][t]
        r = D * np.minimum(1.0, (S + I * K) / np.maximum(self.H * D * K, 1e-12))
        extra = np.minimum(q, D + (S - self.f * cap) / (self.tau * K))
        return np.maximum(r, np.where(D < q, extra, r))


def load_policy(path):
    z = np.load(path)
    if str(z['fset']) == 'tuned_sop':
        return TunedSOP(float(z['f']), float(z['tau']), float(z['H']))
    return NumpyPolicy({k: z[k] for k in z.files if k != 'fset'}, str(z['fset']))
