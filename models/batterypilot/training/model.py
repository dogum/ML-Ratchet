"""BatteryPilot network: features -> marginal value of stored energy at 5 SOC knots (numpy inference; torch for
training lives in train.py and mirrors this exactly)."""
import numpy as np

from battery import Battery, SOC_KNOTS, step_offer
from features import build, N_FEAT, rolling_scale

K = len(SOC_KNOTS)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


class PolicyNet:
    def __init__(self, weights):
        w = dict(np.load(weights)) if isinstance(weights, str) else dict(weights)
        self.mu, self.sd = w['x_mu'], w['x_sd']
        self.hybrid = bool(int(w['hybrid'])) if 'hybrid' in w else False   # v1.2: residual on the DA backbone
        self.min_step = float(w['min_step']) if 'min_step' in w else 0.0
        self.layers = []
        i = 0
        while f'W{i}' in w:
            self.layers.append((w[f'W{i}'], w[f'b{i}']))
            i += 1

    def z_knots(self, X):
        h = (X - self.mu) / self.sd
        for j, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if j < len(self.layers) - 1:
                h = silu(h)
        if self.hybrid:
            u = X[:, -K:] + h
            if self.min_step > 0:
                z = u.copy()
                for k in range(1, K):
                    z[:, k] = np.minimum(z[:, k - 1] - self.min_step, u[:, k])
                return np.clip(z, -12.0, 12.0)
            return np.clip(np.minimum.accumulate(u, axis=1), -12.0, 12.0)
        z0 = h[:, :1]
        steps = softplus(h[:, 1:]) + 0.01
        return np.clip(np.concatenate([z0, z0 - np.cumsum(steps, 1)], 1), -12.0, 12.0)

    def values(self, X, ps):
        """Value curve in $/MWh of stored energy at SOC_KNOTS, decreasing."""
        return np.sinh(self.z_knots(X)) * np.asarray(ps, float)[:, None]


def run_policy(net: PolicyNet, path, bat: Battery, c_lin, soc0=0.0):
    """Run BatteryPilot causally over a whole path (dict/DataFrame with rt, da, df, T, hour, day, dow, doy).
    The value curves depend only on information before each hour, so they are computed for all hours at once; the
    offer-curve execution against each hour's RT price is sequential.  Returns (ch, dis, soc, V)."""
    n = len(path['rt'])
    t = np.arange(n)
    ps = rolling_scale(np.asarray(path['da'], float))
    cl = np.broadcast_to(np.asarray(c_lin, float), (n,)).copy()
    X, ps = build(path, t, bat.d, cl, ps=ps)
    if net.hybrid:
        from backbone import value_curves
        lam = value_curves(np.asarray(path['da'], float), np.asarray(path['hour']), np.asarray(path['day']), cl,
                           bat.d, P=bat.P, eta_rt=bat.eta_rt)
        X = np.concatenate([X, np.arcsinh(lam / ps[:, None])], 1).astype(np.float32)
    V = net.values(X, ps)
    rt = np.asarray(path['rt'], float)
    ch = np.zeros(n); dis = np.zeros(n); soc = np.zeros(n + 1); soc[0] = soc0
    for i in range(n):
        s, c, d = step_offer(bat, soc[i], V[i], rt[i], cl[i])
        soc[i + 1] = s; ch[i] = c; dis[i] = d
    return ch, dis, soc, V
