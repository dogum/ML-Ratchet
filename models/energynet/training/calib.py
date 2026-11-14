"""Error-bar calibration on DEV only (method.md step 6), and the Phase D sigma rung (r10) that chooses between methods.

  'v1'   one widen-only factor per output group: k = max(1, 68th percentile of |log(pred/y)| / sigma) over dev trips.
  'car'  per-architecture-group car term: sigma_tot^2 = (k sigma)^2 + tau^2, where k is v1's factor and tau >= 0 is the
         spread of whole cars that the trips of a few dev cars under-represent: chosen so that the CAR-WEIGHTED share
         of dev trips inside the 68% band is 68% (each car weighs 1, not its trip count).

Judged by leave-one-car-out on the dev VED cars: fit on the other dev cars of the group (the other light-duty dev cars
when the group has fewer than 2 others), count the held-out car's trips inside the 68% band. The method whose pooled
held-out coverage (car-weighted) is closer to 68% on average over the four groups wins; nothing of the exam is read."""
import numpy as np


def z(d, k=1.0, tau=0.0):
    return np.abs(np.log(np.maximum(d.pred.values, 1e-9) / np.maximum(d.y.values, 1e-9))) / np.sqrt((k * d.sig.values) ** 2 + tau ** 2)


def fit_k(d):
    return float(max(1.0, np.quantile(z(d), 0.68))) if len(d) else 1.0


def car_cov(d, k, tau, q=1.0):
    """car-weighted share of trips inside the band (q = 1 for 68%, 1.96 for 95%)."""
    inside = z(d, k, tau) < q
    w = 1.0 / d.groupby('vehicle_id').vehicle_id.transform('size').values
    return float(np.sum(w * inside) / np.sum(w))


def fit_tau(d, k):
    if not len(d) or car_cov(d, k, 0.0) >= 0.68:
        return 0.0
    lo, hi = 0.0, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if car_cov(d, k, mid) < 0.68: lo = mid
        else: hi = mid
    return float(hi)


def fit(d, method):
    k = fit_k(d)
    return (k, fit_tau(d, k) if method == 'car' else 0.0)


def loco(dv, method):
    """dv: dev VED frame (group, vehicle_id, pred, y, sig). -> {group: pooled held-out car-weighted coverage}"""
    rows = []
    for g, dg in dv.groupby('group'):
        for c in dg.vehicle_id.unique():
            others = dg[dg.vehicle_id != c]
            if others.vehicle_id.nunique() < 2:
                others = dv[(dv.vehicle_id != c)]
            k, tau = fit(others, method)
            hc = dg[dg.vehicle_id == c]
            rows.append((g, c, float(np.mean(z(hc, k, tau) < 1.0))))
    out = {}
    for g in sorted(set(r[0] for r in rows)):
        out[g] = float(np.mean([r[2] for r in rows if r[0] == g]))
    return out


def sim_factors(model, version):
    """'simdev' step 1 (simulation only): per architecture, the factor that brings the frozen val2 set to 68% inside
    the band (the networks' own sigma is overconfident on simulated trips too)."""
    import evaluate as EV, dataset as DS, features as F
    B, T, X, E = EV.val_set('val2')
    y = DS.target(B, T, version); ok = y > 0.02
    p, s, c = model.predict(X[ok], E[ok])
    zz = np.abs(np.log(p / y[ok])) / s
    a = B['arch'][ok]
    return {F.ARCHS[i]: float(np.quantile(zz[a == i], 0.68)) for i in range(7) if (a == i).any()}


def apply_sim(d, ks):
    """d: real frame with 'arch' and 'sig' -> sig scaled by the architecture's simulation factor."""
    d = d.copy(); d['sig'] = d.sig * d.arch.map(ks).fillna(1.0)
    return d


class SimScaled:
    """a model whose sigma is first multiplied by its architecture's simulation factor (method 'simv1' = this, then
    v1's dev factor per group). The architecture is read from the input's one-hot (features.vehicle_vec)."""
    def __init__(self, m, ks):
        import features as F
        self.m, self.meta, self.mode = m, m.meta, m.mode
        self.f = np.array([ks.get(a, 1.0) for a in F.ARCHS]); self.c0 = F.NR + F.NF

    def predict(self, X, E):
        p, s, c = self.m.predict(X, E)
        a = np.argmax(np.asarray(X)[:, self.c0:self.c0 + 7], axis=1)
        return p, s * self.f[a], c
