"""Rung 5 (15 Nov 2026): a least-squares polish of the network's planet parameters.

Why: on the exam, a least-squares transit fit beats the network on a/R* at high signal-to-noise (3.7% vs 4.2%), while the
network wins on the radius ratio. A least-squares fit uses every point of the folded transit shape, which matters most
when the noise is low; the network reads a coarse picture. Combining the two by their uncertainties should keep the best
of each. Checked first on the frozen simulated set (val2), where the same comparison can be made.

Method, fixed before scoring:
  1. Start a least-squares fit of the folded, binned transit (the network's local view: the same data the classical fit
     reads) at the network's reading: radius ratio k, impact parameter b, log10 a/R*, a small time shift, and limb
     darkening (Kipping q1, q2 in [0, 1], started at 0.35 / 0.32, the middle of the dev-calibrated prior). Circular orbit.
  2. Its covariance: s^2 (J^T J)^-1, s^2 the reduced chi-square, J at the solution (limb darkening free, so its
     uncertainty is carried into k, b and a/R*).
  3. Combine with the network in (log10 k, b, log10 a/R*): posterior precision = network precision (diagonal, its own
     sigmas) + least-squares precision (full 3x3 block, marginalised over the shift and limb darkening); mean likewise.
     Duration stays the network's. If the fit fails or its covariance is not finite, the network reading is kept.
Rule (fixed before scoring, the ladder's own rule): keep if the val2 objective rises by more than 0.01 AND the dev log-loss
is no more than 0.05 worse AND the dev radius-ratio error no more than 1 point worse.
numpy/scipy only, so it can ship with inference."""
import numpy as np
from scipy.optimize import least_squares
try:
    from . import transitsim as TS          # inside the model folder
except ImportError:
    import transitsim as TS
NL = 201; X = np.linspace(-2.5, 2.5, NL)

def fit(local, scale, P, dur, k0, b0, aR0, q0=(0.35, 0.32)):
    """local: (2, NL) view (values / scale, mask). Returns dict(x=[logk, b, logaR], cov=3x3) or None."""
    v, m = np.asarray(local[0], float), np.asarray(local[1]) > 0
    if m.sum() < 20 or not np.isfinite([k0, b0, aR0, scale, P, dur]).all(): return None
    y = 1 + v[m] * scale; t = X[m] * dur
    def model(th):
        k, b, laR, dt, q1, q2 = th; aR = 10 ** laR; inc = np.arccos(np.clip(b / aR, 0, 1)); s = np.sqrt(q1)
        return TS.planet_flux(t, dt, P, k, aR, inc, 2 * s * q2, s * (1 - 2 * q2), supersample=1)
    x0 = [float(np.clip(k0, 0.003, 0.59)), float(np.clip(b0, 0, 1.4)), float(np.clip(np.log10(aR0), 0.11, 2.9)), 0.0, q0[0], q0[1]]
    try:
        r = least_squares(lambda th: (model(th) - y) / scale, x0, bounds=([0.002, 0, 0.1, -0.3 * dur, 0, 0], [0.6, 1.5, 3.0, 0.3 * dur, 1, 1]), max_nfev=300, x_scale='jac')
    except Exception:
        return None
    n, p = len(y), len(x0)
    if n <= p + 2: return None
    s2 = 2 * r.cost / (n - p); J = r.jac
    try:
        C = s2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    k, b, laR = r.x[:3]; Jt = np.diag([1 / (k * np.log(10)), 1.0, 1.0])          # (k, b, log aR) -> (log k, b, log aR)
    C3 = Jt @ C[:3, :3] @ Jt.T
    if not np.isfinite(C3).all() or np.any(np.diag(C3) <= 0): return None
    return dict(x=np.array([np.log10(k), b, laR]), cov=C3, q=r.x[4:6].tolist())

def combine(mu, sig, f):
    """mu, sig: network (log k, b, log aR) and sigmas; f: fit() result. Returns (mean, sigmas) of the combination."""
    if f is None: return np.asarray(mu, float), np.asarray(sig, float)
    try:
        Pn = np.diag(1 / np.asarray(sig, float) ** 2); Pf = np.linalg.inv(f['cov']); C = np.linalg.inv(Pn + Pf)
        m = C @ (Pn @ np.asarray(mu, float) + Pf @ f['x'])
    except np.linalg.LinAlgError:
        return np.asarray(mu, float), np.asarray(sig, float)
    if not np.isfinite(m).all(): return np.asarray(mu, float), np.asarray(sig, float)
    return m, np.sqrt(np.diag(C))

def polish_readings(R, local, s):
    """R: network readings (dict of arrays: log_k, b, log_aR and _sig); local: (n, 2, NL); s: (n, 10) scalars
    (s[:, 0] log10 P, s[:, 1] log10 duration h, s[:, 2] log10 scale). Returns a new dict and the fits."""
    Q = {k: np.array(v, float, copy=True) for k, v in R.items()}; fits = []
    for i in range(len(local)):
        P, dur, scale = 10 ** s[i, 0], 10 ** s[i, 1] / 24, 10 ** s[i, 2]
        f = fit(local[i], scale, P, dur, 10 ** R['log_k'][i], R['b'][i], 10 ** R['log_aR'][i]); fits.append(f)
        m, sg = combine([R['log_k'][i], R['b'][i], R['log_aR'][i]], [R['log_k_sig'][i], R['b_sig'][i], R['log_aR_sig'][i]], f)
        for j, p in enumerate(('log_k', 'b', 'log_aR')): Q[p][i] = m[j]; Q[p + '_sig'][i] = sg[j]
    return Q, fits
