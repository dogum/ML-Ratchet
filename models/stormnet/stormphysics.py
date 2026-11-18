"""Hourly, pressure-corrected geomagnetic response. Inputs are in nT, km/s and nPa."""
import numpy as np

# injection slope, fast/slow lifetime multipliers, fast share, pressure coefficient, offset
PUBLISHED = [4.4, 1., 1., 1., 7.26, 11.]
TWO_STATE = [4.4, 0.5, 1.7, 0.55, 7.26, 11.]
HISTORY = 36
HORIZON = 3


def drivers(wind):
    return np.maximum(-wind[..., 0], 0) * wind[..., 1] / 1000, np.sqrt(wind[..., 2])


def step(state, field, gain, scale, share):
    """Exact one-hour solution for piecewise-constant convection electric field (mV/m)."""
    tau = scale * 2.40 * np.exp(9.74 / (4.69 + field))
    decay = np.exp(-1. / tau)
    injection = -gain * share * np.maximum(field - 0.49, 0.)
    return decay * state + injection * tau * (-np.expm1(-1. / tau))


def _gamma(params):
    """ring-3 option (17 Nov 2026): injection x (P / 3 nPa)^gamma (Wang, Chao & Lin 2003). Absent = 0."""
    return float(params[6]) if len(params) > 6 else 0.


def _gain(gain, params, pressure_root):
    g = _gamma(params)
    return gain if g == 0. else gain * (np.maximum(pressure_root, 1e-3) ** 2 / 3.) ** g


def reconstruct(field, params, pressure=None):
    gain, tf, ts, share = params[:4]
    fast = np.zeros(len(field))
    slow = np.zeros(len(field))
    for j, e in enumerate(field.T):
        gj = gain if pressure is None else _gain(gain, params, pressure[:, j])
        fast = step(fast, e, gj, tf, share)
        slow = step(slow, e, gj, ts, 1. - share)
    return fast, slow


def assimilated_fraction(field, pressure, past_dst, params):
    """ring-3 option (17 Nov 2026): the fast share of the current at issue time, estimated by least squares
    against the observed Dst of the history window instead of the wind-only reconstruction.

    The two reservoirs evolve linearly given the wind, so every past corrected Dst is (fast0 * a_t + slow0 * b_t
    + driven_t). Solve for the two starting currents, propagate to the issue hour and return fast / total.
    Rows where the fit is degenerate fall back to the wind-only ratio.
    """
    gain, tf, ts, share, b, c = params[:6]
    n, hours = field.shape
    obs = past_dst - b * pressure + c                         # (n, H) corrected Dst, starting at the window start
    fast = np.zeros(n)
    slow = np.zeros(n)
    unit_f = np.ones(n)
    unit_s = np.ones(n)
    a_cols, b_cols, d_cols = [np.ones(n)], [np.ones(n)], [np.zeros(n)]
    for j in range(1, hours):
        e, gj = field[:, j], _gain(gain, params, pressure[:, j])
        fast = step(fast, e, gj, tf, share)
        slow = step(slow, e, gj, ts, 1. - share)
        unit_f = step(unit_f, e, 0., tf, share)
        unit_s = step(unit_s, e, 0., ts, 1. - share)
        a_cols.append(unit_f.copy())
        b_cols.append(unit_s.copy())
        d_cols.append(fast + slow)
    A, B, D = (np.stack(x, 1) for x in (a_cols, b_cols, d_cols))
    y = obs - D
    saa, sbb, sab = (A * A).sum(1), (B * B).sum(1), (A * B).sum(1)
    say, sby = (A * y).sum(1), (B * y).sum(1)
    det = saa * sbb - sab ** 2
    ok = det > 1e-6 * saa * sbb
    f0 = np.where(ok, (sbb * say - sab * sby) / np.where(ok, det, 1.), 0.)
    s0 = np.where(ok, (saa * sby - sab * say) / np.where(ok, det, 1.), 0.)
    fast_end, slow_end = f0 * A[:, -1] + fast, s0 * B[:, -1] + slow
    total = fast_end + slow_end
    frac = np.divide(fast_end, total, out=np.full(n, share), where=np.abs(total) > 1.)
    frac = np.where(ok, np.clip(frac, 0., 1.), np.nan)
    return frac


def projected(field, pressure, dst, params, law, past_dst=None):
    """Condition on the last index; partition observed current using the historical state ratio.

    With past_dst (the window's observed Dst, ring-3) the ratio is assimilated from those observations."""
    gain, tf, ts, share, b, c = params[:6]
    fast, slow = reconstruct(field, params, pressure if _gamma(params) else None)
    total = fast + slow
    fraction = np.divide(fast, total, out=np.full_like(total, share), where=np.abs(total) > 1e-8)
    if past_dst is not None:
        assim = assimilated_fraction(field, pressure, past_dst, params)
        fraction = np.where(np.isfinite(assim), assim, fraction)
    e, p = field[:, -1], pressure[:, -1]
    corrected = dst - b * p + c
    fast, slow = fraction * corrected, (1 - fraction) * corrected
    mean_e, rho_e, mean_p, rho_p = law
    path = []
    for _ in range(HORIZON):
        e = mean_e + rho_e * (e - mean_e)
        p = mean_p + rho_p * (p - mean_p)
        g = _gain(gain, params, p)
        fast = step(fast, e, g, tf, share)
        slow = step(slow, e, g, ts, 1 - share)
        path.append(fast + slow + b * p - c)
    return np.stack(path, axis=1), fraction, total
