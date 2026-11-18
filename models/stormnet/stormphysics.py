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


def reconstruct(field, params):
    gain, tf, ts, share, _, _ = params
    fast = np.zeros(len(field))
    slow = np.zeros(len(field))
    for e in field.T:
        fast = step(fast, e, gain, tf, share)
        slow = step(slow, e, gain, ts, 1. - share)
    return fast, slow


def projected(field, pressure, dst, params, law):
    """Condition on the last index; partition observed current using the historical state ratio."""
    gain, tf, ts, share, b, c = params
    fast, slow = reconstruct(field, params)
    total = fast + slow
    fraction = np.divide(fast, total, out=np.full_like(total, share), where=np.abs(total) > 1e-8)
    e, p = field[:, -1], pressure[:, -1]
    corrected = dst - b * p + c
    fast, slow = fraction * corrected, (1 - fraction) * corrected
    mean_e, rho_e, mean_p, rho_p = law
    path = []
    for _ in range(HORIZON):
        e = mean_e + rho_e * (e - mean_e)
        p = mean_p + rho_p * (p - mean_p)
        fast = step(fast, e, gain, tf, share)
        slow = step(slow, e, gain, ts, 1 - share)
        path.append(fast + slow + b * p - c)
    return np.stack(path, axis=1), fraction, total
