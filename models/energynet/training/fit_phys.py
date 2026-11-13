"""Fit the generic physics-estimate constants (features.PHYS_K) on the SIMULATED training bank only (raw targets,
true road-load parameters). Prints the constants; paste them into features.PHYS_K before training rung 1."""
import _paths as _P  # portable paths (see _paths.py)
import sys
sys.path.insert(0, _P.SRC)
import numpy as np
from scipy.optimize import minimize
import dataset as DS, features as F

B = DS.load_bank('train')
y = B['tg'][:, 0]; ph = B['PH'][:, 0].astype(float); m = B['m_true']; dur = B['dur']; arch = B['arch']
s = (m / 1500.0) ** 0.7
Ep, En = ph[:, 0], -ph[:, 1]


def loss_ice(k):
    e = Ep / k[0] + k[1] * 1000 * s * dur / 1e6
    return np.median(np.abs(np.log(np.maximum(e, 1e-3) / y[g])))


res = {}
g = arch == 0
r = minimize(lambda k: np.median(np.abs(np.log(np.maximum(Ep[g] / k[0] + k[1] * 1000 * s[g] * dur[g] / 1e6, 1e-3) / y[g]))), [0.25, 5.0], method='Nelder-Mead')
res['ice'] = [float(r.x[0]), float(r.x[1] * 1000)]; print('ice', res['ice'], r.fun)
g = np.isin(arch, [1, 2, 3])
f = lambda k: np.median(np.abs(np.log(np.maximum(np.maximum((Ep[g] - k[1] * En[g]) / k[0], 0.2 * Ep[g] / k[0]) + k[2] * 1000 * s[g] * dur[g] / 1e6, 1e-3) / y[g])))
r = minimize(f, [0.30, 0.4, 1.5], method='Nelder-Mead'); res['hev'] = [float(r.x[0]), float(r.x[1]), float(r.x[2] * 1000)]; print('hev', res['hev'], r.fun)
g = (arch == 6) & (y > 0.05)
f = lambda k: np.median(np.abs(np.log(np.maximum(np.maximum(Ep[g] / k[0] - k[1] * En[g], 0.3 * Ep[g]) + k[2] * 1000 * s[g] * dur[g] / 1e6, 1e-3) / y[g])))
r = minimize(f, [0.85, 0.6, 0.6], method='Nelder-Mead'); res['bev'] = [float(r.x[0]), float(r.x[1]), float(r.x[2] * 1000)]; print('bev', res['bev'], r.fun)
print('PHYS_K =', res)
