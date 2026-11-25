"""EnergyNet v2.1 as released (models/energynet, 15 Nov 2026): 5-network ensemble (weights/r15_s*.npz), numpy
forward pass (model_np.py, byte copy), error bars as released (sigma_factors.json: simulation factor per
architecture x the dev factor of the output group, widen-only, fitted by EnergyNet on dev).

Table track. The output group (which dev factor applies) is read from the inputs as EnergyNet's exam did it:
heavy duty by class and architecture (ZTBus trolleybus / Mersin diesel-CNG bus / ICCT truck), an EPA test car by
the dynamometer flag (V[46]) on a light-duty vehicle (EPA_conv, EPA_hev_powersplit for every hybrid, EPA_bev),
otherwise the VED group of the architecture (ICE / HEV / EV / PHEV). Band: log-normal with
sigma = sqrt((k * SIM_arch * s)^2 + tau^2) around the ensemble's median prediction."""
import os, json
import numpy as np
from model_np import Ensemble

HERE = os.path.dirname(os.path.abspath(__file__))
NET = Ensemble(sorted(os.path.join(HERE, 'weights', f) for f in os.listdir(os.path.join(HERE, 'weights')) if f.endswith('.npz')))
K = json.load(open(os.path.join(HERE, 'sigma_factors.json')))
ARCHS = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev']
SIM = np.array([K.get(f'SIM_{a}', 1.0) for a in ARCHS])
QZ = np.array([-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722])
V0 = 90 + 12                       # start of the vehicle vector in the 151 inputs


def _kt(key):
    v = K.get(key, 1.0)
    return (float(v[0]), float(v[1])) if isinstance(v, list) else (float(v), 0.0)


def group_key(arch, cls, dyno):
    if cls == 4:                                   # bus
        return 'HD_ZTBus' if arch == 6 else 'HD_Mersin'
    if cls in (5, 6):
        return 'HD_ICCT'
    if dyno:
        return 'EPA_conv' if arch == 0 else ('EPA_bev' if arch == 6 else 'EPA_hev_powersplit')
    return 'VED_' + ('ICE' if arch == 0 else 'EV' if arch == 6 else 'PHEV' if arch in (4, 5) else 'HEV')


def predict(batch):
    X = np.asarray(batch['X'], np.float64); E = np.asarray(batch['E_phys'], np.float64)
    p, s, c = NET.predict(X, E)
    a = np.argmax(X[:, V0:V0 + 7], axis=1); cl = np.argmax(X[:, V0 + 7:V0 + 14], axis=1); dy = X[:, V0 + 46] > 0.5
    kt = np.array([_kt(group_key(int(x), int(y), bool(z))) for x, y, z in zip(a, cl, dy)])
    S = np.sqrt((kt[:, 0] * SIM[a] * s) ** 2 + kt[:, 1] ** 2)
    Q = p[:, None] * np.exp(QZ[None, :] * S[:, None])
    out = {'pred': p, 'components': c}
    out.update({k: Q[:, j] for j, k in enumerate(['q05', 'q25', 'q50', 'q75', 'q95'])})
    return out
