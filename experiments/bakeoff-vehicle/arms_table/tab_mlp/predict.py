"""MLP control arm: EnergyNet v2.1's network (151 -> 256 -> 256 -> 256 -> 128 -> mean, log-sigma, 5 component shares;
GELU) and training recipe, retrained on the bake-off's shared simulated train bank. Numpy forward pass (model_np.py,
a byte copy of EnergyNet's). The members are weights/*.npz; their log predictions are averaged (an ensemble of one is
a single network).

Output: pred = exp(mean log E) (the median, as EnergyNet); band = log-normal with sigma = sqrt(mean member sigma^2 +
spread of the member means) from the heteroscedastic head, NOT widened (no simulation or dev factors: the conformal
layer of arm C does calibration); components (aero, rolling, brakes, aux, losses) from the component head, MJ."""
import os
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('tab_mlp_model_np', os.path.join(HERE, 'model_np.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
WD = os.path.join(HERE, 'weights')
NET = _m.Ensemble(sorted(os.path.join(WD, f) for f in os.listdir(WD) if f.endswith('.npz')))
QZ = np.array([-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722])


def predict(batch):
    X = np.asarray(batch['X'], np.float64); E = np.asarray(batch['E_phys'], np.float64)
    p, s, c = NET.predict(X, E)
    Q = p[:, None] * np.exp(QZ[None, :] * s[:, None])
    out = {'pred': p, 'components': c}
    out.update({k: Q[:, j] for j, k in enumerate(['q05', 'q25', 'q50', 'q75', 'q95'])})
    return out
