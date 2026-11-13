"""Global calibration constants fitted on the EPA *dev* vehicles only (see road/checks/calibrate_dev.py).
Values in data/calibration.json override the physics defaults below."""
import os, json
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration.json')
DEFAULTS = {'cold_fric_gasoline': 8.0, 'cold_fric_diesel': 8.0, 'cat_warmup_s': 20.0, 'motor_loss_scale': 1.0,
            'bev_motor_loss_scale': 1.0, 'cold_fric_hybrid_factor': 1.0}
_C = None


def get(key):
    global _C
    if _C is None:
        _C = dict(DEFAULTS)
        if os.path.exists(_P):
            _C.update(json.load(open(_P)))
    return _C[key]


def reload():
    global _C
    _C = None
