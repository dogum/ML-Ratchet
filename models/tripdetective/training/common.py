"""TripDetective shared constants and paths."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json
import numpy as np

ROOT = _P.WORK
sys.path.insert(0, _P.VEHICLE + '/roadsim' + ''); sys.path.insert(0, _P.VEHICLE); sys.path.insert(0, f'{ROOT}/src')
os.environ.setdefault('OMP_NUM_THREADS', '1')

ARCHS = ['conv', 'hev_powersplit', 'hev_parallel_p2', 'hev_series', 'phev_powersplit', 'phev_series', 'bev']
VEDTYPE = {'conv': 'ICE', 'hev_powersplit': 'HEV', 'hev_parallel_p2': 'HEV', 'hev_series': 'HEV',
           'phev_powersplit': 'PHEV', 'phev_series': 'PHEV', 'bev': 'EV'}
VEDTYPES = ['ICE', 'HEV', 'PHEV', 'EV']
STRATS = ['conv_rule', 'conv_opt', 'conv_stopstart',
          'aecms', 'ecms', 'powersplit_rule', 'parallel_rule', 'series_thermostat', 'series_power_follower',
          'phev_cdcs', 'phev_blended',
          'bev_regen_series', 'bev_regen_parallel', 'bev_one_pedal']
# strategy families shown on the page
STRAT_FAMILY = {'conv_rule': 'shift rule', 'conv_opt': 'BSFC-optimal shift', 'conv_stopstart': 'stop-start',
                'aecms': 'A-ECMS', 'ecms': 'ECMS', 'powersplit_rule': 'rule', 'parallel_rule': 'rule',
                'series_thermostat': 'rule', 'series_power_follower': 'rule',
                'phev_cdcs': 'CD then CS', 'phev_blended': 'blended',
                'bev_regen_series': 'series regen', 'bev_regen_parallel': 'parallel regen', 'bev_one_pedal': 'one-pedal'}
LEDGER = ['aero', 'rolling', 'grade', 'brake', 'powertrain', 'aux']   # kJ per km, per trip
L = 1024              # seconds per input window
G = 9.81
LHV_GAS = 42.6e3      # J/g (E10 gasoline lower heating value, approx.)
CURB_OFFSET = 136.0   # kg: RoadData convention, spec mass = curb + 136 kg (driver + load)
LB = 0.45359237
WARM = {'engine': 85, 'driveline': 70, 'tire_warm': 1, 'cabin': 21}


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def jsave(o, p):
    tmp = p + '.tmp'
    json.dump(o, open(tmp, 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    os.replace(tmp, p)
