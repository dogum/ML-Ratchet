"""roadsim: a forward-facing, ledger-checked vehicle simulator for every architecture and class.

    from roadsim import simulate, load_vehicle, list_strategies, make_trip
    out = simulate(load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'}), make_trip('udds'))
    out['ledger'], out['summary']
"""
from .sim import simulate, normalize_trip, air_density, isa_density, get_vehicle
from .vehicle import load_vehicle, fill_spec, Vehicle, spec_from_abc, abc_to_si, CLASSES, ARCHS
from .strategies import list_strategies
from .options import DEFAULTS as OPTION_DEFAULTS, PRESETS as OPTION_PRESETS, resolve as resolve_options
from .trips import make_trip, cycle, list_cycles, load_trip, save_trip, sim_to_contract

__version__ = '1.0.0'
PHYSICS_VERSION = 'roadsim-physics-v2'      # simulate()'s default; 'roadsim-physics-v1' stays reproducible by name
