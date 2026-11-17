"""BuildingSim: a verified, fast building-energy simulator for ML Ratchet (buildings project, Phase 1).

    from buildingsim import simulate, generate_buildings, list_systems, ledger
    out = simulate(spec, weather, start, end, dt='15min', controller=None, faults=None, prices=None,
                   init=None, rng=None)
"""
from .simulate import simulate, Simulator
from .spec import fill_spec, list_systems, USE_TYPES, SYSTEMS
from .ledger import ledger
from .weather import read_epw, epw_path, prepare as prepare_weather
from .faults import FAULT_TYPES


def generate_buildings(n, seed=0, **kw):
    from .generator import generate_buildings as _g
    return _g(n, seed=seed, **kw)


__version__ = '1.0'
