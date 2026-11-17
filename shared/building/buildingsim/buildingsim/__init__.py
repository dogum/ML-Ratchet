"""BuildingSim: a verified, fast building-energy simulator for ML Ratchet (buildings project, Phase 1).

    from buildingsim import simulate, generate_buildings, list_systems, ledger
    out = simulate(spec, weather, start, end, dt='15min', controller=None, faults=None, prices=None,
                   init=None, rng=None, options=None)

Phase 2 wrappers (opt-in, README 'Phase 2 wrappers'): `options={...}` for what acts inside the time loop, the hvac
field `district_eff_chw`, and the functions of `buildingsim.phase2`. With none of them used the physics is 1.0.

Physics versions (campus.py; opt-in, the default stays 1.0): options={'physics': '1.1'} (campus and district thermal
plants) or '1.2' (1.1 + weather-insensitive electricity), or the environment variable BUILDINGSIM_PHYSICS.
"""
from .simulate import simulate, Simulator, OPTIONS
from .spec import fill_spec, list_systems, USE_TYPES, SYSTEMS
from .ledger import ledger
from .weather import read_epw, epw_path, prepare as prepare_weather, set_cache_size as set_weather_cache
from .faults import FAULT_TYPES
from . import phase2  # noqa: F401  (opt-in wrappers; importing them changes nothing)
from .campus import PHYSICS_VERSIONS, resolve as physics_version  # noqa: F401

__version__ = '1.2.0'
PHYSICS_VERSION = '1.0'     # the default physics (flags off = the verified v1.0); 1.1 and 1.2 are opt-in (campus.py)
PHYSICS_LATEST = PHYSICS_VERSIONS[-1]


def generate_buildings(n, seed=0, **kw):
    from .generator import generate_buildings as _g
    return _g(n, seed=seed, **kw)


def clear_caches():
    """Drop the prepared-weather cache and the generator's prior tables (bounded-memory mode, with
    set_weather_cache(n))."""
    from . import weather, generator
    weather.clear_cache()
    generator._CACHE.clear()
