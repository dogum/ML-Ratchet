"""WaterSim: a verified daily hydrology simulator for ML Ratchet's Watershed project.

    from watersim import simulate, generate_basins, list_models, routing, reservoir
    out = simulate(spec, forcing, start, end, model='hbv', snow='degree_day', routing='uh')
"""
from ._nb import HAVE_NUMBA, check_cache as _check_cache, clear_cache
_check_cache()
from . import routing, reservoir, pet, snow, metrics, signatures, weather, generator, forcing, realdata, calibrate, noise
from .simulate import simulate, simulate_batch, simulate_many, simulate_network, list_models, prepare_run
from .generator import generate_basins, params_from_attributes
from .weather import generate_weather, fit_climate
from .spec import fill_spec, build_bands, param_table, resolve_params

__version__ = '1.0.0'
__all__ = ['simulate', 'simulate_batch', 'simulate_many', 'simulate_network', 'noise', 'generate_basins', 'list_models', 'routing', 'reservoir',
           'pet', 'snow', 'metrics', 'signatures', 'weather', 'generator', 'forcing', 'generate_weather', 'fit_climate',
           'params_from_attributes', 'clear_cache', 'fill_spec', 'build_bands', 'param_table', 'resolve_params', 'HAVE_NUMBA']
