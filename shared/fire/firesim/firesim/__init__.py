"""FireSim: verified fire behaviour and spread simulator for ML Ratchet (Wildfire, Phase 1).

    from firesim import simulate, behave, observe, generate_landscape, generate_weather, fuel_models
"""
from ._nb import check_cache as _check_cache

_check_cache()

from .fuels import fuel_models, FM, number_of, code_of, NONBURNABLE  # noqa: E402
from .landscape import Landscape, as_landscape, slope_aspect, rasterize_lines  # noqa: E402
from .behave import behave  # noqa: E402
from .simulate import simulate, simulate_batch, prepare, fuel_group, GROUPS, DEFAULTS  # noqa: E402
from .weather import generate_weather, hourly_from_daily, check_weather, ensure_srad  # noqa: E402
from .moisture import dead_moisture, simard_emc  # noqa: E402
from . import rothermel, crown, spotting, moisture, weather, fwi, metrics  # noqa: E402,F401
from .satellite import observe, nominal_overpasses  # noqa: E402
from . import satellite  # noqa: E402,F401
from .generate import generate_landscape  # noqa: E402
from . import replay, realdata  # noqa: E402,F401

__version__ = '1.0.0'
