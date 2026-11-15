"""GridSim: power-system simulator for ML Ratchet (power flow, unit commitment / dispatch, frequency dynamics,
renewables, demand, emissions, fleet and system-state generator).  See README.md and LIMITS.md."""
from .network import Network, load_matpower, powerflow, dc_powerflow, ac_powerflow, dc_opf
from .fleet import prepare_fleet, merge_combined_cycles, fuel_prices_daily, TECH_DEFAULTS, EIA930_FUEL, physics_version
from .dispatch import dispatch, DispatchResult
from .frequency import (frequency_event, make_system_state, inertia_MWs, aggregate_units, event_metrics, TECH_DYN,
                        SYSTEMS)
from . import renewables, demand, market
from .renewables import wind_power, solar_power, farm_curve, air_density, hub_wind, fleet_profiles
from .demand import demand_model, fit_demand
from .generator import generate_fleet, generate_states

demand_model = demand_model
# "1.1" (default): outage defaults from sources published before the exam years (fleet.OUTAGE_DEFAULTS).
# "1.0": as released on 14 Nov 2026; set GRIDSIM_PHYSICS=1.0 to reproduce it.
# "1.2" (opt-in, 15 Nov 2026): 1.1 plus real-time scarcity the day-ahead market cannot see (gridsim.market); every
# other output equals 1.1.  Read once at import.
PHYSICS_VERSION = physics_version()
__all__ = ["Network", "load_matpower", "powerflow", "dc_powerflow", "ac_powerflow", "dc_opf", "dispatch",
           "frequency_event", "make_system_state", "renewables", "demand", "demand_model", "fit_demand",
           "wind_power", "solar_power", "generate_fleet", "generate_states", "prepare_fleet", "market", "PHYSICS_VERSION"]
