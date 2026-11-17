# BuildingSim

A building-energy simulator built from the physics, for training networks on. It runs any building of the shared
contract (a house to a hospital, 12 use types, 13 HVAC system types) through any weather at 15-minute or hourly
steps. It returns zone temperatures and humidity, comfort, HVAC operation, energy by end use and by meter, an AHU/RTU
**sensor trend log**, and an **energy, heat and moisture ledger** that closes on every run. Faults can be injected
for FaultFinder, a controller can drive it for ThermoPilot, and a generator draws random buildings whose
characteristics bootstrap the NREL ResStock/ComStock stock models and real CBECS answers. A building-year takes
about 0.7 s on one thread.

Part of ML Ratchet ("Trained on physics. Graded on reality."), Phase 1 of the buildings project. Phase 2 trains
LoadCast, BuildingDetective, FaultFinder and ThermoPilot on it. Known errors are in [LIMITS.md](LIMITS.md); every
check is a JSON file in `shared/building/checks/` with its numbers.

## Quick start

```python
import sys; sys.path.insert(0, 'shared/building/buildingsim')
from buildingsim import simulate, generate_buildings, list_systems, ledger, read_epw, epw_path

w = read_epw(epw_path('Chicago'))                 # (DataFrame in the contract, meta); 17 bundled TMYx cities
spec = dict(primary_use='office', floor_area_m2=5000, n_floors=3, year_built=2005, climate_zone='5A',
            system='vav_chiller_boiler')          # every other field is defaulted (spec['src_fill'] says how)
out = simulate(spec, w, '2019-01-01', '2020-01-01', dt='15min')
out['summary']['kWh_by_meter'], out['summary']['EUI_kWh_m2_yr'], ledger(out)['ok']
out['ts'][['T_zone', 'RH_zone', 'meter_electricity_kW', 'meter_gas_kW', 'PPD']]   # 35,040 rows, UTC
out['trend'][['OAT', 'RAT', 'MAT', 'SAT', 'OA_damper_pct', 'cool_cmd_pct', 'fan_speed_pct', 'fan_kW']]

specs = generate_buildings(1000, seed=0)          # random buildings of all 12 uses; s['weather_hint'] = a TMYx city
list_systems()
```

`python3 demo.py` runs an office and a house through a winter and a summer week, prints the ledgers and comfort,
injects one fault (office outdoor-air damper stuck open), runs one custom controller (house pre-cooling under a
time-of-use tariff) and writes `demo_weeks.png` (1-2 s).

The package needs numpy, scipy, pandas and pvlib only. CoolProp and pythermalcomfort are used only by the checks,
from the isolated venv `buildingsim/.refenv`.

## API (the shared contract)

```python
out = simulate(spec, weather, start=None, end=None, dt='1h', controller=None, faults=None, prices=None, init=None,
               rng=None, warmup_days=7, sensor_noise=None, record_trend=True, options=None)
```

| argument | meaning |
|---|---|
| `spec` | building spec dict (below). `fill_spec(spec)` shows the completed spec; `src_fill` lists every default and its source. |
| `weather` | `(DataFrame, meta)` in the contract (UTC index; `T_out`, `RH`, `dewpoint`, `GHI`, `DNI`, `DHI`, `wind_speed`, `wind_dir`, `pressure`, optional `cloud`, `IR_horiz`; meta `lat`, `lon`, `elev`, `tz`), a DataFrame with `attrs['meta']`, or an EPW path. A row stamped t describes [t, t+step). Missing DNI/DHI are split with Erbs, missing dewpoint from RH. A typical-year frame wraps around the year end. |
| `start`, `end` | UTC (or tz-aware) timestamps; default the whole weather frame. |
| `dt` | any pandas step: `'15min'`, `'1h'`, `'5min'`... Weather is averaged or interpolated to it. |
| `controller` | callable `obs -> action dict or None`, called every step (see ThermoPilot below). |
| `faults` | list of `{'type', 'start', 'end', **params}` (see FaultFinder below). |
| `prices` | a number ($/kWh), a Series/DataFrame (`elec` column) on any grid, or a TOU dict `{'base', 'periods': [{'months', 'hours': (h0, h1), 'weekdays_only', 'price'}]}`. |
| `init` | `{'T_air', 'T_mass', 'W_air', 'soc'}` or a previous `out['final_state']`. Default: a 7-day warm-up on the first week. |
| `rng` | seed or Generator, for sensor noise. `sensor_noise={'SAT': 0.2, ...}` adds Gaussian noise (K) to trend columns. |
| `options` | opt-in Phase 2 flags that act inside the time loop (see [Phase 2 wrappers](#phase-2-wrappers-opt-in)). `None` or `{}` = v1.0 exactly. Unknown keys raise. |

`out` is a dict:

- `ts`: DataFrame on the simulation grid.
  - Zone: `T_zone`, `T_surface`, `T_mass`, `T_op`, `RH_zone`, `W_zone`, `heat_sp`, `cool_sp`, `occupied`, `occupancy`.
  - HVAC: `hvac_mode`, `Q_hvac_sens_kW`, `Q_heat_kW`, `Q_cool_sens_kW`, `Q_cool_lat_kW`, `plr_heat/cool`, `cop_heat/cool`, `infiltration_m3s`.
  - Comfort: `PMV`, `PPD`.
  - Energy: one `<end use>_<fuel>_kW` column per end use (heating, cooling, fans, pumps, heat_rejection, interior and exterior lighting, interior_equipment, water_systems, cooking, pv, battery) and one `meter_<meter>_kW` per meter (`electricity` net of PV and battery, `gas`, `steam` or `hot_water`, `chilled_water`).
  - Also `pv_kW`, `battery_kW`, `soc`, `price`, the weather, and `fault`: the active fault label, `''` when healthy.
- `trend`: what an AHU/RTU trend log records: `OAT`, `OA_RH`, `RAT`, `MAT`, `SAT`, `ZAT`, `ZAT_heat_sp/cool_sp`,
  `ZONE_RH`, `OA_damper_pct`, `cool_cmd_pct` (compressor runtime or CHW valve), `heat_cmd_pct` (burner, HP or preheat
  valve), `reheat_cmd_pct`, `fan_speed_pct`, `fan_status`, `supply_flow_m3s`, `fan_kW`, `hvac_kW`, `building_kW`,
  `gas_kW`, `occupied`, `fault`. Biased sensors report their biased value.
- `summary`:
  - Energy: `kWh_by_end_use`, `kWh_by_meter`, `site_kWh`, `EUI_kWh_m2_yr`, `peak_electricity_kW`.
  - Comfort: `unmet_heating_h`, `unmet_cooling_h`, `comfort_degree_hours_below_20/above_26`, `mean_PPD_occupied`.
  - Envelope: `UA_envelope_W_K`, `UA_with_air_W_K`, `C_mass_J_K`, `tau_h`.
  - Equipment: `sizing` (design loads and every capacity).
  - Timing: `wall_s`, `x_realtime`.
  - Bill: `electricity_cost` (when prices are given).
- `ledger`: see below. `spec`: the filled spec. `final_state`: to chain runs. `sim`: the Simulator object.

Other entry points:

```python
from buildingsim import Simulator, fill_spec, list_systems, FAULT_TYPES, prepare_weather, read_epw, epw_path
from buildingsim.generator import generate_buildings, labels, weather_hint
from buildingsim.stock import resstock_to_spec, comstock_to_spec, cbecs_to_spec, read_stock_weather_parquet
sim = Simulator(spec, w, start, end, dt='15min', prices=tou)     # step-wise use (ThermoPilot, MPC, RL)
obs = sim.reset(); obs, info = sim.step({'cool_sp': 23.0}); st = sim.get_state(); sim.set_state(st); out = sim.finish()
```

### Building spec (dict)

Contract fields:

| group | fields |
|---|---|
| identity | `id`, `site_id`, `lat`, `lon`, `climate_zone` (IECC) |
| use and size | `primary_use`, `floor_area_m2`, `n_floors`, `year_built` |
| envelope | `wall_U`, `roof_U`, `window_U`, `window_wwr`, `SHGC`, `infiltration_ach`, `thermal_mass` (light/medium/heavy... or kJ/m2K) |
| loads | `internal_gains` {`people` W/person, `people_W_m2`, `lighting`, `plug` (W/m2 at peak), `process`, `cooking`, `exterior_lighting`}, `occupancy_schedule` (id or dict) |
| controls | `setpoints` {`heat`, `cool`, `setback_heat`, `setback_cool` (None = no setback), `schedule`, `recovery_h`} |
| ventilation | `ventilation` {`L_s_person`, `L_s_m2`, `economizer`, `econ_high_limit`, `erv`, `mech_vent`, `natural_ventilation`} |
| HVAC | `system`, `hvac` {efficiencies, capacities, `speed`, `heat_source`, ...} |
| other | `dhw`, `pv` {`kW`, `tilt`, `azimuth`}, `battery` {`kWh`, `kW`, `mode`} |

Unknown fields may be None; they get a documented default.

Extensions:
- `ach50` (blower door) and `infiltration_factor`.
- `geometry` {`aspect_ratio`, `floor_to_floor_m`, `ceiling_m`, `orientation_deg`, `foundation`, `wall_exposed_frac`, `roof_exposed`, `attic`, `window_area_by_facade`, `window_overhang`, `interior_shading`, `constructions` (layers for the multi-node zone), `shelter_class`}.
- `zone_model` ('5R1C' default, or 'multinode').
- `moisture_capacitance`.

**Systems** (`list_systems()`):

| system | what it is |
|---|---|
| `ptac` | PTAC (or PTHP with `heat_source='hp'`, gas or boiler coil). |
| `packaged_rtu` | DX cooling, gas furnace / electric / heat pump / hot-water / district coil, constant-volume fan, economizer. |
| `split_ac_furnace` | Residential split system with gas or electric furnace. |
| `heat_pump_air` | Single, two-stage or variable speed, with defrost and backup strips. |
| `heat_pump_ground` | Water-to-air heat pump with a lumped ground loop. |
| `vav_chiller_boiler` | VAV AHU with preheat, economizer and supply-air reset; reheat boxes and perimeter radiation. Water- or air-cooled chiller with tower, packaged DX (`cooling_source='dx'`), gas, electric or district heat. |
| `fan_coil` | 4-pipe fan coils plus a DOAS. |
| `radiant` | Floor heat into the mass node, from a boiler or heat pump. |
| `district_steam_chw` | VAV with district steam or hot water and district chilled water. |
| `baseboard_resistance` | Electric baseboards, optional window AC. |
| `hydronic_baseboard` | Boiler or district baseboards (extension). |
| `wall_furnace` | Ductless wall furnace (extension). |
| `none` | Free-floating. |
| `ideal` | Unlimited heating and cooling to setpoint (for checks and sizing). |

### Ledger (closes on every run; `ledger(out)['ok']`)

| section | balance |
|---|---|
| `zone_heat_kWh` | conduction (windows, walls, roof, ground) + ISO window radiant loss + infiltration + direct ventilation + solar + internal gains + HVAC delivered = storage change in the mass |
| `zone_moisture_kg` | infiltration + ventilation + supply air + internal latent − condensation = storage |
| `air_side_kWh` | heat delivered to the zone = coil heating − coil sensible cooling + fan heat + outdoor-air sensible + return/supply lag + terminal direct heat; also checked against the zone side |
| `equipment_kWh` | electricity + fuel + district + heat taken from air/water + heat from ambient (heat pumps) = heat to air/water + heat to outdoors (rejected, flue, losses) + fan heat |
| `meters_kWh` | each meter = the sum of its end uses (PV negative, battery charging positive) |
| `battery_kWh` | charge in = discharge + losses + stored change |

`closure_rel` is the worst relative residual (1e-8 or better in every check). `system_lag_share` is the share of
air-side energy carried by the one-step supply-air lag; it is non-zero only when a stuck coil or a heating-mode AHU
makes the supply temperature differ from the one the zone solve assumed, and is 0.1 % or less for healthy systems.

## Using it in Phase 2

All calls are single-threaded numpy (set `OMP_NUM_THREADS=1`). Respect `shared/building/data/proc/splits.json`: real inputs
(weather, tariffs) may be used freely; real building metadata only from non-exam units (BDG2 dev, RECS/CBECS
prior + dev). ResStock/ComStock are model outputs: use them for priors, never as an exam.

```python
import sys; sys.path.insert(0, 'shared/building/buildingsim'); sys.path.insert(0, 'shared/building')
import buildingsim as bs, buildingdata as bd
from buildingsim.generator import labels

# LoadCast (hourly energy from weather + calendar): random buildings x real weather years -> meter series
specs = bs.generate_buildings(20000, seed=1)                        # ~1 s per building-year at 15 min, 0.25 s at 1 h
w = bd.site_weather('Hog'); wm = bd.site_weather_meta('Hog')        # real BDG2 site weather (or any TMYx city)
out = bs.simulate(specs[0], (w, dict(lat=wm['lat'], lon=wm['lon'], elev=wm['elev'], tz=wm['tz'])),
                  '2016-01-01', '2017-01-01', dt='1h')
X = out['ts'][['meter_electricity_kW', 'meter_gas_kW', 'meter_chilled_water_kW', 'meter_steam_kW']]
#   BDG2 campus buildings: infer the system from the meters (see checks/check_bdg2.py:spec_from_bdg2).

# BuildingDetective (UA, time constant, balance point, HVAC type, schedule, use, area from meter + weather)
y = labels(specs[0], out)       # UA_envelope_W_K, UA_network_W_K, C_mass_J_K, tau_h, system, heat_source, fuel,
                                # setpoints, occupancy_schedule, primary_use, floor_area_m2, year_built, wwr, ...
#   use UA_network (what an air-temperature-controlled meter sees; ~7 % below the naive sum of U A, README physics)

# FaultFinder (AHU/RTU fault detection and diagnosis): same building, healthy vs faulted, trend log + labels
f = [dict(type='oa_damper_stuck', position=1.0, start='2016-01-10', end='2016-01-20'),
     dict(type='sensor_bias', sensor='T_mix', bias=2.0, start='2016-03-01')]
out = bs.simulate(spec, wx, '2016-01-01', '2016-04-01', dt='15min', faults=f, sensor_noise={'SAT': 0.15, 'MAT': 0.3})
out['trend']        # OAT RAT MAT SAT ZAT dampers valves fan speed/power + 'fault' label per step
#   grade only on agent B's real/experimental fault runs (bd.faults(split='exam')); add real noise from their
#   normal-operation days.

# ThermoPilot (learned thermostat / heat-pump / pre-cooling controller vs rules and MPC on real tariffs)
tou = bd.tariff_prices('pge_e_tou_c', 2017, tz='America/Los_Angeles')    # real tariff -> price Series
sim = bs.Simulator(house_spec, wx, start, end, dt='15min', prices=tou)
obs = sim.reset()
while obs is not None:
    action = policy(obs)        # {'heat_sp','cool_sp','hvac_off','hvac_on','max_capacity','battery_kW'} or None
    obs, info = sim.step(action)                              # info: electricity_kW, gas_kW, T_zone, price
out = sim.finish()                                            # cost, comfort (PPD, degree-hours), ledger
#   MPC baseline: st = sim.get_state(); roll out candidate actions; sim.set_state(st)  (exact, checked)
#   perfect forecasts: obs['sim'].forecast(k, 24, 'T_out' | 'GHI' | 'price' | 'pv_kW')
```

**Fault types** (`FAULT_TYPES`), each modelled in the physics (see `faults.py` for the equations):

| group | faults |
|---|---|
| dampers and economizer | `oa_damper_stuck(position)`, `oa_damper_leak(leak)`, `economizer_disabled`, `economizer_high_limit(high_limit)` |
| valves | `cooling_valve_stuck(position)` (a stuck compressor on DX units), `cooling_valve_leak(leak)`, `heating_valve_stuck/leak`, `reheat_valve_leak(leak)` |
| coils, refrigerant, airflow | `coil_fouling(severity)`, `refrigerant_charge(charge)` (Kim & Braun 2012 trends), `condenser_fouling(severity)`, `low_airflow(severity)` |
| sensors | `sensor_bias(sensor in T_zone/T_sup/T_mix/T_out/T_ret, bias)` |
| schedules and setpoints | `schedule_override(mode)`, `setpoint_override(heat, cool)`, `sat_setpoint_offset(offset)` |
| terminals and plant | `vav_box_stuck(position)`, `boiler_fouling`, `chiller_fouling` |

**Controller observation** `obs`:
- Zone and time: `k`, `t`, `T_zone`, `T_mass`, `W_zone`, `occupied`, `hour`, `daytype`.
- Weather: `T_out`, `RH_out`, `GHI`.
- Defaults and signals: `heat_sp`, `cool_sp` (schedule default), `price`, `soc`, `pv_kW`, `dt_s`.
- `sim`: for forecasts.

**Action keys:**
- `heat_sp`, `cool_sp`.
- `hvac_off`, `hvac_on` (force the occupied mode: fans, outdoor air).
- `max_capacity` (0-1, limits the first heating/cooling stage).
- `battery_kW` (+ charge).

**Which Phase 2 models it supports best.**
- **FaultFinder.** 26 checked fault signatures on RTUs and VAV AHUs, a trend log with sensor channels, per-step labels and ledger-consistent energy. B has real labelled AHU/RTU runs to grade on.
- **ThermoPilot.** A controller hook and step-wise API with exact save/restore, real TOU tariffs, heat-pump part-load and defrost, PV and battery.
- **BuildingDetective.** Exact ground-truth labels for UA, τ, systems and schedules on generated buildings. Real labels exist only at survey level (RECS/CBECS annual).
- **LoadCast.** Supported, but generic twins miss real campus district-meter magnitudes by about 3× (LIMITS). Its simulated pre-training needs dev calibration (the ratchet's job).

## Phase 2 wrappers (opt-in)

Version 1.0.1 (`bs.__version__`). `bs.PHYSICS_VERSION` is still `'1.0'`: with no wrapper used, every check
reproduces the v1.0 JSONs (`checks/REPRODUCED.md`). The four Phase 2 models had wrapped BuildingSim in their own
training code; those wrappers are ported here (HANDOFF section 4, P0.4), each a copy of the model's v1.0 code, each
off by default. `checks/check_phase2.py` shows every port gives the model's own numbers
(`checks/out/phase2_equivalence.json`), and `check_bdg2.py w2 | w2w3 | lc` runs the 52 BDG2 dev twins with them on.

**One mechanism: opt-in by name.** A v1.0 call never reaches a wrapper. A wrapper is a key of `options=` (only
what must act inside the time loop), an hvac spec field that is absent unless set, or a function you apply
(`buildingsim.phase2`, `buildingsim.stock`, `buildingdata`). This follows RoadSim's `options=` (vehicle platform);
there are no presets yet, because no flag is on in any physics version.

| From | Flag (where) | What it does |
|---|---|---|
| LoadCast | `options={'holidays': 'auto'}` (or `'US'`, `'CA'`, `'UK'`, `'IE'`; `holidays.py`) | Public holidays of the site's country (from the weather `tz`: London -> UK, Dublin -> IE, Toronto -> CA); the US keeps the v1.0 calendar. Holidays run Sunday schedules. |
| LoadCast | `phase2.spec_from_meters(base, meters, knobs)` | BDG2-style twin: the meters a building has decide its plant; knobs `sys`, `sched` ('default' / 'extended' / '24x7'), `oa`, `vav_min`, `sat_reset`, `elec`. |
| LoadCast | `phase2.CAMPUS_KNOBS` | The dev modes: 24x7 (22 of 52 dev buildings), outdoor air x2.5 (37 of 52), VAV minimum 0.3 with reset (43 of 52). |
| LoadCast | `phase2.scale_meters(meters_kW, knobs, area)`, `phase2.DEV_METER_SCALE` | y = a x sim + b x A per meter (b in W/m2); dev medians, e.g. chilled water a 0.92, b 31 W/m2. Applied to copies; the ledger is untouched. |
| LoadCast | `bs.set_weather_cache(n)`, `bs.clear_caches()` | Bounded memory for long bank runs: at most n + 1 prepared weather frames (v1.0: 24 + 1; 0 = none). A year is 3 MB at 1 h, 12 MB at 15 min, so the cache explains at most ~300 MB of LoadCast's out-of-memory kill; the rest was not traced. |
| Detective | `phase2.apply_intensity(filled_spec, phase2.W1_INTENSITY[use])` | W1: one factor per use type on per-area loads and outdoor air (lab 0.42 ... lodging 1.55), fitted to CBECS/RECS prior + dev. |
| Detective | `phase2.make_campus(filled_spec, rng, kind)`; hvac field `district_eff_chw` | W2: campus district service with distribution factors around heat 0.185 and chilled water 0.329 (meter = delivered / factor). `district_eff_chw` (absent = `district_eff`) puts the chilled-water factor in the plant, so the ledger still closes. |
| Detective | `phase2.flat_base_loads(meters, rng)` | W3: moves a share of district energy to a flat base (heat 0.70, chilled water 0.50; annual unchanged) and adds flat electricity (+0.5 x mean, p 0.85). Applied to copies of the meter series. |
| Detective | `phase2.base_share(out)`, `phase2.clean_labels(y)` | Label hygiene: base_share NaN when site energy is ~0 (was -8e12), else clipped to [-0.2, 1]; balance points clipped to [-25, 30] C. |
| FaultFinder | `phase2.log_commands(trend_healthy, trend_faulted, fault, rng)` | Trend logs record commands: stuck actuators log the healthy command (valves wound up to 0 / 100 %), leaks log the signal minus the leak; p 0.2 keeps the position. |
| FaultFinder | `phase2.installed_damper(trend, rng)`, `phase2.damper_command(frac, rng)` | Installed damper characteristic: OA fraction = a + (b - a) cmd^g, so 100 % command mixes 75-100 % outdoor air (dev: 81-90 %). |
| FaultFinder | `phase2.static_offsets(trend, rng, large=True)` | Static sensor disagreement: sd SAT 0.5 K (tail 1.2), MAT 0.8, RAT 0.4, OAT 0.6, ZAT 0.9 K (v3 sizes with `large=False`). |
| FaultFinder | `controller=phase2.OperatorJitter(seed, tz)` | Operator start/stop jitter (sd 2 / 3 steps) and 1-3 h off-blocks (p 0.15) through the controller hook. |
| ThermoPilot | `stock.recs_to_spec(row, seed)`, `stock.recs_comfort_band(row)` | RECS 2020 respondent twin (nearest ResStock prior dwelling, respondent's own area, vintage, stories, occupants) and its comfort band from the respondent's own thermostat settings. |
| ThermoPilot | `bd.tariff_arrays(tariff, times, tz)`, `bd.bill_arrays`, `bd.tariff_prices(key, year, tz, utc=True)` | DST-safe prices: each step takes its own local month / hour / weekday. `tariff_prices` with `tz` was already right on every hour; without `tz` its naive index cannot be localized and reads as UTC in `simulate(prices=)`. `utc=True` gives the UTC frame. |
| ThermoPilot | `bd.rtp_prices(times)` | ERCOT HB_HOUSTON real-time prices + 0.10 $/kWh, weather year Y paired with prices of Y + 2 (728 days, weekdays aligned). Needs GridData. |

Not ported, and why: ThermoPilot's numba DP kernel and its two-pass demand-peak oracle are the model's controller
(its control model of the house, its 36-action menu, its comfort weight), not building physics, and numba is not a
BuildingSim dependency. They stay in `models/thermopilot/training/tp/` (`dp_fast.py`, `mpc.py`,
`episode.oracle_floor`); the billing pieces they need (`tariff_arrays`, `bill_arrays`) are in BuildingData.

## The physics

**Weather** (`weather.py`).
- Reads EPW and the contract format.
- Sun position at interval midpoints (pvlib NREL SPA).
- Perez transposition onto the four facades (rotated by the building orientation), the roof and the PV plane.
- Sky temperature from horizontal infrared, or Clark & Allen (1978) with cloud cover.
- Kusuda & Achenbach ground temperature at 1 m and 3 m.
- Burch & Christensen (2007) mains water temperature.
- Design temperatures (0.4 % and 99.6 %) from the run's own weather.

**Zone** (`zone.py`, ISO 13790:2008 5R1C).
- Nodes: air, star/surface, one mass node.
- Time scheme: Crank–Nicolson on the mass node, generalised to any step; implicit Euler when the mass time constant is under two steps.
- Windows conduct to outdoor air.
- Walls and roof see the sol-air temperature: a/h_o I − ε F_sky σ (T⁴ − T_sky⁴)/h_o, with h_o = 8 + 4 v_local. A vented attic gets 35 % of the roof's sol-air excess.
- Floors are handled by type:
  - slabs: F-factor × perimeter to the 1 m ground temperature;
  - conditioned basements: below-grade walls with ~1.5 m²K/W of soil path, plus the floor, to 3 m ground;
  - crawlspaces, unheated basements and raised floors: steady-state buffer spaces (ISO 13789: H = H_iu H_ue / (H_iu + H_ue), boundary weighted between outdoor air and ground).
- Solar through windows: per facade, beam × SHGC × the Fresnel/absorption angular modifier of pvlib (`iam.physical`, raised to the number of panes) × overhang sunlit fraction, plus sky and ground diffuse with their hemispherical modifiers, × frame/surroundings shading. Residential homes add seasonal interior shading (0.7 summer / 0.85 winter, as ResStock).
- Infiltration: LBL model (ASHRAE Fundamentals ch. 16) Q = A_L √(C_s|ΔT| + C_w U²), with the leakage area from ACH50 (n = 0.65) or from the spec's ACH at 15 K and 4 m/s. Commercial infiltration falls to 25 % while the fans run. Residential exhaust ventilation adds in quadrature.
- Natural ventilation (residential, cooling season, dry outdoor air) is an outdoor-air conductance opened only as far as needed to hold mid-deadband.
- Moisture: one-node implicit balance with 5× air capacitance; latent from people, cooking and outdoor air; condensation when supersaturated.
- The thermostat is solved exactly. The zone gives T_air = a + b q, and every system stage gives q = g (T_s − T) + q0, so the setpoint, the stage and its runtime fraction follow without iteration.
- Multi-node option (`zone_model='multinode'`, `zone_multinode.py`): walls, roof and floor as layered RC ladders (explicit layers, or default stacks that keep the spec U), an air node with furniture capacity, an internal-mass node, and implicit Euler with Sherman–Morrison for the varying air exchange.

**HVAC** (`systems.py`, `plant.py`, `curves.py`).
- DX cooling (EnergyPlus `Coil:Cooling:DX:SingleSpeed` formulation):
  - capacity and EIR biquadratics in entering wet bulb and outdoor dry bulb, plus flow-fraction curves (EnergyPlus example curves `HPACCoolCapFT/EIRFT/CapFFF/EIRFFF`);
  - part-load PLF 0.85 + 0.15 PLR;
  - bypass-factor / apparatus-dew-point latent model, with A0 set from the rated SHR;
  - SEER/EER → COP excluding the fan;
  - two-stage and variable-speed part-load curves (documented assumptions).
- Heat-pump heating:
  - EnergyPlus `HPACHeatCapFT/EIRFT` cubic curves;
  - timed reverse-cycle defrost after the EnergyPlus Engineering Reference (frost multipliers 0.909 − 107.33 Δω and 0.90 − 36.45 Δω, defrost fraction 0.0583);
  - cold-climate units at half the capacity slope;
  - lockout temperature and backup strips.
- Furnaces (AFUE) and boilers:
  - non-condensing: EnergyPlus part-load curve;
  - condensing: efficiency vs return water with hot-water reset.
- Chillers (DOE-2 EIR model, EnergyPlus "DOE-2 Centrifugal" curves):
  - false loading below PLR 0.15;
  - air-cooled curves re-centred at 35 °C (assumption);
  - cooling tower by the Braun, Klein & Mitchell (1989) effectiveness model, with variable-speed fan (cube law) holding max(21 °C, T_wb + 3 K).
- Fans: VAV part-load curve of ASHRAE 90.1 Appendix G; the fan heat goes to the air stream. Pumps: 19 W/gpm design with the 90.1 variable-speed curve; pump heat goes into the water.
- Air side:
  - Unitary systems: continuous fan while occupied (commercial) or cycling (residential, PTAC). Mixing box with minimum outdoor air, differential dry-bulb economizer with high limit, and the return air coupled implicitly to the zone air.
  - VAV: supply-air temperature with outdoor reset. A heating mode (economizer and cooling coil locked out when the zone calls for heat). Coils in air-flow order (mixed → preheat → cooling → fan). Boxes run from minimum flow to dual-maximum heating flow (≤ 50 %) with reheat to 35 °C, plus perimeter radiation for the rest.
- DOAS for fan coils and radiant systems. Residential exhaust fans per ASHRAE 62.2.
- District steam, hot water and chilled water: meters read delivered energy.
- Ground-source heat pump: lumped loop, C dT/dt = Q − UA (T − T_ground).
- DHW (vectorised): storage tank with standby UA; gas, electric, heat-pump water heater (extracts heat from the zone) or district.
- Sizing: ASHRAE-style peak loads from the run's weather; heating ×1.25, cooling ×1.15 (jittered by the generator); 400 cfm/ton for DX, 11 K supply ΔT for VAV.

**On-site** (`simulate.py`).
- PV: pvlib PVWatts DC (−0.37 %/K, Faiman cell temperature), 14 % losses, inverter clipping at DC/AC 1.2.
- Battery: SOC model with 95 % charge and discharge efficiency and a minimum SOC. Modes: self-consumption, TOU (by price percentile) or commanded by the controller.

**Comfort** (`comfort.py`): Fanger PMV/PPD (ISO 7730, vectorised; air speed 0.1 m/s; clothing 0.5–1.0 clo by the 7-day running mean outdoor temperature; met by use) and degree-hours outside 20–26 °C operative.

**Generator** (`generator.py`, `stock.py`, priors in `refdata/priors/`).
- Each building bootstraps one real combination of characteristics, then jitters continuous values (U ±10 %, infiltration ±20 %, lighting and plug ±15 %, people ±20 %, DHW ±25 %, setpoints ±0.5 K, hours ±0.5 h, sizing factors 1.1–1.5).
- Sources by use type:
  - Houses and apartments: NREL ResStock 2025 R1, 120,000 dwellings drawn with stock weights. Mapped: R-values, windows, leakage to outside, foundation and attic types, setpoints and offsets, HVAC type and efficiency, water heater, appliance fuels, PV. Multi-family units become whole buildings.
  - Office, education, lodging, retail, healthcare, warehouse, food: NREL ComStock 2025 R3, 149,272 models. Mapped: type, size, stories, aspect ratio, rotation, energy code of construction / roof / HVAC (envelope and efficiency levels), wall construction, window type, WWR category, airtightness, HVAC system type, fuels, operating hours, setpoints and setbacks, lighting generation.
  - Assembly, lab, other: CBECS 2018 respondents of the frozen *prior* split. Climate is drawn from the stock's distribution.
- Residential plug, appliance and lighting electricity: E = 2,967 + 4.78 A + 500 N kWh/yr (least squares on 147 ResStock dwellings), spread on the ResStock mean hourly shape.
- Autosizing makes every generated building physically consistent with its loads.

## Verification (numbers in `shared/building/checks/*.json`)

| check (file) | what | result |
|---|---|---|
| `analytic.json` | Steady conduction | vs the 5R1C closed form 1.8e-9 (naive ΣUA −6.9 %, see LIMITS). |
| `analytic.json` | RC step response | vs closed form (τ = 37 h): max 0.0035 K at 1 h, 0.0033 K at 15 min. |
| `analytic.json` | Degree-day limit | zero-mass zone vs hourly steady-state network sum, ratio 0.999996; 165 kJ/m²K mass: 0.954. |
| `bestest.json` | ASHRAE 140 BESTEST 600–650, 900–950 and free-float (Denver-Stapleton TMY, the source of DRYCOLD) | 29 of 34 values inside the 1995 reference ranges. Annual heating/cooling all 20 inside; the misses are 900 peak cooling (2.80 vs 2.89–3.93 kW) and 4 free-float minimum temperatures (0.3–1.6 K too cold). |
| `multinode.json` | Steady state (multi-node zone) | 1.7e-9. |
| `multinode.json` | ISO 13786 periodic wall (0.2 m concrete) | amplitude −1.6 %, lag −0.01 h in the frequency domain; time domain −4.7 % (15 min) / −13 % (1 h, implicit Euler damping). |
| `multinode.json` | BESTEST 600/900 (multi-node, layers) | annual heating/cooling all inside; peaks 6–12 % above range. |
| `components.json` | Psychrometrics vs CoolProp | W 0.55 % (enhancement factor), T_wb p95 0.02 K (max 0.40 K at the ice/water branch at 0 °C), T_dew 0.012 K. |
| `components.json` | Comfort vs pythermalcomfort | PMV max error 0.0025. |
| `components.json` | Curves and equipment | all curves 1.000 ± 0.003 at rating; DX reproduces rated SHR 0.750 and COP 3.499; chiller COP at rating 6.004; tower design point closes within 0.9 K. |
| `faults.json` | 26 fault cases (RTU, VAV, heat pump) | all 26 show the expected signature (direction of dampers, mixed/supply temperatures, energy). Ledger closure ≤ 1.7e-9 with faults. |
| `controller.json` | Controller and step-wise API | TOU pre-cooling: peak-period energy −51 %, cost −19 %. Step-wise = callable exactly. get/set_state rollback exact. Demand limit 30 %: peak 4.6 → 3.3 kW. PV+battery ledger closes (2.5e-16). |
| `resstock_twins.json` | 126 ResStock dwellings (simulator vs simulator, same AMY2018 weather) | median sim/ResStock: delivered heating load 1.29 (IQR 1.04–1.51), cooling load 1.22, electricity 1.06 (IQR 0.87–1.26), on-site fuel 1.07, site energy 1.10 (IQR 0.98–1.26, 72 % within ±25 %). Hourly r heating load 0.84, cooling 0.77. Daily r site 0.92. Indoor T −0.15 K. |
| `comstock_twins.json` | 45 ComStock models (3 per building type), a held-out draw (seed 12) after the seed-11 corrections (simulator vs simulator) | median sim/ComStock: site energy 0.86 (IQR 0.68–1.16; aggregate 0.97), electricity 0.83 (aggregate 1.00), heating 0.75 (IQR 0.17–1.01), cooling 0.83, fans 0.80, interior lighting 1.13, water heating 1.07. Hourly r electricity 0.81; daily r electricity 0.89, site 0.89. |
| `bdg2_dev_meters.json` | REAL meters, 52 BDG2 dev buildings, 2016, generic twins from metadata | electricity annual ratio median 0.98 (IQR 0.50–1.42), weekday profile r 0.64, daily r 0.34. Chilled water 0.28×, steam 0.29×, hot water 0.15× (real campus buildings use ~3× more district energy than default twins). Daily r for district meters 0.70–0.88. |
| `generator.json` | 156 generated buildings, one year each | all ledgers close (≤ 8e-9); unmet hours median 0 (p90 27). Site EUI median vs real CBECS/RECS prior medians: office 0.91, warehouse 0.98, retail 1.11, multi-family 1.12, other 1.17, assembly 1.26, single-family 1.40, food 1.53, education 1.81, healthcare 1.91, lab 2.0, lodging 0.66. |
| `energyplus_reference.json` | EnergyPlus on DOE prototypes | **blocked**: no EnergyPlus wheel on PyPI; the openstudio 3.11 wheel has no engine; GitHub blocked. |

Calibrations, all on stock models, never on BDG2 or survey exam data:
- Residential plug model, interior shading and natural ventilation: from ResStock.
- Commercial lighting ×0.66, DHW ×0.6, exterior lighting ×1.8, night base +0.05/+0.1: from the ComStock seed-11 sample. A disjoint seed-12 sample is the reported number.
- Residential infiltration factor: kept at 1.0 (`data/calibration.json`).

## Speed (single thread, shared 2-core Xeon 2.1 GHz; `checks/speed.json`)

| task | time per building-year |
|---|---|
| house, split AC + furnace, 15 min / 1 h | 0.68 s / 0.18 s (≈ 4.6×10⁷ × real time at 15 min) |
| house, heat pump, 15 min | 0.71 s |
| office RTU, 15 min / 1 h | 0.62 s / 0.16 s |
| large office VAV + water-cooled chiller + tower, 15 min | 0.74 s |
| hotel PTAC / school fan coils, 15 min | 0.71 s / 0.38 s |
| 40 generated buildings, mixed, 15 min, incl. weather prep | 1.0 s mean |
| multi-node zone (12-23 nodes), 15 min | ~1.7-2.0 s |
| first run on a new weather frame (pvlib solar, Perez) | + 0.3 s (cached afterwards) |

The 1,000× real-time target (a building-year in ≤ 8.8 h) is exceeded by about 4×10⁴. A generated training bank of
20,000 building-years at 15 min takes ~5.5 h on one core; at 1 h, ~1.5 h.

## Files

```
buildingsim/   the package
  __init__.py simulate.py outputs.py ledger.py                 API, time loop, outputs, ledgers
  weather.py psychro.py comfort.py                              weather + solar (pvlib), psychrometrics, PMV/PPD
  spec.py schedules.py zone.py zone_multinode.py                contract defaults, schedules, 5R1C and multi-node zones
  curves.py systems.py plant.py sizing.py faults.py             equipment curves, HVAC systems, plant, sizing, faults
  generator.py stock.py data/calibration.json                   generator, ResStock/ComStock/CBECS/RECS mapping
  phase2.py holidays.py                                         opt-in Phase 2 wrappers, country holiday calendars
refdata/       weather/epw (17 TMYx US cities, one per IECC zone), weather/bestest (Denver-Stapleton TMY),
               priors/ (resstock, comstock, cbecs and recs prior tables), stock_weather/ (OEDI AMY2018 counties)
tools/         build_priors.py, fetch_stock_weather.py, calibrate_resstock.py (scan kept for the record)
demo.py        two buildings x two weeks, ledgers, a fault, a controller -> demo_weeks.png
.refenv/       isolated venv with CoolProp and pythermalcomfort (checks only)
../checks/     check_*.py, their JSONs, run_all.sh (re-runs all, ~30 min), out/ per-unit results and logs
```

## Sources and licences

Data:
- NREL End-Use Load Profiles for the U.S. Building Stock (ResStock 2025 R1, ComStock 2025 R3, AMY2018 county weather), OEDI, CC-BY 4.0; via agent B's processed copies and the OEDI S3 bucket.
- EIA RECS 2020 and CBECS 2018 microdata (public domain), prior split of agent B.
- Building Data Genome 2 (Miller et al. 2020, CC-BY-SA 4.0), dev buildings via agent B.
- TMYx and the Denver-Stapleton TMY from climate.onebuilding.org (derived from NOAA ISD / ERA5).

Methods and references:
- ISO 13790:2008 (5R1C); ISO 13789 (buffer spaces); ISO 13786 (periodic transmittance); ISO 7730 (PMV).
- ASHRAE Handbook Fundamentals 2017 (psychrometrics ch. 1, infiltration ch. 16, loads ch. 18); ASHRAE 90.1 Appendix G (fan and pump curves); ASHRAE 62.1/62.2 (ventilation); ANSI/ASHRAE 140 (BESTEST).
- EnergyPlus Engineering Reference and example-file curves (DX coils, heat pumps, defrost, DOE-2 chillers, boilers; US DOE, BSD-3).
- Braun, Klein & Mitchell (1989, ASHRAE Trans.), cooling towers.
- Kim & Braun (2012, Int. J. Refrig. 35), refrigerant charge.
- Sherman & Grimsrud (1980), LBL infiltration.
- Kusuda & Achenbach (1965), ground temperature; Burch & Christensen (2007), mains water; Clark & Allen (1978), sky emissivity.
- Building America House Simulation Protocols (Wilson et al. 2014); DOE commercial prototype buildings (PNNL).

Libraries: pvlib (BSD-3) for solar, transposition, IAM and PVWatts. Checks only: CoolProp (MIT) and pythermalcomfort (MIT).
