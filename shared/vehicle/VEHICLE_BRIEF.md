# RoadSim / vehicle project: shared plan (read with /home/claude/AGENT_BRIEF.md)

Greg's next collection project (repo: github.com/dogum/ML-Ratchet, physics first, graded on reality). It is both a
physics problem and a systems problem: whole vehicles at subsystem level, every architecture, many hybrid control
strategies, real roads, real weather, light-duty AND heavy-duty. It is staged:

- **Phase 1 (now, two agents in parallel):**
  - **A: RoadSim**, the shared simulator package, verified.
  - **B: RoadData**, the real-data package with frozen dev/exam splits and the classical baselines' inputs.
- **Phase 2 (later, three agents in parallel, each a full ratchet project):**
  1. **EnergyNet:** predict trip fuel/kWh for any vehicle + route + weather, with error bars.
  2. **Detective:** from a real trip log alone, infer architecture, mass, aero/rolling, hybrid strategy, and the energy ledger.
  3. **Controller:** a learned hybrid energy-management policy vs rule-based / ECMS / DP-optimal, run on real routes.

Root: `/home/claude/road/`.
- A works in `road/roadsim/` (Python package `roadsim`) and `road/checks/`.
- B works in `road/data/` (raw downloads under `road/data/raw/`, processed under `road/data/proc/`) and `road/realdata/` (Python package `realdata`).
- Neither touches the other's folders. Keep a `STATUS.md` in your own folder.

## The shared contract (both agents must follow it exactly, so Phase 2 can plug them together)

### Trip (inputs a simulator needs; what real data provides)
One trip = one `.npz` + one `.json` meta. Arrays at 1 Hz, all SI, all float32:
- `t` [s], `v` [m/s], `grade` [rise/run, signed, along travel], `elev` [m]
- `lat`, `lon` [deg]; `heading` [deg from north, travel direction]
- `T_amb` [°C]; `wind_speed` [m/s]; `wind_dir` [deg, direction the wind comes FROM]
- `rho_air` [kg/m³] (from T and pressure if known, else ISA at elevation)
- `surface` [int code: 0 = dry asphalt, 1 = concrete, 2 = wet, 3 = gravel, 4 = snow]

Measured outputs, where real data has them (NaN-filled arrays if absent):
- `fuel_rate` [g/s]; convert L/h with density 745 g/L for gasoline and 832 g/L for diesel, and record which in meta
- `engine_rpm`, `batt_current` [A, + = discharge], `batt_voltage` [V], `soc` [0–1]
- `aux_power` [W], i.e. AC + heater if logged

Meta JSON fields:
- `trip_id`, `vehicle_id`, `source`, `date`, `distance_m`, `duration_s`, `T_amb_mean`, `notes`
- `split`: `"train_noise"` | `"dev"` | `"exam"`

### Vehicle spec (JSON / dict)
- `id`, `class`: `car` | `suv` | `pickup` | `van` | `bus` | `truck_md` | `truck_hd`
- `architecture`: `conv` | `hev_parallel_p2` | `hev_series` | `hev_powersplit` | `phev_powersplit` | `phev_series` (EREV) | `bev`
- `mass_kg` (test/curb + payload), `payload_kg`, `CdA_m2`, `Crr` (at 20 °C on dry asphalt) and `Crr_speed_coef`
- `road_load_ABC`: optional EPA target coefficients A [lbf], B [lbf/mph], C [lbf/mph²], plus SI conversions
- `wheel_radius_m`, `final_drive`, `driveline_eff`
- `engine`:
  - `fuel` (`gasoline` | `diesel` | `none`), `displacement_L`, `max_power_kW`
  - `bsfc_map` id or a parametric description, `idle_rpm`, `max_rpm`, `warmup_model`
- `transmission`: type (`AT` | `DCT` | `CVT` | `AMT` | `single` | `eCVT`), gear ratios, shift schedule id, efficiency model
- `motors`: list of {role (`P2` | `MG1` | `MG2` | `traction` | `generator`), `max_power_kW`, `max_torque_Nm`, `max_rpm`, efficiency map id}
- `battery`:
  - `chemistry`, `capacity_kWh`, `nominal_V`, `R0_ohm` (and R vs SOC/T), `soc_window`
  - `thermal`: C [J/K], hA [W/K], heater/cooler
- `aux`: `base_W`, `HVAC` model (heat pump vs resistive for EV; engine waste heat for ICE)
- `regen`: max decel share, blend strategy, min speed
- `control`: {`strategy` id, params}

### Simulator API (A builds, B's baselines and Phase 2 call)
```python
from roadsim import simulate, load_vehicle, list_strategies
out = simulate(vehicle_spec, trip_inputs, strategy=None, dt=1.0, init_soc=None, init_temps=None, rng=None)
```
`out` is a dict of 1 Hz arrays:
- road loads, each in N: aero (with the wind component along heading), rolling (`Crr` with temperature, speed and surface effects), grade, inertia
- wheel power; engine speed, torque, power and fuel g/s; motor(s) speed, torque, power and loss
- gear; battery current, voltage, SOC and temperature; aux power; friction-brake power; regen recovered

`out['ledger']` is the **energy ledger** for the trip, in kJ:
- fuel chemical energy
- engine losses, motor/inverter losses, transmission losses, battery I²R losses
- aero, rolling and net grade work
- friction braking
- aux / HVAC
- change in battery stored energy, change in kinetic energy

It must close to < 0.5% (a check A runs on every simulation).

The simulator must be forward-facing, with a driver PI controller that tracks the speed trace. It must be fast:
- target ≥ 200× real time for a 1 Hz car trip in numpy, single thread
- vectorise across a batch of vehicles if helpful

Phase 2 needs millions of simulated trip-seconds.

### Control strategies to implement (A)
- **Conventional:** shift schedules by load/speed; idle stop optional.
- **HEV:**
  - rule-based thermostat (series)
  - power-follower (series)
  - parallel electric-assist / load-levelling rules
  - Prius-style power-split rules with an engine optimal-operating line
  - ECMS with constant equivalence factor
  - adaptive ECMS (SOC-feedback)
  - DP-optimal on a known cycle (offline benchmark, coarse grid; DP is fine being slow)
  - optionally MPC with a short horizon
- **PHEV:** charge-depleting then charge-sustaining, and blended.
- **BEV:** regen blending variants, one-pedal.

### Physics checks A must record (`road/checks/*.json`)
- **Coast-down:** reproduce EPA road-load ABC for ≥ 20 real vehicles from their spec (mass, CdA, Crr) within the test's spread.
- **Cycles:**
  - simulate UDDS / HWFET / US06 for ≥ 20 EPA test vehicles (conventional, HEV, BEV) and compare with their EPA test results (testcar list FE / kWh)
  - report median and spread of the % error
  - **cross-check against NREL FASTSim** (`pip install fastsim`, PyPI is reachable; the GitHub page is blocked) on the same vehicles and cycles
- **Ledger closure** on every check run.
- **Heavy duty:** a class-8 tractor on the HHDDT / CARB / EPA GEM cycles, and a transit bus (diesel, hybrid, electric) on a bus cycle (e.g. Manhattan, OCTA, CBD), against published fuel economy / kWh per mile ranges, with sources cited.

### Real data B must assemble (all reachable endpoints checked by the coordinator)
- **Vehicle Energy Dataset (VED):**
  - Ann Arbor, 2017–18, 383 cars: 264 ICE, 92 HEV, 27 PHEV/EV
  - 1 Hz-ish OBD: speed, fuel rate L/h, rpm, MAF, OAT, AC/heater power, HV battery current/voltage/SOC
  - static specs per vehicle: class, engine, transmission, drive, weight
  - Files: `https://raw.githubusercontent.com/gsoh/VED/master/Data/VED_DynamicData_Part1.7z`, `..._Part2.7z`, `VED_Static_Data_ICE%26HEV.xlsx`, `VED_Static_Data_PHEV%26EV.xlsx`. Licence: Apache 2.0.
  - **Note:** the published GPS was obfuscated (snapped/shifted for privacy; check the paper): judge elevation/grade quality before relying on it.
- **EPA test car list** (`https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy`): yearly CSV/XLSX with test weight, road-load target A/B/C, test results per cycle. 2008–2026 links are on that page.
- **fueleconomy.gov** `vehicles.csv.zip` (all models since 1984): label MPG/kWh, engine, transmission, EV range.
- **EPA dynamometer drive schedules:** UDDS, HWFET, US06, SC03, NYCC, HHDDT, …
- **EPA ALPHA / benchmarking** engine maps (link from `https://www.epa.gov/vehicle-and-fuel-emissions-testing/benchmarking-advanced-low-emission-light-duty-vehicle-technology`) for BSFC maps.
- **Elevation:** USGS EPQS `https://epqs.nationalmap.gov/v1/json?x=lon&y=lat&units=Meters&wkid=4326` (be gentle: cache, sample every ~50–100 m, smooth). OpenStreetMap is reachable for road type/surface if needed; the Overpass API timed out.
- **Weather:** Iowa Environmental Mesonet ASOS (`https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`), e.g. Ann Arbor KARB / Detroit KDTW hourly temperature, wind speed and direction, pressure, for each trip time.
- **Heavy duty:**
  - Find real public data: bus and truck fuel/kWh with drive traces, e.g. NREL Fleet DNA or bus evaluations via data.openei.org (reachable; nrel.gov itself timed out), Zenodo/Mendeley/figshare datasets (Zenodo API works), EPA heavy-duty certification data, CARB.
  - If no labelled real HD trip data is reachable, assemble published per-cycle results (reports) as a small HD exam and say so.
- **Blocked, not to be circumvented:** Argonne D3 dynamometer database (anl.gov returns 403); github.com HTML pages (raw.githubusercontent.com works); nrel.gov.

### Splits (B decides once, freezes, and records in `road/data/proc/splits.json`)
- **Unit:** the vehicle, never the trip. All trips of a vehicle go in one split.
- **dev:** a small fixed-seed sample of vehicles, stratified by architecture (e.g. 8 ICE, 5 HEV, 2 PHEV/EV); only these may be looked at for calibration.
- **exam:** all other labelled vehicles.
- **train_noise:** optional, real trips used only as route/weather inputs (speed and grade traces carry no labels, so using them as simulator inputs is allowed). Phase 2 uses real routes driven by simulated vehicles. Keep exam vehicles' trips OUT of train_noise.
- EPA test-car and fueleconomy.gov vehicles get the same treatment: a dev subset and an exam subset by vehicle, frozen.
- Write the classical baselines' inputs for the exam:
  - EPA label MPG × distance (city/hwy mix by speed)
  - VSP-binned / MOVES-like regression fitted on dev only
  - median L/100 km per class

## Shared-machine rules
As in AGENT_BRIEF.md: 2 cores, one thread each, nice heavy jobs, never kill others' processes, resumable work, ≤ 4 GB disk each for this phase (VED extracted is several GB: process it streaming into compact per-trip npz, then delete the raw CSV if space is short; keep the .7z).
