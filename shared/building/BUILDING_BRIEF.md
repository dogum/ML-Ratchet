# BuildingSim — shared plan (read with docs/agent-brief.md)

> Kept as written on 16-17 Nov 2026 in the cloud project where the buildings work was built, with the owner's name made neutral and the sandbox paths rewritten to repository paths (the sandbox's `bldg/` is `shared/building/`; the model folders are `models/loadcast`, `models/buildingdetective`, `models/faultfinder` and `models/thermopilot`).

The owner's third big-system project for ML Ratchet (github.com/dogum/ML-Ratchet, "Trained on physics. Graded on reality."),
built the same way as RoadSim (vehicles) and GridSim (power grid).

## Phases
- **Phase 1 (now, two agents in parallel):**
  - **A — BuildingSim:** the shared, verified building-energy simulator.
  - **B — BuildingData:** the real-data package, with frozen dev/exam splits and classical baselines.
- **Phase 2 (later):** several ratchet models on top, in parallel. Candidates:
  - **LoadCast:** a building's hourly energy (electricity, gas/steam/chilled water) from weather and calendar.
  - **BuildingDetective:** infer from meter + weather alone:
    - envelope conductance UA and thermal mass/time constant
    - balance points
    - HVAC type: heat pump vs resistance vs gas; chiller type
    - occupancy schedule, primary use, floor area
  - **FaultFinder:** HVAC fault detection and diagnosis on air handlers and rooftop units.
  - **ThermoPilot:** a learned thermostat / heat-pump / pre-cooling controller vs rule and MPC baselines, on real weather and real tariffs.
- **Then:** a `HANDOFF.md` listing what is missing.

## Folders
Root: `shared/building/`.
- **A works in:**
  - `shared/building/buildingsim/` (Python package `buildingsim`)
  - `shared/building/checks/`
- **B works in:**
  - `shared/building/data/` (`raw/`, `proc/`)
  - `shared/building/buildingdata/` (Python package `buildingdata`)
- Neither touches the other's folders. Each keeps a `STATUS.md` in its own folder.
- Earlier work worth reading, read-only:
  - DemandCast's building-stock model: `models/demandcast/training/buildings.py`
  - GridData weather loaders: `shared/grid/griddata/`
  - SunDoctor's pvlib use: `models/sundoctor/` (code only; its data was deleted)

## Shared contract

### Time series
Hourly or sub-hourly UTC-indexed pandas DataFrames / parquet.

| Quantity | Unit |
|---|---|
| Power | kW |
| Energy | kWh |
| Temperature | °C |
| Humidity | relative humidity (%) or humidity ratio (kg/kg) |
| Irradiance | W/m² |
| Wind speed | m/s |
| Pressure | Pa |

### Weather
One frame per site, with columns:
- `T_out`, `RH`, `dewpoint`
- `GHI`, `DNI`, `DHI`
- `wind_speed`, `wind_dir`
- `pressure`
- `cloud` (optional)

Plus a meta json: `lat`, `lon`, `elev`, `tz`, `source`.

### Building spec (dict/JSON)
B fills it from real metadata; A's simulator consumes it and A's generator samples it.
- **Identity and site:**
  - `id`, `site_id`, `lat`, `lon`, `climate_zone` (IECC)
  - `primary_use`: `office` | `education` | `lodging` | `retail` | `healthcare` | `assembly` | `warehouse` | `residential_single` | `residential_multi` | `food` | `lab` | `other`
  - `floor_area_m2`, `n_floors`, `year_built`
- **Envelope:**
  - `wall_U`, `roof_U`, `window_U` (W/m²K), `window_wwr`, `SHGC`
  - `infiltration_ach`
  - `thermal_mass` (light | medium | heavy, or kJ/m²K)
- **Internal loads and schedules:**
  - `internal_gains`: people, lighting, plug, each W/m² with schedules
  - `occupancy_schedule` id, `setpoints`: {heat, cool, setback heat, setback cool, schedule}
  - `ventilation`: {L/s per person, L/s per m², economizer, ERV}
- **HVAC:**
  - `system`: `ptac` | `packaged_rtu` | `split_ac_furnace` | `heat_pump_air` | `heat_pump_ground` | `vav_chiller_boiler` | `fan_coil` | `radiant` | `district_steam_chw` | `baseboard_resistance` | `none`
  - Capacities, efficiencies (COP/EER/AFUE curves vs temperature and part load), fan power, pump power, `dhw` system.
- **Other:**
  - `pv` (kW DC, tilt, azimuth), `battery` (kWh, kW)
  - `meters`: which end uses roll into which meter (electricity, gas, steam, chilled water, hot water)

Unknown fields are None, each with a documented default source.

### Simulator API (A builds; B's baselines and Phase 2 call it)
```python
from buildingsim import simulate, generate_buildings, list_systems, ledger
out = simulate(spec, weather, start, end, dt='15min'|'1h', controller=None, faults=None, prices=None, init=None, rng=None)
```
`out` contains:
- **Zone conditions:** zone temps (one zone or several), humidity, comfort metrics (PMV/PPD or degree-hours outside band)
- **HVAC operation:** HVAC loads (sensible and latent), equipment operation (on/off, part load, COP)
- **Energy:** energy by end use and by meter (kWh), peak kW
- **Ledger:** an energy and heat ledger that must close:
  - conduction, infiltration, ventilation
  - solar gains, internal gains
  - storage in mass
  - HVAC heat added/removed, equipment input energy

`controller` is a callable policy (thermostat setpoints / heat-pump commands) for ThermoPilot.

`faults` injects HVAC faults for FaultFinder: stuck damper, leaking valve, fouled coil, low refrigerant charge, sensor bias, economizer failure, schedule override, …

### Physics A must implement
- **Thermal model:** zone heat balance with envelope conduction (RC network: at least a 3R2C/5R1C-class model, with an option for multi-node walls), solar gains through windows with incidence angle and shading, opaque sol-air temperature, infiltration (wind and stack driven), ventilation and economizer, internal gains with schedules, and moisture balance (latent loads).
- **HVAC:**
  - **Equipment models:** heat pumps and AC with capacity and COP as functions of outdoor and indoor temperature and part load, plus defrost; gas furnaces and boilers; chillers (EIR curves) with cooling towers; VAV/AHU with fans (fan laws), coils and damper positions; DHW.
  - **Behaviour:** controls and setback recovery.
  - **Sources and verification:** curves from DOE/EnergyPlus reference data and ASHRAE literature; use CoolProp and pythermalcomfort from PyPI if helpful.
- **On-site systems:** PV (pvlib) and simple batteries.
- **Weather:** solar position and irradiance processing (pvlib).

### Verification A must record (`shared/building/checks/*.json`)
- **Analytic checks:**
  - steady-state conduction
  - RC step response vs closed form
  - ledger closure
  - degree-day limit
- **ASHRAE Standard 140 (BESTEST) style cases.** EnergyPlus GitHub is blocked here. If a PyPI EnergyPlus wheel works, use it as a reference on a few DOE prototype buildings. Otherwise compare against published BESTEST ranges.
- **Against NREL ResStock/ComStock:**
  - Simulate sampled building types/climates and compare annual and hourly end-use profiles.
  - Data source: OEDI S3, `https://oedi-data-lake.s3.amazonaws.com/?prefix=nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/&delimiter=/` lists releases.
  - These are EnergyPlus-simulated, so this is a simulator-vs-simulator check, not reality.
- **Against real buildings:** the hourly shape and annual totals vs real meters (Building Data Genome 2 and others from B, or your own copy).
- **Speed:** aim for ≥ 1,000× real time for a single-zone building-year at 15 min, single thread.

### Real data B must assemble (coordinator checked reachability on 16 Nov)
- **Building Data Genome 2 (BDG2).**
  - 1,636 non-residential buildings, 2 years (2016–17).
  - Hourly meters: electricity, chilled water, steam, hot water, gas, water, irrigation, solar.
  - Building metadata: primary use, floor area, year built, …
  - Site weather.
  - Location: GitHub `buds-lab/building-data-genome-project-2`. LFS files come via `media.githubusercontent.com/media/...` (200); raw via `raw.githubusercontent.com`. Licence: check (it was CC-BY-4.0 / MIT).
- **NREL ResStock / ComStock** on OEDI S3 (public): building characteristics + 15-min end-use profiles for hundreds of thousands of modelled buildings. Use them as calibration priors, generator priors and the simulator-vs-simulator check, NOT as real truth.
- **EIA RECS 2020** (homes) and **CBECS 2018** (commercial) microdata, eia.gov reachable: real building characteristics + annual energy by fuel. Use them as an annual-level real exam for BuildingDetective (characteristics vs consumption) and as generator priors.
- **LBNL HVAC fault data** (`faultdetection.lbl.gov` reachable, plus OpenEI submissions): AHU/RTU/chiller datasets with labelled faults, some experimental/real. Identify which are real measurements vs simulated, and grade FaultFinder only on the real/experimental ones.
- **Other public real datasets to look for:**
  - Zenodo / Dataverse / figshare: smart thermostat data, residential smart meters with weather, heat-pump field monitoring, e.g. UK RHPP heat-pump field trial or similar open data.
  - Building-level submetering.
  - Use only documented public downloads. If gated (e.g. ecobee Donate Your Data needs a request; Pecan Street is paid), note it for the owner.
- **Weather:**
  - For BDG2 sites: the dataset's own weather + NASA POWER irradiance.
  - Generally: TMY3/TMYx if a documented source is reachable, NASA POWER, Iowa Mesonet ASOS.
- **Tariffs for ThermoPilot:**
  - Real TOU tariff structures, e.g. URDB on OpenEI if reachable.
  - Real wholesale prices: reuse GridData's ERCOT/CAISO prices via `shared/grid/griddata/` (read-only).

### Splits (B decides once and freezes in `shared/building/data/proc/splits.json`)
- **Unit:** the building (or site), never the hour.
  - **dev:** a small stratified set of buildings (by use and climate) for calibration.
  - **exam:** the rest.
  - Also time: e.g. BDG2 2016 as input year and 2017 as forecast year, where a model needs history. Document precisely.
- **RECS/CBECS:** split by respondent, stratified.
- **Fault data:** split by dataset, unit or test run so that no exam run leaks.
- **Simulated training** may use real inputs (weather) freely, and real buildings' metadata only from non-exam buildings.

### Classical baselines (B computes exam numbers now)
- **Hourly load:**
  - persistence and same-hour-last-week
  - ASHRAE Guideline 14 change-point regression (3P/5P) fitted on each building's own training period
  - LightGBM-style regression is optional as a "data-driven" reference, clearly labelled
- **Detective:**
  - PRISM-style change-point balance point and slope (heating/cooling sensitivity)
  - median by use type
- **Faults:** a rule-based AHU fault detector (APAR, NIST rules), plus a majority-class baseline.
- **Thermostat:**
  - fixed setpoint
  - standard setback
  - a simple MPC / rule pre-cool on the same tariffs

## Shared-machine rules
As in AGENT_BRIEF.md:
- 2 cores and 7 GB RAM. Disk: ~13 GB free; you may delete only your own files.
- One thread each; nice heavy jobs.
- Never `pkill -f`; kill only your own PIDs.
- Make work resumable.
- ≤ 5 GB disk each this phase. Stream big files; keep compact parquet.
- Use a subfolder of your own project dir for screenshots and temp files, never generic names in a shared scratchpad.
- Nothing written to the claude.ai project in Phase 1.
