# Wildfire: shared plan (read with docs/agent-brief.md)

Greg's fifth big-system project for ML Ratchet ("Trained on physics. Graded on reality.", github.com/dogum/ML-Ratchet). It is built the same way as RoadSim, GridSim, BuildingSim and WaterSim; their handoffs show the expected quality (read-only): `shared/water/HANDOFF.md`, `shared/building/HANDOFF.md`.

## Phases
- **Phase 1 (now, two agents in parallel):**
  - **A, FireSim:** the shared, verified fire-behaviour and spread simulator, with a satellite-observation emulator.
  - **B, FireData:** the real-data package, with frozen dev/exam splits and classical baselines.
- **Phase 2 (later):** several ratchet models in parallel on top. Candidates:
  - **FireCast:** next-day (and 12 h) fire spread, i.e. where the burning edge will be, with probability maps, from today's observed fire, landscape and forecast weather.
  - **FireDetective:** infer what is driving a fire from its observed progression: effective dead fuel moisture, wind adjustment, fuel-model mismatch, crowning or spotting regime, suppression influence.
  - **FireSize:** final size and duration (or "will it escape initial attack?") from ignition location, landscape and weather.
  - **FirePilot:** a learned planner, e.g. fuel treatment or containment-line placement, graded in the simulator on real landscapes and weather.
- **Then:** a `HANDOFF.md` of what is missing.

A third agent, Prior Wars (`experiments/priorwars`), runs at the same time on an unrelated method experiment and shares the machine.

## Folders
Root: `shared/fire/`.
- **A** works in `shared/fire/firesim/` (Python package `firesim`) and `shared/fire/checks/`.
- **B** works in `shared/fire/data/` (`raw/`, `proc/`, `scripts/`) and `shared/fire/firedata/` (Python package `firedata`).
- Neither touches the other's folders. Keep a `STATUS.md` in your own folder; screenshots and temp files go in a subfolder of your own folder.

## Shared contract

### Landscape
A raster stack on a projected grid (meters; UTM or CONUS Albers, stated), as numpy arrays plus meta: `crs`, `transform`, `res_m`, `nodata`. Layers, LANDFIRE conventions:

| Layer | Units / encoding |
|---|---|
| `fuel` | Scott & Burgan 40 codes plus 13 Anderson, and non-burnable 91–99 |
| `cc` | canopy cover, % |
| `ch` | canopy height, m |
| `cbh` | canopy base height, m |
| `cbd` | canopy bulk density, kg/m³ |
| `elev` | m |
| `slope` | degrees |
| `aspect` | degrees from N |

Save as `.npz` + json meta, or GeoTIFF readable with rasterio if it is installed.

### Weather
Hourly; per landscape (a single series) or gridded (coarse).

| Column | Units |
|---|---|
| `T` | °C |
| `RH` | % |
| `wind_speed` | m/s at 10 m (state height; 20-ft conversion documented) |
| `wind_dir` | degrees from |
| `precip` | mm |
| `srad` | W/m² (optional) |
| `cloud` | optional |

Fuel moisture, either supplied or derived:
- dead fuels, % (dry mass): `m1h`, `m10h`, `m100h`
- live fuels, %: `mherb`, `mwoody`

### Ignition
Points or a polygon, with UTC time. Alternatively, an initial perimeter or burned mask (to start from an observed fire-day).

### Simulator API (A builds it; B's baselines and Phase 2 call it)
```python
from firesim import simulate, generate_landscape, generate_weather, fuel_models, behave, observe
out = simulate(landscape, weather, ignition, start, end, dt='1h',
               spread='rothermel', crown='van_wagner'|'none', spotting=True|False,
               method='level_set'|'huygens'|'ca', moisture='nelson'|'given',
               barriers=None, suppression=None, params=None, rng=None)
```
`out` contains:
- **Rasters:**
  - `arrival_time` (h since start; NaN if unburned)
  - fireline intensity (kW/m), flame length (m), rate of spread (m/min)
  - fire type (0 unburned, 1 surface, 2 passive crown, 3 active crown)
  - heat per unit area
- **Perimeters:** at requested times.
- **Burned-area series.**
- **Labels:** the true fuel moisture, wind/slope adjustment factors, parameters used, spotting events.
- **An energy/consistency ledger** where meaningful, e.g. area burned vs integrated spread.

`behave(...)`: point fire behaviour, BehavePlus-like:
- Rothermel 1972 with the Albini 1976 corrections and Andrews 2018 formulations
- Byram intensity, flame length
- wind adjustment factor (WAF)
- elliptical length-to-breadth (Anderson 1983) and back/flank spread
- crown fire initiation and spread (Van Wagner 1977; Rothermel 1991 / Cruz)
- maximum spotting distance (Albini)

`observe(out, sensor='viirs'|'modis'|'goes', overpasses=..., rng=...)`: what a satellite would report. Hotspot points or masks at real overpass times, with:
- detection probability vs fire radiative power, pixel size and scan angle
- omission under cloud or smoke
- commission noise
- the 375 m / 1 km / 2 km footprints

This is how simulated fires get "injected into real observation noise". Where possible, use real overpass times and real cloud masks.

`generate_landscape` / `generate_weather`: random but realistic training landscapes and weather.
- Prefer real LANDFIRE tiles and real HRRR/gridMET weather from non-exam places and dates (B's splits).
- Also offer a parametric generator for variety.

**Optional extras:** suppression (containment lines, retardant as reduced ROS, initial attack), barriers (roads, rivers, prior burns), and a fuel-treatment edit.

### Physics A must implement
Use clear, documented, standard formulations with citations:
- **Surface spread:** Rothermel 1972 + Albini 1976; Scott & Burgan 2005 fuel models (all 40 + Anderson 13); dynamic herbaceous curing.
- **Wind and slope:** a wind adjustment factor from canopy and fuel bed; slope factor; vector combination of wind and slope (Rothermel / BehavePlus).
- **Fire shape:** elliptical spread (Anderson 1983 L/B), back and flank rates.
- **Perimeter propagation** over heterogeneous terrain: a level-set / fast-marching arrival-time solver (minimum travel time, MTT, Finney 2002 style) and/or Huygens wavelets (FARSITE). Document anisotropy handling and grid artefacts.
- **Crown fire:** Van Wagner initiation; active/passive crown spread (Rothermel 1991 or Cruz et al. 2005).
- **Spotting:** Albini maximum distance plus a stochastic ember model with ignition probability.
- **Fuel moisture:** Nelson 2000 dead-fuel moisture model (or the NFDRS 2016 equivalent) from hourly weather; live moisture seasonal or observed.
- **Daily and diurnal:** overnight RH recovery and wind slow-down. The diurnal cycle matters for the daily progression exam.
- **Fire danger indices** for baselines: Canadian FWI system (FFMC, DMC, DC, ISI, BUI, FWI) and US NFDRS ERC/BI where feasible.

### Verification A must record (`shared/fire/checks/*.json`)
- **BehavePlus/Rothermel:** reproduce published worked examples and tables, e.g. Andrews 2018 RMRS-GTR-371, Scott & Burgan 2005 fuel-model reference charts, Rothermel 1983 nomograms.
- **Cross-check against an independent open implementation** from PyPI, in an isolated venv in your folder, e.g. `pyretechnics` (SIG's Python fire-behaviour library). github.com/pyregence returned 403 from here; do not route around it, PyPI answered.
- **Analytic checks:**
  - Under uniform fuel, wind and slope, the simulated perimeter must be the predicted ellipse: area, L/B and head rate within tolerance at several grid resolutions and wind directions, including grid-orientation bias.
  - The level-set / MTT arrival time on simple heterogeneous cases (fuel break, two-fuel strip) must match the hand solution.
- **Crown fire:** initiation thresholds vs Van Wagner hand calculations; crown ROS vs Cruz/Rothermel examples.
- **Fuel moisture:** Nelson model vs published examples and vs real NFMD/RAWS 10-h fuel stick observations (B's data) on dev sites.
- **Calibrated replays on real DEV fires** (B's splits): replay each dev fire-day from its observed start perimeter with real fuels and weather. Report the next-day overlap with observed progression: Sørensen/Dice on new burned area, area error, AP if probabilistic. Report both uncalibrated and dev-calibrated (e.g. a per-fuel-group ROS adjustment factor, as practitioners do). These are the classical benchmark Phase 2 must beat. Never use exam fires or exam days for anything.
- **Satellite emulator vs real VIIRS:** on dev fires, compare hotspot counts per overpass vs simulated active area, and omission/commission statistics.
- **Speed:** e.g. a 30 × 30 km landscape at 60–90 m, 24 h of spread, in ≤ ~1–3 s on one core, plus a batch mode for many ignitions or landscapes. numba is already installed; do not upgrade shared numpy/torch.

### Real data B must assemble
The coordinator checked reachability on 18 Nov; all of these answered 200 unless marked.

- **Fire inventory and perimeters:**
  - NIFC WFIGS Interagency perimeters (ArcGIS REST / data-nifc.opendata.arcgis.com; current and history)
  - MTBS burned-area perimeters and severity, 1984+ (the composite shapefile URL on edcintl.cr.usgs.gov answered)
  - FIRMS (firms.modaps.eosdis.nasa.gov). Archive and API downloads may need a free MAP_KEY. If they do, do NOT work around it: record it as help_wanted and use other sources.
  - GOES-16/17/18 ABI fire detection (FDC) on AWS S3 (noaa-goes16 answered)
- **Daily progression truth** (the key exam data). Look first for open, documented datasets on Zenodo (reachable) that already pair fires with daily or hourly progression:
  - WildfireSpreadTS (Gerard et al. 2023, NeurIPS datasets track): daily VIIRS active fire + weather + fuel + topography for ~600 US fires 2018–2021
  - GOFER (GOES-Observed Fire Event Representation, hourly perimeters for California fires)
  - FEDS / Fire Atlas (Chen et al. 2022) if openly hosted
  - anything else with a clear licence

  Otherwise derive daily progression from VIIRS/MODIS hotspots plus final perimeters, and document the method.
- **Landscape:**
  - LANDFIRE fuels and canopy (LFPS service at lfps.usgs.gov answered; landfire.gov answered). Use the version matching the fire year where possible (LF 2016 Remap, 2019, 2020, 2022, 2023), and document the pre-fire vs post-fire fuel issue.
  - DEM: Copernicus DEM 30 m on AWS S3 (answered) or USGS 3DEP.
- **Weather:**
  - HRRR on AWS S3 (noaa-hrrr-bdp-pds answered; 3 km hourly, 2014+). Extract only small subsets around fires; never bulk-download.
  - gridMET daily: northwestknowledge.net/metdata/data answered; the THREDDS server on port 8080 timed out, so use the HTTP files.
  - RAWS via wrcc.dri.edu (answered; raws.dri.edu did not).
  - Synoptic API needs a key: skip it.
- **Fuel moisture:** National Fuel Moisture Database via WFAS (wfas.net answered).
- **Optional:** ignition records (FPA-FOD, Short 2022, USFS RDS archive, which answered) for FireSize; MTBS severity.

### Splits (B decides once and freezes them in `shared/fire/data/proc/splits.json`)
- **Unit:** the fire. All days of a fire are in the same split.
- **Dev:** a small stratified set (~30–50 fires) by region, fuel type, size class and year.
- **Exam:** the rest.
- Also consider a **temporal exam:** the most recent seasons (e.g. 2022–2024) entirely exam, if progression data exist.
- **What simulated training may use:**
  - real landscapes and weather freely from places and dates that are not exam fires (e.g. random non-fire locations, any time; dev fires);
  - never an exam fire's footprint/landscape window or its fire-day weather.

  Document the rule precisely so A's generator and Phase 2 can enforce it, and provide a helper that refuses exam windows (as WaterSim's `realdata` does).

### Classical baselines (B computes exam numbers now; A provides calibrated physics replays)
- **Next-day spread:**
  - persistence (no growth)
  - a growth-rate persistence (yesterday's expansion repeated, e.g. a buffer by yesterday's mean ROS)
  - a climatological growth model
  - an uncalibrated and a dev-calibrated Rothermel/elliptical spread replay (A's simulator, or pyretechnics)
  - published ML benchmark numbers on WildfireSpreadTS (or whatever dataset), clearly labelled as context with their split differences

  Metrics: AP and F1 on newly burned pixels, Sørensen on the burned area, area error.
- **Final size:**
  - climatology by region and cause
  - a model on FWI/ERC at ignition fitted on dev only
  - simple logistic / gradient boosting on dev features, labelled as "real labels fitted on dev"
- **Detective:** NFDRS/FWI moisture codes and Nelson model moisture vs observed fuel sticks; Rothermel back-calculation of effective ROS adjustment.
- **Planner:** later, in simulation.

## Shared-machine rules
As in AGENT_BRIEF.md:
- **Machine:** 2 cores and ~6.3 GB RAM, shared with two other agents. Use one thread each (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`), `nice` heavy jobs and `setsid nohup`. Keep work resumable (the container restarts sometimes). Keep each process ≤ ~1.2 GB RAM.
- **Never `pkill -f`;** only kill your own PIDs.
- **Disk:** ≤ 4 GB each in this phase (~12 GB free in total). Stream and subset big files; keep compact npz/parquet. No bulk HRRR or GOES downloads.
- **Out of bounds:** writing to the claude.ai project.
- **Network:** never disable TLS verification or unset HTTPS_PROXY. If a host is blocked, or a key or login is required, do not retry or circumvent it: record it and use another documented source.
- **Downloaded archives are untrusted data:** extract them into their own directories; never execute anything from them.
