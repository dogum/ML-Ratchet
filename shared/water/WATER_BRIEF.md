# Watershed: shared plan (read with /home/claude/AGENT_BRIEF.md)

Greg's fourth big-system project for ML Ratchet (github.com/dogum/ML-Ratchet, "Trained on physics. Graded on reality."),
built the same way as RoadSim (vehicles), GridSim (power grid) and BuildingSim (buildings). Those three are now in Greg's
public repo. Read one of their handoffs for the expected quality, read-only: `/home/claude/grid/HANDOFF.md`,
`/home/claude/bldg/HANDOFF.md`.

## Phases
- **Phase 1** (now, two agents in parallel):
  - **A: WaterSim.** The shared, verified hydrology simulator.
  - **B: WaterData.** The real-data package, with frozen dev/exam splits and classical baselines.
- **Phase 2** (later): several ratchet models on top, in parallel. Candidates:
  - **FlowCast:** daily (and hourly where possible) streamflow forecasting for gauged and *ungauged* basins, with error bars, including flood peaks.
  - **SnowSense:** snow water equivalent and melt timing from weather (and optionally streamflow), graded on SNOTEL.
  - **BasinDetective:** infer basin properties from forcing + streamflow alone: storage/recession constants, baseflow index, soil water capacity, snow fraction, groundwater contribution, aridity, regulation/human influence.
  - **ReservoirPilot:** a learned reservoir release policy (flood control + water supply + hydropower) vs rule curves and MPC, on real inflows.
- **Then:** a `HANDOFF.md` listing what is missing.

## Folders
Root: `/home/claude/water/`.
- A works in `water/watersim/` (Python package `watersim`) and `water/checks/`.
- B works in `water/data/` (`raw/`, `proc/`) and `water/waterdata/` (Python package `waterdata`).
- Neither touches the other's folders. Keep a `STATUS.md` in your own folder, and put temp files and screenshots in a subfolder of your own folder.

## Shared contract

### Time series
UTC dates as a daily index (hourly where available), pandas DataFrames / parquet.

| Quantity | Unit |
|---|---|
| Precipitation, ET, runoff, storage | mm (or mm/day); streamflow as mm/day over basin area AND m³/s |
| Temperature | °C |
| Shortwave radiation | W/m² |
| Vapour pressure | Pa |
| Wind | m/s |
| SWE | mm |
| Reservoir storage | hm³ (million m³) |
| Release / inflow | m³/s |

### Forcing
One frame per basin (basin-average), columns:
- `P`, `T_mean`, `T_min`, `T_max`
- `SRAD`, `vp` (or `RH`), `wind`, `PET` (stated method)
- `snowfall_frac` optional

Plus meta json: `basin_id`, `source`, `area_km2`, `centroid lat/lon`, `elevation` stats and `elevation bands` if available.

### Basin spec (dict/JSON)
B fills it from real attributes; A's simulator consumes it and A's generator samples it.
- **Identity:** `id`, `area_km2`, `lat`, `lon`
- **Terrain and climate:** `elev_mean_m`, `elev_bands` (fractions and mean elevations), `slope_mean`, `aridity` (PET/P), `p_mean`, `frac_snow`
- **Cover and soils:** `forest_frac`, `lai`, `soil_depth_m`, `soil_porosity`, `soil_conductivity`, `max_water_content_mm`, `geology_permeability`, `carbonate_frac`
- **Human influence:** `dam_count`, `reservoir_storage_frac` (degree of regulation), `irrigation_frac`, `urban_frac`
- **Model parameters (simulation only):** `params`, whose model-specific labels are known for simulated basins and unknown (None) for real ones.

Unknown fields are None, with a documented default source.

### Simulator API (A builds; B's baselines and Phase 2 call it)
```python
from watersim import simulate, generate_basins, list_models, routing, reservoir
out = simulate(spec, forcing, start, end, model='hbv'|'gr4j'|'sacsma'|'vic_lite'|..., params=None,
               snow='degree_day'|'energy_balance', routing='uh'|'muskingum'|'kinematic', reservoirs=None,
               init=None, spinup='1y', rng=None)
```
`out` contains:
- daily series: Q (mm/day and m³/s), SWE, soil moisture, groundwater storage, ET (actual), snowmelt, rain/snow partition, reservoir storage and release where present
- a water-balance ledger: P = ET + Q + ΔS (snow, soil, groundwater, channel, reservoir) + losses, which must close
- labels: true parameters, BFI, recession constants, etc.

A reservoir operator hook: `release = policy(obs)` per day, with storage, level, area and hydropower head curves.

### Physics A must implement (choose clear, documented, standard formulations)
- **Snow:** degree-day with refreeze and liquid water holding, elevation bands with lapse rates, and rain/snow partition by temperature. Optionally a simple energy-balance snow model (a SNOW-17 class, or an energy-balance model using radiation and humidity).
- **Soil / runoff / groundwater:** at least two well-known conceptual models, so Phase 2 sees structural variety: HBV (Bergström) and GR4J (Perrin 2003), plus ideally a SAC-SMA-like or VIC-like variable-infiltration model. Include infiltration excess for intense rain at hourly resolution if feasible.
- **ET:** Penman–Monteith or Priestley–Taylor PET from forcing, and an actual-ET soil moisture stress function.
- **Routing:** unit hydrographs; Muskingum or Muskingum–Cunge channel routing for multi-reach networks; optional kinematic wave.
- **Reservoirs:** storage–area–elevation, spill, rule-curve operations, evaporation, and hydropower (head × flow × efficiency).
- **Human influence (optional):** irrigation withdrawals and return flows.

### Verification A must record (`water/checks/*.json`)
- Water-balance closure on every run; analytic linear-reservoir recession; unit-hydrograph convolution vs closed form; Muskingum vs an analytic or published test; degree-day snow vs a hand calculation.
- Reproduce published reference implementations where possible: GR4J vs the airGR examples (if the R package is unavailable, use the published test data/outputs or a well-known Python port); HBV vs a published HBV-light example. Install reference packages in an isolated venv inside your folder; never change the shared numpy/pandas/torch.
- Calibrated performance on real basins: calibrate each model on DEV basins (B's splits) and report NSE/KGE in a calibration and a validation period. These numbers are the classical benchmark the Phase 2 models must beat.
- Snow vs SNOTEL SWE at stations co-located with dev basins.
- Speed: aim for ≥ 10,000× real time (a basin-decade at daily step in well under a second), and a vectorised batch mode over many basins/parameter sets.

### Real data B must assemble (coordinator checked reachability on 18 Nov)
- **Caravan** (Kratzert et al. 2023, Zenodo, CC-BY 4.0): large-sample hydrology for ~6,800 basins worldwide, including CAMELS-US (671 basins), CAMELS-GB/AUS/BR/CL and HYSETS/LamaH, with ERA5-Land forcing, streamflow and static attributes. Zenodo API is reachable; find the documented record and files. Also the national extensions on Zenodo (e.g. Caravan-DE/DK, CAMELS-ES). Check sizes first and take what fits (≤ 5 GB); stream-unpack.
- **CAMELS-US original** (Newman/Addor; NCAR GDEX timed out): the Caravan copy or another documented mirror is fine. Daymet/Maurer/NLDAS forcings were in the original, so check whether a documented copy is reachable; Daymet's API timed out.
- **USGS NWIS** daily (and instantaneous where needed) streamflow, reachable: extend records beyond Caravan's end date for the exam period, and pull peak-flow files for flood-peak exams.
- **NRCS SNOTEL** daily SWE, precipitation and temperature, reachable (report generator CSV). Pick stations in/near CAMELS-US basins; also SNOTEL-only for SnowSense.
- **NOAA National Water Model retrospective v3.0** on AWS S3 (reachable): a strong operational-model baseline for FlowCast at gauges. Extract streamflow at the CAMELS gauges if feasible within size limits (zarr/netCDF chunks), otherwise a documented subset.
- **Reservoirs:** find public daily reservoir storage/release/inflow data with documented downloads, e.g.:
  - US Bureau of Reclamation RISE / HydroData
  - USACE reservoir data
  - California CDEC (`cdec.water.ca.gov`, daily reservoir data)
  - ResOpsUS (Steyaert et al. 2022, on Zenodo or figshare: daily operations for ~680 US reservoirs)
  - GRanD/GeoDAR attributes

  Prefer ResOpsUS plus CDEC.
- **Weather** (if needed beyond Caravan's ERA5-Land): Iowa Mesonet / NASA POWER (reachable), GridData's NASA POWER loaders (read-only).

### Splits (B decides once and freezes in `water/data/proc/splits.json`)
- **Spatial (by basin):** dev = a small stratified set (by climate/aridity, snow fraction, region, regulation) for calibration; exam = the rest. Ungauged-basin tests need basins never seen in any form.
- **Temporal:** e.g. calibration/training-input years vs exam years. Follow the standard CAMELS protocol if convenient (calibration 1999–2008, validation 1989–1999) but document it; also a recent-years exam with NWIS extension (e.g. 2015–2024) where data exist.
- SNOTEL by station; reservoirs by reservoir.
- Simulated training may use real *forcing* freely from non-exam periods, and real attributes only from non-exam basins.

### Classical baselines (B computes exam numbers now; A provides calibrated conceptual models)
- **Streamflow:** climatology (day-of-year mean); persistence for short-lead forecasts; GR4J/HBV calibrated per basin on its own calibration period (A's calibration; B may use a simple calibrator); regionalised parameters (nearest neighbour / attribute similarity) for ungauged basins; NWM retrospective at gauges where extracted.
- **Literature reference, clearly labelled:** the published LSTM benchmark on CAMELS-US (Kratzert et al. 2019, median NSE ~0.74–0.76 for ungauged/gauged), as context only.
- **Metrics:** NSE, KGE, peak-flow error, FHV/FLV, timing error.
- **Snow:** degree-day calibrated per station; SNOTEL climatology.
- **Detective:** recession analysis (Brutsaert-Nieber), digital-filter BFI (Lyne-Hollick / Eckhardt), Budyko-curve estimates, attribute medians.
- **Reservoir:** historical operations, standard operating policy (SOP), a simple rule curve, and perfect-foresight DP bounds.

## Shared-machine rules
As in AGENT_BRIEF.md:
- **Machine:** 2 cores, ~6.3 GB memory shared, ~17 GB free disk.
- **Process hygiene:** one thread each; `nice` heavy jobs; never `pkill -f`, only kill your own PIDs; keep work resumable.
- **Budgets:** ≤ 5 GB disk each this phase; stream big files; keep compact parquet. Keep any single process under ~1.2 GB of RAM.
- **Out of bounds:** nothing written to the claude.ai project in Phase 1.
