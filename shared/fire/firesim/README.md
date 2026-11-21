# FireSim

FireSim is the shared fire simulator for the ML Ratchet Wildfire project (Phase 1, agent A). It covers:

- **Point fire behaviour,** BehavePlus-style: Rothermel 1972 with the Albini 1976 corrections, the Scott & Burgan 40 and Anderson 13 fuel models, wind and slope, the fire ellipse, crown fire and spotting.
- **Spread over real or generated landscapes** under hourly weather, with an arrival-time solver.
- **Dead fuel moisture** from weather (a Nelson-type stick model).
- **The Canadian FWI system,** for baselines.
- **A satellite active-fire emulator** (VIIRS, MODIS, GOES), so simulated fires can be seen through the same sensor as real ones.
- **Generators** of random landscapes and weather for training.

Every number in "Checks" below comes from a script in `shared/fire/checks/` and is stored in the `.json` file named next to it.

```
cd shared/fire/firesim && OMP_NUM_THREADS=1 python3 demo.py     # ~2 s after the first numba compile
```
![demo](demo_out/demo.png)

## Install and use
Pure Python + numpy, scipy, pandas, numba (optional: without numba the same kernels run as slow plain Python),
scikit-image (perimeter contours; matplotlib fallback). Add the folder to the path:
```python
import sys; sys.path.insert(0, 'shared/fire/firesim')
import firesim as fs
```
Run with `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`. numba compiles on first use and caches to disk
(`firesim/__pycache__`); the cache is cleared automatically when a kernel source file changes.

## API (the shared contract)

### Landscape
`fs.Landscape(fuel=..., cc=..., ch=..., cbh=..., cbd=..., elev=..., slope=..., aspect=..., res_m=90, meta={...})`
is a dict of 2-D arrays plus `.meta` (`crs`, `transform` = GDAL 6-tuple, `res_m`, `nodata`). Units:

| Layer | Units / encoding |
|---|---|
| `fuel` | FBFM40 (101-204) or FBFM13 (1-13) numbers; 91-99 non-burnable |
| `cc` | canopy cover, % |
| `ch` | canopy height, m |
| `cbh` | canopy base height, m |
| `cbd` | canopy bulk density, kg/m³ |
| `elev` | m |
| `slope` | degrees |
| `aspect` | degrees from N, the direction the slope faces |

Notes:
- Missing layers get defaults: no canopy, flat terrain. Slope and aspect are derived from `elev` (Horn's method) when absent.
- Scaled LANDFIRE integers (10 × m, 100 × kg/m³) are rejected; converting them is the data package's job.
- `ls.save(path)` / `Landscape.load(path)` use `.npz` + json meta.
- Row 0 is the north edge.

### Weather
Hourly pandas DataFrame with a UTC DatetimeIndex. Columns:

| Column | Units |
|---|---|
| `T` | °C |
| `RH` | % |
| `wind_speed` | m/s at **10 m** |
| `wind_dir` | degrees the wind blows FROM |
| `precip` | mm/h |
| `srad` | W/m², optional; clear-sky × cloud if absent |
| `cloud` | 0–1, optional |

Optional moisture columns:
- dead: `m1h`, `m10h`, `m100h` (%), used with `moisture='given'`
- live: `mherb`, `mwoody` (%)

Wind heights. Rothermel needs the 20-ft wind above the vegetation, so FireSim uses U20ft = U10m / 1.15 (Turner & Lawson 1978; Lawson & Armitage 2008). It then applies a wind adjustment factor (WAF) to get the midflame wind:
- **unsheltered:** Albini & Baughman 1979 / Andrews 2012, from fuel-bed depth
- **sheltered:** from canopy cover, height and crown ratio, when the crown fill fraction is above 5 %, as in FlamMap

Gridded weather is passed as `{'series': [df_zone0, df_zone1, ...], 'zones': int raster}`, e.g. one zone per HRRR 3 km cell. All series must share one hourly index.

### Ignition
Each item can be one of:
- `{'rowcol': [(r, c), ...]}`
- `{'points': [(x, y), ...]}` in the landscape CRS
- `{'polygon': [(x, y), ...]}`
- `{'mask': bool raster}`, an observed burned area. Its edge cells ignite and its interior counts as burned at t = 0.

Each item takes an optional `'time'`: a UTC timestamp, or hours since start. Pass a list for several items.

### simulate
```python
out = fs.simulate(landscape, weather, ignition, start, end, dt='1h', spread='rothermel',
                  crown='van_wagner' | 'none', spotting=False, method='mtt' | 'level_set' | 'ca',
                  moisture='nelson' | 'timelag' | 'given', barriers=None, suppression=None,
                  params=None, rng=None, perimeter_times=None, return_perimeters=True)
```

**`out['arrival_time']`:** hours since `start`; NaN means unburned.

**`out['rasters']`:**

| Key | Contents |
|---|---|
| `ros` | front-normal ROS, m/min |
| `ros_head` | head ROS, m/min |
| `ros_dir` | ROS in the arrival direction, m/min |
| `fireline_intensity` | kW/m, Byram, including the crown fuel burned |
| `flame_length` | m. Byram for surface fires, Thomas 1963 for crown fires |
| `fire_type` | 0 unburned, 1 surface, 2 passive crown, 3 active crown |
| `heat_per_area`, `hpa_surface` | kJ/m² |
| `cfb` | crown fraction burned |
| `phi_w`, `phi_s` | wind and slope factors |
| `lw_ratio` | length-to-breadth ratio |
| `crown_i0` | crown initiation intensity, kW/m |
| `crown_ros_active` | active crown ROS |
| `waf` | wind adjustment factor |
| `arrival_dir_deg` | direction of arrival |
| `source` | 1 ignition, 2 spread, 3 spot fire |

**Other keys:**
- `out['perimeters']`: a list of `{time, hours, area_ha, mask_cells, polygons: [{rowcol, xy}]}`, hourly by default or at `perimeter_times`.
- `out['area']`: burned-area series in ha, at `dt`.
- `out['labels']`:
  - the true hourly dead and live moisture (and per zone)
  - the weather and the 20-ft wind
  - curing
  - all parameters used
  - the spot-fire events table (source and destination cells, distance, ignited or not)
  - mean WAF, phi_w and phi_s over the burned area
  - containment time, when initial attack is on
- `out['ledger']`:
  - burned area by fire type
  - the energy released (surface and crown, GJ)
  - `growth_check`, the area gained each hour vs the line integral of the front-normal ROS along the mid-hour isochrone. The ratio is about 1.05–1.15 on uniform fuels and 1.2–1.4 on rough heterogeneous landscapes, because 1/|∇T| is smoothed by central differences.

**`params`** overrides the entries of `fs.DEFAULTS`:

| Param | Meaning |
|---|---|
| `wind_limit` | `'none'` (default; Andrews et al. 2013), `'rothermel'` (0.9 IR) or `'andrews2013'` |
| `lw` | `'behaveplus'` (1 + 0.25 UE mi/h; default) or `'anderson1983'` |
| `crown_spread` | `'rothermel1991'` (3.34 × FM10 ROS) or `'cruz2005'` |
| `fmc` | foliar moisture, % |
| `mherb`, `mwoody` | live moisture, % |
| `curing` | herbaceous curing fraction |
| `waf` | a number or raster, in place of the per-cell WAF |
| `ros_adjust` | `{'GR': 1.3, 'TU': 0.6, ...}` by fuel group (or fuel number, or `'all'`); the dev calibration knob |
| `ros_factor` | a raster multiplier |
| `stencil` | 8, 16 or 32 |
| `simplex` | two-point updates on or off |
| `moisture_spinup_h` | default 72 |
| `nelson` | Nelson model parameter overrides |
| spotting | `ember_rate`, `spot_med_frac`, `spot_sigma`, `spot_dir_sd`, `spot_p_est`, `spot_delay_h`, `spot_surface_fli`, `shade` |

**Barriers and suppression:**
- `barriers`: a bool raster, `{'mask': ...}` or `{'lines': [...], 'width_cells': 1}`. These cells become non-burnable.
- `suppression`:
  - `{'lines': [{'lines' or 'mask', 'time'}]}`: control lines that block spread after their build time
  - `'ros_factor'`: a raster, e.g. retardant
  - `'initial_attack'`: `{start_h, line_rate_m_per_h, lw}`. Fried & Fried 1996-style containment: line built since `start_h` vs the perimeter of an ellipse of the burned area. The run stops when the line wins.
- `fs.generate.fuel_treatment(ls, mask, kind='thin'|'prescribed_burn'|'masticate'|'fuelbreak'|'nonburnable')` edits a landscape.

**Batch mode:** `fs.simulate_batch(ls, weather, [ign1, ign2, ...], start, end, keep=('arrival_time',))`. It runs moisture and fuel bundles once (`fs.prepare`), then one kernel per ignition. Any `simulate()` call can also reuse a `prepared=fs.prepare(...)`.

### behave (point fire behaviour)
```python
b = fs.behave('SH5', m1=4, m10=5, m100=7, mherb=60, mwoody=80, wind_speed=6.0, wind_height='10m',
              wind_dir=None, slope=20, aspect=180, cc=0, ch=0, cbh=0, cbd=0, fmc=100, elapsed_h=1.0)
```

Returns SI units. `english=True` adds the BehavePlus units.

**Surface fire:**
- head, flank and back ROS
- direction of maximum spread
- L/W and eccentricity
- effective wind, and whether the wind limit applied
- phi_w and phi_s
- reaction intensity (dead and live)
- heat per unit area, residence time
- fireline intensity, flame length
- mixture properties (SAV, packing ratio, live moisture of extinction, ...)
- WAF
- ellipse length, width, area and perimeter after `elapsed_h`

**Crown fire:**
- fire type, final ROS, intensity and flame length
- torching and crowning indices

**Spotting:**
- maximum spot distance, surface and crown
- probability of ignition

### observe (satellite emulator)
```python
ob = fs.observe(out, sensor='viirs'|'modis'|'goes', overpasses='nominal'|'goes'|[(t, sat, scan_deg), ...],
                rng=0, lat=..., lon=..., cloud=None|fraction|{time: mask}, params=None, return_masks=False)
```

Returns:
- `ob['detections']`: one row per fire pixel, with time, satellite, scan angle, day/night, model row/col, x/y, pixel size, FRP (MW) and `true_fire` (False means a false alarm).
- `ob['overpasses']`: per overpass, the number of detections, expected detections, false alarms, cloud-omitted pixels, active and flaming area, total FRP and burned area.

The physical chain, all defaults overridable:
1. **Cell FRP.** The flaming energy is released while the front crosses the cell. Post-frontal combustion follows, as a ratio by fuel group with exponential decay. The radiative fraction is 0.15.
2. **Pixels.** Cells aggregate into sensor pixels at the overpass scan angle: VIIRS 375 m at nadir, growing to about 2x at the edge; MODIS 1 km, growing to about 2 × 4.8 km; GOES 2 km, growing with view zenith.
3. **Detection probability.** Logistic in log FRP, with a threshold that scales with pixel area and differs by day and night.
4. **Cloud.** Given masks, or random correlated clouds.
5. **Smoke.** Optional omission of very bright pixels.
6. **Commission.** Poisson false alarms.

Nominal overpasses come from a sun-synchronous orbit model with the real equator-crossing times. The orbit phase is random.

### Generators
- `fs.generate_landscape(shape, res_m, rng, kind='parametric', region='west_dry'|'mixed_conifer'|'chaparral'|'great_basin'|'southeast', relief_m=None, ...)`:
  - fractal terrain
  - life forms from a site-wetness index (elevation, northness, topographic position, noise)
  - Scott & Burgan models from class-conditional weights per patch
  - correlated canopy layers in LANDFIRE-like ranges
  - non-burnable features: rock, lakes, rivers along flow paths, roads, agriculture, urban
  - latent fields and settings in `.meta['labels']`
- `kind='real'` cuts a non-exam window from B's real landscapes (see "Real data").
- `fs.generate_weather(start, hours, climate=..., lat, lon, rng, **overrides)`:
  - AR(1) daily synoptic anomalies
  - diurnal temperature and RH, with overnight recovery
  - an afternoon wind peak and a night slow-down
  - diurnal direction swing
  - foehn / Santa Ana-like dry-wind events
  - Markov wet days
  - climates: `west_dry_summer`, `california_autumn`, `southwest_spring`, `northern_rockies_summer`, `southeast_spring`
- `fs.hourly_from_daily(daily, lat, lon)` turns daily gridMET-like rows into hourly series.

### Real data (agent B's `shared/fire/firedata`) and replays
- `fs.realdata.landscape_from_firedata(fd.landscape(fid))` turns B's package into a `Landscape`. B's affine transform becomes a GDAL tuple, on EPSG:5070 at 90 m.
- `fs.realdata.weather_series(fd.weather(fid), x, y)` is the HRRR hourly series at the cell nearest (x, y), in the contract. It handles three conversions:
  - **Wind direction:** rotated from true north to grid north (`grid_convergence_deg`). EPSG:5070 grid north differs from true north by up to about 15° in the west, and LANDFIRE aspect is grid-relative.
  - **Precipitation:** missing values are set to 0 and flagged in `precip_missing`.
  - **Cloud:** converted from % to a fraction.
- `fs.realdata.weather_zones(W, ls, coarsen=2)` builds gridded weather from blocks of HRRR cells.
- `fs.generate_landscape(shape, 90, kind='real', rng=...)` cuts a random window of a real dev landscape. The window passes `firedata.guard.check_landscape`; if no clean window is found, exam cells are blanked.
- `fs.generate_weather(start, hours, kind='real', rng=...)` takes real HRRR hours from a random dev package and cell. They pass `firedata.guard.check_weather`, are shifted by whole days to `start`, and include a 72 h spin-up.
- `fs.replay.replay_day(ls, w, burned_now, start, end, params=None, ensemble=True)` replays one fire-day from an observed burned mask. It returns the arrival time, the predicted mask, a score map exp(-arrival/dt) and the ensemble probability.
  - Ensemble members each perturb the wind speed (× LN(0.25)), the wind direction (+ N(0, 20°)) and the ROS (× LN(0.5)).
  - `fs.replay.calibrated_params(json.load(open('shared/fire/checks/replay_calibration.json')))` gives the dev-calibrated per-fuel-group ROS factors.

### Other modules
- `fs.fwi`: Canadian FWI system (FFMC, DMC, DC, ISI, BUI, FWI), hourly FFMC, `fwi_series`, `daily_from_hourly`.
- `fs.moisture`: `dead_moisture`, the Simard EMC, GSI live moisture.
- `fs.crown`: Van Wagner I0, Cruz 2005, Scott & Reinhardt CFB.
- `fs.spotting`: Albini maximum distance, Schroeder probability of ignition.
- `fs.metrics`: Sørensen, Jaccard, P/R/F1, AP, area error, `next_day_scores`.
- `fs.fuel_models()`: the fuel table.

## Physics (what is implemented, with sources)

**Surface fire:**
- Rothermel 1972 with the Albini 1976 corrections, following Andrews 2018, RMRS-GTR-371: weighting factors, the live moisture of extinction, the BehavePlus wind and slope vector addition, and the ellipse.
- Fuel models from GTR-371 Table 8, which matches Scott & Burgan 2005 and Anderson 1982. FM7 and FM9 use the Anderson originals; GR6 heat content is 9000 per S&B Table 7.
- Dynamic herbaceous curing (Scott & Burgan 2005) from live herbaceous moisture.
- No wind limit by default (Andrews et al. 2013); the Rothermel 0.9 IR limit is optional.

**Ellipse.** L/W = 1 + 0.25 UE (UE in mi/h), as in BehavePlus 6 / GTR-371, with Anderson 1983 optional. The back ROS follows from the eccentricity. Directional ROS is the ray speed from the rear focus.

**Spread solver.** Minimum travel time (Finney 2002) on a grid:
- Dijkstra over a 16-cell stencil (default; `level_set` = 32 cells, `ca` = 8 cells).
- Edge times are integrated cell by cell along each straight segment, and through the hourly weather steps.
- Two-point "simplex" updates (a semi-Lagrangian step from the segment between two accepted neighbours) remove most of the stencil's direction bias. Without them, a 16-neighbour MTT underestimates elongated ellipses by up to 36 %; with them, by at most 6 % (see "Checks").
- Segments through a grid corner are blocked when both side cells are non-burnable.
- Edges whose integration already exceeds the current best arrival are pruned. This is exact and gives a 2x speed-up.
- The vertex-based Huygens method (FARSITE) is **not** implemented. `method='huygens'` raises an error and points to `mtt` / `level_set`, which give the same elliptical shapes.

**Crown fire:**
- Van Wagner 1977 initiation, I0 = (0.010 CBH (460 + 25.9 FMC))^1.5.
- Active criterion: R_active = 3 / CBD.
- Crown ROS: Rothermel 1991 (3.34 × FM10 at 0.4 × U20), or Cruz et al. 2005 (active and passive).
- Crown fraction burned: Van Wagner 1993 / Scott & Reinhardt 2001.
- Crown L/W from Rothermel 1991.
- In the solver: the surface ray ROS is compared with the critical ROS for initiation. Passive crown fire blends surface and crown spread by CFB; active crown fire uses the crown ROS.

**Spotting:**
- Albini 1979 maximum distance (the surface form as in BehavePlus), using the fireline intensity for torching and crown sources. This is a documented simplification.
- Schroeder 1969 probability of ignition.
- In the solver, a stochastic ember model:
  - Poisson embers from crowning cells, at a rate proportional to CFB
  - lognormal distances capped at the Albini maximum
  - downwind direction with scatter
  - flight time plus delay
  - ignition with P_ig × an establishment probability

**Fuel moisture:**
- A Nelson 2000-structured stick model, re-derived because the NFDRS4 C++ source was not reachable:
  - wooden cylinders (radius 0.20 / 0.64 / 2.0 cm), radial bound-water diffusion
  - surface exchange toward the EMC at stick temperature, with solar heating of the stick
  - rain wetting
  - the Nelson 1984 isotherm fitted to Simard 1968
- Live moisture from the GSI (Jolly et al. 2005) or constant values.
- The FWI hourly FFMC is also available.

**Diurnal cycle.** It comes from the hourly weather: overnight RH recovery, and slower night winds through the moisture model and the WAF. The generators reproduce it.

**FWI.** Van Wagner & Pickett 1985 / Van Wagner 1987, following Wang et al. 2015.

## Checks (shared/fire/checks/*.json)

| Check | Result | File |
|---|---|---|
| Published Rothermel/BehavePlus tables and worked examples (GTR-371 Tables 8, 12, 13, 17-19, 21-24, text examples; Rothermel 1983 GTR-143) | **436/436** within half the last published digit | `published_examples.json` |
| Cross-check vs pyretechnics 2026.10.20 (independent PyPI implementation, built in `firesim/.refenv`), same fuel table, 6720 flat/slope-only cases × 5 outputs | ROS median abs rel err **4.9e-5**, max 5.2e-5 (pyretechnics is float32); FLI max 1.7e-4; L/W max 3.5e-5 | `pyretechnics_crosscheck.json` |
| ... wind on slope (10080 cases) | median 1.3 %, max 29 %: **by design** (pyretechnics projects wind into the slope plane; FireSim follows BehavePlus horizontal vector addition) | same |
| ... crown I0, crown ROS, WAF | I0 0.5 % (25.9 vs 26 per % FMC constant); crown ROS 1e-7; WAF 80/80 to 1e-8; fire type 100 % agreement | same |
| Ellipse: uniform fuel, 5 L/W (1-3.3), 30/60/90 m, 7 wind directions, 55 cases per solver; burned area / theory | default `mtt` (16 + simplex): mean **0.981**, min 0.940; head ROS 0.993 (min 0.959); L/W err ≤ 5.5 %; orientation bias (area range across directions) ≤ **3.4 %** | `analytic.json` |
| ... `level_set` (32 + simplex) | area mean 0.992, min 0.973; L/W err ≤ 1.5 %; orientation bias ≤ 1.7 % | same |
| ... plain 16-stencil MTT / 8-neighbour CA (for reference) | area mean 0.83 / 0.59, min 0.64 / 0.31; orientation bias 14 % / 12 % | same |
| Two-fuel strip (line fire crossing fuel A then B), arrival vs hand solution | max rel err **2.9e-9** | same |
| Solid fuel break: cells burned beyond it; gap: arrival vs shortest path through the gap (180 targets) | **0** cells; gap error median 0.11 %, max 0.50 % | same |
| Refraction (point in fuel A, targets in fuel B; Fermat minimum over the interface) | median 0.06 %, max 0.25 % | same |
| Time-varying rate (moisture step at 2 h) | median 0.95 %, max 2.6 % | same |
| Crown initiation I0 vs Van Wagner hand calcs (21 CBH × FMC) | exact; VW77 example (CBH 6 m, FMC 100 %) 2476 vs quoted ~2500 kW/m | `crown.json` |
| Cruz 2005 active/passive ROS vs hand calcs; Rothermel 1991 ratio 3.34 | exact | same |
| Simulator head fire type and ROS vs behave (surface / passive / active, winds 2-16 m/s) | types agree; head ROS within 1e-8 | same |
| Moisture: Nelson isotherm vs Simard EMC (T 0-40 C, RH 10-95 %) | RMSE 0.41 %-points, max 1.9 | `moisture.json` |
| Moisture: e-folding time of 1/10/100-h sticks | 1.000 / 10.00 / 100.0 h (rel err < 3e-5) | same |
| Moisture: diurnal cycle and rain (synthetic dry week, 10 mm rain) | 1-h daily range 8.3 %, 10-h 3.7 %, 100-h 1.2 %; 10-h peaks at 24 % after rain and loses half the gain within 1 h (rain water on the surface evaporates fast; likely too fast, to be checked on real sticks) | same |
| Moisture on dev fire sites vs gridMET NFDRS fm100 / fm1000 (real 10-h stick data were not obtainable: NFMD left wfas.net) | 100-h: FireSim is **2.4 %-points drier** (5.8 vs 8.2 %), RMSE 3.0, r = 0.75; 1000-h: 2.9 points drier, r = 0.43 (18 dev fires with ≥ 5 days after a 10-day spin-up, 443 days). Which one is closer to real sticks is unknown. | `moisture_real.json` |
| FWI: standard test row (Van Wagner & Pickett 1985) | all six codes within 0.005 of the published values | `fwi.json` |
| FWI vs xclim 1.0.0 (180-day season with 20 rain days) | FFMC, DC, ISI identical (1e-13); DMC max 0.19, BUI 0.26, FWI 0.05 (xclim's DMC day-length / start-up handling) | same |
| Speed: 30 × 30 km, 24 h, one core | see below | `speed.json` |

## Speed (one core, `speed.json`)
The machine has 2 cores shared with two other agents (load about 2.5 during the run). Timings are medians of 3 and vary by ±30 %.

| Case | Burned | Full `simulate()` |
|---|---|---|
| 500 × 500 at 60 m, mixed conifer, 24 h, crown on, no spotting | 3 700 ha | **0.47 s** |
| same, spotting on | 5 300 ha, 178 spot fires | 0.63 s |
| same, `level_set` (32 stencil) | 4 200 ha | 0.86 s |
| 334 × 334 at 90 m | 215 ha | 0.10 s |
| worst case: 30 × 30 km grass/shrub, 11 m/s, line ignition, about 40 % of the landscape burns in 24 h | 31 000–38 000 ha | **1.1 s (90 m), 1.7–2.0 s (60 m)** |
| batch: 40 random ignitions on one landscape (`simulate_batch`) | median 300–400 ha | 0.09 s (90 m), 0.23 s (60 m) per fire |

The full call includes the moisture model (about 0.02 s per day of weather), hourly perimeters and the ledger. The kernel processes 35 000–55 000 burned cells per second.

Dev replays of large fires are slower: 20–100 s per fire-day for 13 runs. The time goes to post-processing and scoring on windows of up to 1.3 million cells.

## Dev-fire replays (the classical benchmark; `checks/replay_dev.json`, `checks/replay_calibration.json`)

**Data.** All 40 dev fires (B's frozen split), 470 growth fire-days. Exam fires were never loaded.

**Setup.** Each replay starts from the observed FEDS burned area at a day's afternoon (PM) S-NPP overpass. It runs to the next PM overpass and predicts the newly burned 90 m cells.
- **Landscape:** LANDFIRE.
- **Weather:** the hourly HRRR series at the cell nearest the fire, with winds rotated to grid north. Where HRRR has no precipitation (14 fires, 2017–18), gridMET daily rain is spread over the day.
- **Moisture:** the Nelson model with a 72 h spin-up.
- **Fire behaviour:** crown fire on, no spotting, no suppression.

**Scoring.** B's exam metric code (`firedata.metrics.nextday` / `summarize`), on growth days only.

**Calibration (dev only):**
- One ROS factor per fuel group, read from the dominant fuel in a 900 m ring around the fire. Each factor is picked from a sweep of 0.05–2 to maximise mean F1.
- A separate factor for a fire's first observed day.
- Fitted on 39 fires. Creek 2020 (39 growth days) was still running and is held out.

| Next-day spread, dev growth days | F1 new cells (mean / median) | AP (mean / median) | Sørensen, total area (median) | log10(pred / obs new area), median | abs area error, km², median |
|---|---|---|---|---|---|
| Persistence (no growth) | 0 | n/a | **0.96** | n/a (predicts 0) | = new area |
| Uncalibrated, deterministic (40 fires, 470 days) | 0.153 / 0.108 | 0.156 / 0.105 (score exp(-arrival/dt)) | 0.71 | **+1.08 (12x too much)** | 17.1 |
| Uncalibrated, 12-member ensemble | 0.150 / 0.107 | 0.134 / 0.086 | 0.70 | +1.13 | 18.5 |
| Dev-calibrated, deterministic (39 fires, 431 days, in-sample) | **0.192** / 0.157 | 0.138 / 0.086 | 0.85 | +0.64 | 4.4 |
| Dev-calibrated, ensemble (in-sample) | **0.194** / 0.160 | **0.154** / 0.103 | 0.84 | +0.70 | 5.0 |
| Dev-calibrated, leave-one-fire-out (deterministic, via the sweep) | 0.184 / 0.150 | 0.138 | 0.84 | +0.71 | |
| Creek 2020, held out: calibrated / uncalibrated | 0.146 / 0.148 | 0.101 / 0.102 | 0.94 / 0.94 | +1.08 / +1.08 | 158 / 158 |

**Fitted factors:**

| Fuel group | Factor | Fire-days |
|---|---|---|
| GR | 0.5 | 21 |
| GS | 0.35 | 80 |
| SH | 0.35 | 31 |
| TL | **1.0** | 141 |
| TU | 0.35 | 158 |
| first observed day | 0.7 | 38 |

The default factor is 0.5. The factor that makes the median predicted area match the observed area is 0.2. F1 prefers about 0.5, because missing cells costs more than over-predicting them. On the 39 calibrated fires, the uncalibrated runs score F1 0.154, Sørensen 0.69 and log ratio +1.07. Creek is timber litter, whose factor is 1, so calibration leaves it unchanged.

What this says:
- **Uncalibrated physics over-predicts badly.** The median new area is about 12x the observed. This is the familiar result: there is no suppression, no fuel-moisture or fuel-map errors, and the HRRR winds are 10 m winds over open terrain.
- **Halving to thirding the ROS** (shrub, grass-shrub, timber understory) roughly doubles the precision. The factors also absorb the fact that FireSim's dead fuels run about 2.4 %-points drier than NFDRS (see `moisture_real.json`).
- **On total-area Sørensen, persistence (0.96) beats every physics replay (0.71–0.85).** Most of the next-day burned area is already burned today, and over-predicted growth costs more than none. Phase 2 must report all of F1/AP and Sørensen against both persistence and the calibrated replay.
- **Big wind-event days are missed entirely.** Riverside 2020, day 0, burned 496 km² overnight in an east-wind event, and the replay F1 was 0.05. Late, mostly contained days are over-predicted.
- About 36 % of the predicted new cells burn as crown fire (uncalibrated).

`fs.replay.replay_day(..., params=fs.replay.calibrated_params(cal, fire_day_index=i), ensemble=True)` reproduces the calibrated method. B / Phase 2 can run it on exam fires. Figure (a seeded draw of 3 dev fire-days, small, medium and large): `demo_out/replay_examples.png`.
![replays](demo_out/replay_examples.png)

## Satellite emulator vs real VIIRS (`checks/observe_viirs.json`)

**The data.** The test uses all 40 dev fires and 1570 S-NPP overpasses: 777 by day and 793 by night. The real data are the VIIRS 375 m fire pixels that FEDS assigned to each fire, 69,535 in all. The FEDS 12-hourly steps are split into single overpasses by pixel size.

**How the emulator was driven.** To test the sensor model rather than the spread model, the emulator was driven by the **observed** progression:
- arrival times interpolated between the FEDS perimeters, with an afternoon-peaked diurnal weighting
- ROS from the gradient of the observed arrival field
- Rothermel heat per unit area from the LANDFIRE fuel model
- the real overpass pixel size
- no clouds

| Per-overpass detections (emulated vs real) | Uncalibrated | Dev-calibrated (in-sample) | 2-fold by fire |
|---|---|---|---|
| median log10((emu+1)/(real+1)) | -0.14 (emulator detects 27 % fewer) | +0.01 | +0.03 / -0.05 |
| Spearman correlation of counts | 0.79 | 0.80 | 0.77 / 0.80 |
| overpasses within a factor 2 | 60 % | 62 % | 68 % / 56 % |
| total detections, emulated / real | 37,362 / 69,535 | 52,384 / 69,535 | |
| overpasses with zero detections, real / emulated | 12 % / 27 % | 12 % / 18 % | |
| FRP per detected pixel, median, day (real / emu) | 16.3 / 8.9 MW | 16.3 / 11.2 MW | |
| FRP per detected pixel, median, night (real / emu) | 2.9 / 2.6 MW | 2.9 / 2.8 MW | |
| FRP per detected pixel, 90th percentile, all | 42 / 26 MW | 42 / 38 MW | |

Calibration has two steps, both now the defaults in `firesim.satellite`:
1. **Energy multiplier.** This multiplies the heat per unit area, standing for the energy of large fuels, duff and canopy that Rothermel's fine-fuel HPA leaves out. It is set to match the median FRP of detected pixels: chosen value 2 from {1, 2, 4}.
2. **Night detection threshold.** The FRP50 multipliers are set so the median count ratio is about 1. Day keeps 1.2 MW; night drops from 0.5 to 0.25 MW.

The 2-fold split by fire picked (e, day, night) = (2, 1, 0.5) and (4, 2, 2), so the fit is only loosely determined.

**Limits of this test:**
- The real counts include cloud and smoke omission, and the emulator here has no cloud. Part of the calibrated low night threshold may compensate for the emulator's smoother fire edge.
- Commission (false alarms) cannot be measured, because FEDS keeps only the pixels it assigned to a fire.
- Overpass times are approximate: the FEDS half-day label + 1.5 h − lon/15, and +100 min for the second orbit.
- Large fires are undercounted the most. The totals stay 25 % low even when the median overpass matches.

## Deviations from the contract
- `method='huygens'` is not implemented: arrival-time solvers only. `level_set` is a 32-neighbour MTT with simplex updates. It is not a PDE level-set; it is the same minimum-travel-time solution on a finer angular stencil.
- The Nelson model is re-derived from the published structure: the NFDRS4 source was not reachable. It is checked by its definition (timelags), by its isotherm vs Simard, and against gridMET's NFDRS fm100 / fm1000 on dev sites. Real fuel-stick observations (NFMD) were not obtainable: they left wfas.net.
- US NFDRS ERC/BI are not implemented. gridMET distributes ERC and BI (fuel model G) directly; B's baselines can use those.
- Spotting from crown fires uses the Albini surface-fire plume relation with the crown intensity. Albini's torching-tree and crown-fire spotting models need species tables that are not reproduced here.
- Real 10-h fuel-stick observations were not available (NFMD), so the moisture check is against gridMET's NFDRS fm100 / fm1000.
- The dev-calibrated replay ensemble was run on 39 of the 40 dev fires. Creek 2020 has only the deterministic calibrated score (held out, via the factor sweep), for compute time.

## Limitations
**Spread solver:**
- **Grid MTT.** With the default 16-cell stencil, the area of elongated ellipses (L/W ≥ 3) is up to 6 % low, and the head ROS up to 4 % low in some directions. Use `level_set` (1-3 %) when that matters.
- **Spread rasters.** Front-normal ROS and intensity come from the arrival-time gradient, so they are noisy where fronts merge. The ledger's growth check is about 1.05–1.15 on uniform fuels and 1.2–1.7 on rough landscapes or with spotting.
- **Wind.** There is no fire-atmosphere coupling, no plume-driven winds and no terrain wind downscaling. Weather is one hourly series per zone; the replays use one HRRR cell per fire-day.
- **Spotting** parameters (ember rate, establishment) are uncalibrated defaults. Replays run without spotting.

**Fuel moisture:**
- FireSim's 100-h moisture is 2.4 points drier than gridMET NFDRS. Without real sticks it is unknown which is right.
- Live moisture is constant (60 / 90 %) unless given. The GSI model exists but is not driven in the replays.

**Replays:**
- **Uncalibrated replays over-predict** the next-day area about 12x. The calibrated ones still over-predict about 4–5x in the median (F1-optimal factors).
- The per-fuel-group factors are coarse. They are picked from a ring of fuel around the fire, and the first-day factor is pooled across groups.
- No suppression, containment or night-time inactivity is modelled beyond what weather and moisture produce.
- Wind-event days (e.g. Riverside 2020) are under-predicted by an order of magnitude.

**Satellite emulator.** It is calibrated only for VIIRS, and only on per-overpass counts and FRP, with no real cloud masks. MODIS and GOES keep literature thresholds.

**Generated landscapes** are plausible, not real. Their fuel-model mixtures and canopy statistics are hand-set. The `kind='real'` windows come from the 40 dev landscapes only (90 m).
