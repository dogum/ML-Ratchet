# RoadData: real data for the vehicle project (Phase 1, agent B)

Everything Phase 2 is graded on lives here. It is frozen: splits, cleaning rules and baselines below were decided once
(seed 20261112) and must not be changed after Phase 2 starts. Build scripts are in `road/data/scripts/`; the loader
package is `road/realdata/` (numpy/pandas/scipy only).

```python
import sys; sys.path.insert(0, 'shared/vehicle')   # or: pip install -e ".[vehicle]" at the repo root
import realdata as rd
for meta, arr in rd.load_trips('dev', arch='conv', label_ok=True, limit=20, seed=0): ...   # VED trips
spec  = rd.load_vehicle('VED_455')          # vehicle spec dict (contract format, unknowns = None)
veh   = rd.ved_vehicles()                   # VED vehicle table: static data, arch guess, fueleconomy label match, split
idx   = rd.ved_index()                      # one row per VED trip: meta + quality flags
epa   = rd.epa_testcars(split='dev')        # EPA test cars 2008-2026, one row per vehicle configuration
fe    = rd.fueleconomy()                    # fueleconomy.gov vehicles.csv (cleaned) + split
cyc   = rd.cycles(duty='heavy')             # dynamometer schedules in trip format
hd    = rd.hd_data()                        # ZTBus bus legs/missions, Mersin bus trips, ICCT truck results
for meta, arr in rd.load_trips('exam', source='ZTBus'): ...                               # bus legs (1 Hz)
base  = rd.baselines('ved_exam'); summ = rd.baselines('summary')
sp    = rd.splits()                         # the frozen split file (proc/splits.json)
```

**Rule for Phase 2:** only `dev` may be looked at for calibration, gates and any fitting. `exam` is for the final
report only. `train_noise` trips carry real speed/grade/weather inputs; their labels are never used (they have none).

## 1. What is where

| Path | What |
|---|---|
| `proc/ved/trips/VED_<veh>/<trip_id>.npz/.json` | 33,015 VED trips at 1 Hz in the trip contract, one meta JSON each |
| `proc/ved/index.parquet` | all trip metas + quality flags (`q_label_ok`, `q_route_ok`, ...) |
| `proc/splits.json` | frozen splits (VED by vehicle; EPA/fueleconomy by model family; HD) + sha256 |
| `proc/epa/testcars_tests.parquet` | EPA test car lists 2008-2026, one row per test (80,227) |
| `proc/epa/testcars_vehicles.parquet` | one row per test vehicle configuration (28,553), ABC + per-cycle results |
| `proc/fe/fe_vehicles.parquet`, `ved_label_match.parquet` | fueleconomy.gov table; label match for each VED vehicle |
| `proc/cycles/*.npz/.json`, `index.parquet` | 24 dynamometer schedules in trip format |
| `proc/hd/ztbus/*.npz/.json`, `ztbus_legs.parquet`, `ztbus_missions.parquet` | Zurich bus legs (1 Hz) and 1,409 mission labels |
| `proc/hd/mersin_trips.parquet` | 1,456 diesel/CNG bus trip records (no traces) |
| `proc/hd/published_truck_results.csv` | measured tractor-trailer fuel use on 5 cycles (ICCT 2018) |
| `proc/baselines/` | baseline predictions (`ved_exam.parquet`, `ved_dev.parquet`) and `summary.json` |
| `proc/checks/` | `dem_vs_epqs.csv`, `qc_summary.json`, `ztbus_trace_vs_meta.csv` |
| `proc/dem_annarbor_2m.npz` | 2 m bare-earth elevation grid over the VED area (UTM 17N, uint16 cm) |
| `raw/` | original downloads (VED .7z kept; extracted CSVs deleted after conversion) |

## 2. Sources, licences, citations

| Source | Licence | Citation / URL |
|---|---|---|
| Vehicle Energy Dataset (VED) | Apache-2.0 | G. S. Oh, D. J. LeBlanc, H. Peng, "Vehicle Energy Dataset (VED), a large-scale dataset for vehicle energy consumption research", IEEE T-ITS 2020, doi:10.1109/TITS.2020.3035596, arXiv:1905.02081. Data: raw.githubusercontent.com/gsoh/VED |
| USGS 3DEP 1 m DEM, MI_31Co_Washtenaw_2016 (4 tiles) | US public domain | USGS The National Map, via TNM Access API (prd-tnm.s3.amazonaws.com). Checked against USGS EPQS point service |
| ASOS hourly weather KARB, KYIP, KDTW | free use, credit IEM | Iowa Environmental Mesonet, mesonet.agron.iastate.edu/cgi-bin/request/asos.py |
| EPA "Data on Cars Used for Testing Fuel Economy" 2008-2026 | US public domain | epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy |
| fueleconomy.gov vehicles.csv | US public domain | fueleconomy.gov/feg/epadata/vehicles.csv.zip (US DOE/EPA) |
| EPA dynamometer drive schedules | US public domain | epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules |
| 40 CFR 1037 App. A (HD transient) and App. D (HD grade profile) | US public domain | eCFR API, ecfr.gov |
| FASTSim 2.1.5 cycle resources (WLTC 3b, long-haul truck day) | Apache-2.0 (NREL / Alliance for Sustainable Energy) | pypi.org/project/fastsim/2.1.5 (wheel resources) |
| ZTBus | CC-BY-4.0 | F. Widmer et al., ZTBus dataset, ETH Zurich, doi:10.3929/ethz-b-000626723; files from the Zenodo mirror doi:10.5281/zenodo.17470816 (10 sample missions + metadata of all 1,409) |
| Mersin route 33M bus trips | CC-BY-4.0 | Y. Abut, Zenodo doi:10.5281/zenodo.22716221 (assoc. manuscript in Energy Sources Part A, submitted) |
| ICCT tractor-trailer tests | ICCT publication (cite) | F. Rodriguez, O. Delgado, R. Muncrief, "Fuel consumption testing of tractor-trailers in the European Union and the United States", ICCT Briefing, May 2018, Table 3/4, Fig. 8 |

## 3. Trip format (shared contract) and VED processing

Arrays (float32, 1 Hz, SI): `t, v, grade, elev, lat, lon, heading, T_amb, wind_speed, wind_dir, rho_air, surface,
fuel_rate [g/s], engine_rpm, batt_current [A, + = discharge], batt_voltage [V], soc [0-1], aux_power [W]`.
Extras on VED trips: `oat` (car's own outside-air sensor, degC), `valid` (uint8, see gaps). Extras on ZTBus legs:
`elec_power [W]` (the label), `passengers`, `traction_force [N]`, `valid`. Missing outputs are NaN arrays.

VED steps (`ved_stage1.py`, `ved_stage2.py`):
1. **Time.** `DayNum` is days since 2017-11-01 00:00 **UTC** (checked: the OAT-vs-ASOS lag is best at 0 h, and the trip
   hour histogram has its minimum at 06-08 UTC = 01-03 local). `meta.date` is UTC; `meta.hour_local` is Detroit time.
2. **Resampling.** The logger writes a row only when some value changes and repeats (holds) the others. Each signal is
   interpolated through its change points (plus the end of long plateaus) onto a 1 Hz grid.
3. **Gaps.** A trip is split into segments at gaps > 60 s (2.8% of trips; ids get `_k`). A second is `valid` if a
   row is within 5 s, or if the car is stationary at both ends of the gap (quiet stop: a hybrid with its engine off
   writes no rows for up to a minute). Measured outputs are NaN in invalid seconds; inputs are interpolated.
4. **Fuel.** VED paper Algorithm 1: OBD fuel rate L/h if logged (Volts, one diesel) x 745 g/L (gasoline) or 832 g/L
   (diesel); else MAF x (1 + (STFT + LTFT)/100) / 14.08 (E10 stoichiometric; trims averaged over banks); else
   absolute load x 1.184 g/L x displacement x rpm/120 x same correction. Fuel = 0 when rpm < 100 (engine off: the MAF
   PID holds a stale ~0.9 g/s). `meta.fuel_source` says which (MAF: 298 vehicles; load x rpm: 53; fuel-rate PID: 13).
5. **Battery.** VED current is negative on discharge; flipped to the contract. Power = V x I (sampled ~5 s).
6. **Aux.** `aux_power` = AC kW x 1000 (Volt-type) or AC W, plus heater W (EVs). Logged only on PHEV/EV.
7. **Elevation / grade** (section 5).
8. **Weather** (section 6).
9. **Quality flags** in meta: `q_len` (>= 1 km and >= 120 s), `q_gaps` (invalid seconds <= 5% and <= 5 s of speed
   dropout, i.e. v=0 while GPS moves > 8 m/s), `q_fuel` (coverage >= 95%, rate < 25 g/s, ICE: 2 < L/100 km < 60),
   `q_elec` (coverage >= 95%), `q_elev` (on the DEM). `q_label_ok` = all that apply to the architecture;
   `q_route_ok` = length, gaps, elevation (use for route inputs). Other meta: `fuel_g`, `elec_kWh_net_discharge`,
   `L_per_100km`, `v_start/v_end`, `idle_frac`, `gap_frac`, `quiet_held_s`, `speed_dropout_s`, `accel_spikes`,
   `gps_len_over_odo`, `elev_closing_frac`, `elev_gain_m`, `oat_mean`, `wet_frac`, `snow_frac`, `aux_logged`.

## 4. Coverage

VED as published covers 383 cars (384 ids), Nov 2017 - Nov 2018, Ann Arbor only. After the publisher's privacy
trimming (geofence + trip ends cut, section 7) it holds **33,015 trips, 104,300 miles, 4,743 h** (the paper's 374,000
miles is the raw collection). Trips are short: median 4.2 km / 408 s (10-90%: 1.3-10 km).

Label-quality trips (`q_label_ok`) by contract architecture and split:

| architecture | split | vehicles | trips | miles |
|---|---|---|---|---|
| conv | dev | 8 | 645 | 2,128 |
| conv | exam | 240 | 14,469 | 49,183 |
| hev_powersplit | dev | 5 | 275 | 913 |
| hev_powersplit | exam | 84 | 7,994 | 27,878 |
| hev_parallel_p2 | exam | 2 | 247 | 738 |
| phev_series (Volt-type) | dev / exam | 1 / 11 | 177 / 977 | 420 / 3,303 |
| phev_powersplit (Energi/Prius PHV) | dev / exam | 1 / 11 | 143 / 1,939 | 391 / 8,187 |
| bev (2013 Leaf) | dev / exam | 1 / 2 | 258 / 167 | 931 / 417 |

`train_noise`: 17 vehicles (15 ICE, 2 HEV) with no fuel signal at all: 1,092 route-quality trips, ~5,000 miles.
Total label-quality: 27,291 trips, 94,490 miles. Ambient temperature (ASOS) over trips: -26.4 to +33.9 degC
(5/50/95%: -8.9 / 7.6 / 28.5 degC); 12% of trips wet, 6% in snow (surface code from ASOS weather).

EPA test cars: 80,227 tests on 28,553 vehicle configurations, MY 2008-2026: 25,242 conv, 1,303 HEV, 457 PHEV,
1,502 BEV, 49 FCEV. Target road-load ABC present for all; FTP results for 26.6k, HWY 19.6k, US06 4.2k, SC03 3.0k,
cold FTP 2.9k configurations; charge-depleting UDDS/HWY kWh/mi for ~850 BEV/PHEV.

## 5. Elevation and grade: verdict = trustworthy (after cleaning)

- **GPS was not shifted.** The VED paper (Sec. III) de-identifies by *suppression*: random "fog" removes the start/end
  of each trip, a rectangular geofence keeps only Ann Arbor, and trips are cut to their first/last major intersection.
  Points that remain are real positions, so elevation lookup is meaningful. GPS updates every ~3-5 s (interpolated).
- **DEM.** USGS 3DEP 1 m lidar (Washtenaw 2016), block-averaged to 2 m and stored locally. It is the same data EPQS
  serves: 40 EPQS points agree to 0.023 m median, 0.155 m max (`checks/dem_vs_epqs.csv`). A bulk tile download was used
  instead of hundreds of thousands of EPQS calls, to be gentle on the service.
- **Cleaning.** Sample every 5 m along the speed-integrated distance; morphological closing over 120 m (bare-earth DEMs
  show the river/road *under* bridges, and GPS points sometimes fall into ditches: both make narrow false dips); Gaussian
  smoothing sigma 40 m; grade = dz/ds clipped to +-15%. Closing changes elevation by > 2 m on 1.6% of a typical
  track (95th pct 5.8%).
- **Checks** (`checks/qc_summary.json`): on 46,119 road cells (20 m, 8 heading sectors) driven by >= 5 trips of >= 3
  vehicles, the between-trip std of grade is **0.36% grade** (p90 0.9%) against a 2.1% spread between cells: **93% of
  the variance is repeatable road signal**. Opposite directions on 20,100 cells: grade(fwd) vs -grade(back)
  correlation **0.97**, median |sum| 0.22% grade. Typical |grade| 1.1% (p90 3.5%, p99 6.8%); 7.6 m climb per km.
- **Limits.** Repeatability does not rule out a common bias (e.g. a long bridge wider than the 120 m closing is still
  a dip; real sags under rail bridges are partly filled). GPS path length / speed-integrated distance: median 1.011
  (5-95%: 0.98-1.06), so speedometers read ~1% low or GPS cuts corners; distance uses the speedometer.

## 6. Weather join: verdict = good for temperature and pressure, rough for wind

- KARB (Ann Arbor airport, ~6 km south of downtown) supplies 99.1% of trips; KYIP then KDTW fill gaps > 2 h.
  Hourly obs are linearly interpolated in time (wind via u/v components). `rho_air` = dry + vapour partial densities
  from station pressure (from altimeter setting), dew point and T, with pressure moved hypsometrically to the car's
  elevation. `surface`: 2 (wet) if the last two ARB reports within 75 min had rain/drizzle/thunder or precip > 0;
  4 (snow) for snow/ice pellets/freezing codes; else 0 (no snow-on-ground information).
- Spatial representativeness: KARB vs KYIP (17 km apart) differ by 1.7 degC median (p95 3.9 degC), wind 1.0 m/s,
  pressure 0.25 hPa. Treat T_amb as +-2 degC. Wind is airport 10 m wind: wind at car height in town is lower (often
  x0.5-0.7) and gusty; a simulator should treat wind as uncertain.
- **VED's own OAT** (100% for EV/PHEV, ~60% for ICE/HEV, 1 degC resolution) is a poor reference: on 3,000 trips
  (second half of trip) OAT - ASOS = +1.2 degC median, MAD 3.7 degC, slope 0.90, corr 0.86; 36 of 213 vehicles are
  biased by > 5 degC (sensor heat soak, sun, bad PIDs). T_amb therefore comes from ASOS; OAT is kept as `oat`.
- AC/heater power columns exist only for PHEV/EV (Volt: AC kW; Ford/Toyota PHEV and Leaf: AC W, Leaf heater W).
  ICE/HEV HVAC load is not logged: Phase 2 must model it from T_amb.

## 7. Data problems Phase 2 must know

1. **VED trips are fragments.** 96% start with v > 1 m/s; the engine is usually warm; a trip's start/end kinetic
   energy is not zero. Simulate from the logged initial speed. Cold starts are mostly cut away.
2. **Fuel is estimated, not metered**, for 351 of 384 vehicles (MAF x trims / AFR, or load x rpm). MAF-based fuel
   ignores deceleration fuel cut-off only partly (MAF drops, but not to 0) and assumes stoichiometric mixture, so
   heavy-load enrichment is missed. No vehicle logs both a fuel-rate PID and MAF, so the bias cannot be measured here.
   As a sanity check the fueleconomy label baseline is only +1% biased on ICE exam trips and +5% on HEV. The
   load x rpm vehicles (`fuel_source == 'absload_rpm_displacement'`, 53 ICE) are less reliable.
3. **VED hides make/model/year.** Architecture of HEV/PHEV is inferred from engine size (confidence in
   `ved_vehicles().arch_confidence`); mass is "Generalized_Weight" (curb weight rounded to 500 lb). CdA, Crr, gearing are
   unknown. fueleconomy.gov matches are candidate *sets* (median label; `match_confidence`: 29 high, 174 medium,
   180 low, 1 none).
4. **Logging gaps.** Moving gaps (> 5 s) are real dropouts; they are NaN in outputs. HEVs needed the quiet-stop rule.
5. **PHEV mode is unknown** per trip (charge-depleting vs sustaining). Total energy = fuel x 42.0 MJ/kg + battery
   kWh x 3.6 is the PHEV target used in the baselines.
6. **Few EVs**: 3 Leafs (1 dev, 2 exam with 167 label trips). EV exam numbers are thin.
7. **EPA test-car quirks** fixed: 2008-09 files use coded column names and cubic inches; EVs carry a 100 L
   placeholder displacement; electric FE values are labelled "MPG" but mix MPGe, values of 10-50 (probably kWh/100 mi
   or blended PHEV mpg; `fe_unit_flag = ambiguous_10_50`, not converted) and 0/999/1000 placeholders; 107 conventional
   values of 10000 or < 3 mpg dropped. Architecture: BEV/PHEV from test fuels and displacement; HEV and some PHEV
   from name patterns (`arch_source`). EV kWh/mi = 33.705 / MPGe (EPA definition; recharge-energy based).
8. **Units of fueleconomy labels**: `city08/highway08` are window-sticker (5-cycle adjusted) values; `UCity/UHighway`
   are unadjusted lab values, closer to dyno physics. Label kWh/100 mi includes charging losses (wall to wheels).
9. **Heavy duty is thin** (section 8): 2 buses with traces, 1 bus route without traces, 3 trucks as published numbers.

## 8. Heavy duty

- **ZTBus (real, 1 Hz):** two articulated electric trolleybuses with traction batteries (VBZ Zurich, buses 183 and 208).
  The full dataset (1,409 missions) was not downloadable: the ETH repository answered HTTP 429 and was not retried
  repeatedly. The Zenodo mirror provides 10 complete missions (1 Hz: speed, electrical power demand, GNSS
  lat/lon/altitude, passengers, traction force, brake, door, ambient T) and mission-level metadata for all 1,409
  missions (distance, energy, passengers, ambient T). Integrated power reproduces the metadata energy to 1e-5
  (`checks/ztbus_trace_vs_meta.csv`). Missions are split into 83 legs at layovers > 5 min. Label = `elec_kWh`
  (total electrical demand, traction + auxiliaries, whether from the overhead line or the battery). Leg energy
  1.48 kWh/km median (0.76-3.4). Grade from smoothed GNSS altitude (no lidar DEM for Zurich here); no wind data (0).
- **Mersin 33M (real, trip-level):** 728 diesel and 728 CNG bus trips on one 23.4 km route with 41-64 stops: fuel per
  trip (diesel litres from depot metering; CNG m3 from tank pressure drop, integer m3, i.e. ~6% resolution), duration,
  stops, passengers, AC flag. Diesel ~65 L/100 km. No traces: Phase 2 needs a synthetic route (stops, speed
  profile) with these trip statistics.
- **Trucks: no public trip data with fuel was reachable.** Fleet DNA on OpenEI only links to the NREL/NLR site (blocked;
  not used); the MOVES HD activity set has starts/idle only; Zenodo/figshare searches found nothing with fuel + traces.
  Fallback: **measured chassis-dynamometer results** of 3 tractor-trailers (ICCT 2018, testing by WVU and TU Graz):
  L/100 km on 55 mph with the 40 CFR 1037 grade profile, 55 mph flat, ARB transient (all three traces are in
  `cycles()`), plus EU Long Haul / Regional Delivery (traces not available). Payload 17.2 t (US cycles), curb mass with
  trailer and 500 L fuel 14,865-17,208 kg, coast-down Cd 0.438-0.471, 12-speed AMT, 300-330 kW. 15 numbers in total.
- Cycles with HD relevance: LA92 (class 3 HD), HD transient (1037 App. A), 55/65 mph cruise with grade (1037 App. D),
  55 mph flat, FASTSim long-haul day (83,000 s with grade). The EPA heavy-duty UDDS is "[Reserved]" in today's CFR
  (and in every eCFR version back to 2017) and was not found elsewhere; HHDDT creep/cruise modes and the Manhattan /
  OCTA / CBD bus cycles were not found at a documented public source (A can define CBD from SAE J1376 if needed).

## 9. Splits (`proc/splits.json`, seed 20261112)

- **VED, by vehicle**, drawn before any label was computed: dev = 8 ICE, 5 HEV, 1 Volt-type PHEV, 1 power-split PHEV,
  1 EV (16 vehicles, from labelled vehicles with >= 20 trips): VED_149, 241, 250, 258, 283, 345, 375, 435, 440, 546,
  577, 599, 606 (ICE+HEV), VED_537 (Volt gen 2), VED_11 (Ford Energi), VED_455 (Leaf). exam = the other 351 labelled
  vehicles. train_noise = the 17 vehicles without any fuel signal.
  *Why this design:* the unlabelled vehicles are a free, leak-proof source of real Ann Arbor routes, so no labelled
  vehicle has to be sacrificed. Phase 2 route inputs for training = these 1,092 trips + all dynamometer cycles + synthetic
  routes. Note that exam trips drive the same city roads (route distribution is shared, labels are not). One EV is in
  dev (instead of a second PHEV) because the BEV physics needs at least one calibration unit.
- **EPA test cars + fueleconomy.gov, by model family** (make + base model name, shared by both tables, so the same car
  line is never in dev and exam): 10% of families per architecture stratum = dev (210 families; 3,095 conv, 73 HEV,
  61 PHEV, 140 BEV EPA configurations in dev).
- **HD:** ZTBus by mission (only 2 physical buses exist): one seeded traced mission per bus is dev, everything else exam
  (78 exam legs, 1,407 exam missions under the v1 rule; the full set uses the by-month split below). ICCT trucks: BIC-US is dev, BIC-EU and AVG-EU exam. Mersin: the 7-day
  short-term subsample (16-22 Feb 2026) is dev, the rest exam.

## 10. Baselines (fitted on dev only) and their exam errors

Target per VED trip (`q_label_ok`): ICE/HEV fuel g, EV battery kWh, PHEV total MJ. MdAPE = median |pred - y| / y over
trips; "veh" = median error of per-vehicle totals. Predictions and labels both use only `valid` seconds.

| VED exam | n trips (veh) | label MPG x distance | VSP op-mode (MOVES-like) | class median |
|---|---|---|---|---|
| ICE fuel | 14,469 (240) | MdAPE 13.0%, MAPE 16.5%, veh 8.6% | **10.3%**, 12.1%, veh 9.6% | 22.8%, 24.1%, veh 22.0% |
| HEV fuel | 8,241 (86) | 16.2%, 24.2%, veh **8.9%** | **15.3%**, 20.1%, veh 12.7% | 17.2%, 24.3%, veh 9.0% |
| EV kWh | 167 (2) | 21.5%, 28.2%, veh 15.1% | **19.0%**, 23.2%, veh 16.6% | 27.1%, 31.9%, veh 21.9% |
| PHEV total MJ | 2,916 (22) | 46.1%, 63.2%, veh **14.4%** | **41.3%**, 40.5%, veh 35.7% | 42.7%, 42.5%, veh 52.4% |

- *Label:* fueleconomy.gov window-sticker city/highway MPG (or kWh/100 mi), mixed in fuel-per-distance with city weight
  w = clip((48.3 - v_mean_mph)/(48.3 - 21.2), 0, 1). PHEV: battery share of total energy fitted on dev (0.73).
- *VSP:* VSP = v(1.1a + 9.81 grade + 0.132) + 0.000302 v^3 (Jimenez-Palacios); 29 op-modes (idle, braking,
  3 speed bands x 9 VSP bins); mean rate per op-mode per VED type from dev; per vehicle x exp(c)(displacement/2 L)^gamma
  (dev fit: gamma 0.56 ICE, 0.59 HEV).
- *Class median:* dev median energy per km for (VED type, body class).
- EPA exam (`summary.json`): log-linear regression on dev (log ETW, road-load force at 50 mph, displacement, rated hp)
  gives MdAPE 8.8% (FTP gal/mi), 7.0% (HWY), 6.9% (US06) for conv; 21.5% / 16.1% for HEV FTP/HWY; 9.8% / 6.5% for BEV
  CD UDDS/HWY kWh/mi. Dev-median: 17-50%.
- HD exam: ZTBus legs, dev kWh/km x distance: MdAPE 22.1% (missions: 14.9%); Mersin dev median fuel per trip: 8.6%
  diesel, 7.1% CNG; ICCT trucks, dev truck's value on the same cycle: 7.0% (6 numbers).

## 11. Rebuild

```
python3 -I scripts/ved_stage1.py raw/ved/x1   # after extracting each .7z part (py7zr); deletes CSVs
python3 -I scripts/ved_survey.py; scripts/make_splits_ved.py; scripts/dem_mosaic.py
FORCE=1 python3 -I scripts/ved_stage2.py; scripts/ved_index.py
scripts/epa_testcars.py; scripts/fe_labels.py; scripts/cycles.py; scripts/hd.py; scripts/make_splits_all.py
scripts/baselines.py; scripts/qc.py
```

### One command: `mlratchet data vehicle` (13 Nov 2026)

```bash
mlratchet data vehicle                 # everything; safe to re-run, resumes where it stopped
mlratchet data vehicle --list          # stages, and which downloads are still missing
mlratchet data vehicle --only tables,verify    # or --from stage2, --offline, --jobs 8
```

The work is in `data/build.py` (also runnable as `python -I shared/vehicle/data/build.py`). It needs `py7zr`,
`rasterio` and `openpyxl` on top of numpy/pandas/scipy/pyarrow. Stages, in order:

| Stage | What | Time here (16 cores) |
|---|---|---|
| `fetch` | every raw file missing from `raw/<source>/` (2.05 GB; resumes partial files; generic User-Agent; IEM one request at a time) | 21 min (the four DEM tiles, 1.74 GB) |
| `stage1` | VED .7z -> `proc/ved_stage1/`, CSV shards in parallel | 31 s |
| `survey`, `splits_ved` | signal survey; `make_splits_ved.py` must reproduce the frozen VED split (it does, 384 vehicles), then the frozen part is used | 20 s |
| `dem` | `proc/dem_annarbor_2m.npz` (tiles deleted afterwards) | 8 s |
| `stage2`, `index` | 33,015 trips in `proc/ved/trips/` (16 buckets in parallel), `proc/ved/index.parquet` | 102 s + 6 min |
| `tables` | EPA, fueleconomy, cycles, HD, splits, baselines, QC, built in a shadow root `data/work/rebuild/` | 2.5 min |
| `refdata` | `roadsim/refdata/` copied from the same raw files, `roadsim/tools/epa_candidates.py`; shadow runs of `build_cycles.py` and `build_generator_calib.py` | 30 s |
| `verify` | splits.json sha256, rebuilt vs shipped tables, a dev-trip load; report in `data/work/rebuild_report.json` | 5 s |

Tracked files are never overwritten: tables are rebuilt in the shadow root and compared. `proc/splits.json` is never
written; the rebuilt one is compared and the frozen one is used. Result of the first full run (Windows 11, Python 3.14,
pandas 3.1): splits sha256 d917df78… unchanged, and the rebuilt splits file has the same content. All 236 shipped
tables match: `baselines/summary.json`, `ved_exam/ved_dev.parquet`, every EPA/fueleconomy/cycle/ZTBus/Mersin table and
the roadsim `cycles.npz`/`generator_calib.json` are equal in value. The only differences are four grade statistics in
`checks/qc_summary.json`, at about 1e-8 relative (float noise). `checks/dem_vs_epqs.csv` is not rebuilt (it needs 40
EPQS calls; run `scripts/dem_check_epqs.py` by hand).

Choices the scripts did not record, fixed by matching the shipped tables: IEM ASOS with routine and special reports
(routine only moves trip T_amb by up to 1.3 degC), 2017-10-31 to 2018-11-13 UTC (gives QC's 8,830 joint ARB/YIP hours);
eCFR at the latest issue date. `build_maps.py` needs the EPA ALPHA packages in `roadsim/refdata/alpha/`, which this
command does not fetch.

## Fixes after Phase 2 (13 Nov 2026, found by the EnergyNet agent; splits.json unchanged, sha256 d917df78…)
- **EPA test-car cycle labels were swapped.** Procedure code 3 is HWFE (highway) and code 21 is the federal 2-day FTP; the
  build mapped them the other way. Fixed in `scripts/epa_testcars.py` and rebuilt. Conventional medians are now FTP 24.5 mpg,
  HWY 37.6 mpg. The EPA baselines were recomputed: conventional regression FTP 7.1 %, HWY 5.1 %, US06 6.9 %; HEV FTP 23.4 %,
  HWY 21.5 %; BEV CD UDDS 9.8 %, HWY 6.5 %. Phase 2 EnergyNet graded on corrected labels (its own `epa_fix.py`). TripDetective
  and HybridPilot did not use EPA cycle results.
- **VED vehicle mass counted the 136 kg driver load twice** (`mass_kg` already included it and `payload_kg` added it again
  inside roadsim). `mass_kg` is now curb mass only. Phase 2 models were trained or graded before this fix. TripDetective grades
  curb mass and is unaffected. EnergyNet and HybridPilot twins of VED cars ran 136 kg heavy.

## ZTBus full data set: all 1,409 missions (13 Nov 2026)

RoadData v1 had 10 traced ZTBus missions (Zenodo mirror; the ETH host answered HTTP 429 from the cloud sandbox). From
this machine the ETH Research Collection answered, and the full data set is now built.

```python
zl = rd.ztbus_full()                         # 7,567 legs, one row each (split, label, passengers, route, season, ...)
zm = rd.ztbus_full(table='missions')         # 1,409 missions: ETH metaData.csv + trace checks + split
for meta, arr in rd.load_trips('dev', source='ZTBus_full', label_ok=True, route_ok=True): ...   # 1 Hz legs
m, a = rd.load_trip('ZTBusFull_B183_2021-10-05_13-53-20_2021-10-05_18-14-53_leg00')
```

**Source.** F. Widmer, A. Ritter, C. H. Onder, "ZTBus: A Large Dataset of Time-Resolved City Bus Driving Missions",
Scientific Data 10 (2023), doi:10.1038/s41597-023-02600-6 (cite this paper). Data: ETH Research Collection,
doi:10.3929/ethz-b-000626723 (hdl:20.500.11850/626723), licence CC BY 4.0. Two VBZ Zurich trolleybuses (HESS lighTram 19
DC: single-articulated, ~19 m, curb ~19 t, ~160 passengers max, traction battery for a few km off the wire), bus 183
(864 missions, Apr 2019 - Dec 2022) and bus 208 (545 missions, 2021-2022).

**Download** (`mlratchet data vehicle --only ztbus_full`; files in `raw/hd/ztbus/full/`, git-ignored). The DSpace REST
API lists the item's files with MD5s (`/server/api/core/bitstreams/<uuid>/content`). Fetched one request at a time,
resumable, with a back-off of 2, 5, 10, 20, 30, 30 min on HTTP 429 (then recorded as blocked in
`work/build_state.json`):

| file | bytes | MD5 | used |
|---|---|---|---|
| `ZTBus_compressed.zip` | 2,006,877,986 | 65e265fd… (checked) | yes: all 1,409 mission CSVs, deflated |
| `README.txt` | 15,754 | bf852195… (checked) | column definitions |
| `ZTBus.zip` | 8,893,495,239 | f71dc952… | no: the same CSVs stored without compression |
| `metaData.csv` | 272,886 | 7dd57860… | the copy already in `raw/hd/ztbus/` has the same MD5 |

The download took 17 min at ~2 MB/s. The host returned 429 once, during the first look at the item (about 10 API calls
within a minute). It served the files after a few minutes' pause.

**Processing** (`scripts/hd.py ztbus_full`, 45 s on 15 processes). The CSVs are read straight out of the zip. Each
mission is cut into legs with the v1 code: legs split at stops > 300 s (a v < 0.3 m/s run), 30 s of the stop kept on each
side, legs >= 120 s. Arrays and label are as in section 8. Extra arrays: `grid_available` (1 = current collector on the
overhead line) and `door_open`. Extra leg meta: `regen_kWh`, `passengers_mean/max`, `route` (most frequent ITCS route),
`grid_available_frac`, `door_open_frac`, `n_door_openings`, `idle_frac`, `month`, `season` (meteorological, UTC month),
coverage fractions (`gnss_frac`, `passengers_frac`, `T_amb_frac`: share of seconds with a logged value), `q_label_ok`
(v1 rule: > 1 km and > 95% of seconds logged) and `q_route_ok` (>= 90% logged GNSS seconds, so grade and position are
real).
Outputs: `proc/hd/ztbus_full/trips/B183|B208/*.npz/.json` (1.3 GB, git-ignored), `legs.parquet`, `missions.parquet`,
`summary.json` and `proc/hd/ztbus_full_splits.json` (tracked).

**Checks.** All 1,409 missions in the zip match `metaData.csv`, and none failed. Integrated power / metadata energy:
1.00001 mean, 1.00011 max. Integrated speed / metadata distance: 1.00004 mean, 1.0001 max. The 10 sample missions give
the same 83 legs as v1 (`proc/hd/ztbus/`). Seven are bit-identical; three differ by at most 6e-11 (float noise; v1 was
built on another machine). The v1 HD tables still rebuild unchanged (`verify`).

**Size.** 1,409 missions, 7,567 legs (7,490 `q_label_ok`; the other 77 are <= 1 km), 12,814 h, 207,528 km, 328 MWh net.
Legs: median 15.2 km / 56 min (p10-p90 9.0-63.5 km). Missions last 3-24 h (median 8.9 h) with 1-31 legs. Routes by
leg-km: 83 (48%), 33, 72, 46, 32, 31, night routes N1-N4.

**Split** (`proc/hd/ztbus_full_splits.json`; sha256 of canonical JSON (sorted keys, no whitespace, UTF-8) without the sha
field = `18e0fead…`; written once, then only recomputed and compared). **Changed on 13 Nov 2026, in the open, with the
owner's approval.**

- **v1 rule** (`proc/splits.json` HD.ZTBus, unchanged and still hashed): one seeded traced mission per bus is dev, all
  other missions are exam. Applied to the full set it gives 2 dev missions (5 legs) and 1,407 exam: the simulator could
  calibrate on only 2 bus missions.
- **In force since 13 Nov 2026: a by-month split**, fixed on 13 Nov before any full-data trace was processed. The unit is
  the (bus, calendar month) block. Within each (bus, season) stratum, max(1, round(0.2 n)) months are drawn with seed
  20261113, plus the months that hold the two v1 dev missions; every mission takes the side of its month, **except the
  v1 trace-exam sample missions, which stay exam** (one of them, `B208_2021-05-27…`, fell in a dev month and is pinned),
  so v1's ZTBus trace exam (78 legs) is unchanged.

| side | missions | legs |
|---|---|---|
| dev | 294 | 1,662 |
| exam | 1,115 | 5,905 |

292 missions moved from v1-exam to dev. They had never been processed as traces, but EnergyNet v1 graded the
**mission totals** of all 1,409 missions as exam. So every v1 number on ZTBus mission totals is re-scored on the 1,115
missions that are exam now, before any v1-vs-v2 comparison. `missions.parquet` keeps the v1 rule as `split_v1_rule`;
loaders use `split`.

**Energy per km** (legs with `q_label_ok`, pooled kWh / km; all splits, descriptive only):

| season | legs | km | kWh/km | median (p10-p90) | mean T_amb |
|---|---|---|---|---|---|
| winter (DJF) | 846 | 25,702 | 1.89 | 1.88 (1.53-2.21) | 6.3 degC |
| spring | 2,062 | 53,299 | 1.58 | 1.55 (1.17-2.02) | 13.8 degC |
| summer | 2,372 | 62,253 | 1.48 | 1.46 (1.09-1.82) | 23.9 degC |
| autumn | 2,210 | 66,257 | 1.56 | 1.54 (1.18-1.93) | 14.0 degC |

| mean passengers on leg | 0-10 | 10-20 | 20-40 | 40-60 | 60+ |
|---|---|---|---|---|---|
| legs | 2,025 | 3,582 | 1,814 | 67 | 2 |
| kWh/km (all seasons) | 1.55 | 1.56 | 1.62 | 1.78 | 2.49 |
| winter / summer kWh/km | 1.80 / 1.37 | 1.89 / 1.47 | 1.93 / 1.55 | 2.20 / 1.71 | - / 2.61 |

By leg mean temperature: 1.98 kWh/km at 0-5 degC, 1.40 at 15-25 degC, and 1.56 above 25 degC (heating and AC both cost
energy). Bus 183: 1.60, bus 208: 1.55 kWh/km. Regeneration returns 30% of the positive electrical energy. The bus is on
the overhead line for a median 66% of a leg. Mission level (metadata, all 1,409): winter 1.94, summer 1.48 kWh/km.

**What is missing or odd.**
- **Passengers are logged only at stops**: 0.9% of seconds (the ITCS count at each served stop). Like v1, the
  `passengers` array is linearly interpolated between stops, so it ramps where the true count steps. Treat it as a
  per-stop signal. 29 legs have zero passengers all through (probably empty runs to or from the depot).
- **No ambient temperature below 0 degC** in 4 winters. The minimum is exactly 273.15 K in 5 missions, and the sensor has
  1 K resolution and sits behind the front bumper. Cold readings are probably clipped at 0 degC or warm-biased, so
  winter T_amb is not reliable at the low end.
- **GNSS gaps**: 418 legs have < 90% logged GNSS (`q_route_ok` False), and one mission has 0.9%. On those legs grade
  and position come from interpolation or edge-hold.
- **Net-negative legs exist**: 2 legs with net regeneration, e.g. a 9.2 km leg that descends 223 m. One more is below
  0.1 kWh/km, and 46 legs are below 0.5 kWh/km. The two negative ones are real downhill runs; check the rest before calling them errors.
- No battery state (SOC, current) and no split between traction and auxiliary power: `elec_power` is the total demand
  (line + battery). No vehicle mass beyond "~19 t curb". No wind; grade comes from GNSS altitude (no lidar DEM for
  Zurich here).
- Time is UTC. `season` and `month` use the UTC start of the leg (local time is UTC+1/+2).
