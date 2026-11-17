# LoadCast v1.0 — model card

**What it does.** Forecasts a building's hourly energy on one meter (electricity, chilled water, steam, hot water or gas) for any span of hours, from weather, calendar and what is known about the building. Each hour gets a mean (kW) and a 90 % band.

Two modes:
- **cold** (no history): metadata only (use type, floor area, optional year built, which meters exist, site latitude and climate).
- **warm** (history): the same plus the meter's own past readings. Trained for 2 weeks, 8 weeks or a year of history. In warm mode the network predicts a *correction* to a TOWT model (time of week and temperature) fitted on that history.

**How it learned.** Only from simulated buildings (BuildingSim), with real noise added. No real meter reading is a training sample.

## Inputs

- `weather`: hourly UTC DataFrame with `T_out` (°C), `dewpoint` (°C), `GHI` (W/m²), `wind_speed` (m/s). Cover the history and forecast windows plus 7 days before each. The annual climate summary (mean T, HDD18, CDD18) comes from the forecast's calendar year(s) in the frame, so give at least a year (or a TMY).
- `tz`, `lat`; holidays are built in for the US, Canada (Ontario), UK and Ireland.
- `use`: office, education, lodging, retail, healthcare, assembly, warehouse, food, lab, other.
- `area_m2`, `meter`, `meters_present`, optional `year_built`, `hvac` (inferred from the meters by default), `sched` (unknown / standard / extended / 24x7).
- `history`: optional hourly kWh Series (UTC) of the same meter.

## Outputs

`kW`, `sigma`, `lo90`, `hi90` per hour. Error bars are widened by one factor per mode and meter type, fitted on the 52 development buildings only (the rule only widens):

| mode | electricity | chilled water | steam | hot water | gas |
|---|---|---|---|---|---|
| cold | 2.0 | 2.9 | 7.75 | 1.0 | 1.5 |
| warm, 1 year | 1.05 | 1.0 | 1.15 | 1.2 | 1.0 |
| warm, 8 weeks | 1.0 | 1.65 | 1.0 | 1.25 | 1.0 |
| warm, 2 weeks | 1.3 | **30 (cap)** | 1.25 | 4.0 | **30 (cap)** |

The two capped factors mean the 2-week band is unusable for chilled water and gas when the history is winter. Pass `calibrated=False` for the raw bands.

## Physics and simulation

- **Simulator:** BuildingSim (ML Ratchet Phase 1): ISO 13790 5R1C zone, DX / heat pump / chiller (DOE-2 EIR) / boiler / district plant, VAV and unitary air sides, DHW, real weather with pvlib solar. Checks (Phase 1, `shared/building/checks/*.json`, arriving with the platform zip): steady conduction vs closed form 1.8e-9; RC step response 0.0035 K; BESTEST 29 of 34 values inside the reference ranges; ResStock twins site energy median 1.10, ComStock twins 0.86 (aggregate 0.97).
- **LoadCast wrappers (no edit to the shared package, `training/lcsim.py`):**
  - country holidays (US / CA / UK / IE) swapped into BuildingSim's calendar per site;
  - a BDG2-style spec maker: which meters exist decides the plant (district steam/hot water/chilled water, gas boiler or RTU, or unknown);
  - operating knobs: HVAC for electricity-only buildings, operating hours (default / extended / 24-7), outdoor air ×1 or ×2.5, VAV minimum flow 0.3 or 0.8 with fixed 12.8 °C supply air;
  - meter calibration `y = a · sim + b · A`.
- **Dev calibration (`training/calib.py`, 52 BDG2 dev buildings, 2016 only):** each dev twin was simulated on the knob grid; per meter `a, b ≥ 0` were fitted hourly; the knob set with the lowest mean CV was kept.

  | meter (n) | generic twin sim/real | best knobs sim/real | a (median) | b W/m² (median) | hourly CV generic → calibrated (2016, in sample) |
  |---|---|---|---|---|---|
  | electricity (47) | 0.97 | 0.96 | 0.43 | 6.3 | 62 → 24 % |
  | chilled water (20) | 0.29 | 0.47 | 0.92 | 31 | 93 → 53 % |
  | steam (10) | 0.29 | 0.87 | 0.36 | 3.5 | 113 → 62 % |
  | hot water (7) | 0.18 | 0.57 | 0.40 | 6.0 | 171 → 79 % |
  | gas (2) | 0.14 | 0.14 | 0.0 | 62 | 115 → 76 % |

  Best knobs: 24-7 operation in 22 of 52 buildings, extended hours in 9; outdoor air ×2.5 in 37. Real campus electricity is flatter than simulated (a = 0.43 plus a large base load).
- **Training bank:** 1,600 generated buildings (BuildingSim generator: ComStock / CBECS-prior specs) at the 19 BDG2 sites under their real 2016-2017 weather, hourly. Use mix like BDG2; meter sets drawn from the dev buildings (80 %) or a fixed list; half the floor areas from the dev log-area distribution. Each building draws its knobs and `a, b` from a dev donor (60 % same use), jittered ×exp(N(0, 0.2)).
- **Real noise (`training/noisebank.py`, `dataset.py`):** for 85 dev building-meters (q_ok both years), the hourly residual of an in-sample regression (TOWT + 24 h mean temperature + dewpoint + solar) divided by the year mean: occupancy, campus breaks, events, meter quirks. One residual series is added to each simulated meter on the same local calendar dates, plus a real 2017/2016 level change (whole year or a step), occasional quantisation and the real gap pattern.
- History and target years: each simulated building is used 2016 → 2017 or 2017 → 2016 at random.

## Network

- Two MLPs (cold, warm): 3 hidden layers of 192 SiLU units, dropout 0.1, Gaussian head (mean, log σ). Cold 86 k parameters, warm 90 k.
- Inputs per hour: 38 metadata features, 10 weather features (T, 24 h and 168 h means, 72 h exponential mean, daily max, dewpoint and humid excess, GHI and 24 h mean, wind), 16 calendar features (hour harmonics, day of week, holiday, Christmas break, season, mid-summer). Warm adds 19 history features (TOWT and change-point predictions from the history, time-of-week profile, history mean, weeks of history, zero fraction, change-point coefficients, history temperature range, weekend / night / holiday ratios, recent level, day-to-day variability, temperature outside the history range).
- Units: cold predicts kW / (A × I_ref) (I_ref 15 W/m² for electricity, chilled water, steam; 10 for hot water, gas); warm predicts kW / history mean.
- Loss: weighted MSE on the mean (weight 1 / ybar², floored) + 0.1 × Gaussian NLL for σ with the mean detached. AdamW, cosine schedule, 30 epochs of fresh random hours, per-row jitter of the metadata (log area ±0.1, year ±6, climate ±0.5 K / ±8 %) to stop the network memorising simulated buildings.
- numpy == torch: max difference 1.5e-6 on real dev rows. The browser port (JavaScript) matches Python to 3e-4 relative (weather stored at 0.01 °C).

## Ratchet

Frozen set: 300 simulated buildings (v1 physics, seed 99, real noise seed 12345), 650 building-meters. Objective = mean over meter types of (cold skill + warm skill)/2, where cold skill = 1 − median CV / median CV of the set's own median-intensity × median-shape guess, and warm skill = 1 − median CV / median CV of TOWT on the same history. Keep a rung only if the objective rises by > 0.01 and the dev score (mean of cold and warm median CV over 84 dev building-meters, 2016 → 2017) is no more than 2 % worse. The objective definition changed twice before R3 (floored pooled skills → unfloored → per meter type); every rung was re-scored each time (`ratchet.json`).

| rung | change | frozen objective | dev score (CV %) | decision |
|---|---|---|---|---|
| R0 | generic Phase 1 physics, 800 buildings, real dev noise | −0.174 | 59.9 | kept (first) |
| R1 | dev-calibrated physics (knobs + meter a, b from the 52 dev fits), 800 buildings | −0.118 | 54.9 | kept |
| R2 | + anti-memorisation: metadata jitter, dropout 0.1, weight decay 1e-3 | −0.018 | 51.9 | kept |
| R3 | + full bank (1,600 buildings) | **+0.014** | **49.2** | **kept (released)** |
| R4 | cold loss weight 1/ybar^0.5 (aim magnitude at the median) | −0.026 | 52.2 | rejected (both tests) |
| R5 | real noise matched by use type | +0.006 | 51.7 | rejected (both tests) |

**Twin test (dev, R3):** each dev building's calibrated twin, given another dev building's real residuals, read by the networks like the real meter. Median CV real vs twin: cold electricity 62.5 vs 51.8 (per-building correlation of log CV r = 0.86), chilled water 101 vs 99 (r = 0.96), steam 104 vs 108 (r = 0.95); warm electricity 22.3 vs 17.8, chilled water 43.2 vs 41.7, steam 61.7 vs 59.9 (r ≈ 0). Cold errors are reproduced building by building (they are magnitude errors); warm errors match in size but not in which building is hard, since the noise is borrowed.

## Exam (BDG2 exam buildings, forecast local 2017 with actual weather, history = 2016)

1,502 building-meters with q_ok in both years. Median CV(RMSE) % [IQR], median NMBE %, share meeting G14 hourly (CV ≤ 30 %, |NMBE| ≤ 10 %), mean 90 % coverage. LoadCast uses p = 0, BuildingData's fitted baselines n − p. Persistence rows read the real 2017 meter (nowcasts, not forecasts).

| meter (n) | LoadCast cold | LoadCast warm 2 wk | warm 8 wk | **warm 2016** | median guess | TOWT | change point | persistence 24 h* |
|---|---|---|---|---|---|---|---|---|
| electricity (856) | 59.3 | 30.1 | 23.8 | **18.8** [12.9–26.7] | 53.8 | 19.6 | 19.1 | 18.6 |
| chilled water (323) | 104.1 | 87.3 | 62.6 | **42.9** [28.2–67.1] | 101.6 | 45.7 | 42.4 | 40.3 |
| steam (225) | 102.8 | 57.1 | 49.1 | **45.5** [33.5–71.4] | 111.6 | 39.8 | 40.5 | 41.9 |
| hot water (72) | 134.0 | 93.6 | 86.5 | **94.5** [59.7–154] | 207.6 | 83.1 | 82.6 | 69.4 |
| gas (26) | 181.5 | 159.9 | 133.1 | **110.5** [78–131] | 190.4 | 95.7 | 95.8 | 87.7 |

| meter | G14 pass: warm 2016 / TOWT / change point | NMBE median: cold / warm 2016 | 90 % coverage: cold / warm 2016 / 8 wk / 2 wk |
|---|---|---|---|
| electricity | 70 % / 70 % / 71 % | +24.3 / −0.5 | 0.88 / 0.92 / 0.90 / 0.89 |
| chilled water | 20 % / 22 % / 24 % | +46.9 / −1.0 | 0.90 / 0.92 / 0.90 / 0.91 |
| steam | 10 % / 25 % / 23 % | +59.2 / +8.8 | 0.84 / 0.88 / 0.85 / 0.87 |
| hot water | 1 % / 1 % / 0 % | +57.5 / −1.1 | 0.81 / 0.92 / 0.90 / 0.98 |
| gas | 0 % / 0 % / 0 % | +14.9 / −0.9 | 0.74 / 0.73 / 0.72 / 0.95 |

Daily totals (warm 2016 vs change point daily): electricity 14.1 vs 17.2, chilled water 32.4 vs 32.4, steam 36.0 vs 30.6. Monthly: electricity 9.5 vs 11.0, chilled water 22.3 vs 21.6, steam 29.2 vs 24.2.

**Reading it.** With a year of history LoadCast ties the classical models on electricity (slightly lower median CV, same G14 pass rate) and chilled water, and loses on steam, hot water and gas. With no history it is no better than a median-intensity guess and it underpredicts campus thermal meters by about half (positive NMBE = real higher than forecast). Its error bars are its most useful output: 88-92 % coverage of the 90 % band on electricity, chilled water and (warm) hot water.

## Limits

1. Cold forecasts lose to a median guess on electricity and chilled water and underpredict thermal meters ~2×: calibrating on 52 buildings closed only part of the simulator's district-energy gap, and the CV²-weighted loss pulls unknown magnitudes low (R4 tried the fix and failed both tests).
2. Warm forecasts lose to TOWT on steam, hot water and gas (heating meters with seasonal shutdowns and plants serving several buildings).
3. Two weeks of winter history cannot forecast summer chilled water (CV 87 %); its dev widening hit the cap of 30.
4. Not a nowcaster: yesterday's real reading is as good as any 2016-based model for electricity.
5. Dev set is small (only 5 hot water and 2 gas meters); the noise bank holds dev 2017 residuals, so dev-gate numbers are a little optimistic (exam unaffected).
6. Single-zone simulator; no multi-zone reheat; district meters calibrated empirically (a, b), not physically.
7. Trained on BDG2's 19 sites and their 2016-17 weather; other climates and years are extrapolation. Non-residential only.

## Data credits and licences

- Building Data Genome 2 (Miller et al. 2020, *Scientific Data* 7:368), **CC BY-SA 4.0**: dev metadata, meters and residuals for calibration and noise; exam meters; the example files in `examples/bdg2/`. See *Data and licences* below.
- Weather: BDG2 site weather (NOAA ISD), Iowa Environmental Mesonet ASOS gap fill, NASA POWER irradiance (via BuildingData).
- Generator priors: NREL ResStock / ComStock (CC BY 4.0; "Data includes information from the ResStock™ / ComStock™ datasets developed by the National Renewable Energy Laboratory"), EIA CBECS 2018 (public domain).
- Methods: ASHRAE Guideline 14 (CV(RMSE), NMBE); TOWT after Mathieu et al. (2011) / LBNL; change-point models after ASHRAE RP-1050 / PRISM.
- BuildingSim, BuildingData: ML Ratchet Phase 1. Code: same licence as the ML Ratchet repository.

## Data and licences

- **CC BY-SA 4.0 (BDG2-derived), each folder with a `LICENSE-DATA.md`:**
  - `examples/bdg2/`: the three exam meters (2016-2017 hourly kWh), their site weather and `examples.json` (metadata and BuildingData's 2017 baseline scores).
  - `lab/index.html`: the original page embeds 16 exam meters' local 2017 readings, their site weather, metadata, 2016 history encodings and change-point fits.
  - `site/public/instruments/loadcast/` (written by `web/export_instrument.py`): the same 16 meters, weather and encodings, and LoadCast's outputs on them.
- **Logs in `training/`** (`ratchet.json`, `exam_r3.json`, `twin_r3.json`, `widen_r3.json`, `calib_summary.json`) hold summary statistics computed on BDG2 dev and exam meters (medians, skills, coverage, fitted factors), not readings.
- **Weights** (`loadcast_weights.npz`): trained on BuildingSim simulations only; the dev calibration (knobs, a, b), the noise residuals and the widening factors that shaped them come from BDG2 dev buildings. Because real BDG2-derived series went into the training samples, the weights are **CC BY-SA 4.0** (`LICENSE-WEIGHTS.md`; decided 17 Nov 2026, as the v1.0 zip's README already stated).
- **Other sources** (credited in `model.yaml`): NOAA ISD station weather (public domain, inside BDG2), Iowa Environmental Mesonet ASOS (credit), NASA POWER (credit), NREL ResStock / ComStock (CC BY 4.0), EIA CBECS 2018 (public domain).
- **Code** (`loadcast.py`, `pipeline.py`, `demo.py`, `training/`, `web/`): MIT, like the rest of the repository.
