# LoadCast v1.0.1 — model card

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

Frozen set: 300 simulated buildings (v1 physics, seed 99, real noise seed 12345), 650 building-meters. Objective = mean over meter types of (cold skill + warm skill)/2, where cold skill = 1 − median CV / median CV of the set's own median-intensity × median-shape guess, and warm skill = 1 − median CV / median CV of TOWT on the same history. Keep a rung only if the objective rises by > 0.01 and the dev score (mean of cold and warm median CV over 84 dev building-meters, 2016 → 2017) is no more than 2 % worse. The objective definition changed twice before R3 (floored pooled skills → unfloored → per meter type); every rung was re-scored each time (`ratchet.json`). Checked on 17 Nov 2026 (`training/optimism.py`, `optimism_r3.json`): two retrains of the R3 recipe on a rebuilt bank differ by 0.020 in the frozen objective (more than the 0.01 margin) and by 1.8 points in the dev score (the 2 % gate is about 1 point), so single-seed decisions on this ladder are within training noise.

| rung | change | frozen objective | dev score (CV %) | decision |
|---|---|---|---|---|
| R0 | generic Phase 1 physics, 800 buildings, real dev noise | −0.174 | 59.9 | kept (first) |
| R1 | dev-calibrated physics (knobs + meter a, b from the 52 dev fits), 800 buildings | −0.118 | 54.9 | kept |
| R2 | + anti-memorisation: metadata jitter, dropout 0.1, weight decay 1e-3 | −0.018 | 51.9 | kept |
| R3 | + full bank (1,600 buildings) | **+0.014** | **49.2** | **kept (released)** |
| R4 | cold loss weight 1/ybar^0.5 (aim magnitude at the median) | −0.026 | 52.2 | rejected (both tests) |
| R5 | real noise matched by use type | +0.006 | 51.7 | rejected (both tests) |

**Round 2 (17 Nov 2026, `ratchet.yaml` second ladder, `training/r2.py`, `r2_ladder.json`).** Rebuilt, reproducible
frozen set `v2_val` (deterministic spec per building); every rung judged on the mean of 3 training seeds against the
released R3 re-scored there (−0.0081; dev 49.2). r6 (BuildingSim physics 1.1 district plants, calibration re-run):
+0.011 on its own frozen set but dev 52.8, rejected. r7 (median-aimed magnitude head): −0.028, dev 50.4 (gate 50.2),
rejected. r8 (no-history mode anchored on the dev-calibrated twin): −0.005, dev 53.9, rejected; the twin alone scores
dev no-history CV 65.4 % against 78.2 % for r8 and 70.6 % for R3. The R3 recipe retrained (3 seeds) scores −0.0076 and
dev 53.3, so the gate set by R3's selected seed is about 3 points below its own recipe. v1.0 stays.

**Twin test (dev, R3):** each dev building's calibrated twin, given another dev building's real residuals, read by the networks like the real meter. Median CV real vs twin: cold electricity 62.5 vs 51.8 (per-building correlation of log CV r = 0.86), chilled water 101 vs 99 (r = 0.96), steam 104 vs 108 (r = 0.95); warm electricity 22.3 vs 17.8, chilled water 43.2 vs 41.7, steam 61.7 vs 59.9 (r ≈ 0). Cold errors are reproduced building by building (they are magnitude errors); warm errors match in size but not in which building is hard, since the noise is borrowed.

**What the network adds (dev twin test, 17 Nov 2026, `training/twin_anchor.py`, local 2017 on the dev meters).**
Cold: anchor = the dev-calibrated twin from metadata only (the exam's "calibrated twin"). Median CV anchor → network:
electricity 53.4 → 62.5 % (47 meters), chilled water 71.1 → 101.0 % (20), heating meters 96.0 → 103.2 % (17); the
network beats the twin on 43 %, 55 % and 47 % of meters. Its corrections correlate with what reality needed (truth
minus twin) at r = 0.73, 0.70 and 0.18 hourly, mostly through the level across buildings, but on the median it moves
the level the wrong way: reality is 13 % above the twin on electricity and 2 % on chilled water; the network goes
11 % and 55 % below it. What it misses: real campus electricity has a stronger weekday-day / weekend swing than the
24-7-calibrated twin (+11 % / −8 % of the mean; the network adds about 0), and heating meters peak harder in winter
(+56 % in February; the network adds +7 %). It does reproduce the chilled-water season the twin gets wrong (July
needs −68 %, the network gives −52 %). Warm (a year; anchor = its own ridge TOWT): median CV 22.3 → 22.3 % on
electricity (r = 0.15 hourly), 48.1 → 43.2 % on chilled water (r = 0.50), 65.9 → 64.5 % on heating meters
(r = 0.32). The warm correction gets holidays right on electricity (−14.6 % needed, −13.7 % given) and little else.

## Exam (BDG2 exam buildings, forecast local 2017 with actual weather, history = 2016)

1,502 building-meters with q_ok in both years. Taken once in the build session, after R3 was chosen. **Re-scored on
17 Nov 2026 (seen; nothing chosen from it)** with the released weights through `training/rescore_seen.py`: every one
of the 635 statistics in `exam_r3.json` reproduces to 5e-6 (float32 numpy vs torch), the dev gate to 1e-5 and the
widening factors exactly.

Median CV(RMSE) %. All scores use p = 0: BuildingData's fitted baselines (n − p in v1.0's table) are converted with
CV × √((n − p)/n), which lowers its monthly change-point CVs by up to a fifth (electricity 11.0 → 9.3 %). "Best" is
the baseline with the lowest median on the exam (picking it there can only favour the baselines). Interval: 95 %
paired bootstrap of median(LoadCast) − median(best) over building-meters, 2,000 resamples, seed 20261117. Win if the
interval is below zero, loss if above, tie if it spans zero. Four exam buildings at the Bobcat site have no twin
(BuildingSim stops with `KeyError('kW')` on their specs); twin rows are paired over the rest. NMBE > 0 means the
forecast is too low.

Baselines. **Calibrated twin:** each exam building's BuildingSim twin from its BDG2 metadata only, with the knobs
chosen most often on the 52 dev buildings (24-7 operation, outdoor air ×2.5, VAV minimum 0.3, district HVAC for
electricity-only buildings) and the dev median meter fit `a, b` (`training/twin_knobs.json`), run through 2016-2017
with the real weather; **by use:** knobs and `a, b` per use type where dev has ≥ 3 buildings. **Median guess:** dev
median intensity × median time-of-week shape per use and meter. **Dev intensity × TOWT shape:** the same intensity
times a TOWT shape (median over dev meters) driven by the exam building's weather. **TOWT, change point (BD):**
BuildingData, fitted on 2016. **Ridge TOWT:** LoadCast's own anchor fitted on the same history (a year, 8 or 2 weeks).

| meter (n) | history | LoadCast | best baseline | LoadCast − best [95 % CI] | verdict (v1.0) | NMBE | G14 pass LoadCast / best | 90 % cover |
|---|---|---|---|---|---|---|---|---|
| electricity (856) | a year | 18.8 | change point (BD) 19.1 | -0.3 [-0.8, +0.1] | **tie** (win) | -0.5 | 70 / 71 % | 0.92 |
| electricity (856) | 8 weeks | 23.8 | ridge TOWT, same 8 wk 26.2 | -2.4 [-3.3, -1.6] | **win** | +1.5 | 53 / 48 % | 0.90 |
| electricity (856) | 2 weeks | 30.1 | ridge TOWT, same 2 wk 36.5 | -6.4 [-7.7, -5.2] | **win** | +8.2 | 29 / 22 % | 0.89 |
| electricity (856) | none | 59.3 | calibrated twin by use 52.4 | +6.8 [+4.7, +9.5] | **loss** (loss) | +24.3 | 5 / 6 % | 0.88 |
| electricity (856) | a year, daily | 14.1 | ridge TOWT (anchor) 14.5 | -0.4 [-0.9, -0.1] | **win** | -0.5 | – | – |
| electricity (856) | a year, monthly | 9.5 | change point (BD) 9.3 | +0.2 [-0.1, +0.5] | **tie** | -0.4 | – | – |
| electricity (856) | none, daily | 54.7 | calibrated twin by use 46.4 | +8.1 [+5.0, +10.7] | **loss** | +24.3 | – | – |
| electricity (856) | none, monthly | 51.8 | calibrated twin by use 43.5 | +8.2 [+4.9, +11.2] | **loss** | +24.5 | – | – |
| chilled water (323) | a year | 42.9 | change point (BD) 42.4 | +0.5 [-1.5, +2.9] | **tie** (loss) | -1.0 | 20 / 24 % | 0.92 |
| chilled water (323) | 8 weeks | 62.6 | ridge TOWT, same 8 wk 68.1 | -5.5 [-9.1, -0.1] | **win** | +19.0 | 8 / 7 % | 0.90 |
| chilled water (323) | 2 weeks | 87.3 | calibrated twin 85.8 | +1.4 [-8.0, +7.9] | **tie** | +42.1 | 3 / 2 % | 0.91 |
| chilled water (323) | none | 104.1 | calibrated twin 85.8 | +18.2 [+9.5, +25.1] | **loss** (loss) | +46.9 | 0 / 2 % | 0.90 |
| chilled water (323) | a year, daily | 32.4 | change point (BD) 32.2 | +0.1 [-2.2, +2.8] | **tie** | -0.7 | – | – |
| chilled water (323) | a year, monthly | 22.3 | change point (BD) 18.1 | +4.2 [+1.9, +6.1] | **loss** | -1.0 | – | – |
| chilled water (323) | none, daily | 95.5 | calibrated twin by use 73.9 | +21.5 [+15.4, +27.3] | **loss** | +46.9 | – | – |
| chilled water (323) | none, monthly | 90.8 | calibrated twin by use 66.8 | +23.9 [+17.5, +29.9] | **loss** | +46.3 | – | – |
| steam (225) | a year | 45.5 | TOWT (BD) 39.4 | +6.0 [+2.9, +8.2] | **loss** (loss) | +8.8 | 10 / 26 % | 0.88 |
| steam (225) | 8 weeks | 49.1 | ridge TOWT, same 8 wk 45.6 | +3.6 [-1.3, +6.4] | **tie** | +11.4 | 4 / 22 % | 0.85 |
| steam (225) | 2 weeks | 57.1 | ridge TOWT, same 2 wk 60.5 | -3.3 [-8.7, +2.9] | **tie** | +20.1 | 1 / 12 % | 0.87 |
| steam (225) | none | 102.8 | calibrated twin by use 99.7 | +3.1 [-1.1, +6.8] | **tie** (win) | +59.2 | 0 / 0 % | 0.84 |
| steam (225) | a year, daily | 36.0 | change point (BD) 30.5 | +5.5 [+2.9, +7.5] | **loss** | +10.0 | – | – |
| steam (225) | a year, monthly | 29.2 | change point (BD) 20.0 | +9.2 [+6.7, +13.2] | **loss** | +8.7 | – | – |
| steam (225) | none, daily | 96.6 | calibrated twin 92.7 | +3.9 [-1.0, +10.3] | **tie** | +59.2 | – | – |
| steam (225) | none, monthly | 93.2 | calibrated twin 87.1 | +6.1 [+1.8, +12.2] | **loss** | +58.9 | – | – |
| hot water (72) | a year | 94.5 | ridge TOWT (anchor) 80.5 | +14.0 [+0.3, +23.9] | **loss** (loss) | -1.1 | 1 / 1 % | 0.92 |
| hot water (72) | 8 weeks | 86.5 | ridge TOWT, same 8 wk 106.8 | -20.3 [-27.3, -2.1] | **win** | +8.8 | 0 / 0 % | 0.90 |
| hot water (72) | 2 weeks | 93.6 | calibrated twin 124.4 | -30.8 [-50.1, -10.3] | **win** | +8.4 | 0 / 0 % | 0.98 |
| hot water (72) | none | 134.0 | calibrated twin 124.4 | +9.6 [-10.6, +28.2] | **tie** (win) | +57.5 | 0 / 0 % | 0.81 |
| hot water (72) | a year, daily | 69.8 | change point (BD) 59.2 | +10.5 [+0.4, +22.8] | **loss** | +1.6 | – | – |
| hot water (72) | a year, monthly | 57.3 | ridge TOWT (anchor) 40.8 | +16.5 [+7.3, +27.1] | **loss** | -1.0 | – | – |
| hot water (72) | none, daily | 115.5 | calibrated twin by use 87.4 | +28.1 [-4.4, +40.5] | **tie** | +60.1 | – | – |
| hot water (72) | none, monthly | 108.9 | calibrated twin 85.1 | +23.8 [+2.4, +34.0] | **loss** | +57.9 | – | – |
| gas (26) | a year | 110.5 | TOWT (BD) 94.6 | +15.9 [-4.9, +27.2] | **tie** (loss) | -0.9 | 0 / 0 % | 0.73 |
| gas (26) | 8 weeks | 133.1 | ridge TOWT, same 8 wk 110.4 | +22.7 [+0.8, +43.8] | **loss** | -30.5 | 0 / 0 % | 0.72 |
| gas (26) | 2 weeks | 159.9 | ridge TOWT, same 2 wk 131.3 | +28.5 [+4.6, +62.3] | **loss** | +1.3 | 0 / 0 % | 0.95 |
| gas (26) | none | 181.5 | calibrated twin 179.8 | -10.3 [-65.2, +94.7] | **tie** (win) | +14.9 | 0 / 0 % | 0.74 |
| gas (26) | a year, daily | 75.5 | change point (BD) 61.7 | +13.8 [-1.0, +28.5] | **tie** | +2.3 | – | – |
| gas (26) | a year, monthly | 44.7 | change point (BD) 36.2 | +8.6 [+0.8, +21.6] | **loss** | +0.3 | – | – |
| gas (26) | none, daily | 131.8 | calibrated twin 127.8 | +1.3 [-77.5, +98.9] | **tie** | +28.6 | – | – |
| gas (26) | none, monthly | 96.5 | calibrated twin 92.5 | +1.3 [-54.9, +67.0] | **tie** | +10.7 | – | – |

All baselines, hourly (median CV %; persistence rows read the real 2017 meter, so they are nowcasts):

| meter | median guess | dev intensity × TOWT shape | calibrated twin | twin by use | TOWT (BD) | change point (BD) | ridge TOWT (anchor) | persistence 1 h* / 24 h* / last week* |
|---|---|---|---|---|---|---|---|---|
| electricity | 53.8 | 53.8 | 53.5 | 52.4 | 19.5 | 19.1 | 19.5 | 9.6 / 18.6 / 19.5 |
| chilled water | 101.6 | 92.4 | 85.8 | 87.5 | 45.2 | 42.4 | 45.2 | 18.8 / 40.3 / 51.8 |
| steam | 111.6 | 104.9 | 100.2 | 99.7 | 39.4 | 40.5 | 41.4 | 24.9 / 41.9 / 56.6 |
| hot water | 207.6 | 192.9 | 124.4 | 125.2 | 82.2 | 82.6 | 80.5 | 52.8 / 69.4 / 93.8 |
| gas | 190.4 | 190.4 | 179.8 | 179.8 | 94.6 | 95.8 | 94.7 | 64.3 / 87.7 / 94.1 |

**Reading it.** 6 wins, 16 ties, 18 losses. The network earns its keep with short histories: with 2 or 8 weeks it
beats TOWT fitted on the same weeks on electricity, chilled water (8 weeks) and hot water. With a year of history it
ties the classical models on electricity and chilled water and loses on steam and hot water; the electricity row that
v1.0 called a win is a tie (−0.3 points, interval −0.8 to +0.1). With no history the dev-calibrated BuildingSim twin
alone beats LoadCast on electricity and chilled water and ties it elsewhere, so the v1.0 cold "wins" against the
median guess came from the calibrated physics, not the network: the network forecasts thermal meters at about half
the real load (NMBE +47 % chilled water, +59 % steam) where the twin is roughly level. Its error bars stay its most
useful output: 88-92 % coverage of the 90 % band with a year of history on electricity, chilled water, steam and hot
water.

## Fresh exam (17 Nov 2026): I-BLEND, 3 Delhi campus buildings no version had seen, scored once

I-BLEND (IIIT Delhi; figshare 10.6084/m9.figshare.6007637, CC0), weather from NOAA ISD-Lite (Delhi IGI) and NASA
POWER. Three buildings (academic, library, facilities) passed the 85 % good-hours rule; history 2016, forecast 2017,
electricity only. Frozen in commit 3d9f949 before the round-2 decision and scored once after it, for v1.0 (no round-2
release; `training/exam2_iblend_score.py`, `exam2_iblend_results.json`). Intervals: 95 % bootstrap over the 3
buildings (2,000 resamples, seed 20261117) on median(LoadCast) minus median(best baseline).

| Row, median CV(RMSE) | LoadCast v1.0 | Best baseline | Minus best | Verdict |
|---|---|---|---|---|
| No history, hourly | 125.7 % | BDG2 dev median guess 89.8 % | +35.9 [−33.5, +78.8] | tie |
| A year of history, hourly | 41.0 % | change-point (BuildingData) 40.2 % | +0.7 [−2.2, +1.0] | tie |
| 8 weeks, hourly | 41.1 % | ridge TOWT on the same weeks 46.2 % | −5.1 [−5.1, +2.5] | tie |
| 2 weeks, hourly | 43.2 % | ridge TOWT on the same weeks 46.7 % | −3.5 [−4.1, −3.5] | win |
| No history, daily | 120.7 % | dev TOWT-shape guess 82.3 % | +38.4 [−39.4, +78.6] | tie |
| No history, monthly | 119.6 % | dev TOWT-shape guess 80.7 % | +39.0 [−42.0, +79.5] | tie |
| A year, daily | 17.4 % | TOWT (BuildingData) 17.9 % | −0.5 [−2.4, −0.3] | win |
| A year, monthly | 12.6 % | ridge TOWT 13.0 % | −0.5 [−1.9, −0.4] | win |

With no history LoadCast forecasts about 2.2 times the real load (median NMBE −120 %); Delhi's climate and Indian
holidays are outside what it was trained on. With history it is level with the fitted models and a few points better
than its anchor on short histories. Three buildings cannot rank models: the "wins" are 0.5-3.5 points and their
intervals come from resampling three values, so read them as "not worse". The dev-calibrated BuildingSim twin
baseline was not run: the twin builder takes BDG2 buildings only.

## Limits

1. Cold forecasts lose to the dev-calibrated BuildingSim twin on electricity (59.3 vs 52.4 %) and chilled water (104.1 vs 85.8 %) and tie it on steam, hot water and gas; the network sets thermal meters at about half the real load (the CV²-weighted loss pulls unknown magnitudes low; R4 tried the fix and failed both tests). On dev the network makes the twin's cold forecast worse on every meter type.
2. With a year of history it ties TOWT and the change-point model on electricity and chilled water and loses on steam (to TOWT) and hot water (to its own ridge TOWT anchor); gas is a tie with 26 meters.
3. Two weeks of winter history cannot forecast summer chilled water (CV 87 %, no better than the twin with no history); its dev widening hit the cap of 30.
4. Not a nowcaster: yesterday's real reading is as good as any 2016-based model for electricity.
5. Dev set is small (5 hot water and 2 gas meters). The noise bank holds dev 2017 residuals, but the leak is below the noise: with 2016-only residuals the retrained dev score moved −0.1 points (95 % CI −2.7 to +2.7), against 1.8 points between two training seeds. The released dev score (49.2) is optimistic by about 4 points through selection: two retrains of the R3 recipe score 53.8 and 52.0 on dev, while R3's exam score (52.8) sits between theirs (53.8, 51.2).
6. Single-zone simulator; no multi-zone reheat; district meters calibrated empirically (a, b), not physically.
7. Trained on BDG2's 19 sites and their 2016-17 weather; other climates and years are extrapolation. Non-residential only.
8. Every training building had an electricity meter. v1.0 put a meter list without one far outside training (cold forecasts 24-50× too high); v1.0.1 (27 Nov 2026) adds the meter by default (`assume_electricity=True`). Re-scored on the seen exam, the 19 building-meters this touches went from median cold CV 597 % to 114 % and warm (a year) 51 % to 35 %; the exam rows for chilled water and steam moved 0.1-3.3 points down, no verdict changed (training/rescore_noelec_r3.json). The exam tables above are v1.0's.

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
