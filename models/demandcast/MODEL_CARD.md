# DemandCast v1.1 model card

Part of ML Ratchet (github.com/dogum/ML-Ratchet): trained on physics, graded on reality. Built on the shared grid
platform (GridSim + GridData, `shared/grid/`), phase 2.

## What it does
Day-ahead forecast of hourly electricity **demand**, **wind** and **solar** output for a US balancing authority (BA),
with error bars. Forecast issued at 10:00 local standard time on day D-1 for the 24 hours of day D (local standard
time). Covers the 8 focus BAs: ERCO, CISO, PJM, MISO, SWPP, ISNE, NYIS, BPAT.

## Inputs
- Weather for D (actual, or day-ahead forecast) and the past week: population-weighted temperature, humidity and
  irradiance; hub-height wind, temperature, pressure and humidity at the wind points; irradiance at the solar points
  (GridData weather points, NASA POWER / ASOS).
- Calendar (DST-aware civil hour, weekday / Saturday / Sunday-holiday, US holidays).
- The grid's static descriptors: building-stock parameters fitted on dev years 2019 and 2021 (heating electrification,
  heat-pump share, conductance, cooling setpoint distribution, rooftop PV), the real EIA-860 wind and solar fleet (MW,
  locations, dates, hub heights, tracking) and a dev-calibrated wind/solar scale.
- The last 168 h of real EIA-930 demand / wind / solar before issue (compared against the physics).

## Outputs
24 hourly means and sigmas per output. Demand: `MW = prior_MW * exp(mu)`, sigma on the log scale. Wind/solar:
capacity factor of the fleet online at issue. Sigmas include one widen-only factor per output fitted on dev
(demand 1.14, wind 1.00, solar 1.37; v1.0 1.34).

**v1.1 (15 Nov 2026), solar only.** EIA-930's solar timestamps move between years, so v1.1 reads the reporting offset of
the solar history at issue. The 168 h reported-minus-physics error is fitted by least squares on the physics diurnal
profile p and its slope p' (a lag of r hours gives e ~ (kappa - 1) p - kappa r p'); r_hat = -beta / (1 + alpha), minus
the grid's dev-year value (`solar_offset_ref` in `demandcast_meta.json`: CISO 0.46 h, the others 0.04-0.19 h), rounded
to -1, 0 or +1 h (|r_hat| <= 0.5 -> 0). The last 24 h of reported solar are put back on the physics clock, the same r8
network runs, and its forecast is moved by r onto the reported clock. No learned parameters; only inputs available at
issue. `predict_cf(..., align=False)` gives v1.0.

## Physics
- **Demand: building-stock thermal model** (`training/buildings.py`). Heating need = UA x E[max(T_set - T_eff, 0)] over
  a Normal spread of thermostat settings (closed form), cooling likewise plus latent load (ventilation air dried from
  q_out to q_in, 0.75 K per g/kg = h_fg/c_p x 0.3 ventilation share). T_eff = sol-air temperature through a fast + slow RC
  envelope lag + internal gains + lagged window gains. Night setback / set-up from occupancy. Electricity: resistance heat
  (COP 1), heat pumps (COP 2.6 + 0.08 T, 1-4.5), non-electric heat (3 % fans), AC (COP 4.2 - 0.07 (T - 25), 2-5.5).
  Base load = residential / commercial / industrial occupancy schedules x a water-heating seasonal term; lighting =
  occupancy x darkness from irradiance; minus rooftop PV (pvlib, fixed 20 deg).
- **Wind**: every EIA-860 farm, on/off by its dates; hub-height speed from the point's own 10 m / 50 m shear; moist-air
  density; IEC density-equivalent speed; GridSim's pitch-regulated farm power curve from specific power (tabulated);
  wake 10 %, availability 97 %, electrical 2 %; cold shutdown < -30 C.
- **Solar**: pvlib (Erbs, Perez, single-axis tracking with backtracking or fixed 25 deg, SAPM cell temperature, PVWatts,
  clipping) per weather point x the online MW of each tracking type.

Checks (`training/checks/physics.json`): RC lag vs an ODE solver 3e-7 K; stock-averaged hinge vs 1e6-draw Monte Carlo
0.0035 K; degree-day limit exact (1,080,000 vs 1,080,000 MWh); specific humidity vs PsychroLib (ASHRAE) 0.22 % (T >= 0 C);
COP at rating points 3.26 (8.3 C), 1.94 (-8.3 C), AC 3.5 (35 C); wind table vs GridSim farm curve mean 0.0005 CF.
Wind/solar physics checked in GridSim phase 1 vs ERCOT (hourly r 0.90 / 0.97-0.98).

## Simulator and training data
182,500 simulated region-days (6 banks). Each simulated region-year takes one focus BA's real weather from a
**training-input year** (2018, 2020, 2022), perturbed (temperature offset N(0, 2 K) and anomaly scale 0.85-1.2,
humidity x0.85-1.15, cloudiness x0.85-1.1, wind x0.9-1.1); random true building parameters (80 % around one BA's dev
fit, 20 % broad uniform); descriptor parameters = true ones with drift (what a fit 2-4 years earlier would give).
**Real noise**: demand gets real relative residuals of building-model fits to the training-input years (whole days,
local-hour aligned). Wind truth uses true hub wind = reanalysis x exp(point bias + correlated AR(1) errors); solar a
cloud error; random losses, specific-power and hub shifts, reporting scale, curtailment; fleet = the BA's real fleet at
a 2018-2022 date with random per-group scaling. Half of the samples get synthetic day-ahead forecast errors.
No real labelled example is ever a training sample.

## Network
Three MLPs (two hidden layers of 512, SiLU), inputs standardised: demand 329 inputs / 456k parameters, wind 164 /
372k, solar 156 / 368k. Gaussian negative log-likelihood, AdamW, one-cycle LR, 40 epochs. Numpy release equals torch to
3e-7 (`training/checks/numpy_vs_torch.json`); the original page's JS port (`lab/index.html`) equals Python to 1e-4
(`training/checks/js_vs_python.json`).

## Ratchet (`training/ratchet.json`)
Objective: frozen simulated validation set (val_v3, 4,000 region-days), mean over outputs of skill = 1 - error / physics
error. Keep a rung only if objective > incumbent + 0.01 and dev gate (30 seeded real days per BA, 2019/2021, cross-year
descriptors) mean relative error <= +2 %. Re-scored on each new frozen set when the simulator changed.

| Rung | Change | Objective (val_v3) | Dev demand / wind / solar % | Decision |
|---|---|---|---|---|
| r1_base | 3 MLPs 256-256, sim v1 | 0.333 | 2.06 / 21.4 / 15.8 | kept (first) |
| r2_simv2 | sim v2: drift/noise/error widths calibrated to dev | 0.338 | 1.98 / 21.5 / 21.1 | rejected (solar broke on real data) |
| r3_robust | + clamp negative night output, input-scale floor | 0.339 | 1.98 / 21.3 / 15.3 | rejected (+0.006 < 0.01) |
| r4_more | v2 + v1 banks, 384-384 | 0.348 | 1.98 / 21.4 / 15.4 | **kept** |
| r5_simv3 | sim v3 (twin test: noise missing in ~60 % of hours) | 0.348 | 1.99 / 21.8 / 15.0 | rejected (no gain) |
| r6_all | v3x2 + v2 + v1, 384-384 | 0.356 | 1.95 / 21.5 / 15.5 | rejected (+0.008) |
| r7_hist | full 168 h history inputs | 0.355 | 2.27 / 22.4 / 16.2 | rejected (dev +8 %) |
| r8_wide | all 6 banks, 512-512 | 0.362 | 2.00 / 21.4 / 15.6 | **kept = v1.0** |

Dev calibration rungs outside the network: building model fitted per BA on 2019/2021; wind/solar scale per BA; CISO and
PJM EIA-930 solar shifted by 1 h (dev correlation peak); sim v2 widths matched to dev; sim v3 noise fix from the twin test.

**v1.1 ladder (15 Nov 2026), frozen set val_v4** (simulator v4: v3 plus a solar reporting offset of -1 / 0 / +1 h with
probability 0.15 / 0.70 / 0.15 per region-year; 4,000 region-days, seed 404). Only r8 could be re-scored; earlier
checkpoints are not in the repository.

| Rung | Change | Objective (val_v4) | Dev demand / wind / solar % | Decision |
|---|---|---|---|---|
| r8_wide | v1.0, re-scored | 0.333 (solar skill 0.386) | 2.00 / 21.4 / 15.60 | incumbent |
| r9a | + reporting offset read at issue, counted from the physics clock | 0.375 (0.513) | 2.00 / 21.4 / 16.11 | replaced by r9b (CISO dev days flipped) |
| r9b | + offset counted from each grid's dev-year reference | 0.375 (0.513) | 2.00 / 21.4 / 15.68 | **kept = v1.1** (+0.042; dev +0.2 %) |

On val_v4 the offset is read right on 3,886 of 4,000 region-days; solar error on regions reported 1 h early / on time /
1 h late: v1.0 36.2 / 14.5 / 31.8 %, v1.1 18.4 / 14.7 / 15.8 %. Dev days with the reported solar moved by -1 / 0 / +1 h
(30 seeded days per grid): v1.0 22.8 / 15.6 / 37.7 %, v1.1 15.6 / 15.7 / 16.4 %, persistence 16.6 %.

## Exam (2023-2025, never used for choosing)
Demand MAPE (%), actual weather (an upper bound: grids' own forecasts use forecast weather), all hours where every
method has a value:

| Grid | DemandCast | Grid DA | Persistence 24 h | Same hour last week | Physics alone | In 68 / 95 % band |
|---|---|---|---|---|---|---|
| ERCO | 2.25 | 2.40 | 4.56 | 8.36 | 6.04 | 68 / 92 |
| CISO | 3.37 | 6.86 | 4.90 | 6.46 | 6.00 | 46 / 74 |
| PJM | 2.10 | 3.49 | 4.95 | 8.32 | 3.24 | 69 / 92 |
| MISO | 1.88 | 2.71 | 4.17 | 6.98 | 2.98 | 67 / 92 |
| SWPP | 2.41 | 4.31 | 4.32 | 8.21 | 5.64 | 66 / 89 |
| ISNE | 3.00 | 2.56 | 7.27 | 10.64 | 5.51 | 61 / 88 |
| NYIS | 1.97 | 2.97 | 5.34 | 8.22 | 5.16 | 68 / 93 |
| BPAT | 2.10 | 2.00 | 3.69 | 6.42 | 5.58 | 68 / 92 |
| all | 2.38 | 3.41 | 4.90 | 7.95 | 5.02 | 64 / 89 |

By year and with real **day-ahead forecast weather** (Open-Meteo previous runs, day-1 lead; 2024 only, top population
points; the archive's rate limit stopped further downloads). DemandCast / grid DA:

| Grid | 2023 | 2024 | 2025 | 2024, forecast weather |
|---|---|---|---|---|
| ERCO | 2.14 / 2.60 | 2.14 / 2.16 | 2.47 / 2.46 | 2.98 / 2.05 |
| CISO | 2.90 / 5.50 | 3.42 / 6.97 | 3.78 / 8.13 | 3.73 / 7.00 |
| PJM | 2.02 / 3.46 | 2.31 / 3.56 | 1.97 / 3.44 | 2.84 / 3.57 |
| MISO | 1.87 / 2.98 | 1.91 / 2.78 | 1.85 / 2.37 | 2.15 / 2.80 |
| SWPP | 1.96 / 2.97 | 2.54 / 2.32 | 2.72 / 7.68 | 2.74 / 2.22 |
| ISNE | 2.77 / 2.40 | 3.13 / 2.77 | 3.10 / 2.52 | 3.93 / 2.79 |
| NYIS | 1.74 / 2.41 | 1.93 / 3.29 | 2.23 / 3.21 | 2.57 / 3.28 |
| BPAT | 2.08 / 2.04 | 2.23 / 1.97 | 1.99 / 1.99 | 2.44 / 1.92 |

Pooled forecast-weather days: DemandCast 2.92 % vs grid DA 3.20 % (same days with actual weather: 2.44 %); bands 72 / 94 %.
Note CISO's published DA forecast is misaligned by 1 h and SWPP's runs 6.5 % high in 2025 (GridData DATA.md), which
flatters DemandCast there.

**Second look: forecast weather 2024-2025** (run once on 14 Nov 2026 from the shipped weights, after the archive was
fetched in full; reported next to the released 2024 row, which stays; nothing was chosen from it). Open-Meteo
previous-runs day-1 forecasts at the shipped fetch list's points (top 4 population, 3 wind, 3 solar per grid; the
released row's cache held up to 8 population points in 2024). 2023 cannot be graded this way: the archive holds no
humidity or radiation forecasts before 2024. Demand MAPE, DemandCast / grid DA:

| Grid | 2024 | 2025 | 2024-2025 |
|---|---|---|---|
| ERCO | 3.05 / 2.05 | 3.60 / 2.46 | 3.33 / 2.26 |
| CISO | 3.75 / 7.01 | 4.09 / 8.13 | 3.92 / 7.58 |
| PJM | 2.91 / 3.57 | 2.52 / 3.44 | 2.71 / 3.50 |
| MISO | 2.21 / 2.79 | 2.09 / 2.37 | 2.15 / 2.58 |
| SWPP | 2.80 / 2.22 | 3.10 / 7.68 | 2.96 / 5.01 |
| ISNE | 3.97 / 2.78 | 4.38 / 2.52 | 4.18 / 2.65 |
| NYIS | 2.62 / 3.28 | 3.05 / 3.21 | 2.84 / 3.24 |
| BPAT | 2.42 / 1.92 | 2.24 / 1.99 | 2.33 / 1.96 |
| all | 2.97 / 3.20 | 3.13 / 3.97 | 3.05 / 3.59 |

Pooled 3.05 % (135,700 h) vs grid DA 3.59 %, persistence 4.91 %, physics alone 5.88 %; same days with actual weather
2.48 %; bands 71 / 93 %. Wins in 5 of 8 grids, but without CISO and SWPP (flawed published forecasts) it is 2.92 % vs
the grids' 2.70 %, a loss. Wind 26.7 % vs physics 39.0 %, persistence 42.9 %, climatology 40.6 % (same days with actual
weather 19.2 %); solar 34.7 % vs physics 39.2 %, persistence 18.9 % (a loss), climatology 41.1 %. v1.1 (run once, 15 Nov
2026): solar 18.2 % vs persistence 18.9 %, a narrow win (CISO 28.3 %, persistence 10.5 %; PJM 15.1 %, 27.4 %); bands
72 / 89 %. Demand and wind are unchanged.

Wind and solar, error = sum |error| / sum output (%), actual weather; DemandCast / physics / persistence 24 h / dev-year
CF climatology (68 / 95 % band coverage). Solar: v1.0 as released, then v1.1 (run once, 15 Nov 2026):

| Grid | Wind | Solar v1.0 | Solar v1.1 |
|---|---|---|---|
| ERCO | 17.8 / 21.6 / 42.9 / 42.2 (78/97) | 10.6 / 20.3 / 20.9 / 27.1 (77/92) | 10.6 / 20.3 / 20.9 / 27.1 (77/93) |
| CISO | 32.7 / 40.6 / 42.2 / 43.5 (66/89) | 59.4 / 47.9 / 11.4 / 40.7 (50/72) | 24.9 / 47.9 / 11.4 / 40.7 (64/79) |
| PJM | 24.0 / 24.4 / 54.8 / 48.2 (72/95) | 28.1 / 30.8 / 28.0 / 43.3 (58/78) | 11.9 / 30.8 / 28.0 / 43.3 (69/88) |
| MISO | 16.0 / 17.1 / 43.4 / 39.0 (74/97) | 10.4 / 57.9 / 22.5 / 64.7 (74/90) | 10.4 / 57.9 / 22.5 / 64.7 (75/91) |
| SWPP | 17.2 / 22.6 / 36.5 / 36.0 (77/95) | 41.5 / 41.1 / 29.1 / 47.4 (68/86) | 41.5 / 41.1 / 29.1 / 47.4 (68/86) |
| ISNE | 28.3 / 35.0 / 62.6 / 55.2 (72/92) | 16.4 / 39.7 / 39.0 / 52.4 (71/92) | 16.5 / 39.7 / 39.0 / 52.4 (72/92) |
| NYIS | 26.4 / 29.1 / 68.9 / 60.7 (72/95) | not reported in EIA-930 | not reported |
| BPAT | 50.1 / 55.4 / 88.4 / 86.5 (57/92) | 51.8 / 21.0 / 30.5 / 34.9 (59/74) | 52.0 / 21.0 / 30.5 / 34.9 (59/74) |
| all | 19.2 / 22.7 / 43.5 / 41.4 (71/94) | 29.9 / 36.3 / 19.1 / 39.2 (65/84) | **16.1** / 36.3 / 19.1 / 39.2 (69/86) |

Extremes (demand MAPE, DemandCast / grid DA): Winter Storm Uri (dev) ERCO 14.5 / 18.6 (DemandCast forecast ~85 GW demand; ERCOT
served 69 GW under load shed); Winter Storm Elliott (training-input year) PJM 3.05 / 3.78, ERCO 4.18 / 5.54; Texas heat
Aug 2023 ERCO 1.96 / 1.59 (peak 88.5 predicted vs 85.4 GW); Eastern heat dome Jun 2024 PJM 1.98 / 4.50, MISO 1.56 / 4.01;
Arctic outbreak Jan 2025 PJM 1.83 / 4.17, ISNE 4.78 / 1.27. Error by temperature band is 2.2-2.7 % from
-10 C to 35 C (grid DA 2.9-3.9 %); below -10 C (913 h) 3.25 % (grid DA 3.33 %); above 35 C (1,211 h) 2.73 % against
the grids' 2.55 %, a loss. Hottest/coldest 2 % of days per grid and all named periods in `lab/index.html`
and `training/exam_r8_wide.json` / `training/exam_r8_wide_events.json`.

## Limits
1. Actual weather is an upper bound; with forecast weather (2024) DemandCast loses to ERCOT, SPP, ISO-NE and BPA's own
   forecasts; over 2024-2025 (second look) to ERCOT, ISO-NE and BPA, and to the 6 grids without CISO and SWPP pooled.
   No forecast-weather grade for 2023 (the archive has no humidity or radiation forecasts before 2024).
2. Solar timestamps: CISO solar led the physics by 1 h in 2019/2021, matched in 2023, lagged by 1 h in 2024-25 (PJM
   changed in 2025). v1.0's fixed dev shift was wrong in the exam (29.9 % vs persistence 19.1 %). v1.1 reads whole-hour
   offsets from the week before issue (16.1 %), but picks a change up only once that week shows it, and CISO's
   dev-year profile sits about half an hour off, so CISO days near the boundary flip; CISO solar still loses to
   persistence (24.9 % vs 11.4 %). BPAT's tiny solar fleet is badly served (52.0 % vs 30.5 %), and so is SWPP (41.5 %
   vs 29.1 %).
3. Bands are too narrow on real data (demand 64 / 89 % for 68 / 95 %; CISO 46 / 74).
4. Wind limited by 50 km reanalysis (BPAT 50 %, CISO 33 %); the network trims physics error by ~15 %.
5. Needs the last week of metered output; physics alone is ~5 % off (load growth since 2019-21).
6. Building parameters are effective, several fits sit on bounds; the heating/cooling split is indicative.
7. Over-predicts heat-wave peaks by ~3 %.

## Data credits and licences
EIA-930 Hourly Electric Grid Monitor and EIA-860/923 (US public domain; via Catalyst Cooperative PUDL, CC-BY-4.0);
NASA POWER hourly (MERRA-2, CERES; NASA LaRC POWER Project); ASOS via Iowa Environmental Mesonet; Open-Meteo Previous
Runs API (CC-BY-4.0, free non-commercial use); EIA-861 and Census for weather weights; GridData / GridSim (ML Ratchet
grid phase 1); pvlib (BSD-3); PsychroLib (MIT) for checks.

## In this repository (14 Nov 2026)

- Layout per the model contract: `demandcast.py` (numpy only), `demandcast_weights.npz` + `demandcast_meta.json`,
  `calib/`, `pipeline.py`, `demo.py`, `examples/exam_days.npz` (16 real exam days, a fixed random draw: two per grid,
  seed 16), `training/` (every script as run; `training/_paths.py` sends outputs to `work/`), `lab/index.html` (the
  original page), `web/export_instrument.py` (the live instrument on the website).
- `demo.py` prints the same output from the repository as from the release zip (identical text).
- The released exam numbers above match the raw exam output, kept as `training/exam_r8_wide.json` (as written by
  `training/exam.py`, recovered from the page's embedded copy), to the printed precision. Two corrections against that
  file: specific humidity matches PsychroLib to 0.22 % (`training/checks/physics.json`; "0.3 %" above), and below
  -10 C (913 h) the demand error is 3.25 % (grid DA 3.33 %), so "flat at 2.2-2.7 %" holds from -10 C up.
- The 2023-2025 actual-weather exam was re-scored from the shipped weights (`training/rescore_release.py`: the whole
  pipeline rebuilt in `work/`, then `exam.py`'s own metrics with a numpy stand-in for the torch loader). All 125
  compared numbers (pooled and per grid: demand, wind, solar, every baseline, coverage, hours, the dev-fitted sigma
  scales) reproduce to the printed precision, e.g. pooled demand 2.38 % vs grid DA 3.41 %, wind 19.2 %, solar 29.9 %,
  bands 64 / 89 %. The NASA POWER per-point series it needs are not in the GridData v1.0 zip; they come from the data
  rebuild's `power` stage. Physics checks 1-3, the COP points and the wind-table check reproduce
  `training/checks/physics.json` exactly.
- The forecast-weather numbers were re-scored once the Open-Meteo archive was fetched (`mlratchet data grid --only
  forecastwx`, 14 Nov 2026, no refusals): with the release's own cache reconstructed (`training/rescore_forecast.py
  release2024`) all 338 forecast-mode numbers of the 2024 row reproduce to 1e-4 (pooled 2.92 % vs 3.20 %, every grid,
  wind 31.2 %, solar 35.1 %, coverage, hours). That cache held more points than the shipped fetch list: all 8
  population points in 6 grids, 5 in NYIS, 4 in BPAT, and one ERCOT wind point; with the shipped list's top 4 the 2024
  row is 2.99 % vs 3.20 %. The 2024-2025 second look above is `rescore_forecast.py full`.
- Live instrument: the three networks (float16) on the 112 exam days of the original page (14 per grid), re-running the
  building physics for the 38 h after issue and the tabulated wind/solar physics when temperature, humidity, sunshine
  or wind change. Browser = Python to 2.3e-7 on 80 cases (`site/tests/parity.test.ts`). On those 112 days with the
  controls neutral, demand MAPE is 2.49 % against 3.51 % for the grids' own forecasts.
