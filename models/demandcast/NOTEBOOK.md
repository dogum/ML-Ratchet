> **Lab notebook.** DemandCast was built in a cloud sandbox session (13–14 Nov 2026) on the shared GridSim + GridData
> platform, now in `shared/grid/`. The first entry is that build's status document as written, with two edits: the
> page link and sandbox paths are replaced by where those things now live, and parallel runs are called builds. Its paths
> (`src/`, `data/`, `checks/`) refer to the sandbox work folder `grid/demandcast/`; in this repository the code is in
> `training/` and outputs go to `work/`. `MODEL_CARD.md` holds every number.

## 14 Nov 2026 — build notes (demandcast-status.md, as written)

# DemandCast status (GridSim phase 2)

Day-ahead hourly electricity demand, wind and solar for the 8 focus balancing authorities (ERCO, CISO, PJM, MISO, SWPP,
ISNE, NYIS, BPAT), with error bars. Trained only on simulated regions; graded on 2023-2025.

- Page: the original page, now `lab/index.html` (lab: pick a grid and exam day, drag temperature, humidity,
  sunshine, wind; physics + network run in the browser)
- Release: `demandcast-v1.0.zip` (numpy model, demo on 16 real exam days, model card, training/)
- Work folder: the sandbox's `grid/demandcast/` (STATUS.md has the pipeline order; kept as `training/STATUS.md`)

## Recipe as built
- Physics: building-stock thermal model (UA x stock-averaged setpoint hinge, fast + slow RC envelope lag, sol-air and
  window gains, latent load, resistance / heat-pump / AC COPs vs temperature, occupancy schedules, daylight lighting,
  pvlib rooftop PV). Wind/solar: GridSim farm physics on every EIA-860 farm with real online dates.
- Checks: RC lag vs ODE 3e-7 K; stock hinge vs Monte Carlo 0.0035 K; degree-day limit exact; humidity vs PsychroLib 0.3 %;
  COP rating points; numpy == torch 3e-7; JS == Python 1e-4.
- Real noise: relative residuals of building-model fits to the training-input years 2018/2020/2022.
- Simulator: real training-year weather, perturbed; random building stocks around dev fits; drifted descriptors; real
  fleets re-scaled; wind/solar weather errors sized to dev mismatch; synthetic forecast errors on half the samples.
- Dev calibration (2019/2021 only): per-BA building fits (in-sample 2.2-3.5 %), cross-year descriptors for the dev
  bank, wind/solar scale per BA, CISO/PJM solar 1 h timestamp shift, sim error widths matched to dev, twin test.
- Network: 3 MLPs 512-512 (456k / 372k / 368k parameters), Gaussian NLL, 182,500 simulated region-days.

## Rungs (objective on frozen val_v3; gate = 30 seeded dev days per BA, <= +2 %)
| Rung | Change | Objective | Dev d/w/s % | Kept |
|---|---|---|---|---|
| r1 | MLPs 256-256, sim v1 | 0.333 | 2.06/21.4/15.8 | yes |
| r2 | sim v2 (calibrated widths) | 0.338 | 1.98/21.5/21.1 | no (solar broke on real) |
| r3 | + clamp negative night output, scale floor | 0.339 | 1.98/21.3/15.3 | no (+0.006) |
| r4 | v2 + v1 banks, 384-384 | 0.348 | 1.98/21.4/15.4 | yes |
| r5 | sim v3 (twin test found missing noise) | 0.348 | 1.99/21.8/15.0 | no |
| r6 | 4 banks, 384-384 | 0.356 | 1.95/21.5/15.5 | no (+0.008) |
| r7 | full 168 h history inputs | 0.355 | 2.27/22.4/16.2 | no (dev +8 %) |
| r8 | 6 banks, 512-512 | 0.362 | 2.00/21.4/15.6 | yes = v1.0 |

## Exam 2023-2025 (actual weather = upper bound)
Demand MAPE: ERCO 2.25 (grid DA 2.40), CISO 3.37 (6.86), PJM 2.10 (3.49), MISO 1.88 (2.71), SWPP 2.41 (4.31),
ISNE 3.00 (2.56), NYIS 1.97 (2.97), BPAT 2.10 (2.00); pooled 2.38 vs 3.41; persistence 4.90; physics alone 5.02.
Bands: 64 / 89 % inside 68 / 95 % (too narrow; CISO 46 / 74).
Forecast weather (Open-Meteo previous runs, 2024 only): pooled 2.92 vs grid DA 3.20 (2.44 with actual weather on the
same days); wins in CISO, PJM, MISO, NYIS; loses in ERCO (2.98 vs 2.05), SWPP, ISNE, BPAT.
Wind (sum|err|/sum output): 19.2 % vs physics 22.7, persistence 43.5, climatology 41.4; coverage 71 / 94.
Solar: 29.9 % vs physics 36.3, persistence 19.1 (loses), climatology 39.2. Good in ERCO 10.6, MISO 10.4, ISNE 16.4;
fails in CISO 59.4 and BPAT 51.8.
Extremes: Uri ERCO 14.5 vs 18.6; Elliott PJM 3.05 vs 3.78; heat dome Jun 2024 PJM 1.98 vs 4.50; Texas Aug 2023
1.96 vs 1.59 (peak over-predicted 3 %). Error flat 2.2-2.7 % across temperature bands.

## Limits
1. Actual weather is an upper bound; forecast mode only 2024 (archive rate limit), loses to 4 of 8 grid forecasts.
2. Solar: EIA-930 CISO solar timestamps moved (-1 h dev, 0 in 2023, +1 h 2024-25; PJM in 2025); dev shift wrong in exam.
3. Error bars too narrow on real data.
4. Wind limited by 50 km reanalysis in complex terrain (BPAT 50 %, CISO 33 %).

## Upstream candidates
- `buildings.py` building-stock demand model for GridSim (replaces the hybrid regression; 1.7-3.0 % cross-year on dev).
- GridData: flag EIA-930 solar timestamp convention changes (CISO 2023/2024, PJM 2025) and negative night solar.
- GridData: an archived day-ahead forecast-weather table (Open-Meteo previous runs; T from 2022, all vars from 2024).
- `renew.py` tabulated farm curve + fleet grouping (fast fleet physics).

## Next rungs
1. Learn the reporting lag in context: simulate random 0/+-1 h timestamp shifts so the net reads the lag from the week.
2. Real forecast-error noise (2022 T library) and a complete 2024-25 forecast-weather exam once the archive is fetched.
3. Better bands: per-BA noise scale variety in the simulator, or a dev-fitted conformal widening.

## Sandbox gotchas
- Open-Meteo returns HTTP 429 after ~1-25 year-requests per half hour on the shared proxy IP; fetch is resumable
  (`src/fetch_openmeteo.py`, cache data/openmeteo/), keep top-weighted points only.
- The session scratchpad is shared between builds: use a project subfolder.
- Bank generation is ~1 s per region-year on an idle core, ~3 s when four builds run.

## 14 Nov 2026 — brought into the repository

**Layout.** The release zip (`demandcast-v1.0`) became `models/demandcast/`: `demandcast.py`, `demandcast_weights.npz`
(4.8 MB, float32; the name is kept because `demandcast.py` and `pipeline.py` load it), `demandcast_meta.json`, `calib/`,
`pipeline.py`, `demo.py`, `examples/exam_days.npz`, `MODEL_CARD.md`, and `training/` with every script, `STATUS.md`
(the sandbox's resume notes), `ratchet.json` and `checks/`. `model.yaml` and `ratchet.yaml` are new; the rungs in
`ratchet.yaml` use the raw log's numbers (objective to 4 decimals, dev errors to 2).

**Paths.** `training/_paths.py` is new: `WORK` = `models/demandcast/work/` (or `$ML_RATCHET_WORK/demandcast`),
`GRID` = `shared/grid/` (or `$ML_RATCHET_GRID`), with `GRID/gridsim` and `GRID` on `sys.path`. `common.py` now sets
`ROOT = WORK` instead of the sandbox folder, so every output (`data/`, `ckpt/`, `checks/`, `web/`, `release/`) lands in
`work/`. `physics_check.py` reads GridSim's `shared/grid/checks/renewables.json`; `export.py` copies scripts from
`training/`; `real.py` falls back to the released `calib/*.json` when `work/data/calib/` has not been rebuilt.
`pipeline.py` puts `training/` on `sys.path` only when it runs, so importing it does not shadow other models' modules.

**Reproduced.**
- `demo.py` from the repository and from the extracted release print identical text: 16 exam days, demand MAPE 2.26 %
  against 3.77 % for the grids' own forecasts on those days.
- Physics checks 1-3 (RC lag vs ODE 3.2e-7 K, step 3.6e-15 K, hinge vs Monte Carlo 0.0035 K, degree days 1,080,000 MWh),
  the COP points (3.264, 1.936, 3.5) and the wind table (mean 0.00051 CF, max 0.038 away from cut-out) re-run to the
  same digits. The PsychroLib check was not re-run (the package is not installed here); the raw file says 0.22 %, not
  the 0.3 % in the notes above.
- The released exam numbers in the card and in `model.yaml` match the raw exam output to the printed precision. That
  file was not in the zip; it is recovered from the page's embedded copy into `training/exam_r8_wide.json` and
  `training/exam_r8_wide_events.json`. One claim in it is looser than the notes say: below -10 °C (913 h) the demand
  error is 3.25 % (grids 3.33 %); the 2.2-2.7 % band holds from -10 °C up.

**Not reproduced: the 2023-2025 exam re-score.** `training/realbank.py exam` needs the weather bundles from
`training/wxprep.py`, which read NASA POWER hourly series at every GridData weather point
(`gd.weather_point(pid)` -> `shared/grid/data/raw/weather/power/<pid>.parquet`). The GridData v1.0 zip ships only the
BA-level weather (`data/proc/weather/<BA>.parquet`), not `raw/`, so the bundles cannot be built without the full data
rebuild (`mlratchet data grid`, hours of downloads). The forecast-weather rows also need the Open-Meteo cache
(`data/openmeteo/`), which is not shipped. `exam.py` itself also loads the torch checkpoint; a numpy re-score would use
`exam.block()` with `demandcast.py`. Nothing was re-scored and no released number was changed.

**Spot check instead.** The original page holds 112 exam days (the release's `training/webdata.py` draw, seed 2024:
one random day per grid per quarter of each exam year, plus each grid's hottest and coldest), with the inputs the
pipeline built. With the shipped weights, demand MAPE on those 2,688 hours is 2.49 % against 3.51 % for the grids' own
forecasts, 4.87 % for persistence and 5.12 % for physics alone — in line with the exam's 2.38 / 3.41 / 4.90 / 5.02.

**Live instrument.** `web/export_instrument.py` reads those 112 days from `lab/index.html` and writes
`site/public/instruments/demandcast/` (3.8 MB: weights 2.4 MB in float16, days 0.8 MB, reference 0.6 MB). All three
networks run live. The demand physics for the 38 h from issue to the end of day D is stepped from the filter states at
issue; that stepper matches `buildings.demand()` to 2.7e-16 of total demand on real weather and a real calendar. Wind
and solar physics stay tabulated in Python over weather multipliers (wind × 0.5-1.5, irradiance × 0.3-1.5, steps of
0.1) and are interpolated, as on the original page. With neutral controls the stepper reproduces the page's stored
prior to 1.3e-4 (prior stored to 1 MW) and the physics features to 5e-4 (temperature stored to 0.01 K). Float16 weights
move demand by at most 6e-5. Browser vs Python on 80 cases (16 days × 5 control settings): worst 2.3e-7 relative.

## 14 Nov 2026 — exam re-scored from the shipped weights

NASA POWER hourly series for all 184 GridData weather points were fetched into `shared/grid/data/raw/weather/power/`
(the data rebuild's `power` stage; the v1.0 zip does not carry them), which removed the blocker above. The pipeline was
rebuilt in `work/` from the scripts as run: `wxprep.py` (8 grids in parallel, about 2 min), `renew.py`,
`calib_demand.py dev2019` / `dev2021` and `calib_renew.py dev2019` / `dev2021` (the per-year fits the cross-year dev
bank needs; the zip ships only the two-year `dev` fits, which the exam uses), `realbank.py dev` (5,840 days) and
`realbank.py exam` (8,760 days). Then `training/rescore_release.py` (new) ran `exam.main()` unchanged, with a numpy
stand-in for `evaluate.py` (load / predict / sigma_scale on `demandcast.py` and `demandcast_weights.npz`; `exam.py` and
`evaluate.py` need the torch checkpoint). About 25 min wall clock.

- The rebuilt physics matches the release: the wind and solar dev scales from the rebuilt priors equal the released
  `calib/renew_dev.json` to 4 decimals for all 8 grids, and the rebuilt exam rows for the 16 `examples/` days equal
  `examples/exam_days.npz` exactly (inputs, physics prior, truth, grid forecast, wind and solar; difference 0).
- Re-fitted sigma scales on the rebuilt dev bank: demand 1.14078 (released 1.14077), wind 1.0 (1.0), solar 1.33997
  (1.33996).
- All 125 compared numbers reproduce to the printed precision (`work/data/exam/release_vs_released.json`): pooled demand
  2.38 % vs grid DA 3.41, persistence 4.90, same hour last week 7.95, physics alone 5.02, bands 64 / 89 %, 209,340
  hours; wind 19.2 % vs physics 22.7, persistence 43.5, climatology 41.4, bands 71 / 94 %; solar 29.9 % vs physics
  36.3, persistence 19.1, climatology 39.2, bands 65 / 84 %; per grid demand ERCO 2.25 (DA 2.40), CISO 3.37 (6.86),
  PJM 2.10 (3.49), MISO 1.88 (2.71), SWPP 2.41 (4.31), ISNE 3.00 (2.56), NYIS 1.97 (2.97), BPAT 2.10 (2.00), and every
  per-grid wind, solar, persistence, physics and coverage value. Largest difference: solar 68 % coverage 65.42 vs
  65.43 %. The temperature bands agree to 2e-7 (3.25 % below -10 °C, as noted above).
- Not re-scored: the forecast-weather mode (2024: 2.92 % vs 3.20 %, wins in 4 of 8 grids). The Open-Meteo cache is not
  shipped, so `real.Forecasts` finds no archived forecasts and `realbank.py exam` builds actual-weather rows only. The
  named-period numbers (`events.py`) were not re-run. No released number changed.

## 14 Nov 2026 — the GridSim outage-rate leak (HANDOFF P0.5): DemandCast is not exposed

GridSim v1.0's default forced and planned outage shares for coal, combined cycles, gas turbines and gas steam came from
Texas RE's 2024/2025 assessments (data from the exam years); GridSim physics 1.1 replaces them with pre-2023 sources.
DemandCast uses GridSim only for weather-to-power physics (`gridsim.renewables`: air density, farm curves) and the demand
module, never dispatch or the outage shares, so nothing it learned or chose touched them. No number changes.

Two other GridSim citations reach into the exam years and do touch DemandCast's physics: the turbine specific-power trend
(LBNL Land-Based Wind Market Report, 2023 edition, published Aug 2023) describes turbines installed up to 2022, so it
carries no exam-period measurement; and the EIA-930 solar timestamp offsets now flagged in GridData
(`gd.eia930_solar_qc()`) confirm the cause of the solar loss found after the exam (CISO's offset flips sign in 2024).

## 14 Nov 2026 — forecast weather fetched; released 2024 row reproduced; second look 2024-2025

**Fetch.** GridData's build gained an optional stage `forecastwx` (`mlratchet data grid --only forecastwx`) that runs
`fetch_openmeteo.py`'s request list (same variables, points, years, endpoint `previous-runs-api.open-meteo.com`,
default model) into `shared/grid/data/raw/openmeteo/<pid>_<year>.json`, one year-request at a time, 15 s apart, waiting
on HTTP 429 and stopping on a daily limit. From the build machine it answered every request: 220 year-requests, 92 MB,
about 1.5 h, no HTTP 429. Coverage checked first with two 2-day requests: the archive holds day-1 temperature forecasts
from 2022 but humidity, radiation and wind forecasts only from 2024, so 2023 rows (and the dev years) cannot be graded
in forecast mode; that is why the release planned 2024-2025 plus 2022 temperature. `real.py` reads the cache from
`$DC_OPENMETEO`, else `work/data/openmeteo/`, else the grid data root (new `om_dir()`; nothing else changed).

**Released 2024 row: reproduced.** A first re-score with the shipped list's points (top 4 population points per grid,
ERCO's 3 wind points) gave the same rows and hours (66,079 demand hours) but not the same numbers: pooled 2.99 % vs
3.20 % (released 2.92 %), every grid but BPAT worse by 0.05-0.10, ERCO wind 26.7 % (released 31.2 %). A re-fetch of
one file an hour later was identical to the cache, so the archive is stable. The difference is the release's cache:
matching the forecast-mode physics-prior error per grid (inputs only, no network) finds one set of points. ERCO with
all 8 population points gives 6.497 % (released 6.497; top 4: 6.513); NYIS needs the top 4 plus NYIS_pop_0
(5.29966 vs released 5.29966; no top-k matches); BPAT the top 4; ERCO wind its top point only (31.24 % vs physics
47.86 %, exact; 2 points 27.5 %, 3 points 26.7 %). That fits an early pass in file order (pid 0-7, grid by grid) that
the rate limit stopped after NYIS_pop_0, then the top-4 pass. With that cache (`training/rescore_forecast.py
release2024`; the stage fetches population points 5-8 of the 7 grids for this) all 338 forecast-mode numbers of the
released 2024 row reproduce to 1e-4 (largest difference: same-days demand 68 % coverage 63.285 vs 63.286 %): pooled
2.92 % vs 3.20 %, wins CAISO, PJM, MISO, NYISO, losses ERCO 2.98 vs 2.05, SPP, ISO-NE, BPA. No released number changed.
The points tried to find the cache are a diagnosis of the archive, not of the model; nothing in the model was chosen.

**Second look, forecast weather 2024-2025, run once (`rescore_forecast.py full`).** The shipped fetch list (top 4
population, 3 wind, 3 solar points per grid, every grid) for both years, same weights, same dev sigma scales (1.14078,
1.0, 1.33997). Demand 3.05 % on 135,700 hours vs grid DA 3.59 %, persistence 4.91 %, physics alone 5.88 %, bands
71 / 93 %; same days with actual weather 2.48 %. By year 2.97 / 3.20 (2024; one more day than the released row, since
2025 files now cover 31 Dec) and 3.13 / 3.97 (2025). Wins CAISO 3.92 vs 7.58, PJM 2.71 vs 3.50, MISO 2.15 vs 2.58, SPP
2.96 vs 5.01, NYISO 2.84 vs 3.24; loses ERCO 3.33 vs 2.26, ISO-NE 4.18 vs 2.65, BPA 2.33 vs 1.96. The pooled win rests
on the two grids with flawed published forecasts (CAISO misaligned by 1 h, SPP 6.5 % high in 2025): the other 6 pooled
give 2.92 % vs 2.70 %, a loss. Wind 26.7 % (131,024 h, all 8 grids) vs physics 39.0 %, persistence 42.9 %, climatology
40.6 % (actual weather on the same days 19.2 %); solar 34.7 % vs persistence 18.9 %, a loss. Reported next to the
released row in `model.yaml` and `MODEL_CARD.md`; it chose nothing.

**2022 library.** `make_tfc.py` built `work/data/noise/tfc_2022.npz` (363 days per grid): day-D temperature forecast
error RMS 1.28 K (MISO) to 2.01 K (CISO), mean bias -0.85 K (PJM) to +1.38 K (CISO). Not used in training; a rung that
uses it would be judged on dev.

**Card corrections (site honesty audit).** Each was a claim the exam file does not support, or first-person copy:
- `wins` said the error "stays at 2.2-2.7 % from -10 °C to above 35 °C". Above 35 °C (1,211 h) it is 2.73 % vs the
  grids' 2.55 %, a loss: moved to `losses`; the win now says -10 to 35 °C (2.2-2.7 % vs 2.9-3.9 %), and names PJM for
  the cold snap (the same Jan 2025 outbreak in ISO-NE is a loss, 4.78 vs 1.27, already in `losses`).
- `MODEL_CARD.md`: PsychroLib check 0.3 % -> 0.22 % (`training/checks/physics.json`); "flat at 2.2-2.7 % from < -10 C
  to > 35 C" -> 2.2-2.7 % from -10 to 35 C, 3.25 % below -10 C (913 h; grids 3.33 %), 2.73 % above 35 C (grids 2.55 %);
  the "all" row's same-hour-last-week cell filled (7.95, from `exam_r8_wide.json`); "ours", "us", "we" -> DemandCast.
- `help_wanted`: the Open-Meteo line now says what was found (fetched in full; no forecasts of humidity, radiation or
  wind before 2024).

## 15 Nov 2026 — v1.1: the solar reporting offset is read at issue

**The loss.** Solar lost to persistence in the exam: 29.9 % vs 19.1 % with actual weather, 34.7 % vs 18.9 % with
forecast weather (2024-2025). The cause was found after the exam: EIA-930's solar timestamps move between years, and
v1.0 shifts CISO and PJM by one fixed hour learned on 2019/2021.

**Same failure outside the exam.** GridData's `eia930_solar_qc()` flags (allowed for training-input and dev years): in
2018-2022, 9 of 35 BA-years with solar are 1 h off (CISO 2018-21, PJM 2018-22), and CISO moved from -1 h (2019, 2021)
to 0 h in 2022. With the fixed dev shift, v1.0 on CISO 2022 (a training-input year; its solar output was never a
training sample) is 22.8 % vs 12.7 % for persistence; PJM 2022 13.8 % vs 30.0 %. On the 30 seeded dev gate days per
grid, moving the reported solar by -1 / 0 / +1 h gives v1.0 22.8 / 15.6 / 37.7 % against persistence 16.6 %.
`eia930_solar_qc()` itself is not usable at forecast time: each flag is fitted on a whole calendar year, the days after
issue included.

**Fix (rung r9).** `demandcast.solar_offset` reads the offset from the 168 h before issue only. If the report lags the
physics by r hours, the reported-minus-physics error is e ~ (kappa - 1) p - kappa r p'; a least-squares fit on the
physics diurnal profile p and its slope p' (both from the last 24 h, by LST hour) gives r_hat = -beta / (1 + alpha),
rounded to -1, 0 or +1 h at 0.5. On a synthetic diurnal curve with 10 % noise it reads -0.97 / 0.00 / +0.96 for true
-1 / 0 / +1 h. The last 24 h of reported solar are put back on the physics clock (the hour outside the history is
the physics CF times the week's ratio), the r8 network runs unchanged, and its forecast is moved by r. No learned
parameters; the exam day's own output never enters.

- r9a, counted from the physics clock: on dev it moved 201 of 730 CAISO days by an hour. CAISO's dev-year r_hat has
  median +0.46 h (after the fixed shift): its reported solar sits about half an hour off the physics, so 0.5 is a coin
  toss there. Dev solar 15.60 -> 16.11 %.
- r9b, counted from each grid's dev-year median r_hat (`solar_offset_ref`: CISO 0.46, SWPP 0.19, MISO 0.11, PJM 0.09,
  ISNE 0.08, ERCO 0.05, BPAT 0.04 h), the clock the fixed dev shift left each grid on. A dev calibration, like the
  shift itself. Dev solar 15.60 -> 15.68 %, dev gate +0.2 %. On the moved dev days: 15.6 / 15.7 / 16.4 % for -1 / 0 / +1 h. CISO 2022: 15.6 %
  (persistence 12.7 %, still a loss there).

**Frozen set.** The measurement model changed, so simulator v4 = v3 plus a reporting offset of the simulated solar
(-1 / 0 / +1 h with probability 0.15 / 0.70 / 0.15, constant per region-year; `sim.py`, `DC_SIM=v4`), and a new frozen
set val_v4 (`sim.py val_v4 404 100 40`: 4,000 region-days). The r1-r7 checkpoints are not in the repository, so only the
release r8 is re-scored on it: objective 0.3329 (solar skill 0.386). r9b: 0.3752 (solar skill 0.513), +0.042. The
offset is read right on 3,886 of 4,000 region-days (true -1 h: 475 / 45 / 0 read -1 / 0 / +1; true 0: 35 / 2,954 / 11;
true +1 h: 0 / 23 / 457). Solar error on regions reported 1 h early / on time / 1 h late: v1.0 36.2 / 14.5 / 31.8 %,
r9b 18.4 / 14.7 / 15.8 %. Kept under the rule (+0.01 and dev <= +2 %) and released as v1.1. The solar sigma scale is
re-fitted on dev: 1.3400 -> 1.3686. Demand and wind are unchanged. The release choice was committed before the exam.

**Work folder rebuilt** (it was empty): `wxprep.py`, `renew.py`, `calib_demand.py dev2019 / dev2021 / train`,
`calib_renew.py dev2019 / dev2021`, `realbank.py dev / exam`. pvlib was missing from the main venv; installed from the
project's `grid` extra (pvlib 0.16.1). The re-scored v1.0 reproduces every released number below.

**Exam, run once (`training/rescore_v11.py actual` and `forecast`; v1.0 re-scored on the same rows).**

| Row | v1.0 | v1.1 | Persistence | Physics alone | Verdict |
|---|---|---|---|---|---|
| Solar, 7 grids, actual weather (183,858 h) | 29.9 % | **16.1 %** | 19.1 % | 36.3 % | loss -> win |
| Solar, forecast weather 2024-2025 (119,348 h) | 34.7 % | **18.2 %** | 18.9 % | 39.2 % | loss -> win (by 0.7 points) |
| Solar bands, daylight, actual weather (68 / 95 %) | 65 / 84 % | 69 / 86 % | | | |

Per grid, actual weather, v1.0 -> v1.1 (persistence): CISO 59.4 -> 24.9 % (11.4 %), PJM 28.1 -> 11.9 % (28.0 %), ERCO
10.6 -> 10.6 %, MISO 10.4 -> 10.4 %, ISNE 16.4 -> 16.5 %, SWPP 41.5 -> 41.5 % (29.1 %), BPAT 51.8 -> 52.0 % (30.5 %).
Offsets read at issue: CISO 2023 +1 h on 232 of 365 days, 2024 on 362 of 366, 2025 on 356 of 364; PJM 2025 on 311 of
364; the other grids 0 on 350-366 days a year. Small changes the wrong way: ISNE +0.1, BPAT +0.2, MISO 2023 +0.1, SWPP
2023 +0.3 points (a few days read as 1 h off). CISO and BPA solar still lose to persistence, and so does SPP. Every
demand and wind row is identical to v1.0 (2.38 %, 19.2 %; forecast weather 3.05 %, 26.7 %). Wins in model.yaml: 12 ->
14 (the two solar rows); no row got a worse verdict.

Golden demo output regenerated: only the CISO solar days change (26.8 % and 51.2 %, were 53.3 % and 82.3 %). The live
instrument stores the offset each page day's history shows (read in Python from the exam bank's full feature vector)
and does the same alignment and shift in the browser; parity 2.3e-7.

## 27 Nov 2026 — audit: inputs the simulator never varies (warnings added; repair queued)

The two-reviewer audit (`docs/audits/demandcast-v1.1.md`) found two inputs the released networks over-read. Reported
solar at dark hours (20:00 LST twilight) pinned two BPAT exam days at the 1.2 cap. Missing demand-history hours sit
at |z| up to 286. Counted on the exam bank's inputs: 1,734 of 12,642 solar rows and 365 of 14,448 demand rows are
affected. The exam scored them as they were. The reader now warns on both and offers an opt-in `z_clip`. The repair
(night-masked solar history, simulated missing demand hours, higher floor on count-like inputs, ±2 h solar offsets)
needs a retrain and is queued, with a dated re-score after it on the owner's approval.

## 27 Nov 2026 — dated z_clip re-score (v1.1.1), the repair rung r10, and its exam (v1.2)

The owner approved the queue: a dated re-score of v1.1 with `z_clip=8`, then the repair rung judged by the ratchet,
with the exam opened once only if the ratchet promoted it.

**1. Re-score of v1.1 with z_clip=8** (`training/rescore_zclip.py`; the released weights and arithmetic, every
standardised input clipped at ±8; sigma scales re-fitted on dev as exam.py does; nothing chosen, the 8 was the
audit's). Counts first: the audit's 1,734 solar rows are those with output at 20:00 LST (feature 66) over every row of
the bank (actual and forecast weather); any dark hour from 16:00 to 07:00 LST gives 4,029 bank rows, 2,350 of them
actual-weather. The 365 missing-history demand rows are 201 actual-weather rows and 164 in the 2024-2025
forecast-weather bank.

| | v1.1 | v1.1.1 (z_clip 8) |
|---|---|---|
| Demand, actual weather (209,340 h) | 2.382 % | 2.323 % |
| — the 201 rows with missing history hours | 5.96 % | 2.58 % |
| — the other 8,559 rows | 2.32 % | 2.32 % |
| Demand, forecast weather 2024-2025 | 3.051 % | 2.961 % |
| — the 164 rows with missing history hours | 7.37 % | 3.25 % |
| Demand, forecast weather 2024 (released row) | 2.923 % | 2.834 % |
| Solar, actual weather | 16.14 % | 16.12 % |
| — the 2,350 rows with output at a dark hour | 25.24 % | 25.20 % |
| — the 973 rows with output at 20:00 LST | 26.29 % | 26.20 % |
| Wind, actual weather | 19.21 % | 19.21 % |

No grid got worse; ERCOT 2.25 -> 2.17, PJM 2.10 -> 1.91, CAISO 3.37 -> 3.24, MISO 1.88 -> 1.79 (actual weather). The
clip repairs the missing-hour rows. It does nothing for the dark-hour solar rows, as the audit said: there the inputs
sit at |z| ~3 under the 0.02 floor and the amplification is in the first-layer weights. It held, so it became the
default as **v1.1.1** (`demandcast_meta.json` `z_clip: 8`, sigma scales d 1.1406, s 1.3672).

**2. Rung r10, the repair** (`training/repair.py`, `train.py --repair --base`, `sim.py` with `DC_SIM=v5`). Fixed
before training:
- Simulator v5 = v4 + offsets -2..+2 h (3 / 12 / 70 / 12 / 3 %), output at dark edge hours next to daylight in 30 % of
  region-years (2-15 % of the neighbouring hour), and a 1-48 h gap in the demand history of 8 % of samples.
- New frozen set val_v5 (`sim.py val_v5 505 100 40`, 4,000 region-days, 295 with a gap), built before any training.
  Training banks train_v5a-c (seeds 5101-5103, 1,500 regions × 40 days each, 180,000 region-days; ~52 min on 3 cores).
- Solar: offsets read to ±2 h (`solar_offset(rmax=2)`), history aligned, then the reported last 24 h set to 0 where the
  physics CF < 1e-6 (`mask_dark_solar`); the same transform runs on the training rows, whose target is put on the
  physics clock by the offset read. Count-like inputs (demand day-type one-hot, DST, forecast flag, miss_frac; wind and
  solar flag and missing fraction) get an sd floor of 0.05 (was 0.001 for demand: miss_frac's training sd was exactly
  the floor). Inputs clipped at |z| 8. Three 512-512 MLPs on base features, 40 epochs, lr 2e-3 (as r8); 6 min on the GPU.

Ratchet (training/ratchet.json; incumbent v1.1.1):

| | val_v5 objective | demand / wind / solar err on val_v5 | demand, gap samples | dev demand / wind / solar |
|---|---|---|---|---|
| v1.1 | 0.3488 | 2.98 / 26.7 / 16.0 % | 11.18 % | 2.00 / 21.44 / 15.68 % |
| v1.1.1 | 0.3908 | 2.34 / 26.7 / 15.9 % | 2.59 % | 1.98 / 21.44 / 15.67 % |
| r10 | 0.4054 | 2.22 / 26.6 / 15.4 % | 2.36 % | 2.05 / 21.23 / 13.11 % |

Solar error on val_v5 by true offset -2 / -1 / 0 / +1 / +2 h: v1.1.1 33.7 / 20.9 / 14.3 / 17.0 / 32.8 %, r10 18.3 /
20.2 / 14.2 / 15.9 / 12.6 %. Offsets read on val_v5 (true -> read): -2 h 55 of 80 right, +2 h 32 of 40. On val_v4:
0.3752 / 0.3751 / 0.3774. Decision: objective +0.015 (margin 0.01), dev errors -4.8 %: **promoted**. The dev gate
already showed demand 3 % worse (1.98 -> 2.05 %); the pooled gate let solar's 16 % dev gain carry it.

Weights exported (`demandcast_weights.npz`; v1.1's kept as `demandcast_weights_v1_1.npz` with
`demandcast_meta_v1_1.json` = v1.1.1), sigma scales fitted on dev (d 1.3417, w 1.0, s 1.1204), the class checked against
the rung's arithmetic (6e-8 on dev rows), and the choice committed (285eedd) before the exam.

**3. Exam of v1.2, run once** (`repair.py exam`: actual weather, forecast weather 2024-2025, the released 2024 row;
`repair.py events`; side by side in `training/exam_v1_2.json`):

| | v1.1 | v1.1.1 | v1.2 |
|---|---|---|---|
| Demand, actual weather | 2.38 % | 2.32 % | 2.49 % |
| Demand, forecast weather 2024-2025 | 3.05 % | 2.96 % | 3.14 % |
| Demand, 6 grids without CAISO / SPP, forecast weather (grids 2.70 %) | 2.92 % | 2.84 % | 2.98 % |
| Demand, forecast weather 2024 (grids 3.20 %) | 2.92 % | 2.83 % | 3.00 % |
| Grids beating their own forecast, actual weather | 6 of 8 | 6 of 8 | 5 of 8 (ERCOT 2.41 vs 2.40) |
| Wind, actual / forecast weather | 19.2 / 26.7 % | 19.2 / 26.7 % | 19.3 / 26.9 % |
| Solar, actual weather (persistence 19.1 %) | 16.1 % | 16.1 % | 10.6 % |
| Solar, forecast weather 2024-2025 (persistence 18.9 %) | 18.2 % | 18.1 % | 12.0 % |
| Solar CAISO / BPA / SPP, actual weather | 24.9 / 52.0 / 41.5 % | 24.9 / 51.7 / 40.4 % | 11.2 / 20.8 / 27.0 % |
| Demand bands 68 / 95 % | 64 / 89 % | 64 / 89 % | 64 / 89 % |
| Solar bands 68 / 95 % | 69 / 86 % | 69 / 86 % | 72 / 90 % |

What it says: the repair fixed solar. It now beats persistence in every grid, with CAISO only just ahead (11.2 vs
11.4 %); BPA's two pinned days are gone. Demand got worse in every grid than v1.1.1 (ERCOT 2.17 -> 2.41, ISO-NE 3.00 ->
3.32, NYISO 1.97 -> 2.17), as the dev gate's demand column said it would. The demand network changed in three ways at
once (v5 banks only instead of r8's v1-v3 mix, gap samples, the 0.05 floors), so which one cost demand is not known.
Events: Texas heat Aug 2023 1.96 -> 2.08 %, PJM heat dome Jun 2024 1.98 -> 2.37 %, ISO-NE Arctic outbreak Jan 2025 4.78
-> 4.49 %, Uri (dev) 14.5 -> 16.9 %.

The exam chooses nothing, so v1.2 ships as the ratchet promoted it, and the card reports the demand loss. The lesson,
the same one BatteryPilot v1.3 taught with years: a pooled gate lets one strong component carry a loss elsewhere. Next
ladder: a per-output gate fixed before training, and a rung that pairs v1.1.1's demand network with v1.2's solar.

Golden demo output regenerated on purpose (new weights). The instrument was re-exported with the v1.2 weights; the
browser clips inputs at |z| 8, aligns ±2 h offsets and masks dark hours as `demandcast.py` does.

## 27 Nov 2026 — per-output dev gate (exam-informed rule change); rung r11 rejected

**Rule change.** The owner adopted a per-output dev gate for DemandCast: a rung must keep each output's dev-gate error
(demand, wind, solar) no more than 2 % worse than the incumbent's, separately, so a gain in one output cannot carry a
regression in another. This is an exam-informed rule change. It was motivated by v1.2's exam, where demand got worse
(2.49 % vs 2.32 % for v1.1.1) while solar improved (10.6 % vs 16.1 %). The old pooled gate had already shown dev demand
+3 % (1.98 -> 2.05 %) and passed r10 on the mean (-4.8 %). The rule applies from the next rung on; v1.2 stays released
under the rule it was judged by. Recorded as a new ladder in ratchet.yaml ("DemandCast v1.3 ladder").

**Rung r11** (`training/repair.py add_per_output r11_split`): v1.1.1's demand network (r8, inputs clipped at |z| 8) with
r10's wind and solar networks and r10's solar input changes (±2 h offsets, dark-hour mask). No training. Incumbent v1.2.

| | val_v5 objective | demand skill / err on val_v5 | dev demand / wind / solar |
|---|---|---|---|
| v1.2 (r10) | 0.4054 | 0.560 / 2.22 % | 2.046 / 21.23 / 13.11 % |
| r11 | 0.3973 | 0.535 / 2.34 % | 1.984 / 21.23 / 13.11 % |

Per-output gate: demand -3.0 %, wind +0.0 %, solar +0.0 %: passed. Objective -0.008 (needs +0.01): **rejected**. On
val_v4: 0.3774 vs 0.3765. No exam was run and no weights changed. The frozen set and the dev years disagree on which
demand network is better: val_v5's simulated history gaps (295 samples, demand 2.59 % for r8 vs 2.36 % for r10) and its
v5 noise favour r10, the real dev days favour r8. Had r11 been promoted, both halves' exam scores were already known
from v1.1.1 and v1.2, so its exam would have added little; the frozen set has to be brought into line with dev on demand
before a demand rung can be judged fairly.
