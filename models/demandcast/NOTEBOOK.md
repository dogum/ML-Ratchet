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
