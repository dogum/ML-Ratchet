> **Lab notebook.** SnowSense was built in a claude.ai cloud session on 18 Nov 2026 on the Watershed platform
> (WaterSim + WaterData, now in `shared/water/`). The first entry is that session's status document as written; its
> paths (`results/`, `banks/`, `release/`, `web/`) refer to the sandbox workspace, now `work/` (see
> `training/_paths.py`). `MODEL_CARD.md` holds every number; the import entry at the end says what was checked.

## 18 Nov 2026 — build notes (snowsense-status.md, as written)


Daily snow water equivalent (SWE) at a mountain snow-pillow station, read from that station's own rain gauge and thermometer. It also gives each winter's peak SWE, peak date and melt-out date, each with an error bar. It was trained only on WaterSim snow simulations and never sees a snow measurement.

- Page (the network runs live in the browser: station explorer, a weather lab, exam, ratchet): https://claude.ai/artifact/QQdrSbSKeN7SH2eqLFLbEW
- Release: `snowsense-v1.0.zip`, containing:
  - numpy-only inference (`snowsense.py`) and weights
  - `pipeline.py` (CSV in, CSV out) and `demo.py` (three real exam stations, about 2 s)
  - model card
  - all training and diagnostic scripts, plus `ratchet.json`

### Recipe as built

- **Inputs:** the SNOTEL station's own daily P, T_mean, T_min and T_max, plus latitude and elevation. Missing P counts as 0 and missing T is interpolated; both are flagged. Each water year is read from 1 Aug to 1 Oct (427 days).
- **Physics:** WaterSim snow only (`model='none'`), one band, Hargreaves radiation.
  - degree-day (HBV-light): exact against a hand calculation; equal to TUWmodel to 1e-12 mm
  - daily energy balance: terms exact against hand arithmetic
  - each simulated winter uses one of the two, 50/50
- **Weather:** real P and T from the 33 usable dev stations, WY ≤ 2010 only (WY2011–2025 is the dev gate). Six dev stations are held out to make the frozen simulated validation sets. Climate shifts: P × e^N(0,0.3), T + N(0,1.5) °C.
- **Real noise:** the simulated snowpack feels the "true" weather, while the network reads the gauge record:
  - wet-day P × lognormal(0.25)
  - one-day timing slips
  - T error AR(1) 0.6 °C plus a 0.5 °C station bias
  - injected data gaps
- **Calibrated on dev:** both routines calibrated on each dev station's own SWE (WY ≤ 2010, wider bounds). A multivariate-normal prior was fitted to the 36 parameter sets per routine.
- **Network:** 2-layer GRU, 64 hidden (51k parameters), five quantiles of √SWE, pinball loss. It was trained on 48k simulated station-winters, on one CPU thread.
- **Ratchet:** a rung is kept if it scores more than +0.01 median per-series NSE on the newest frozen sim set and the 38 dev stations get no more than 5% worse in (1 − NSE).
- **Checks:**
  - numpy vs torch: 1.2e-6 in z (5e-4 mm)
  - JS vs numpy: 0.005 mm

### Rungs (final re-scoring on val_p4; the exam was run once, at the end)

| Rung | Change | Sim (val_p4) | Dev NSE | Exam NSE | Exam melt-out MAE |
|---|---|---|---|---|---|
| R1 | WaterSim parameter ranges, uniform; no measurement error | 0.744 | 0.728 | 0.734 | 11.7 d |
| R2 | Gauge error between what the snowpack feels and what the network reads, plus gaps | 0.769 | 0.805 | 0.765 | 12.7 d |
| R3 | Dev-fitted parameter prior (×1.3 inflation) | 0.810 | 0.880 | 0.854 | 14.3 d |
| R4 | Quantile head + pinball loss instead of a Gaussian | 0.822 | 0.882 | 0.886 | 8.3 d |
| R5 | Prior as fitted (inflation 1.0). **Rejected:** 0.831 is not > 0.822 + 0.01 | 0.831 | 0.887 | 0.883 | 8.5 d |
| R6 | Second p4 bank (48k windows), 6 more epochs from R5. **Released** | 0.839 | 0.888 | 0.888 | 8.3 d |

- **Frozen sets:** val_p1 to val_p4, one per physics version. Every rung was scored on each, and the decisions were re-taken in order on the newest set.
- **Twin test** (dev stations, calibrated energy-balance twins; the twin itself scores 0.917 against the pillows):
  - **R1:** read the twin at 0.80 and real at 0.73, with peaks 21% below the twin. The gap was the parameter prior, which led to R3.
  - **R3:** melted out 10 days after the twin. A Gaussian mean over many zero-SWE days drags the curve late, which led to R4.
  - **R6:** reads the twin at 0.914 and real at 0.888. Peak bias is −4% against the twin and −9% against real.

### Against classical methods (exam: 621 stations, WY2011–2025, B's protocol, medians)

| Method | Protocol | NSE | RMSE | Peak SWE \|err\| | Melt-out MAE |
|---|---|---|---|---|---|
| **SnowSense** | weather only | **0.888** | 51.7 mm | 17.5% | 8.3 d |
| Energy balance, default parameters | weather only | 0.853 | 57.2 mm | 23.7% | 9.1 d |
| Degree-day, default parameters | weather only | 0.691 | 80.0 mm | 28.5% | 21.9 d |
| Climatology (own 2001–2010 SWE) | uses SWE | 0.701 | 84.6 mm | 28.2% | 11.4 d |
| Degree-day calibrated per station (gauged ceiling) | uses SWE | 0.925 | 41.2 mm | 12.5% | 6.1 d |

- **Station wins:** SnowSense has the higher NSE at 75% of stations against the energy balance and 34% against the calibrated ceiling.
- **Peak SWE:** median bias −9%; peak date median error 3 days.
- **Weakest groups:** stations below 1500 m (0.787) and the Southwest (0.776).
- **Error bars:** all widen factors were fitted on dev and came out at 1.0. Exam coverage:

  | Output | 68% bar | 90% bar |
  |---|---|---|
  | Daily SWE | 78% | 96% |
  | Peak SWE | 76% | 93% |
  | Melt-out | 86% | 95% |

  The bars are conservative; the widen-only rule cannot narrow them.

### Upstream candidates for WaterSim

- **Energy-balance turbulent transfer is too strong at SNOTEL points.** Dev calibrations put TURB at a median of 0.17 (default 1.0) and GHEAT at 0.32. Default EB peaks come out 20% low on the exam.
- **Albedo ageing:** the dev median ALB_DECAY is 2.4 days (default 12), which compensates for the lower turbulence.
- **Degree-day TT at points:** +1.0 °C (dev, wider bounds). A found +1.3 °C.
- **Generator guard:** cold-shifted forcing can grow perennial packs (SWE in metres after years). Drop or cap winters that start with snow on 1 Aug.

### Next rungs worth trying

- **Peak undercatch:** a wind or exposure-dependent snowfall factor (SFCF) prior, or a drift term. The −9% peak bias is the main miss.
- **Rain-on-snow and thin packs below 1500 m:** sub-daily melt-refreeze, or a rain-energy term in the degree-day routine.
- **Gridded forcing mode** (ERA5-Land at basin bands), so SnowSense can feed FlowCast. It needs ERA5 point extraction or basin-band training.
- **Sharper error bars:** a per-station-climate conditional prior. Dev correlations were weak (|ρ| ≤ 0.43), so collect more dev evidence first.

### Rule notes

- Dev station SWE was used three ways:
  - WY ≤ 2010 for the parameter prior
  - WY2011–2025 for the gate, sigma calibration and twin test
  - the dev weather only up to WY2010 for training
- Exam stations' weather was never used for training.
- The R6 network was initialised from the rejected R5 weights. Only the decision compares against the incumbent.
- val_p3 existed before R2 was decided, so R2 was decided on it. Under the re-take rule everything now stands on val_p4.
- The release demo was test-run on three exam stations before the final exam, to check code only. Nothing was chosen on it.

### Sandbox gotchas

- **Lazy npz reads:** do not rebuild an `.npz` that a running trainer reads lazily (the crash showed as BadZipFile CRC). The trainer now loads val into RAM and memory-maps the training banks (`banks/*_X.npy`).
- **`setsid` from the tool shell** prints "Done" at once. The job is still running; check with `ps`.
- **Machine load:** 4–7 load on 2 cores, so a step took 0.6 s alone and 1.2–1.9 s shared. Each 6-epoch rung took 20–30 min.

## 18 Nov 2026 — imported into ML Ratchet

What changed on import (no weights, inference numbers or training results changed):

- Folder layout follows `docs/model-contract.md`. File names kept. The release README became `model.yaml` (the README is
  generated). `examples/README.md` credits SNOTEL. `tests/golden/snowsense.txt` is the demo's output.
- `snowsense.py`: `DatetimeIndex.dayofyear` -> `day_of_year` (the same values; pandas printed a deprecation warning per
  call). `demo.py`: `stations.json` is opened with a context manager and UTF-8 (a ResourceWarning). Demo output is
  unchanged: NSE 0.820, 0.980, 0.822 for the three stations, as before.
- `training/_paths.py` maps the sandbox workspace to `WORK` (`models/snowsense/work/` or `$ML_RATCHET_WORK/snowsense/`)
  and the platform to `shared/water/` (code) and `$ML_RATCHET_WATER` (data root, default `shared/water`). `common.py`
  sets `ROOT = WORK`; the snow-baseline table path is `SNOW_SCORES`; `webdata.py` and `export.py` import `snowsense.py`
  from the model folder (they imported a sandbox `release/` copy). No absolute sandbox path is left. The scripts were
  not run here; they need the SNOTEL series (`mlratchet data water`), torch, and numba for speed.
- `training/check_js.cjs` and `snowsense.js` were in the sandbox's `web/` folder; `check_js.cjs` still expects
  `weights.json` and `data.json` beside it (written by `export.py` and `webdata.py` under `WORK/web/`).
- New: `web/export_instrument.py` (the live instrument's packs and parity cases), the TypeScript instrument
  (`site/src/instruments/snowsense/`), `web/build_lab.mjs` and `lab/`. The original artifact page
  (https://claude.ai/artifact/QQdrSbSKeN7SH2eqLFLbEW) and its `data.json` were not in the zip, so `lab/index.html` is a
  new page built from the site's instrument, not the original.
- Instrument parity: the browser port of the GRU, the bands, the annual readings and WaterSim's degree-day and energy
  balance matches Python to 9e-15 relative over 54 cases (3 stations x 2 water years x 9 what-ifs).

Numbers checked against the shipped JSONs:

- Exam medians (NSE, RMSE, peak error and bias, melt-out MAE and bias) for SnowSense and all four baselines, the win
  fractions (463, 210 and 523 of 621 stations), coverage, the region and elevation tables, the share below NSE 0.5,
  every rung's sim, dev and exam numbers, and the twin-test medians: all as stated in the card, the status doc and the
  handoff.
- The calibrated degree-day and climatology columns reproduce exactly from WaterData's per-station table
  (`shared/water/data/proc/baselines/snow_scores.parquet`), including every region and elevation median.
- The handoff's WaterSim snow check (degree-day 0.89 / 0.70, energy balance 0.93 / 0.82 calibrated / default)
  matches `shared/water/checks/snotel.json` (validation period 0.892 / 0.700 and 0.930 / 0.824).

Numbers that do not reproduce, or could not be checked:

- **"Dev calibrations put TURB at a median of 0.17, ALB_DECAY 2.4 d, GHEAT 0.32"** (status doc, handoff section 4).
  These are the prior's means in transformed space (geometric means for TURB and ALB_DECAY) over the 36 calibrations
  with NSE > 0.5 (`prior_p4.json`). The per-station medians in `twin_r6.json` (38 dev stations) are TURB 0.199,
  ALB_DECAY 1.40 d and GHEAT 0.060. The direction holds (turbulence about 5x weaker than the default 1.0, albedo ageing
  much faster than 12 d), but the GHEAT claim does not: the median is 0.06, below the default 0.173, not 0.32 above it
  (the prior's GHEAT sd is 0.39, so the calibrations disagree widely). The card's "Means" wording is correct.
- Calibration quality "median dev NSE 0.92 degree-day, 0.95 energy balance" (card): `calib_dev.json` was not shipped.
- Twin numbers for R1 (net vs twin 0.80, real 0.73, peaks 21 % below) and R3 (melt-out 10 days late): `twin_r1.json`
  and `twin_r3.json` were not shipped. Only R6's twin file is.
- numpy vs torch 1.2e-6 and JS vs numpy 0.005 mm: `export_check.json` and `js_check.json` were not shipped.
- Station counts differ between documents: 39 dev stations in the split, 38 in the gate, 36 calibrations in the prior,
  "33 with usable records" for training weather (card), and "30 development stations" in the page text
  (`webdata.py`). The 33 and 30 cannot be checked without the series.
- The ">2800 m" median NSE (0.9101399479953898) equals the Central Rockies median to 16 digits. Both groups have an odd
  count, so each median is a single station's score; it is plausibly the same high Central Rockies station. Cannot be
  checked without per-station scores.

The fair baseline that was not run: the energy balance with the dev-calibrated parameters. It needs the SNOTEL time
series, which are not on disk; it was deliberately not run here. It is the first thing to do once the data are
rebuilt, paired station by station with SnowSense. Expect the energy-balance win to shrink, possibly to a tie.

Upstream candidates for WaterSim (from the build, unchanged): EB turbulent transfer too strong at points (TURB well
below 1.0), faster albedo ageing, degree-day TT about +1 C at points, and a perennial-snow guard in the generator.
GHEAT should not be raised on the strength of the "0.32" figure (see above).
