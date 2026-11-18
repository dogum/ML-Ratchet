# BasinDetective lab notebook

Watershed Phase 2, project 3. Built on 18 Nov 2026 in a cloud sandbox (workspace `water/detective/`, next to the
Watershed platform in `water/`), then imported into ML Ratchet the same day. Historical entries are kept as written;
add new dated entries below, do not rewrite these.

## 18 Nov 2026 — build notes (basindetective-status.md, as written)

Reads nine years of a river basin's forcing and streamflow plus the forcing of another decade. It says how six flow
signatures change in that other decade, whether a reservoir sits upstream, and four hidden properties, each with an
error bar. Trained only on WaterSim simulations and graded on 619 real CAMELS-US basins.

### Targets and why
- **Out-of-period change** (real truth: each basin's observed 1989-99 record vs its 1999-2008 record): mean flow,
  runoff ratio, BFI, Q5, Q95, half-flow date. The basin's own record already predicts the levels at rho 0.96-0.98, but
  its answer for the change is "none", and on dev basins calibrated GR4J predicts the changes at rho 0.25-0.74.
- **Regulation** (real truth: HydroATLAS storage / annual flow; USGS peak codes 5/6). No standard classical detector.
- **Hidden properties** (slow-store time constant, groundwater share, soil capacity, precipitation multiplier): no real
  truth in the data package, so graded on dev twins only.

### Recipe as built
- **Physics:** WaterSim (four runoff models, two snow routines, reservoirs), Phase 1 checks (GR4J vs airGR 1e-7 mm, HBV vs
  TUWmodel 1e-12, closure 1e-15).
- **Bank:** 10,000 basins per physics version + 1,000 frozen. Real ERA5-Land forcing 1951-1988 (donor and random CAMELS
  basins) and stochastic weather; layout spin-up 1 y / target decade 10 y / gauged 9 y; random model and snow routine;
  TRUE forcing drives the simulator, SEEN forcing goes to the network; gauge model on Q.
- **Features:** ~150; signature code is an exact port of B's (`features.sigs`, 1.3e-6).
- **Network:** MLP 2 x 256 GELU, Gaussian heads + logistic head, 5 seeds (110k weights each).
- **Dev set:** 52 dev basins, gauged decade 1999-2008 -> non-exam 1980-89 window; 260 dev twins (5 calibrated setups each).
- **Ratchet:** frozen-sim skill margin +0.01, dev composite tolerance 0.02.

### Rungs
| Rung | Change | Physics | Sim skill (own set) | Dev composite | Decision |
|---|---|---|---|---|---|
| R1 | MLP 2x256 GELU, ~150 features, bank v1 (10,000 basins) | v1 | 0.546 | 0.479 | kept: first rung |
| R1b | wider/deeper MLP 3x384 | v1 | 0.552 | 0.413 | rejected: +0.005 (need > 0.01); dev composite FAIL |
| R1e | 5-seed ensemble of R1 | v1 | 0.561 | 0.461 | kept: +0.015; dev ok |
| R2 | physics v2: dev-fitted gauge records + dev-measured precipitation bias | v2 | 0.578 | 0.456 | kept: +0.081 on val-v2 |
| R3 | physics v3: decadal drift of the forcing precipitation, delta ~ N(0.029, 0.038) | v3 | 0.551 | 0.461 | kept: +0.025 on val-v3 |
| R4 | physics v4: year-to-year P error, e_y ~ N(0, 0.10) | v4 | 0.509 | 0.461 | rejected: +0.0003 on val-v4 |
| R5 | banks v3 + v4 together (20,000 basins), 60 epochs | v4 | 0.540 | 0.451 | kept: +0.032 on val-v4 (released) |

Re-scores when the physics changed (sim skill / dev composite): val-v2: R1 0.476, R1b 0.499, R1e 0.497; val-v3: R1
0.416, R1b 0.441, R1e 0.434, R2 0.526; val-v4: R1 0.410, R1b 0.423, R1e 0.420, R2 0.494, R3 0.508.

### Twin tests and dev calibrations (what each physics rung came from)
- v1 twin test: real dev records have more flat days and fewer rising days than twins with WaterSim's default gauge
  noise. v2 `calib_gauge.py`: grid over AR(1) rho, cv and rounding; best rho 0.999, cv 0.02, 3 significant figures
  (gap score 5.84 -> 4.63). `calib_pbias.py`: Budyko-required precipitation multiplier, sim vs dev per snow group ->
  log m ~ N(-0.076 + 0.30 frac_snow, 0.116).
- v3 `calib_drift.py`: calibrated twins over-predict the 1980s flow relative to the gauges; implied spurious
  precipitation shift median 5.7 % over 19 years (78 % of basins positive), robust sd 7.6 %, scaled linearly to the
  9.5-year layout: delta ~ N(2.9 %, 3.8 %).
- v4 `calib_annual.py`: real annual flow correlates less with ERA5-Land annual P than twins do (0.855 vs 0.945):
  per-year multiplicative P error sd 0.10.
- Final twin test (`twin_final.json`, R5): change rho on twins 0.33-0.85 vs on real 0.25-0.61; hidden properties on
  twins rho 0.70-0.79; p_regulated median 0.06 on twins vs 0.21 on real dev basins (0.18 on the unregulated ones);
  largest feature gaps real - twin (standardised): q_rise -0.68, q_flat +0.50, q_annpcorr -0.33, q_fdc10 -0.32.

### Exam (619 basins, run once)
Change rho (network / GR4J model-implied): mean flow 0.76 / 0.79, runoff ratio 0.45 / 0.50, BFI 0.20 / 0.23, Q5
0.44 / 0.42, Q95 0.71 / 0.72, half-flow date 0.63 / 0.57. 68 % coverage 73-83 %, 95 % coverage 95-99 %. Levels beat
B's GR4J table. Regulation AUC 0.49 (dor > 0.1), 0.53 (> 0.5), 0.44 (any), 0.71 (NWIS peak codes); simulated 0.91.
Hidden properties on 260 dev twins: rho 0.73 (log_kslow), 0.79 (gw_share), 0.70 (log_soilcap).

### Limits
- Ties calibrated GR4J on the changes; does not beat it on rank correlation. Better MAE on runoff ratio, Q5, Q95.
- Regulation does not transfer (sim AUC 0.91, exam 0.49).
- Error bars too wide on exam (73-83 % at 68 %), from dev-window widening (widen-only).
- Drift rung measured on the 1980s and scaled linearly.
- ERA5 undercatch basins (gauged RR > 1): every method fails (mean-flow rho 0.22).

### Next rungs
1. **Realistic regulation in the bank**: fit reservoir size/position/policy to the 15 dev ResOpsUS dams and the 5
   regulated dev basins; add hydropower peaking and withdrawal-dominated operation; check with a twin of each
   regulated dev basin.
2. **Measure the forcing drift per region / era** (non-exam windows 1980-89 and 2008-15 both), instead of one linear
   scaling; this also sets the error bars better.
3. **Sequence encoder** on the daily record (small 1-D CNN) next to the features, and an ungauged mode (attributes only).

### Upstream candidates for WaterSim
- Gauge noise: default `noise.streamflow_obs` (rho 0.9, cv 0.08) makes records too jittery; dev fit rho 0.999, cv 0.02,
  3 significant figures in cfs (`calib_gauge.round_sig`).
- Forcing error model: add a decadal drift (ERA5-Land P about 6 % higher in 1980-89 relative to 1999-2008, from dev
  twins) and a 10 % year-to-year P error.
- Precipitation multiplier prior from the Budyko-required multiplier on dev (mu -0.076 + 0.30 frac_snow, sd 0.12).
  The handoff asks for this to be reconciled with FlowCast's water-yield multiplier into one documented prior.

### Sandbox gotchas
- ERA5-Land P is not stationary between decades; any change prediction must allow for it.
- `Detective.norm` on an ensemble uses member 0's normalisation (all members share it; checked on import: identical).
- Real Caravan Q is float32: features computed from float32 vs float64 differ by up to 0.03 days on half-flow date.
- Shared machine: bank shards ran at 35-68 s per 250 basins depending on load; training is ~1.5 min per seed.

## 18 Nov 2026 — rule notes from the build handoff (section 6, BasinDetective parts)
- Dev data used both to fit physics and to gate rungs (partly in-sample): BasinDetective's decadal-drift mean. The
  exam was untouched.
- Free-window forcing from exam basins was used in training banks, as the brief allows. No exam Q was used for
  training.

## 18 Nov 2026: imported into ML Ratchet

What changed on import (nothing that moves a number):
- Folder layout kept as shipped: the inference is the package `basindetective/` (`__init__.py`, `features.py`,
  `model.py`) and the weights stay in `weights/` (`basindetective_v1.npz`, `basindetective_v1_sigma.json`), so
  `files.inference` lists the package files. The shipped `README.md` is replaced by the generated one (its content is
  in `model.yaml` and this notebook).
- `training/_paths.py` added; `training/common.py` sets `D` to WORK (`models/basindetective/work` or
  `$ML_RATCHET_WORK/basindetective`), imports WaterSim and WaterData from `shared/water`, creates WORK's
  subfolders and copies the four shipped dev knob files (gauge_v2, pbias_v2, drift_v3, annual_v4) into
  `WORK/models/` when absent. Every sandbox path (under the old `water/` root) became `_paths.DEV_PARAMS` (`shared/water/checks/
  dev_calibrated_params.json`), `WATER_PROC` (`$ML_RATCHET_WATER/data/proc`) or `D`. `ratchet.json`'s model paths are
  now relative to D, and `ratchet.py` resolves them there. `release_examples.py` imports the package from the model
  folder; `buildpage.py` reads the template and scripts from `training/`. `pipeline.py`'s optional CAMELS-id mode
  imports waterdata from `shared/water` (the CSV mode and the package need numpy, scipy and pandas only).
- Added: `training/import_check.py` (re-derives baselines, computes the intervals; writes nothing),
  `web/export_instrument.py` and `web/build_lab.mjs` (the live instrument and `lab/index.html`), `examples/README.md`.
- No training script was run. The demo output and `examples/expected.json` agree exactly (max difference 0).

What reproduced (training/import_check.py, from WaterData's shipped `signatures.parquet`):
- All six "GR4J vs own record" rhos and median errors, all six no-change median errors, the own-value and GR4J-sim
  level rhos, and the four examples' observed changes: exact.
- Every number in the status doc, the card and the handoff section 5 matches `exam.json`, `ratchet.json`,
  `twin_final.json` and the calibration JSONs, with the exceptions below.

What did not reproduce, or is labelled wrong:
- "MAE" everywhere is the **median** absolute error (`exam.py` and `evaluate.py` use `np.median`). Renamed in the card.
- The dev composite and every dev statistic are over **51** dev basins (one has no usable 1980s record); the rule text,
  `calib_sigma.py`'s "dev real (52)" and the status doc say 52.
- `calib_drift.py`'s docstring: "twins say +21 % more flow in 1980-89 (GR4J +17 %)". `calib_drift.json`'s rows give a
  median twin log ratio of 0.153 (+16.5 %; GR4J setups 0.137, +14.7 %), and the gauges +0.057 (+5.9 %, the "6 %"). The
  21 % / 17 % may come from the v1 twin run, whose log was not shipped. "ERA5-Land has 8.6 % more precipitation in the
  1980s" is in no shipped file.
- `calib_gauge.py`'s docstring: "+1.4 IQR flat days, -1.2 IQR rising days". `calib_gauge.json` has no such numbers; its
  default-noise row (rho 0.9, cv 0.08) shows q_flat -0.62 and q_rise +1.29 IQR (twin minus real).
- `calib_annual.py` as shipped writes the best grid value (s = 0.12) to annual_v4.json; the shipped file holds 0.101,
  the linear interpolation to the real median between the 0.08 and 0.12 grid points (`interpolated_sd` in
  calib_annual.json). The interpolation step is not in the script.
- The card's "HBV vs TUWmodel <= 1e-12 mm" is for the snow and soil states only; runoff differs by design (NSE 0.999,
  `shared/water/checks/reference_models.json`).
- Not checkable without the Caravan time series or unshipped logs: numpy == torch < 6e-6, the shipped page's JS ==
  Python 2e-7 sigma, the signature port's 1.3e-6, the dev GR4J change rhos quoted in the card (0.28 / 0.25 / 0.31 /
  0.38 / 0.74 / 0.63).
- Claims softened on import: "better MAE on runoff ratio, Q5, Q95" are gaps of 11 %, 23 % and 2 % in medians with no interval; on Q5
  and BFI the network does not beat "no change". "Levels clearly better than B's table (BFI 0.98 vs 0.79, Q5 0.975 vs
  0.78)" is true against GR4J only: the basin's own 1999-2008 value scores 0.978 and 0.971.
- Ratchet creep: each gate compares with the incumbent only, so R5's dev composite (0.451) is 0.027 below R1's (0.479).

Honest reporting (model.yaml): intervals as the import brief prescribes. Rho: Fisher z; differences: Zou's interval
for two independent correlations (conservative, since the rhos are paired over the same 619 basins); AUC:
Hanley-McNeil (approximate, not in the brief's list, labelled); coverage: exact binomial; median errors: no interval.
Verdicts: six change rhos tie (best baseline: calibrated GR4J, or GR4J vs own record on BFI and half-flow date), six
median-error rows tie, levels five ties and one narrow win (Q95: lower bound +0.00001), regulation four ties (three at
chance). No NWM, regionalisation or dev-tuned baseline exists for the change task in the shipped files.

The live instrument (site/src/instruments/basindetective/, lab/index.html):
- The whole of `features.py` is ported to TypeScript (`features.ts`), not only the target-decade climate part the
  shipped page recomputed, so the network re-runs on perturbed flow as well as perturbed weather. Parity: 36 reference
  cases (4 basins × 9 what-ifs) on features and outputs, worst relative difference 3.5e-15; tolerance set to 1e-6.
- Weights: the matrices go to float16 (1.1 MB pack), normalisation and scales stay float32; float16 moves the
  outputs by at most 3.7e-4 relative (in sigma units 1.8e-4).
- The examples are stored as int32 hundredths of the CSV values, so the browser starts from the same doubles Python
  parses. The 1989-99 daily flow is not in the release, so the scope shows its observed levels (gauged value × observed
  change), not the hydrograph.
- First finding from the what-ifs: a 60-day linear reservoir on the gauged flow (storage 0.16 of the annual flow,
  "regulated" by the exam's definition) lowers p_regulated on three of the four basins (Guadalupe 0.89 -> 0.38, Revais
  Creek 0.41 -> 0.17, Umatilla 0.13 -> 0.11) and raises it on one (Tiffin 0.03 -> 0.13). The network's notion of a reservoir is WaterSim's
  operated reservoirs, not a smooth store.
