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

## 18 Nov 2026: reproduction re-score (seen)

Approved by the owner after the import: re-run the released weights on the exam to replace the unpaired intervals
with a paired bootstrap. This is reproduction, not a rung: nothing was changed or chosen from the result, and the
exam had already been taken once (so every number below is seen).

- Data: the Caravan stream wrote all 671 CAMELS-US basins (ERA5-Land forcing + CAMELS Q) to the git-ignored data root
  `work/water` (ML_RATCHET_WATER). NWIS and NWM were not fetched. `python -I shared/water/data/build.py --only verify`:
  1,170 of 1,170 shipped files identical, splits.json sha256 33f88ca0... OK, field ed8109e5... OK. The scored basins are
  exactly splits.json camels_us exam (619; asserted in the script).
- Code: `training/rescore.py` (new) calls the shipped `realfeats.build('exam')` and `exam.main(weights/basindetective_v1.npz)`
  unchanged, in `.venv-train` (numba). Three stand-ins, all in the script:
  1. WaterData's daily GR4J simulations (`proc/baselines/sims/gr4j_gauged.parquet`) were not shipped. Re-made from the
     shipped calibrated parameters exactly as `run_gr4j.py` writes them (gr4j_snow, 1980-10-01..2023-12-30, float32),
     under WORK.
  2. NWIS peaks were not fetched. An empty peaks table stands in, so the peak-code row comes out empty and is not
     re-scored.
  3. Neither venv has scikit-learn. A Mann-Whitney AUC stands in for `roc_auc_score` (identical for a binary truth).
- It reproduces: every number in the shipped exam.json comes back to 1e-9 relative, except the two Budyko-Fu R2 values
  (0.44741085 vs 0.44741096 and 0.11001094 vs 0.11001095, from the brentq fit of w) and the NWIS row (not scored).
  The re-made GR4J simulations therefore match the ones the exam used.
- Committed outputs: `training/exam_per_basin.csv` (619 rows: predicted change and sd, observed change, GR4J
  model-implied and own-record changes, Budyko-Fu, levels, p_regulated, log storage, dor and the regulated flag, the
  GR4J misfit and damping scores, snow fraction and aridity; no time series) and `training/exam_rescore.json`
  (intervals). Intermediates are in `models/basindetective/work/` (git-ignored).
- Paired bootstrap: 10,000 resamples of the 619 basins with replacement, numpy default_rng(20261118), the same
  resamples for every row; 95 % percentile intervals of the model minus each baseline. Verdict: loss if any baseline
  wins with an interval excluding 0, win only if every baseline is beaten that way, else tie.
- Results, changes (rho, minus GR4J model-implied): mean flow -0.036 [-0.063, -0.008], **loss** (it was a tie with the
  unpaired interval [-0.081, +0.009]); runoff ratio -0.046 [-0.101, +0.008] tie; BFI -0.029 [-0.111, +0.052] tie; Q5
  +0.019 [-0.047, +0.087] tie; Q95 -0.013 [-0.045, +0.019] tie; half-flow date +0.065 [+0.005, +0.127] (a win against
  the model-implied GR4J) but -0.049 [-0.106, +0.007] against GR4J run on its own record: tie. All six median-error rows
  tie. Against no change, the median error is better on mean flow, runoff ratio, Q95 and half-flow date, and level with
  it on BFI (+0.0004 [-0.0015, +0.0020]) and Q5 (-0.003 [-0.020, +0.022]).
- Levels: wins on mean flow, runoff ratio, Q5 and Q95 (both baselines beaten; e.g. Q95 +0.0066 [+0.0015, +0.0125]
  against GR4J, where the unpaired lower bound was +0.00001); ties on BFI (own value +0.0011 [-0.0014, +0.0042]) and
  half-flow date (GR4J +0.0038 [-0.0197, +0.0259]).
- Regulation: AUC 0.485 [0.395, 0.574] (storage > 10 %), 0.532 [0.394, 0.673] (> 50 %), 0.441 [0.357, 0.527] (any).
  All three are indistinguishable from chance (AUC minus 0.5 includes 0): no skill. All three are ties with GR4J misfit
  and seasonal damping too.
- Coverage of the 68 % bars, bootstrap: 81 % [78, 84], 77 % [74, 81], 77 % [73, 80], 78 % [74, 81], 73 % [70, 77],
  83 % [80, 85]. All are too wide, as before.
- The import-time Fisher-z / Zou / Hanley-McNeil intervals stay in model.yaml's notes and the card as a cross-check.
  They were conservative as expected: every paired interval is narrower.
- Not changed: the weights, the inference code and the released numbers.

Two places outside this folder still describe the import-time verdicts. The coordinator was told; this re-score
does not edit them:
- site/src/instruments/basindetective/ui.ts says "it ties calibrated GR4J (ρ 0.76 vs 0.79)".
- site/src/lib/model-families.ts's gap line says "Ties calibrated GR4J on decade-to-decade change".

## 18 Nov 2026: NWIS peak-code row re-scored (seen)

USGS NWIS was fetched later the same day and built into the data root (`work/water/data/proc/nwis/peaks.parquet`:
48,776 annual peaks at 669 gauges with their qualification codes, from `data/scripts/build_nwis.py`). The earlier
re-score had used an empty stand-in for the peaks, so the regulation row that uses NWIS peak codes 5/6 as its label
had kept the shipped value and an unpaired interval. This is reproduction, not a rung: nothing was changed or chosen
from it.

- `training/rescore.py` now uses the data root's peaks when present (the empty stand-in only when absent). It records
  which source built the cached exam features (`WORK/feats/real_exam.peaks`) and rebuilds them when the source
  changes. Same run, same rules: splits.json sha256 33f88ca0... checked by build.py verify earlier the same day, the
  619 exam basins asserted against splits.json, released weights through the shipped `realfeats.build('exam')` and
  `exam.main`.
- It reproduces: 11 positives, AUC 0.7112739 (shipped 0.7112739), GR4J misfit 0.6075 (0.6075), seasonal damping
  0.6389055023923444 (shipped ...446, 1e-15), precision in the top 11 0.00 (0.00). The only differences left in all of
  exam.json are the two Budyko-Fu R2 values at 1e-7. Every other row's bootstrap result is identical to the earlier
  run (same seed, same resamples). `exam_per_basin.csv` gains two columns, `peak_reg_frac` and `reg_peak_code`.
- Paired bootstrap (10,000 resamples, seed 20261118): AUC 0.711 [0.543, 0.863]. Against chance: +0.211 [+0.043,
  +0.363], so it is above chance, unlike the three HydroATLAS rows. Against GR4J calibration misfit (0.608): +0.104
  [-0.106, +0.291], tie. Against seasonal damping (0.639): +0.072 [-0.062, +0.217], tie. Row verdict: **tie**
  (unchanged). With only 11 positives, the interval is wide.
- Cross-check: the unpaired Hanley-McNeil interval at import was [0.538, 0.885].

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). predict() refuses calibration flow with under 365 finite days or no positive flow. The negative bias in the direction of change (all four packaged basins decline with identical forcing) needs a matched retrain with a symmetric precipitation prior; it is queued.

## 27 Nov 2026 — R6s: symmetric precipitation-drift prior, promoted; v1.1 examined once

Owner-approved follow-up to the audit (`docs/audits/basindetective-v1.0.md`): with identical forcing in both periods
v1.0 predicted a mean-flow decline on all four packaged basins, and one basin declined in both directions.

**What changed.** The v3 bank prior on decadal precipitation drift was asymmetric: TRUE target-decade P = SEEN /
exp(delta), delta ~ N(0.029, 0.038). New physics versions v3s and v4s (`simbank.py`, `PHYS=v3s|v4s`) use
delta ~ N(0, 0.038). Nothing else changes; the draw is the same generator call, so every basin, model, forcing window
and noise draw is the same as in v3 / v4 (matched basin for basin).

**Rebuild check.** No bank was shipped. The rebuilt v4 frozen set (shards 900-903) gives R5 a sim skill of 0.540498,
the shipped 0.540498; the rebuilt dev features give R5 a dev composite of 0.451457, as shipped; the rebuilt dev twins
(`twin.py build`, PHYS=v4) give exactly R5's shipped widen-only factors. Banks built: v4 900-903, v3s 0-39, v4s 0-39
and 900-903 (21,000 basins, about 0.25 s each here, no failures).

**Rung R6s** = R5's recipe (`train.py --phys v3s --train 0-39 --extra v4s:0-39 --valphys v4s --epochs 60`, seeds 0-4,
about 45 s each), scored by `training/rung_r6s.py` (`training/rung_r6s.json`):

| | sim val-v4s | sim val-v4 | dev composite | dev ρ mean flow | dev ρ RR | dev ρ BFI | dev ρ Q5 | dev ρ Q95 | dev ρ HFD | dev AUC reg | paired dev sum Δlog Q (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R5 (v1.0) | 0.5154 | 0.5405 | 0.4515 | 0.609 | 0.253 | 0.268 | 0.586 | 0.510 | 0.350 | 0.791 | −0.149 |
| R6s | 0.5348 | 0.5221 | 0.4392 | 0.601 | 0.213 | 0.281 | 0.542 | 0.528 | 0.429 | 0.740 | −0.040 |

**Decision: kept** (sim +0.0194 on the new frozen set, needs > 0.01; dev composite −0.012, allowed −0.02). The paired
dev check (`training/symmetry.py`: forward = 1999-2008 in, 1980-89 out; reverse = 1980-89 in, 1999-2008 out; each uses
only its own input period's flow) fell from −0.149 to −0.040: closer to 0, still negative. Basins where both directions predict a decline: 16 of
51 (31 %) → 4 of 51 (8 %). Identical forcing on the four packaged basins: v1.0 −0.032, −0.130, −0.002, −0.045; v1.1 +0.006, −0.039,
+0.086, +0.015.

**Release steps before the exam (dev only).** Widen-only factors refitted with the shipped `calib_sigma.py`: d_logq
×1.39 (v1.0 1.15), d_rr ×1.20 (1.05), d_bfi ×1.38 (1.49), d_logq5 ×1.00, d_logq95 ×1.19 (1.00), d_hfd ×1.71 (1.73),
log_kslow ×1.15 (1.12); regulation temperature 1.5 (1.3). Packed as `weights/basindetective_v1_1.npz` (sha256
bc59f9d6...).

**Exam, run once** (`training/exam_v11.py`: the shipped `realfeats.build('exam')` and `exam.main` with the same
stand-ins as the 18 Nov re-score; run-once guard and `exam.started` line in `work/exam_v1.1/`). Then the 18 Nov paired
bootstrap code on its outputs (`exam_v11_rescore.json`, `exam_v11_per_basin.csv`). 619 basins:

| change | ρ v1.0 → v1.1 | GR4J ρ | verdict v1.0 → v1.1 | median error v1.0 → v1.1 | no change | verdict v1.0 → v1.1 |
|---|---|---|---|---|---|---|
| mean flow | 0.757 → 0.767 | 0.793 | loss → tie (−0.025 [−0.051, +0.001]) | 0.083 → 0.092 | 0.184 | tie → loss (vs Budyko-Fu +0.015 [+0.006, +0.022]) |
| runoff ratio | 0.453 → 0.439 | 0.499 | tie → loss (−0.059 [−0.111, −0.008]) | 0.0296 → 0.0324 | 0.0381 | tie |
| BFI | 0.203 → 0.213 | 0.232 | tie | 0.0185 → 0.0198 | 0.0181 | tie |
| Q5 | 0.440 → 0.460 | 0.421 | tie | 0.126 → 0.120 | 0.129 | tie |
| Q95 | 0.709 → 0.729 | 0.722 | tie | 0.108 → 0.117 | 0.204 | tie |
| half-flow date | 0.630 → 0.648 | 0.565 | tie | 5.97 → 6.07 d | 8.28 | tie |

68 % coverage of the change bars: 73-88 % (v1.0 73-83 %); regulation still at chance (AUC 0.483 against storage > 10 %;
100 false alarms for 5 hits). Levels: every verdict unchanged. Why the errors grew while the ranks improved: the
median predicted mean-flow change is +0.197 (v1.0 +0.133) against an observed +0.173. v1.0's asymmetric prior pulled
predictions down; the symmetric one overshoots. Nothing was chosen from this; it is recorded as the reason a
per-region, per-era drift measurement is the next rung.

**Release.** `basindetective.load()` now loads v1.1; `bd.V1_0` is the v1.0 file, kept for its exam record.
`examples/expected.json` is v1.1's output on the same four basins (v1.0's is `expected_v1.0.json`); golden output and
the site instrument regenerated. `lab/index.html` is still the v1.0 page. `model.yaml` rows are v1.1's with v1.0's
value and verdict in each note; the hidden-property rows (dev twins) are v1.0's and were not re-run.
