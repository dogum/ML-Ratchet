# BasinDetective v1.0: model card

Imported into ML Ratchet on 18 Nov 2026 from the shipped `basindetective-v1.0` release. The text below is the
shipped card with paths fixed, the "MAE" columns renamed to what they are (median absolute error), and four sections
added: the weather-data caveat, the exam with intervals and verdicts, rule notes, and the import check. Nothing was
re-trained; weights and inference numbers are the release's.

## The weather-data caveat (read first)
The forcing is ERA5-Land (via Caravan). It under-catches mountain and Pacific Northwest precipitation: 50 of the 671
CAMELS-US basins have a 1989-99 runoff ratio above 1 (44 in 1999-2008; 42 of the 619 exam basins in the gauged
decade), which is impossible with correct precipitation. It also drifts between decades: on the dev basins the 1980s
carry about 6 % more precipitation relative to 1999-2008 than the gauges support (calib_drift.json: median implied
shift 5.7 % over 19 years). This caps every method, the network and calibrated GR4J alike, and literature numbers from
Daymet- or NLDAS-forced LSTMs (NSE about 0.69-0.76) are not comparable with anything here.

## What it does
Reads a river basin's daily forcing (precipitation P, mean temperature, FAO-56 potential evaporation) and streamflow for
a gauged period of about nine years (trained on 9 water years), plus the forcing of another decade (trained on 10 water
years, immediately before the gauged one). Outputs, each with a 1-sigma error bar (5-member ensemble, Gaussian heads):

- **Out-of-period response (graded on real truth).** The change from the gauged decade to the other decade of: mean flow
  (log ratio), runoff ratio, baseflow index (Lyne-Hollick), low flow Q5 (log ratio), high flow Q95 (log ratio) and the
  half-flow date (days). Adding the change to the gauged decade's own value gives the other decade's signature.
- **Regulation (graded on real truth).** Probability that upstream reservoir storage exceeds 10 % of the mean annual flow,
  and log10 of that storage ratio.
- **Hidden properties (no real truth; checked on dev twins).** Slow-store (groundwater) recession time constant, the
  slow-store share of flow, the soil water capacity, and the ratio of true to forcing precipitation in the gauged decade.

Protocol: **gauged** (it reads the basin's own calibration-period Q), so its classical comparison is per-basin calibrated
GR4J. Inputs on the exam: 1999-10..2008-09 forcing + Caravan Q, and 1989-10..1999-09 forcing.

## Why these targets
B's Phase 1 table shows the basin's own calibration-period record already predicts validation-period runoff ratio, BFI,
recession K, Q5 and Q95 at Spearman 0.96-0.98, so re-predicting those levels is not informative. The **change** between
decades is different: the own-record answer is "no change" (zero skill by construction), and on the dev basins
(1999-2008 -> 1980-89) the classical answers are weak: calibrated GR4J predicts the change of runoff ratio at rho 0.28,
BFI 0.25, Q5 0.31, half-flow date 0.38 (mean flow 0.74, Q95 0.63). Real truth exists for every exam basin (its observed
1989-99 record). Regulation has real truth (HydroATLAS / GRanD storage, B's `dor_frac`; USGS peak codes 5/6) and no
standard classical detector. Recession and storage constants, groundwater share and soil capacity have no real truth for
CAMELS-US in this data package, so they are reported on simulated twins only.

## How it was made (the ML Ratchet recipe)
1. **Physics.** WaterSim (Phase 1): HBV, GR4J, SAC-SMA, VIC-lite; degree-day and energy-balance snow on elevation bands;
   UH routing; reservoirs (SOP / rule curve / Hanasaki) and irrigation. Verified in `shared/water/checks/` (GR4J vs airGR 1.0e-7 mm,
   HBV snow and soil states vs TUWmodel <= 1e-12 mm (runoff routing differs by design, NSE 0.999), water balance
   1.4e-15 relative on 121 runs, analytic recession / UH / Muskingum checks).
2. **Simulated data in real conditions.** 10,000 basins per physics version (bank shards 0-39) + 1,000 frozen validation
   basins (shards 900-903). Forcing: 40 % camels prior (dev-basin attributes, jittered) with the donor's own ERA5-Land
   forcing from 1951-1988; 35 % default-prior basins on a random CAMELS basin's 1951-1988 forcing (attributes reset to that
   forcing, parameters regenerated); 25 % stochastic weather. Layout: 1 spin-up year, a 10-year target decade (forcing only)
   and 9 gauged years. Random model (4) and snow routine (2) per basin; target-decade climate perturbed in half the basins
   (P x U(0.88, 1.12), T + U(-1, 1) C). The simulator is driven by TRUE forcing; the network reads the SEEN forcing
   (persistent multiplier, daily error cv 0.25, snow undercatch, T offset, and from v3/v4 decadal drift and annual error).
   Streamflow goes through a gauge model (rating errors and revisions, 3 significant figures in cfs from v2, gaps).
   Labels use an exact port of B's `waterdata.signatures` (`features.sigs`, max relative difference 1.3e-6, only in the
   unused elasticity).
3. **Network.** ~150 engineered features (climate of both decades and their differences; gauged signatures, FDC, monthly
   regime, recession, baseflow filters, rainfall-flow lag correlations, autocorrelation, seasonal damping). MLP 2 x 256
   GELU (dropout 0.1, 110k weights) with Gaussian NLL heads for 11 regression outputs and a logistic head; five seeds
   averaged (variance = mean member variance + spread). numpy forward == torch (float32) to < 6e-6; JS port == Python to 2e-7 sigma
   (90 checks).
4. **Ratchet.** Keep if frozen-sim skill (mean of max(R2, 0) over 11 outputs and 2 AUC - 1) beats the incumbent on the
   current physics' frozen set by > 0.01 AND the real dev composite (mean Spearman of the 6 changes on the 52 dev basins,
   1999-2008 -> 1980-89, plus 2 AUC - 1 for regulation) drops by <= 0.02. New frozen set per physics version; all rungs
   re-scored.
5. **Calibrated on dev only.** Twin test: each dev basin re-simulated with its five dev-calibrated model setups
   (`shared/water/checks/dev_calibrated_params.json`), real forcing and gauge noise, then read by the network next to the real record.
   - v2 `calib_gauge.py`: real records have more flat days (+1.4 IQR) and fewer rising days (-1.2 IQR) than twins with
     watersim's default gauge noise. Grid fit: AR(1) rho 0.999, cv 0.02, 3 significant figures (gap score 5.84 -> 4.63).
     `calib_pbias.py`: Budyko-required precipitation multiplier, sim vs dev, per snow group -> log m ~ N(-0.076 + 0.30
     frac_snow, 0.116).
   - v3 `calib_drift.py`: twins say the dev basins carried 21 % more flow in 1980-89 than in 1999-2008 (GR4J: 17 %); the
     gauges say 6 %. ERA5-Land has 8.6 % more precipitation in the 1980s. Implied spurious precipitation shift: median
     5.7 % over 19 years (78 % of basins positive), robust sd 7.6 %; scaled linearly to the 9.5-year exam separation:
     delta ~ N(2.9 %, 3.8 %) on the target decade.
   - v4 `calib_annual.py`: real annual flow correlates less with ERA5-Land annual P than twins (0.855 vs 0.945): per-year
     multiplicative P error sd 0.10.
6. **Error bars.** Widen-only factor per output: changes fitted on the 52 real dev basins, hidden properties on the 260
   dev twins; regulation temperature on dev (5 positives).

## Ratchet (every rung; `ratchet.json`)
| Rung | Change | Physics | Sim skill (own frozen set) | Dev composite | Decision |
|---|---|---|---|---|---|
| R1 | MLP 2x256 GELU (dropout 0.1), Gaussian heads + regulated logit; ~150 features; bank v1 shards 0-39 (10,000 simulated basins) | v1 | 0.546 | 0.479 | kept: first rung |
| R1b | wider/deeper MLP 3x384 (dropout 0.15), same bank v1 0-39 | v1 | 0.552 | 0.413 | rejected: sim skill 0.5516 vs incumbent R1 0.5464 on val-v1 (delta +0.0052, need > 0.01); dev composite 0.4133 vs 0.4786 (FAIL) |
| R1e | 5-seed ensemble of the R1 configuration (mean of means; variance = mean variance + spread of means) | v1 | 0.561 | 0.461 | kept: sim skill 0.5612 vs incumbent R1 0.5464 on val-v1 (delta +0.0147, need > 0.01); dev composite 0.4612 vs 0.4786 (ok) |
| R2 | physics v2: dev-fitted gauge records (AR(1) rho 0.999, cv 0.02, 3 significant figures in cfs; calib_gauge.py) + dev-measured precipitation bias log m ~ N(-0.076 + 0.30 fsnow, 0.116) (calib_pbias.py); 5-seed ensemble, bank v2 0-39 | v2 | 0.578 | 0.456 | kept: sim skill 0.5776 vs incumbent R1e 0.4966 on val-v2 (delta +0.0810, need > 0.01); dev composite 0.4564 vs 0.4612 (ok) |
| R3 | physics v3: decadal drift of the forcing precipitation measured by dev twins (TRUE target-decade P = SEEN / exp(delta), delta ~ N(0.029, 0.038); calib_drift.py); 5-seed ensemble, bank v3 0-39 | v3 | 0.551 | 0.461 | kept: sim skill 0.5513 vs incumbent R2 0.5263 on val-v3 (delta +0.0251, need > 0.01); dev composite 0.4613 vs 0.4564 (ok) |
| R4 | physics v4: year-to-year forcing precipitation error, TRUE P x exp(e_y), e_y ~ N(0, 0.10) per water year, fitted on dev twins to the real annual flow-vs-P correlation (calib_annual.py); 5-seed ensemble, bank v4 0-39 | v4 | 0.509 | 0.461 | rejected: sim skill 0.5085 vs incumbent R3 0.5082 on val-v4 (delta +0.0003, need > 0.01); dev composite 0.4612 vs 0.4613 (ok) |
| R5 | more data: banks v3 + v4 together (20,000 simulated basins), 60 epochs; 5-seed ensemble | v4 | 0.540 | 0.451 | kept: sim skill 0.5405 vs incumbent R3 0.5082 on val-v4 (delta +0.0323, need > 0.01); dev composite 0.4515 vs 0.4613 (ok) |

Re-scores when the physics changed (sim skill / dev composite):

- val-v2: R1 0.476 / 0.479; R1b 0.499 / 0.413; R1e 0.497 / 0.461
- val-v3: R1 0.416 / 0.479; R1b 0.441 / 0.413; R1e 0.434 / 0.461; R2 0.526 / 0.456
- val-v4: R1 0.410 / 0.479; R1b 0.423 / 0.413; R1e 0.420 / 0.461; R2 0.494 / 0.456; R3 0.508 / 0.461


## Exam: out-of-period change (619 CAMELS-US exam basins, 1999-2008 -> 1989-99, gauged)

| Change of | BasinDetective rho | median abs err | 68 % cover | 95 % cover | GR4J model-implied rho / med. abs err | GR4J vs own record rho / med. abs err | Budyko-Fu rho / med. abs err | no-change med. abs err |
|---|---|---|---|---|---|---|---|---|
| mean flow (log ratio) | **0.76** | 0.083 | 81 % | 97 % | 0.79 / 0.086 | 0.70 / 0.111 | 0.77 / 0.077 | 0.184 |
| runoff ratio | **0.45** | 0.030 | 77 % | 96 % | 0.50 / 0.033 | 0.33 / 0.043 | 0.47 / 0.031 | 0.038 |
| baseflow index (LH) | **0.20** | 0.018 | 77 % | 96 % | 0.23 / 0.021 | 0.25 / 0.065 | – | 0.018 |
| low flow Q5 (log ratio) | **0.44** | 0.126 | 78 % | 95 % | 0.42 / 0.163 | -0.04 / 0.634 | – | 0.129 |
| high flow Q95 (log ratio) | **0.71** | 0.108 | 73 % | 96 % | 0.72 / 0.111 | 0.67 / 0.120 | – | 0.204 |
| half-flow date (days) | **0.63** | 6.0 | 83 % | 99 % | 0.57 / 5.8 | 0.68 / 6.9 | – | 8.3 |


The dumb baseline (no change) is the last column (median absolute error only; rank correlation undefined). Every
"MAE" in the shipped files is a median of |error| (`training/exam.py`, `evaluate.py`: `np.median`).

## Exam with intervals and verdicts (added on import)

The release ships medians and rank correlations, not per-basin outputs, and the real time series are not in the
repository, so no paired bootstrap is possible yet. Methods, per row: Spearman rho with a Fisher-z 95 % interval; the
difference to a baseline with Zou's (2007) interval for two **independent** correlations built from the Fisher-z
intervals (conservative: both are on the same 619 basins, so the paired interval is narrower); AUC with a Hanley-McNeil
interval from the AUC and class counts (approximate, unpaired); coverage with an exact binomial interval; median
absolute errors with no interval ("per-unit scores not shipped; needs the data rebuild, `mlratchet data water`"), a
verdict from the median only when the gap is large and its direction certain. `win` only if the model beats every
baseline on the row. Computed by `training/import_check.py`.

| Change of | BasinDetective rho [95 %] | Best baseline (rho) | Difference [95 %, conservative] | Verdict | Median abs err: net / best baseline / no change | Verdict |
|---|---|---|---|---|---|---|
| mean flow | 0.757 [0.721, 0.789] | GR4J model-implied 0.793 | -0.036 [-0.081, +0.009] | tie | 0.083 / Budyko-Fu 0.077 / 0.184 | tie |
| runoff ratio | 0.453 [0.388, 0.513] | GR4J model-implied 0.499 | -0.046 [-0.133, +0.040] | tie | 0.0296 / Budyko-Fu 0.0313 / 0.0381 | tie |
| baseflow index | 0.203 [0.126, 0.277] | GR4J vs own record 0.251 | -0.048 [-0.154, +0.058] | tie | 0.0185 / no change 0.0181 / 0.0181 | tie |
| low flow Q5 | 0.440 [0.374, 0.502] | GR4J model-implied 0.421 | +0.019 [-0.072, +0.110] | tie | 0.126 / no change 0.129 / 0.129 | tie |
| high flow Q95 | 0.709 [0.668, 0.746] | GR4J model-implied 0.722 | -0.013 [-0.068, +0.042] | tie | 0.108 / GR4J 0.111 / 0.204 | tie |
| half-flow date | 0.630 [0.580, 0.676] | GR4J vs own record 0.679 | -0.049 [-0.113, +0.015] | tie | 6.0 / GR4J 5.8 / 8.3 d | tie |

Coverage of the 68 % bars, exact binomial 95 % intervals: mean flow 81 % [78, 84], runoff ratio 77 % [74, 81], BFI
77 % [73, 80], Q5 78 % [74, 81], Q95 73 % [70, 77], half-flow date 83 % [79, 86]. All lie above 68 %: the bars are too
wide. The 95 % bars cover 95-99 %.

Levels (rho; difference to the best baseline): mean flow 0.982 vs GR4J 0.978, +0.004 [-0.001, +0.008], tie; runoff
ratio 0.976 vs own value 0.972, +0.004 [-0.002, +0.010], tie; BFI 0.979 vs own value 0.978, +0.001 [-0.004, +0.006],
tie; Q5 0.975 vs own value 0.971, +0.005 [-0.001, +0.011], tie; Q95 0.973 vs GR4J 0.966, +0.0066 [+0.00001, +0.0135],
**win** at the edge (also beats the own value, +0.010 [+0.003, +0.017]); half-flow date 0.913 vs GR4J 0.909, +0.004
[-0.015, +0.023], tie. WaterData's ungauged kNN on attributes scores 0.55-0.80 on these levels.

Regulation (AUC, Hanley-McNeil 95 %): storage > 10 % 0.485 [0.390, 0.580] vs GR4J misfit 0.433 [0.342, 0.525] and
seasonal damping 0.469 [0.375, 0.563], tie, at chance; > 50 % 0.532 [0.386, 0.678], tie; any reservoir 0.441 [0.363,
0.519], tie; USGS peak codes 0.711 [0.538, 0.885] vs 0.608 [0.429, 0.786] and 0.639 [0.460, 0.817], tie (above
chance, 11 positives). Log storage among the 53 regulated basins: rho 0.236 [-0.037, 0.476].

No baseline exists in the shipped files for the change task from NOAA's National Water Model, from regionalisation
(U-dev, U-k12) or from dev-tuned rules: the fairest available is the per-basin calibrated GR4J, which reads the same
1999-2008 flow (same gauged protocol).

## Exam: validation-decade signature levels (cal value + predicted change), compared with B's table

| Validation-decade level | BasinDetective rho / R2 | own calibration value rho / R2 | GR4J sim rho / R2 |
|---|---|---|---|
| q_mean | 0.982 / 0.986 | 0.971 / 0.961 | 0.978 / 0.982 |
| runoff_ratio | 0.976 / 0.971 | 0.972 / 0.961 | 0.965 / 0.953 |
| bfi_lh | 0.979 / 0.960 | 0.978 / 0.958 | 0.787 / 0.523 |
| q5 | 0.975 / 0.965 | 0.971 / 0.944 | 0.778 / 0.477 |
| q95 | 0.973 / 0.971 | 0.963 / 0.946 | 0.966 / 0.974 |
| hfd_mean | 0.913 / 0.880 | 0.870 / 0.774 | 0.909 / 0.839 |


## Exam: regulation

| Regulation truth | positives | BasinDetective AUC | GR4J misfit AUC | seasonal-damping AUC | precision in top n |
|---|---|---|---|---|---|
| dor>0.1 (mid+high) | 37 | **0.49** | 0.43 | 0.47 | 0.05 |
| dor>0.5 (high) | 16 | **0.53** | 0.43 | 0.48 | 0.00 |
| dor>0 (any) | 53 | **0.44** | 0.39 | 0.51 | 0.08 |
| NWIS peak code 5/6 in >=1/3 of WY1990-2008 peaks | 11 | **0.71** | 0.61 | 0.64 | 0.00 |

log10(storage / annual flow) among the 53 exam basins with any upstream reservoir: rho 0.24, median |error| 0.48 (log10), 68 % coverage 28 %. Confusion at p > 0.5: {'tp': 5, 'fp': 92, 'fn': 32, 'tn': 490}.


## Exam by stratum (Spearman rho of the change, network / GR4J model-implied)

| Stratum | n | mean flow (log ratio) | runoff ratio | baseflow index (LH) | low flow Q5 (log ratio) | high flow Q95 (log ratio) | half-flow date (days) |
|---|---|---|---|---|---|---|---|
| snow (frac_snow>=0.15) | 210 | 0.84 / 0.82 | 0.50 / 0.44 | 0.35 / 0.31 | 0.41 / 0.46 | 0.80 / 0.80 | 0.78 / 0.68 |
| rain (frac_snow<0.15) | 409 | 0.73 / 0.79 | 0.43 / 0.53 | 0.13 / 0.19 | 0.47 / 0.40 | 0.66 / 0.68 | 0.57 / 0.53 |
| arid (PET/P>1) | 214 | 0.67 / 0.71 | 0.43 / 0.49 | 0.21 / 0.11 | 0.38 / 0.23 | 0.66 / 0.58 | 0.74 / 0.44 |
| humid (PET/P<=1) | 405 | 0.71 / 0.79 | 0.46 / 0.49 | 0.23 / 0.35 | 0.50 / 0.51 | 0.61 / 0.67 | 0.61 / 0.66 |
| regulated (dor>0.1) | 37 | 0.68 / 0.74 | 0.37 / 0.58 | 0.21 / 0.22 | 0.32 / 0.42 | 0.58 / 0.74 | 0.71 / 0.64 |
| runoff ratio > 1 in cal | 42 | 0.22 / 0.21 | 0.11 / 0.18 | 0.24 / 0.39 | -0.06 / 0.08 | 0.24 / 0.37 | 0.90 / 0.90 |


## Hidden properties: parameter recovery on dev twins

| Hidden property (dev twins, 260) | Spearman rho | R2 | median abs err | bias | 68 % cover |
|---|---|---|---|---|---|
| log_kslow | 0.73 | 0.42 | 0.903 | 0.148 | 68 % |
| gw_share | 0.79 | 0.60 | 0.124 | -0.024 | 69 % |
| log_soilcap | 0.70 | 0.51 | 0.387 | -0.145 | 71 % |
| log_pmult | – | – | 0.054 | -0.011 | 79 % |

Sigma factors (widen-only, dev): d_logq 1.15, d_rr 1.05, d_bfi 1.49, d_logq5 1.00, d_logq95 1.00, d_hfd 1.73, log_kslow 1.12, gw_share 1.00, log_soilcap 1.00, log_pmult 1.00, log_dor 1.00; regulation temperature 1.30.

Plausibility of hidden outputs on exam (no truth): log_pmult vs calibration runoff ratio (rho): 0.65; median predicted P multiplier, cal RR>1 basins: 1.16; median predicted P multiplier, all: 1.04; log_kslow vs observed recession K (rho): 0.28; gw_share vs observed BFI (rho): 0.80


Frozen simulated set v4 (release model): R2 per output d_logq 0.59, d_rr 0.30, d_bfi 0.38, d_logq5 0.40, d_logq95 0.56, d_hfd 0.69, log_kslow 0.51, gw_share 0.68, log_soilcap 0.51, log_pmult 0.58, log_dor 0.46; regulated AUC 0.91; skill 0.540.

## Rule notes (from the build handoff, section 6, the parts about BasinDetective)
- Dev data were used both to fit physics and to gate rungs (partly in-sample): BasinDetective's decadal-drift mean
  (physics v3) was fitted on the dev twins and the rungs were then gated on the same dev basins. The gauge model (v2),
  the precipitation prior (v2) and the annual error (v4) were also fitted on dev. The exam was untouched in every case.
- Free-window forcing from exam basins (ERA5-Land 1951-1988, before any exam window) was used in the training banks,
  as the brief allows. No exam Q was used for training.
- Found on import: the dev gate is scored on 51 dev basins (one has no usable 1980s record), not the 52 the rule text
  and the sigma JSON state. The gate compares each rung only with the current incumbent, so the released R5's dev
  composite (0.451) ended 0.027 below the first rung's (0.479), more than one gate tolerance (0.02).

## Import check (18 Nov 2026)
- Every exam number in this card matches `training/exam.json`. The GR4J-vs-own-record and no-change numbers, the level
  baselines and the four examples' observed changes re-derive exactly from WaterData's `signatures.parquet`
  (`training/import_check.py`). The network's own exam numbers cannot be re-derived without the Caravan time series.
- The demo reproduces `examples/expected.json` exactly (max difference 0). The site instrument's TypeScript port of the
  150 features and the ensemble equals the Python package to 3.5e-15 on 36 cases (4 basins × 9 what-ifs); float16
  weights move the outputs by at most 3.7e-4 relative.
- Not reproducible from the shipped files (they need the Caravan data or logs that were not shipped): numpy == torch
  < 6e-6, JS == Python 2e-7 sigma (90 checks), the 1.3e-6 signature-port check, the dev GR4J change rhos (0.28, 0.25,
  0.31, 0.38, 0.74, 0.63), "twins say 21 % more flow in 1980-89 (GR4J 17 %)" (calib_drift.json's rows give a median
  twin log ratio of 0.153, +16.5 %, and 0.137 for the GR4J setups) and "ERA5-Land has 8.6 % more precipitation in the
  1980s". The "+1.4 IQR flat days, -1.2 IQR rising days" of calib_gauge.py's docstring is not in calib_gauge.json (the
  default-noise row there shows q_flat -0.62 and q_rise +1.29 IQR, twin minus real).

## Limits
- On the decade-to-decade changes it **ties** calibrated GR4J (rank correlation within 0.05 everywhere; 0.44 vs 0.42 on low
  flow is a tie too); it does not beat it. Its median error on BFI and Q5 changes is no better than "no change". It needs no per-basin calibration and its error bars hold (too wide on exam: 73-83 % inside the
  68 % bars, 95-99 % inside 95 %).
- **Regulation does not transfer**: sim AUC 0.91, exam AUC 0.49 vs HydroATLAS storage (37 positives), 0.71 vs USGS peak codes
  5/6 (11 positives). log10 storage among regulated basins: rho 0.24, 68 % coverage 28 % (sigma not calibrated: no dev truth).
  Simulated reservoirs are generic policies; CAMELS is "minimally impacted"; HydroATLAS upstream volumes can be wrong where the
  pour point snaps to a larger river.
- The drift (v3) was measured on 1980-89 and scaled to 1989-99 with a linear-drift assumption.
- Basins where ERA5-Land undercatches (gauged runoff ratio > 1, 42 exam basins): mean-flow change rho 0.22 (GR4J 0.21).
- Error-bar factors were fitted on the dev 1980s window, which had more drift than the exam decade; widen-only keeps them wide.
- Hidden properties: no real truth; "slow store" differs between model structures (HBV lower box, GR4J routing store
  linearised, SAC-SMA lower primary, VIC-lite ARNO baseflow). P-multiplier twins all have m = 1, so only error, not ranking.
- Gauged mode only; trained on 9 + 10 water-year windows of daily data; CAMELS-US / ERA5-Land only.

## Data and licences
None of these sources is share-alike, so the weights stay MIT like the rest of ML Ratchet.
Caravan v1.6 (Kratzert et al. 2023, Sci. Data 10, 61; CC BY 4.0) with CAMELS-US streamflow (Newman et al. 2015; Addor et al.
2017) and ERA5-Land forcing (Munoz-Sabater et al. 2021; Copernicus licence); HydroATLAS (Linke et al. 2019; CC BY 4.0) with
GRanD volumes (Lehner et al. 2011); USGS NWIS annual peaks (public domain). Baselines from WaterData (Phase 1 agent B);
simulator WaterSim (Phase 1 agent A).
