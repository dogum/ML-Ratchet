> **Lab notebook.** FaultFinder was built in a claude.ai cloud session (16-17 Nov 2026) on the shared BuildingSim +
> BuildingData platform (to land in `shared/building/`). The first entry is that session's status document as
> written; its paths (`bldg/faultfinder/`, `release/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 17 Nov 2026 — build notes (faultfinder-status.md, as written)

### FaultFinder (BuildingSim Phase 2) — status

**What it is.** Air-handler / rooftop-unit fault detection and diagnosis from one day of BMS trend log (OA/RA/MA/SA temperatures, setpoint, damper and valve commands, fan speed/power, occupancy). Outputs P(fault), a probability over 18 fault classes (BuildingSim's 20 fault types mapped onto the 27-type exam taxonomy; 13 exam types are not simulable and are graded for detection only), a severity and an energy-waste estimate. Trained only on BuildingSim simulations.

- Page: the original page, now `lab/index.html` (real-day explorer with rule vs FaultFinder verdicts, AHU schematic, fault-injection lab, network running in the browser; JS == numpy to 5e-6 on all 217 exam days)
- Release: faultfinder-v1.0.zip (numpy-only `faultfinder.py`, `pipeline.py`, weights, `demo.py` on 4 real exam days drawn with a fixed seed, MODEL_CARD.md, training/)
- Working folder: the sandbox workspace bldg/faultfinder/ (STATUS.md, ratchet.json, exam/exam.json)

### Recipe as built
- **Physics:** BuildingSim (Phase 1 checks: RC step 0.003 K, 31/36 BESTEST inside range, psychrometrics 0.02 K vs CoolProp, 26/26 fault signatures in the expected direction).
- **Bank:** random RTU / VAV buildings x 17 TMYx cities x any season; 12-day windows simulated healthy and faulted (fault from day 5). Two banks of 38,987 days (v1 generic, v3 dev-calibrated).
- **Real noise:** residual library from real fault-free dev days, static offsets, fan-off sensor drift, gaps, per-source channel patterns; Korean office AHUs (unlabelled) for start/stop irregularity and SAT tracking tail.
- **Dev calibration (20 dev unit-days only):** twin tests found (1) BMS logs commands not positions (FLEXLAB valve stuck at 100 % logs 0 %) -> `log_commands`; (2) real damper at 100 % command mixes only 81-90 % outdoor air, Nesbitt 15 % mixes 14-28 % -> installed damper characteristic; (3) sensors disagree by 1-1.5 K on normal days -> larger static offsets. Plus operator jitter, schedule-shift faults, 24x7 units. Calibration temperature T = 4.3 (widen-only) on dev.
- **Network:** 1-D CNN, 96 x 15-min bins, 35 feature rows, 161k weights.

### Ratchet (decided on frozen sim val v4; keep if skill > incumbent + 0.01 and dev AUC within 0.05)
| Rung | skill v4 | dev AUC | decision |
|---|---|---|---|
| r1 Gaussian noise | 0.415 | 0.60 (dev FPR 0.91) | kept |
| r2 real noise | 0.454 | 0.74 (FPR 0.45) | kept |
| r3 calibrated bank v3 | 0.500 | 0.71 | kept |
| r4 damper characteristic + offsets | 0.511 | 0.73 (FPR 0.36) | kept |
| r5 both banks | 0.518 | 0.74 | rejected (+0.007) |
| r6 wider net, 10 epochs | **0.543** | 0.71 | **kept, final** |

### Exam (217 real days, run once)
| | FaultFinder | APAR rules | majority |
|---|---|---|---|
| AHU AUC (200 days, 91 faulty) | **0.78** [0.72-0.83] | 0.69 | - |
| AHU balanced acc. | 0.70 | 0.65 | 0.50 |
| AHU recall at 2 / 5 / 10 % false alarms | 23 / 27 / 47 % | 32 / 41 / 45 % | - |
| AHU FPR at 0.5 | 33 % | 2 % | 0 |
| RTU AUC (17 days) | 0.65 | 0.72 | bal. acc. 0.50 |
| **Nesbitt Hall (real building, 76 days) AUC** | **0.78** | 0.54 | - |
| FLEXLAB / Iowa ERS AUC | 0.93 / 0.73 | 0.67 / 0.73 | - |
| Diagnosis top-1 / top-3 (66 simulable faulty days) | 20 % / 48 % | - | 21 % top-1 |
| ECE (5 bins) all / AHU | 0.12 / 0.13 | | |

Calibration: 164/217 days in P 0.4-0.6 (temperature from 20 days over-shrinks; Brier 0.216 vs 0.204 uncalibrated). Energy-waste head: Spearman 0.19 on 46 ERS paired days (not useful).

### Limits
1. Diagnosis no better than the majority guess at top-1; cc valve stuck (50 % top-1) and zone sensor (100 %) work, OA damper stuck does not (0 %).
2. Many false alarms at P = 0.5; rules win at strict alarm rates.
3. RTU worse than the one-line rule; one day at a time, single-zone simulator, no exhaust damper / return fan / duct model.

### Next ideas
1. Multi-day context: compare a day with the unit's own recent days (most real false alarms are unit-specific habits such as 24x7 operation with the chiller off at night).
2. Simulate the missing plant faults (chiller off / CHW sensor) and exhaust/return-fan dampers in BuildingSim, so 13 more exam types become diagnosable.
3. Calibrate with a 2-parameter (Platt) map or a larger dev set; the 20-day temperature over-shrinks.

### Sandbox gotchas
- Container restart mid-run: banks are sharded per 100 windows and training checkpoints per epoch; resumed cleanly.
- Machine load 8-10 on 2 cores from four agents: one bank (3000 windows) takes ~1 h, one training rung 25-30 min.
- `data.json` must not contain NaN (node JSON parse); sanitised in webdata.py.

## 17 Nov 2026: imported into ML Ratchet

- **Layout.** The release zip (faultfinder-v1.0.zip) became `models/faultfinder/`: `faultfinder.py`, `pipeline.py`,
  `faultfinder_weights.npz` (+ its `.check.json`, numpy vs torch 9.5e-7), `demo.py`, `examples/` (four exam days and
  `labels.json`), `MODEL_CARD.md` and every training script in `training/` with `configs/`, the raw ratchet log
  (`training/ratchet.json`), the exam results (`training/exam.json`) and the browser port (`training/ffnet.js`), all
  as shipped. The inference files, weights and examples are byte-identical to the zip. The original page was fetched
  from its claude.ai artifact and saved byte for byte as `lab/index.html` (3.1 MB; it embeds all 217 exam days and the
  weights). `build.py` refers to `page.html` and `exam.js`, which were not in the zip; it is kept as shipped.
- **Paths.** New `training/_paths.py` (SRC, MODEL, REPO, WORK, BLDG, LAB). `common.py` now takes `ROOT` from
  `_paths.WORK` (`models/faultfinder/work/` or `$ML_RATCHET_WORK/faultfinder/`) and puts `BLDG` and `BLDG/buildingsim`
  on `sys.path` instead of `/home/claude/bldg`. `exam.py` and `webdata.py` read `faults_summary.json` from
  `BLDG/data/proc/baselines/`; `examples.py` writes to `WORK/release/examples`; `noise.py`'s default output is
  `WORK/data/noise_v1.npz`; `export.py` and `webdata.py` import `faultfinder.py` from the model folder (it used to sit
  in `release/`); `ratchet.py` reads `training/configs/ladder.json` (it used to be `ROOT/configs`). No other change.
- **Waits on the platform zip.** Every training script needs `buildingsim` and `buildingdata` (shared/building/,
  from buildingsim-buildingdata-v1.0), and the banks, noise model and checkpoints are not in the repository, so no
  training step was re-run. Inference, the demo and the instrument exporter need only numpy and pandas.
- **What was re-run.** The demo, from the repository folder: deterministic, under a second; the zip ships no demo
  output to compare with, and the files it runs are byte-identical to the zip's, so this is the shipped demo's
  output (`tests/golden/faultfinder.txt`). The 217 exam days embedded in the lab page, re-scored with the shipped
  weights: P(fault) matches the page's stored Python values to 5e-6 (they were stored to 5 decimals), the detection
  AUC is 0.7747 for all units (exam.json 0.7747), 0.7796 for AHUs (0.7795; the page's inputs are rounded) and 0.65
  for the RTU, and diagnosis top-1/top-3 is 19.7 %/48.5 % on the 66 simulable faulty days, as released.
- **Licences.** Nothing FaultFinder uses or ships is BDG2-derived, so there is no CC BY-SA folder. The handoff called
  the fault subset CC0; the release's own credits say the LBNL synthesis (FLEXLAB, ORNL, MZVAV-2) is CC0 but the
  ASHRAE RP-1312 ERS and Nesbitt Hall data come from Ghalamsiah's figshare package under CC BY 4.0. They are credited
  in `model.yaml`, in a new "Data and licences" section of `MODEL_CARD.md` (with a file-by-file list) and on every
  day of the live instrument.
- **Inconsistencies left as shipped.** `ratchet.json` describes r6 as "179k weights"; the exported network has
  160,871 parameters (160,103 after BatchNorm folding), and the status doc says 161k. The lab page's "Use it" section
  says 101k weights. The model card's r3 sim AUC on v4 is 0.69; `ratchet.json` has 0.6849, so `ratchet.yaml` says 0.68.
- **Exam rows.** Every row of the card's exam table went into `model.yaml`, with the per-source rows. One baseline
  was added at import: always naming the three commonest exam classes gets 36 of 66 days (55 %) at top-3, against
  48 % for FaultFinder, so top-3 is a loss (the release reported it without a baseline). The fixed-false-alarm recall
  rows set their thresholds on the exam's own healthy days and are marked report-only.
- **Rule notes.** No exam result was used to choose anything. The 20 dev unit-days set the noise library, the twin
  tests, the dev gate and the calibration temperature. Each bank or noise change got a new frozen simulated set
  (v1-v4); all rungs were re-scored on all four and the decisions re-taken in order on v4. The fixed-false-alarm
  detection rates are report-only (handoff section 5).
- **Live instrument.** `web/export_instrument.py` writes `site/public/instruments/faultfinder/` (0.51 MB): the
  network in float16, twelve exam days (the four examples, read through `raw_day` as the demo reads them, plus one
  faulty and one healthy day per source drawn with seed 20261117 from the page's exam days) and 108 parity cases.
  `site/src/instruments/faultfinder/net.ts` ports `featurize` and the network from `training/ffnet.js` onto the
  shared ops; it matches Python to 1.4e-6. Controls: a sensor offset on one temperature, the outdoor-air damper stuck
  (mixed air from the mixing balance, the logged command unchanged), extra sensor noise and removing the mixed-air
  sensor. float16 weights move P(fault) by at most 8.5e-5.

## 2026-11-17 — Integrity pass: the plan, written before anything is re-scored

The building round-2 review asked for the same integrity pass WakeNet had. The exam (217 days) is already seen: v1.0
took it once. Every number below is labelled "seen", and nothing is chosen from it. No new rung runs in this pass.

**1. Re-score the exam with the released weights.** The released network is `faultfinder_weights.npz` (numpy, BatchNorm
folded); the torch checkpoint `r6_wide.pt` is not in the repository. `exam.py` gets a numpy path: a model name ending
in `.npz` runs `faultfinder.forward` instead of torch (and `evaluate.py` imports torch only when it needs it). Then
`exam.py ../faultfinder_weights.npz` on the unpacked fault data, with the dev temperature re-fitted by the same rule.
FaultFinder's own numbers must reproduce `training/exam.json`; I report the largest difference. Every row exam.py
computes goes into `model.yaml` and `MODEL_CARD.md`: graded rows for all units, AHU, RTU and each source (AUC,
balanced accuracy, false-alarm rate and recall at P = 0.5; report-only recall at 2/5/10 % false alarms), and info rows
for accuracy, by dataset, season, severity, fault type, diagnosis by class and source, calibration and energy.

**2. Fair baselines, tuned on the 20 dev days only.** The APAR rules ran with textbook tolerances (temperature
tolerance 1.5 K, supply-fan heat 1.1 K, 15-min persistence); only the score threshold was tuned on dev.
- *APAR dev-tuned.* A parameterised copy of `shared/building/data/scripts/baselines.py::apar` (checked first: the
  textbook setting must reproduce `faults_apar.parquet` scores). Grid, 54 settings: temperature tolerance
  {0.75, 1.0, 1.5, 2.0, 2.5, 3.0} K × fan heat {0.5, 1.1, 1.7} K × persistence {15, 30, 60} min. Chosen by AUC on the
  18 AHU dev days; ties by dev balanced accuracy, then by closeness to the textbook setting. Threshold: the one with the
  best dev balanced accuracy among 0 and the midpoints of the dev scores (ties: the lowest, as `baselines.py` does).
  RTU keeps the fan-on-while-unoccupied rule (no tolerance to tune).
- *Mixing-box residual detector* (per unit-day, 15-min bins, fan on). Three residuals: (C1) outdoor-air fraction
  residual, median |(MA−RA)/(OA−RA) − f(cmd)| over bins with |OA−RA| > 3 K, where f(cmd) = a + (b − a)·cmd^g is fitted
  by least squares on the healthy dev days' bins; (C2) coil-off residual, median |SA − MA − δ| over bins with both
  valves ≤ 0.02, δ the healthy-dev median; (C3) supply-air tracking, median |SA − SP|. Each needs 4 bins. Each is
  standardised by the healthy dev days' median and IQR (floors 0.02 and 0.2 K); score = the largest available z.
  Threshold by dev balanced accuracy on all 20 dev days.
- Scored on the same 217 rows as FaultFinder. "Best baseline" on a row is the best classical value on that row
  (textbook APAR at its dev threshold, dev-tuned APAR, mixing-box residual, fan rule for the RTU). Picking the best on
  the exam favours the baselines, not the model.

**3. Intervals.** For every graded row, model minus best baseline, 95 % bootstrap, 2,000 resamples, seed 20261117,
resampling unit-days within the row's subset (the exam unit). Also a cluster bootstrap over facility-days (`group`:
ERS AHU-A and AHU-B, and Nesbitt AHU-1 and AHU-2, share a day). Verdict from the unit-day interval: win if it excludes
zero on the good side, loss if on the bad side, tie if it spans zero; a row the dumb baseline wins is a loss. Where the
facility-day interval disagrees, the note says so. The recall-at-fixed-false-alarm rows set thresholds on the exam's own
healthy days (re-set in every resample) and stay report-only. Diagnosis rows: against always naming the commonest exam
class (top-1) or the three commonest (top-3), resampling the 66 days. Info rows (calibration, Brier) get an interval on
the value alone.

**4. Dev twin test of what the network adds** (20 dev days). Anchor = the best dev-tuned rule, turned into a
probability by a two-parameter logistic fit on dev (AHU: dev-tuned APAR; RTU: fan rule). Correlate the network's
correction (P_net − P_rule, P_net at T = 4.3) with what reality needed (truth − P_rule). List the days where they
disagree with the truth.

**5. Unit habit, measured on dev only** (for next round's multi-day rung). Units are dataset + unit name; three dev
units have two or more healthy days (ERS AHU-B 3, Nesbitt AHU-1 3, Nesbitt AHU-2 2).
- Day features: mean and standard deviation of each of the 35 network feature rows over fan-on bins, standardised
  over the 20 dev days. For each healthy day: distance to the nearest other day of the same unit vs the nearest day of
  another unit.
- The network's detection log-odds on healthy days: share of variance between units vs within units.
- Leave-one-out unit offset: subtract the mean log-odds of the unit's other healthy dev days; count false alarms at the
  dev threshold before and after (diagnostic only, not a rung).
Then the rung plan goes in the NOTEBOOK as a plan (not run): what the network sees from the unit's recent days, how
the bank supplies them, and how the exam supplies them causally. Exam metadata (dates per unit, no labels) may be
counted to size the plan.

**6. Readiness.** On this Windows machine, at most 3 processes: `noise.py` (noise model from dev), `simbank.py` for a
few windows into `work/`, a one-epoch `train.py` on that bank. Time each and extrapolate two 38,987-day banks and the
frozen val v4. List blockers.

**7. Paperwork.** `model.yaml` (rows, intervals, verdicts, "seen"), `MODEL_CARD.md`, a ladder note in
`ratchet.yaml`, a results entry here; `readme --write faultfinder`; validate; demo test.

## 2026-11-17 — Fresh exam frozen before round-2 rungs: Wang 2024 (Korean office) and Nehasil (CTU Prague)

Written before either dataset is downloaded. Nothing is scored in this step: no model, no baseline score. The exam is
frozen in `training/split_exam2_wang.json` and `training/split_exam2_nehasil.json` (rows, baselines and interval rule
inside), built by `training/exam2_wang.py` and `training/exam2_nehasil.py` into `work/exam2/` (git-ignored). It is
scored once for v1.0 and once for any round-2 release, after the round-2 release decision.

**Data.**
- Wang et al. 2024, figshare 10.6084/m9.figshare.25909813.v1 (CC BY 4.0, `combined_FDD.xls`, 0.8 MB): 20 AHUs of a
  large office in Gwacheon, hourly BMS, Jul-Oct 2023 per the record, expert labels (normal, fan, sensor, valve fault).
  **Not unit-clean.** The same 20 AHUs, Nov 2023-May 2024 (figshare 25932409.v2, no labels), set two noise statistics
  in v1.0 training (start/stop jitter, supply-air tracking tail). Labels were never seen and the months differ, so this
  is a time-fresh exam on units whose unlabelled operation shaped the noise model. It is labelled that way.
- Nehasil, CTU Prague, Mendeley 10.17632/jkw85sm4v4.1 (CC BY 4.0): AHU1, AHU2, AHU8, AHU18, AHU23, AHU24 of a real
  BMS, manually induced faults, `Faults_schedule_and_description.xlsx`. Unit-clean: no Czech data in training or exam 1.

**Input contract.** `faultfinder.py` reads 13 channels per local day in 15-min bins; any may be missing (value 0 and
mask 0 in the features). Wang has T_ra, T_sa, T_sa_sp, supply fan on/off, a valve position, heating-water supply
temperatures and heating-pump total: no T_oa, T_ma, damper, cooling valve, fan speed or power, occupancy. Mapping as
BuildingData's `korea_office` reader: T_ra, T_sa, T_sa_sp (0 = off -> missing), fan_sa_status; the valve stays missing
(meaning unclear), the rest go to `x_` columns. So FaultFinder sees 4 of 13 channels in 24 of 96 bins (hourly logger);
the APAR rules that need T_oa, T_ma or the damper cannot fire. Nehasil: points are mapped from the place / quantity /
type tags only where a tag names a contract channel unambiguously; the map is written into the split file.

**Unit-day labels.** Wang: AHU x local date; fault day if >= 1 hour is labelled a fault, normal day if >= 20 hours are
labelled normal and none faulty, otherwise excluded (mostly unlabelled). The day's fault type is its most frequent
fault label. Nehasil: from the fault schedule; fault day if a scheduled fault on that AHU overlaps the local day,
normal if the AHU has no scheduled fault that day; per-row labels in the files, if present, take precedence and
disagreements are listed. Days on a fault's start or end date with < 1 h of fault are excluded.

**Day quality control** (as `common.real_raw` / `to_day`): temperatures < -45 or > 70 degC -> missing; a temperature
constant all day -> missing; T_sa_sp < 1 -> missing; duplicate timestamps keep the first. A day is gradable if T_sa is
present in >= 75 % of the logger's slots (Wang >= 18 of 24 hours; Nehasil >= 72 of 96 15-min bins) and the fan runs
>= 1 h (fan status, or T_sa varying when no fan point exists). Every dropped day is counted by reason.

**Split, by unit, seed 20261117.** Wang: 5 dev AHUs, 15 exam AHUs, drawn 25 % from each of two strata (AHUs with any
fault day / none), at least one per non-empty stratum. Dev only tunes the tracking-rule baseline below. Nehasil: all 6
AHUs are exam (too few to split); its baselines use the platform dev runs.

**Rows**, per dataset, never pooled: (1) detection AUC over unit-days; (2) balanced accuracy at P = 0.5; (3)
false-alarm rate at P = 0.5 on normal days; (4) diagnosis top-1 on fault days whose label maps to a FaultFinder class
(Wang: SAT sensor -> sensor_bias_T_sa, RAT sensor -> sensor_bias_T_ra; fan, pump and unnamed valve faults -> other;
Nehasil: through `common.EXAM2CLASS` by the schedule's description), info only if n < 10. Model input: the released
weights through `FaultFinder.read_day` on the contract columns, with the shipped temperature; no refit on these data.

**Baselines.** APAR rule set (`shared/building/data/scripts/baselines.py` `run_score`, 15-min persistence) with the
platform dev threshold 0.2322 (`faults_summary.json`), unchanged; Wang only, a supply-air tracking rule tuned on the 5
dev AHUs (score = share of fan-on hours with |T_sa - T_sa_sp| > theta, theta in 0.5-5 K by 0.5, day threshold for best
dev balanced accuracy; ties to the smaller theta, then the larger threshold); majority guess (Wang dev majority;
Nehasil platform dev majority, "no fault").

**Intervals.** 95 % bootstrap over exam AHUs (all days of a drawn AHU move together), 2,000 resamples, seed 20261117,
on model minus the best baseline of the row. Win if the interval is entirely on the good side of zero, loss if on the
bad side, tie if it spans zero. n AHUs and n days stated. With 6 and 15 AHUs most rows will be ties.

## 2026-11-17 — Integrity pass results: fair baselines, intervals, twin test, unit habit, readiness

Everything below followed the plan in the entry above; nothing was changed after a result was seen, with one
correction before the final run: the first run scored the RTU's textbook fan rule at the midpoint threshold and let
the mixing-box residual (which cannot exist on the RTU) win the RTU false-alarm row with "never alarms". The final run
uses the released RTU threshold for the textbook rule and leaves the residual out of the RTU rows. Outputs:
`work/exam/` and `work/integrity/` (`rows.json`, `integrity.json`).

- **Re-score reproduces.** `exam.py ../faultfinder_weights.npz` (new numpy path) re-fits T = 4.3 on dev and
  reproduces every number in `training/exam.json` to 6e-8 (495 numeric fields; float32 rounding). The parameterised
  APAR copy reproduces `faults_apar.parquet` exactly on all 237 dev and exam days.
- **Fair baselines.** APAR grid on the 18 AHU dev days: best is tolerance 1.5 K, fan heat 0.5 K, 15 min (dev AUC 0.775
  against 0.731 textbook), threshold 0.037. Mixing-box residual: damper characteristic 0.88·cmd^1.2 from 416 healthy
  dev bins, threshold 3.84, dev AUC 0.65. The coils-off residual had only 2 healthy dev days to set its scale.
  On the exam (seen): AHU AUC textbook 0.69, tuned 0.69, mixing box 0.64; Nesbitt 0.54 / 0.54 / **0.71**; balanced
  accuracy AHU 0.65 / 0.68 / 0.66; AHU false alarms at the dev threshold 2 / 17 / 15 %.
- **Verdicts (unit-day intervals).** None of v1.0's wins survives as a win. AHU AUC +0.09 [−0.00, +0.18], all units
  +0.08 [−0.00, +0.16]: ties (the facility-day intervals [+0.00, +0.17] and [+0.01, +0.16] would say win). Balanced
  accuracy: AHU +0.02 [−0.08, +0.11], all units −0.00 [−0.08, +0.08]: ties with the dev-tuned rules. Nesbitt AUC +0.07
  [−0.05, +0.19] against the mixing box: tie. FLEXLAB +0.19 [−0.03, +0.43]: tie. False alarms stay losses (AHU +31
  points [+23, +41]). RTU balanced accuracy stays a loss (0.47 below the majority guess); RTU AUC, diagnosis top-1
  (−2 [−17, +14]) and top-3 (−6 [−23, +9]) become ties. New wins: recall at P = 0.5, all units +17 [+5, +29], AHU
  +21 [+7, +34], Nesbitt +23 [+4, +42]. Every fixed-false-alarm row is a tie (report-only).
- **Dev twin test.** Correlation of the network's correction to the dev-tuned rule with what the truth needed: +0.07
  over 20 dev days. Network right 13 of 20 at 0.5, rule 14. The network adds the two Nesbitt AHU-2 faulty days the
  rules score 0 on; it loses two healthy days sitting at P = 0.501 and a Nesbitt healthy day at 0.79.
- **Unit habit (dev only).** Nearest day in feature space is the same unit's for 4 of 10 healthy days (median distance
  0.81 against 0.82 to other units). Healthy-day log-odds: 38 % of variance between units. A leave-one-out unit offset
  raised dev false alarms from 3 to 4 of 8. So the v1.0 claim that most false alarms are unit habits is a hypothesis
  the dev days do not support; two of the four dev false alarms are calibration (P = 0.501). The model card and
  model.yaml now say so.
- **Readiness (Windows, one process each).** `noise.py` 2 s. `simbank.py` 0.69 s per 12-day window (100 windows in
  69 s plus 27 s to draw building specs): a 38,987-day bank (about 3,000 windows) takes about 35 min on one process,
  both training banks plus a ~920-window validation bank about 80 min on one process or about 35 min on three. One
  epoch of `train.py` (r6 network, 1,013 days) took 4 s (1 s noise + features, 3 s training, one thread): about
  2 min per epoch on a full bank (about 30,000 days after the effect filter), so r6's 10 epochs take about 20 min.
  Building frozen val v4 (noise + features) is seconds even at 12,000 days. A full rebuild fits in about an hour.
- **Blockers found.**
  1. The worktree's `shared/building/buildingsim` was being edited by another agent (`from . import phase2` failed;
     `phase2.py` appeared later). The smoke run used the committed BuildingSim exported to a scratch folder. Any
     BuildingSim physics change needs new frozen sets (HANDOFF section 4).
  2. `.venv-train` has torch but no pyarrow, so `train.py`'s dev gate (`bd.faults` reads parquet) fails. It ran by
     appending `.venv`'s site-packages (same Python 3.14.9 and numpy 2.5.4). Fix: install pyarrow in `.venv-train`.
  3. The r1-r6 torch checkpoints are not in the repository. r6 can be re-scored on any new frozen set through the new
     numpy path (`evaluate.NpzModel`); r1-r5 cannot without retraining.
  4. For the multi-day rung: banks do not record the window or the day within it, and `Noise.apply` draws static
     offsets per day, so a unit's habits do not persist across its days. Both need changing (plan below).

## 2026-11-17 — Plan for the multi-day rung (r7_context), not run

A plan for next round, written now so it is fixed before any bank is built. The dev measurement above did not
support "most false alarms are unit habits", so this rung is a test of that idea, not a fix known to work.

**What the network sees.** Today's 35 × 96 feature rows as now, plus a reference built from the same unit's most
recent 1-3 logged days before today (within 7 days): the per-bin median of each feature row over those days, today
minus that median for the 11 temperature and physics rows, and one row saying how many reference days there are
(0 = none). 35 + 35 + 11 + 1 = 82 input rows, the same CNN. With no reference the extra rows are zero and the
network is in single-day mode, so v1.0's use stays covered. No label of a reference day is ever an input.

**How the bank supplies recent days (bank v5, simulator unchanged).**
- `simbank.py` saves the window id and day index for every row, and keeps every day of both runs (healthy days 1-11,
  faulted days 5-11), so a reference can be drawn from the same window.
- The noise model gets a per-window state: the static sensor offsets, the damper characteristic, the residual-library
  draw and the operating habits (24 × 7, night chiller off, operator jitter) are drawn once per window and kept for
  its days. Without this the bank has no unit habits to learn.
- Training samples: for day k, the reference is 1-3 days drawn from days max(1, k − 7)..k − 1 of the same run. A
  faulted day early in the fault has healthy references (onset); a later one has faulted references (a persistent
  fault looks like the unit's habit). 20 % of windows start the fault on day 1, so some faults are present in every
  reference. 30 % of samples get no reference.
- New frozen set v5 (seed 505) built the same way, and r6 re-scored on it from the numpy weights.

**How the exam supplies them causally.** For each exam unit-day, the reference is the same dataset and unit's days
with an earlier date, at most 7 days earlier, the most recent 3, chosen by date alone. Their labels are never read
(a reference day may be faulty, as in deployment). The ERS sister unit on the same day is not used. From the exam
metadata (dates only): 70 % of exam days have an earlier same-unit day within 7 days and 41 % have three or more;
ERS RP-1312 86 of 92, FLEXLAB 21 of 23, ORNL 12 of 17, Nesbitt only 30 of 76, MZVAV-2 4 of 9. Results will be
reported for days with and without a reference.

**Decision.** Same rule as the ladder: keep if skill on v5 beats r6 by more than 0.01 and the dev AUC passes the gate.
The dev days give only a small check: 12 of the 20 have an earlier same-unit dev day, but within 7 days only
Nesbitt's two 3 Aug 2017 days (reference 29 Jul) do. The dev gate runs every dev day with whatever reference the rule
gives and reports how many had one. Any exam result is labelled seen.

## 2026-11-17 — r7 plan fixed before bank v5 is built; r7b added

Written before any v5 bank, frozen set or rung exists. It adds to the plan above; where they differ, this entry holds.

**Rule (fixed now).** A rung is kept if its skill on frozen v5 (0.5 detection AUC + 0.5 diagnosis accuracy on
visible faults, as the ladder) is more than 0.01 above the incumbent's AND its dev AUC is at least
max(0.60, incumbent dev AUC - 0.05). The incumbent is r6 from the released numpy weights (dev AUC 0.707, so the gate is
0.657). Every rung also reports its dev false-alarm rate at P = 0.5 (11 healthy dev days; r6: 4 of 11, two of them at
P = 0.501). The false-alarm rate is reported, not gated. r1-r5 cannot be re-scored on v5: their checkpoints are not in
the repository.

**r7b: run it, and it cannot be kept.** r7b replaces the single temperature T = 4.3 with a two-parameter (Platt) map
P = 1 / (1 + exp(-(a d + b))) on r6's detection log-odds d, fitted by maximum likelihood on the 20 dev days (the card's
own `next` item). A monotone map changes neither the AUC nor the diagnosis, so r7b's frozen skill equals r6's exactly
and the rule rejects it. That is written here so it cannot be argued after the result. It is still run, because it
costs seconds and answers the dev question directly: how many of the 4 dev false alarms does a 2-parameter map remove?
Reported in-sample and leave-one-out (fit on 19 days, score the 20th), with the Brier score and a, b. If it helps,
calibration needs its own rule next round; it is not slipped into a release here.

**r7 seeds.** r6's training config (width 64, 10 epochs, lr 2e-3) with seeds 1 and 2. The decision uses the mean
over seeds (skill and dev AUC). If the margin (mean r7 skill - r6 skill) is under twice the seed spread (|s1 - s2|),
seed 3 runs and the decision uses the 3-seed mean. If r7 is kept, the released weights are seed 1, fixed now: no
seed is picked by its score. Information only: (a) r7 scored on frozen v5 with its references removed (single-day
mode); (b) a control, r6's 35-row network retrained on bank v5 with noise v5 (seed 1), to separate "new bank" from
"reference days".

**Bank v5 (`training/simbank5.py`, BuildingSim 1.0.1).** simbank.py's buildings, cities, control priors and fault draws
(calib_v3.json priors), with the dev-calibration wrappers taken from `buildingsim.phase2`: `OperatorJitter` as the
controller and `log_commands` on the faulted trend frame, from the fault start only. One feedback draw per window
(v3 drew it per day; a unit either has position feedback or not). The fault starts on day 1 in 20 % of windows,
otherwise day 5. Every row records window id, day (1-11), run (healthy / faulted) and a sample flag. Kept: healthy days
1-11 and faulted days from the fault start. Sample rows are v3's: faulted days, healthy days before the fault, and
healthy days 10-11, so 80 % of windows give v3's class mix; the other healthy days serve only as references. Train bank
3,000 windows (seed 5050), validation bank 920 windows (seed 5051), on 3 processes.

**Noise v5 (`noise.py`, new `window_state`).** Drawn once per window and kept for all its days: the static sensor
offsets (`phase2.static_offsets`, v4 sizes), the damper characteristic (`phase2.damper_command`, one draw per
window), which real residual day each channel uses and its gain, the fan-off blend, the channel pattern and the 6 %
extra missing channels. Drawn per day: the residual's time shift and logger gaps. Training redraws every window's
state each epoch.

**Check against v4, before training.** (a) A 300-window bank from the shipped `simbank.py` with calib_v3 (seed 5051,
BuildingSim 1.0.1) against val v5: rows per window, faulty share, share of visible faults (effect >= 1), median
effect, fan-on hours, logged actuator signal on stuck and leak days, healthy supply-air minus setpoint. (b) Noise v4
against noise v5 on the same val v5 days: mean and SD of the 35 feature rows over fan-on bins. The marginals should
match; v5 should differ only in that offsets repeat within a window. Differences are reported; nothing is tuned on
them.

**Frozen v5 (seed 505, never regenerated).** From val v5: sample rows with class 0 or effect >= 1, at most 12,000,
drawn with seed 505. Window states and references are drawn by the training rule: 30 % of samples get none, the rest
1-3 days drawn from days max(1, k - 7)..k - 1 of the same run (faulted run from the fault start, healthy before). The
file stores the 82-row input; r6 reads rows 0-34.

**Input (82 rows).** Rows 0-34: today's features. 35-69: per-bin median of the reference days' 35 rows. 70-80: today
minus that median for rows 0-10 (6 temperatures, 5 physics differences). 81: number of reference days / 3. With no
reference, rows 35-81 are 0 (single-day mode).

**Dev.** Reference days: same dataset and unit, earlier dev dates within 7 days, the most recent 3, chosen by date;
their labels are never read. Expected: 2 of 20 dev days have one (Nesbitt AHU-1 and AHU-2, 3 Aug 2017, from 29 Jul).
The dev result is reported for all 20 days and for those with a reference.

**Order.** No decision is written until a fresh-exam split file (`training/split_exam2_*.json`) is committed. The
fresh exam is not scored here.

## 2026-11-17 — Fresh exam frozen: results (nothing scored)

Built by `training/exam2_wang.py` and `training/exam2_nehasil.py` (shared helpers in `exam2_common.py`). No model or
baseline was run on either dataset. `--check` rebuilds reproduce both split files.

- **Wang 2024: downloaded, rejected as an exam.** `combined_FDD.xls` (md5 matches figshare). It holds January 2024
  only: 5,471 hourly rows in 20 blocks with no AHU name, labelled "Fan fault" (4,246) or "Sensor fault" (1,225). There
  is no normal row, so AUC, balanced accuracy and false-alarm rate cannot be computed. 5,436 rows (99.4 %) are the
  same readings (time, T_ra, T_sa, T_sa_sp) as BuildingData's `korea_office`, which fitted v1.0's noise model; each
  block maps to one AHU (AHU-101 to 112 without 106, AHU-201 to 209). The record's "Jul-Oct 2023" does not match the
  file. `split_exam2_wang.json` records this with per-block hashes and freezes no rows (sha256 `e1184417…`). The Wang
  dev split and the tracking-rule baseline of the plan were therefore not built.
- **Nehasil: downloaded, frozen.** All 8 files match Mendeley's sha256. 5-min logs, 1 Mar - 31 Aug 2019, with a
  per-row `fault` column that agrees with the schedule sheet on every AHU-day. These are 100 % outdoor-air units with
  heat recovery: no T_oa, T_ma, supply setpoint, zone temperature or occupancy point. Mapped: T_sa, T_ra, OA damper
  (open/closed command), heating and cooling valve, supply fan speed and status. AHU2 and AHU8 log only water-side
  points (no T_sa on any day): dropped. Kept AHU1, AHU18, AHU23, AHU24: **513 gradable days, 54 fault days, 459
  normal, 34 fault days with a FaultFinder class.** 219 days dropped as "fan off all day" (one of them a fault day on
  AHU18). Fault days have 1.1-3.4 h of fault. The two "heating valve ... during cooling regime" labels name a
  different valve than the schedule ("cooling valve"), so they are left out of the diagnosis row.
  `split_exam2_nehasil.json` sha256 `b84c05f508e92411eb35b40fca15e663211a3c53cbf16bf5e9e7f64ae5ff438e`
  (units block `31f05ccb…`).
- **What to expect.** Four AHUs give the bootstrap four distinct units: the intervals will be wide and most rows will
  be ties. APAR can fire only R23 (simultaneous heating and cooling) on these channels.

## 2026-11-17 — r7_context and r7b results: nothing released, v1.0 stays

Everything followed the two plan entries above; nothing was changed after a result was seen. The fresh-exam split
(`training/split_exam2_nehasil.json`, `split_exam2_wang.json`) was committed (3d9f949) before this decision was
written. The fresh exam was not scored. Raw scores: `work/r7/` (git-ignored); the record is `training/ratchet_r7.json`.
Code: `training/simbank5.py`, `noise.py` (`window_state`, `apply_window`), `context.py`, `train7.py`, `r7.py`.

**Bank v5 (BuildingSim 1.0.1, `phase2` wrappers).** Train: 3,000 windows on 3 processes in 11 min; 2 windows failed
inside BuildingSim (a negative input to a log) and were skipped, as simbank.py does. 2,998 windows, 56,144 rows,
29,283 training samples (46 % faulty). Validation: 920 windows, 17,244 rows. Frozen v5 (seed 505): 9,145 samples,
4,309 faulty, 5,771 (63 %) with reference days. It has fewer than v4's 12,000 because the validation bank is smaller.

**Check against v4 (before training).**
- Paired with the shipped `simbank.py` (calib_v3, same seed 5051, first 300 windows): the 242 windows whose fault
  starts on day 5 draw the same building, weather, controls and operator jitter. Their 1,452 healthy days are identical
  (max difference 0.0). 1,573 of 1,694 faulted days are identical; the rest differ only in the logged actuator, where
  `log_commands` now draws position feedback once per window instead of once per day.
- Bank statistics, shipped code vs v5 sample rows (300 windows each): 13.0 rows per window both; faulty share 0.54 vs
  0.60 (expected: 20 % of v5 windows fault from day 1); visible share of faulty days 0.61 vs 0.60; median effect 3.87
  vs 3.84; fan-on hours on healthy days 19.5 vs 19.6; healthy SAT minus setpoint median 0.0 both, IQR 0-1.64 vs
  0-1.62 K. Logged actuator on stuck or leaking days moves by 0.01-0.06 (about 100 days each from about 15 windows).
- Noise v4 vs v5 on the same 2,000 val v5 days: feature-row means within 0.07, SDs within 12 %. The largest gaps are
  the availability masks (up to 5 points) and the temperature rows that depend on them: v5 draws the channel pattern
  once per window, so 2,000 days carry only about 900 independent patterns. r6 on those days: skill 0.543 with v4
  noise, 0.553 with v5 noise. Nothing was tuned on these.

**Scores (frozen v5, 9,145 samples; dev, 20 days, 11 healthy).**

| Rung | skill v5 | AUC | diag | skill with ref / without | dev AUC | dev false alarms |
|---|---|---|---|---|---|---|
| r6 (released weights, incumbent) | **0.561** | 0.743 | 0.378 | 0.560 / 0.561 | 0.707 | 4 of 11 |
| r7b (r6 + Platt on dev) | 0.561 | 0.743 | 0.378 | same as r6 | 0.707 | 1 of 11 (in-sample and leave-one-out) |
| r6v5 control (35 rows, bank v5) | 0.554 | 0.740 | 0.369 | 0.556 / 0.550 | 0.677 | 6 of 11 |
| r7_context seed 1 | 0.564 | 0.759 | 0.369 | 0.569 / 0.545 | 0.677 | 6 of 11 |
| r7_context seed 2 | 0.558 | 0.754 | 0.363 | 0.561 / 0.546 | 0.626 | 4 of 11 |
| r7_context seed 3 | 0.552 | 0.745 | 0.358 | 0.559 / 0.528 | 0.677 | 3 of 11 |
| **r7_context, 3-seed mean** | **0.558** | 0.753 | 0.363 | 0.563 / 0.539 | 0.660 | 4.3 of 11 |

r1-r5 were not re-scored: no checkpoints. r6 scores 0.561 on v5 against 0.543 on v4.

**Decisions.**
- **r7_context: rejected.** Two seeds gave a margin of +0.0006 over r6 with a seed spread of 0.0057; the margin was
  under twice the spread, so seed 3 ran as planned. The 3-seed mean is 0.558, 0.003 below r6 (seeds 0.552-0.564).
  The dev AUC mean, 0.660, passes the 0.657 gate only just.
- **r7b: rejected,** as written before it ran: its frozen skill and dev AUC are r6's. On dev it does what the card
  hoped. Platt a = 0.302, b = -0.596 moves the alarm threshold from log-odds 0 to 1.97. False alarms drop from 4 to 1
  of 11, in-sample and leave-one-out; the one left is Nesbitt AHU-2 on 29 Jul (log-odds 5.6). Detections drop from 6
  to 4 of 9. Brier: temperature 0.224 in-sample, 0.246 leave-one-out; Platt 0.208 in-sample, 0.258 leave-one-out.
  The two-parameter map overfits 20 days at least as much as the one-parameter one.
- **Released: nothing.** v1.0 stays; the ladder logs r7b, r7_context and the control.

**What the reference days do.** Within r7, the references help: on the 5,771 frozen days with a reference, r7 scores
0.563 (3-seed mean) and 0.537 with the references zeroed. Detection AUC there is 0.749-0.762 against r6's 0.738.
But r7 without references is worse than r6 (0.539 vs 0.561 on days without one), and so is the control (0.554), which
retrains r6's network on bank v5 alone. Single-day accuracy fell by 0.006 (control) to 0.023 (r7 with its references
zeroed) against r6. The reference gain roughly cancels that loss, and the 0.01 margin is not met. A second likely cause is the dev split:
only 2 of 20 dev days have a reference (Nesbitt AHU-1 and AHU-2, 3 Aug 2017, from 29 Jul). The labelled AHU-1 fault
day moved from log-odds -0.63 (r6) to +0.07, +0.32 and -0.82 across the three seeds; AHU-2 stayed at 2.2-2.9.
Dev says nothing about unit habits beyond those two days. The dev false-alarm count is driven by single-day mode and
varies from 3 to 6 of 11 across seeds of the same rung.

**For next round.** (1) The reference input helps on simulated days with a reference (+0.026 skill over the same
network without it). A rung that keeps r6's single-day accuracy, for example by fine-tuning r6 with the extra rows
zero-initialised, would test it more fairly. That is a new plan, not run here. (2) Calibration needs its own rule
and a larger dev set: 20 days cannot tell a one-parameter map from a two-parameter one.
