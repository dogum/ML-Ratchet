# Prior Wars: pre-registration

Method experiment C1 for ML Ratchet. Physics prior vs generic prior, on the collection's real exams.

**How this file is frozen.** It is written in sections. Before any exam prediction or score for a model,
`scripts/freeze.py` hashes the whole file as it stands (SHA-256 of the first N characters) and appends
`{models, n_chars, sha256, time}` to `work/prereg_log.json`. `harness.py exam_predict` and `harness.py score`
refuse to run for a model unless its frozen prefix still hashes to the logged value. Text is only ever appended
after a freeze. Corrections go in a dated "Amendment" section and never change an earlier section.

## 0. The question

Every ML Ratchet model learns only from simulated physics. TabPFN learns only from synthetic data too, but from a
generic prior over random causal models, not from a domain simulator. Does a domain simulator beat a generic synthetic
prior on the collection's real exams, and how many real labels does the generic prior need to catch up?

## 1. What was already seen before this file was written

- **P's exam numbers are public.** Every model's own exam was run once by its builder and is in the repo
  (`model.yaml`, status docs). I read those summary numbers (P and its baselines) while choosing models. I did not
  look at any exam label for this experiment. Before freezing, the only thing done with exam data was to recompute P's
  exam predictions from the released weights and compare them with the shipped predictions (inputs only, no labels).
- **Dev labels were used** for tuning (section 4) and sanity checks, as the method allows.

## 2. Arms

For each model, on its own exam rows, with its own metrics:

| Arm | What it is | Real labels in its context |
|---|---|---|
| **P** | The released simulation-trained model, recomputed from the shipped weights | 0 |
| **G-dev** | TabPFN, context = the model's real dev rows (inputs exactly as the model's pipeline builds them) | k = all dev rows |
| **G-sim** | TabPFN, context = simulated rows from the model's own training bank, subsampled | 0 |
| **G-sim+dev** | TabPFN, context = G-sim's simulated rows + every dev row, plus an `is_real` column (0 sim, 1 dev and exam) | k |
| **C-dev** | Ridge / logistic regression fitted on the dev rows (only where the card already has a dev-fitted classical baseline) | k |
| **LC** | G-dev with k random dev rows, k in {5, 10, 20, 40, 80, 160, 320} below the dev size, 3 random subsets per k | k |

The card's own classical and dumb baselines are re-scored beside the arms on the same rows where they can be rebuilt.

## 3. TabPFN

- Package `tabpfn` 10.2.0 from PyPI (code: Apache-2.0), in an isolated venv (`/home/claude/priorwars/venv`, system
  torch 2.15 CPU, numpy 2.5.3, scikit-learn 1.9.1 untouched).
- **Weights: TabPFN-v2** (`Prior-Labs/TabPFN-v2-reg/tabpfn-v2-regressor.ckpt`,
  `Prior-Labs/TabPFN-v2-clf/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, the package's v2 defaults), under the Prior
  Labs License 1.1 (Apache 2.0 plus an attribution clause). Downloaded without a login.
- **Not used: TabPFN-3.5 (the package default), 3, 2.6 and 2.5.** Their weights are non-commercial, and the client
  requires a Prior Labs account token and licence acceptance in a browser before downloading (`browser_auth.
  ensure_license_accepted`). Per the brief this was recorded and not worked around. v2 is the openly downloadable fallback.
- CPU, one thread, `n_estimators = 2` for every arm (CPU budget), `random_state = 0` (LC subset r uses seed r),
  `ignore_pretraining_limits = True` (contexts above v2's 1,000-row CPU guideline). Regression point prediction = the
  mean of TabPFN's predictive distribution (package default). Classification: P(class 1), or argmax for multi-class.
  TabPFN-v2 takes at most 10 classes. A context with one class predicts that class.
- One TabPFN fit per target (TabPFN is single-output). Rows with a missing target are dropped from that target's context.

## 4. Settings, chosen before the exam and never from it

- **G-dev preprocessing** in {`raw` = the model's feature table as is (TabPFN preprocesses it itself), `norm` = the
  model's own input normalisation}: chosen by 5-fold cross-validation inside dev (grouped by unit where units repeat;
  2 repeats, 1 for dev sets above 1,000 rows) on the model's headline. `raw` unless `norm` wins by more than 0.01.
- **G-sim preprocessing and context size**: chosen on 400 rows (seed 123) of the model's frozen simulated validation set.
  First `raw` vs `norm` at 1,024 simulated context rows (`raw` unless `norm` wins by > 0.01). Then 2,048 rows with the
  winner, kept only if it beats 1,024 by more than 0.01 headline skill. Context rows are a seeded random draw (seed 0)
  from the model's training bank.
- **G-sim+dev**: no tuning. G-sim's preprocessing; the first n of G-sim's simulated rows + every dev row, total at most
  2,560 rows (n shrinks, minimum 256) + the `is_real` column.
- **LC**: G-dev's settings; subsets drawn with seed 1000 k + r, r = 0, 1, 2. The LC point is the mean headline over the
  three subsets. The full-dev point is G-dev itself.
- **C-dev**: median imputation, standardisation, `RidgeCV(alphas = 10^-3..10^3)` for regression targets,
  `LogisticRegression(C = 1)` for classes, fitted on all dev rows.

## 5. Scoring (once)

- Each model's own exam metrics on its own exam rows, as listed in its section.
- A **headline** per model, fixed in its section; errors are negated so higher is better.
- **Paired bootstrap**: B = 1,000 resamples of the model's exam units (seed 0), with every arm scored on the same
  resample. Report 95 % percentile intervals for each metric and for each arm minus P.
- **Verdict** for a contrast on a model: *win* if the 95 % interval of the headline difference excludes 0 in the arm's
  favour, *loss* if it excludes 0 against, *tie* otherwise. The same rule is reported per metric.

## 6. Contrasts and the cross-collection verdict

- **C1, P vs G-dev**: a domain simulator with 0 real labels vs a generic prior given the k dev labels. This is the headline question.
- **C2, P vs G-sim**: the same simulated data, the model's purpose-built network vs a generic in-context learner (0 real labels each).
- **C3, G-sim+dev vs G-dev**: does the simulator, delivered as data, add to the real labels?
- **C4, G-sim+dev vs P.**
- **k\***: the smallest LC k whose mean headline reaches P's exam headline ("> k_max" if none, "≤ 5" if k = 5 already does).
- **Cross-collection sentence**: "the domain simulator beats the generic prior" if P wins C1 on more models than it loses,
  "the generic prior beats the domain simulator" if it loses on more, "split" otherwise. The count, not the sentence,
  is the result.

## 7. Expectations, written down before scoring

These are guesses, recorded so they can be wrong in public:

1. G-dev beats P where dev is large next to the task (RECS 1,848 homes; EnergyNet 1,496 trips) and loses where dev is
   tiny (GridPulse 4 events; BasinDetective 51 basins; BDG2 52 buildings).
2. G-sim loses to P on most models: 1-2 thousand simulated rows and n_estimators = 2 vs a network trained on 10-30 thousand.
3. G-sim+dev beats G-dev where dev is small, and adds little where dev is large.

## 8. Known limits, fixed in advance

- TabPFN-v2, not the current TabPFN-3.5 (section 3). n_estimators = 2 and contexts of at most 2,560 rows, on one CPU
  core: this is a lower bound on what TabPFN can do, not its best.
- Exams too large for CPU TabPFN are subsampled once, with a fixed seed, and P is scored on the same rows (EnergyNet).
- Only tabular models whose exam inputs and dev rows exist on this machine are in scope.

## 9. Amendment A1 (2026-11-18 18:50, after a container restart; before any freeze and before any exam prediction)

The first runs put several TabPFN processes on the shared machine at once and filled its 6.27 GB memory group,
starving two other agents. A restart then killed every job. Nothing had been frozen, predicted or scored. These rules
replace the matching parts of sections 3, 4 and 8. They are fixed by memory alone, never by any label.

- **One TabPFN process at a time.** It runs under a 2 GB address-space cap (`prlimit --as=2000000000`), with `nice -n 10`
  and one thread, and only after a check of the shared group's memory. TabPFN runs with `memory_saving_mode=True` and
  `fit_mode="low_memory"`. These change memory use only, not predictions (apart from floating-point rounding).
- **Row budget.** Measured under the cap, (context rows + test batch) x (features + 8) of about 220k cells fits. So any
  TabPFN context holds at most R_max(F) = min(2,560, floor((210,000 / (F + 8) - 128) / 64) x 64) rows. The test batch
  fills the rest of the 210k cells, and halves if an allocation still fails. Batching does not change predictions,
  because test rows attend only to the context and preprocessing is fitted on the context. R_max: BasinDetective
  (150 features) 1,152; GridPulse (86) 2,048; RECS (12) 2,560; BDG2 (112) 1,600; PlantDetective (171) 1,024;
  EnergyNet (149) 1,152.
- **Contexts above R_max** are a seeded random subsample of R_max rows (seed = the fit's seed). This touches only
  EnergyNet: its G-dev context is 1,152 of the 1,496 dev trips, and its dev-CV training folds are capped the same way.
  Its "k = all dev" learning-curve point is therefore k = 1,152.
- **G-sim tuning:** the 2,048-row step runs only where 2,048 <= R_max(F). Elsewhere G-sim uses 1,024 rows.
- **G-sim+dev:** total rows <= R_max(F + 1), replacing the fixed 2,560. Every dev row is kept if that leaves at least
  256 simulated rows. Otherwise the dev rows are subsampled to R_max - 256 (seed 0); this happens for EnergyNet only.
  The simulated rows are the first n of G-sim's context.
- **PlantDetective exam subsample** (the section 8 rule, now also used here): a fixed draw of at most 2 year windows per
  exam unit (seed 20261118), which gives 781 of the 2,835 rows over all 395 units. P is scored on the same rows.
- **GridPulse frozen simulated set:** simulated loss labels are exact, so on the frozen set they are scored as the
  "medium" tier. This lets the headline items exist there. It fixes a crash and affects tuning only.
- Tuning results computed before the restart (BasinDetective and GridPulse `tune_dev`) are kept. They used the same
  TabPFN settings, without the memory flags.

## 10. Amendment A2 (2026-11-18 19:05, before any freeze)

The first jobs under A1 failed inside TabPFN's forward pass with an allocation error ("could not create a primitive")
at R_max contexts. The harness process carries more than the bare probe did. So:

- The cap is now 2 GiB (`prlimit --as=2147483648`).
- The first test batch is sized from a 180k-cell budget, with at least 64 rows.
- Any allocation failure halves the batch, down to 16 rows.

R_max and every other A1 rule are unchanged.

## 11. Amendment A3 (2026-11-18 19:20, before any freeze)

The GridPulse 2,048-row G-sim step ran for 12 minutes without finishing one target. Under the 2 GiB cap, a 2,048-row
context leaves room for test batches of only about 64 rows, and every batch re-encodes the whole context. That made a
2,048-row context about 10 times slower per prediction than a 1,024-row one. The exam predictions would have taken
hours per model. This is decided on cost alone; no 2,048-row score was seen. So:

- **G-sim uses 1,024 simulated rows for every model.** The 2,048 step is dropped.
- **G-sim+dev uses at most G-sim's 1,024 simulated rows** (the first n of them), plus the dev rows, within R_max(F + 1)
  as in A1. Resulting contexts:

  | Model | Simulated rows | Dev rows |
  |---|---|---|
  | GridPulse | 1,024 | 8 |
  | BasinDetective | 1,024 | 52 |
  | PlantDetective | 964 | 60 |
  | BDG2 | 1,024 | 52 |
  | RECS | 712 | 1,848 |
  | EnergyNet | 256 | 896 (subsampled, A1) |

- The first test batch is sized from the 210k-cell budget again (A2 had 180k). The A2 halving fallback is kept.

## 12. GridPulse  (`gridpulse`)

GridPulse (grid platform) adapter, GB events. Read-only use of /home/claude/grid/gridpulse and shared griddata.

Rows = (event, N) with N = seconds of frequency seen after onset, N in {10, 60} (the exam's inertia rows).
Features = the model's own 86 inputs (dataset.features with ctx: window, is_gb, N, demand).
Targets = the model's own log_loss (MW) and log_E (inertia, MW s).
Dev = the 4 GB dev events (GB_DEV); exam = the 412 GB exam events (410 with NESO inertia; loss labels by tier).
G-sim context = simulated GB events from the release's banks (train_p1 + train_p1b) built exactly as train.py builds
an epoch (DS.make: phase, real quiet noise, onset; demand x N(1, 0.03), 25 % unknown), N fixed to 10 or 60 per row.

- Feature table: 86 columns. Dev rows: 8. Exam rows: 824 (412 bootstrap units).
- Fitted targets: `log_E` (reg), `log_loss` (reg).
- Scored items (metric): inertia N=10 (mape); inertia N=60 (mape); MW lost medium N=60 (mape); MW lost medium N=10 (mape); MW lost low N=60 (mape); MW lost low N=10 (mape).
- Headline: mean of MdAPE (%) over GB inertia at N=60 s, inertia at N=10 s and MW lost (medium-confidence labels, N=60) (negated).
- Bootstrap unit: per exam unit; 412 units.
- Card baselines re-scored beside the arms: constant 230 GW s / dev-median loss (dumb), k x demand / RoCoF with 230 GW s (classical).
- C-dev: no (the card has no dev-fitted classical learner).
- LC k values: none (dev too small).

**Tuning results (dev and frozen simulated set only):**

- P on dev: headline -17.7326. P on the 400 frozen-sim rows: -14.5519.
- G-dev cross-validation inside dev: raw -26.7374; norm -26.5889 -> **norm**.
- G-sim on the frozen simulated set: raw@1024 -15.7067; norm@1024 -15.6176 -> **norm @ 1024 rows**.


## 13. BasinDetective  (`basindetective`)

BasinDetective (Watershed phase 2) adapter. Read-only use of /home/claude/water/detective.

Rows = CAMELS-US basins. Features = the model's own ~150-feature vector (feats/real_*.npz, bank shards).
Targets = the six out-of-period change targets (real truth) + regulated (dor > 0.1), exactly the model's dev composite.
Dev: 52 dev basins, calibration 1999-2008 -> early window 1980-89 (the model's own dev protocol).
Exam: 619 basins, 1999-2008 -> 1989-99.

- Feature table: 150 columns. Dev rows: 52. Exam rows: 619 (619 bootstrap units).
- Fitted targets: `d_logq` (reg), `d_rr` (reg), `d_bfi` (reg), `d_logq5` (reg), `d_logq95` (reg), `d_hfd` (reg), `reg` (clf).
- Scored items (metric): d_logq (rho, mdae); d_rr (rho, mdae); d_bfi (rho, mdae); d_logq5 (rho, mdae); d_logq95 (rho, mdae); d_hfd (rho, mdae); reg (auc).
- Headline: mean over the 6 change targets of Spearman rho and (2 AUC - 1) for regulated (= the model's dev composite).
- Bootstrap unit: per basin; 619 units.
- Card baselines re-scored beside the arms: GR4J model-implied (classical), no change (dumb).
- C-dev: no (the card has no dev-fitted classical learner).
- LC k values: [5, 10, 20, 40].

**Tuning results (dev and frozen simulated set only):**

- P on dev: headline 0.4515. P on the 400 frozen-sim rows: 0.7064.
- G-dev cross-validation inside dev: raw 0.3348; norm 0.3394 -> **raw**.
- G-sim on the frozen simulated set: raw@1024 0.6238; norm@1024 0.6272 -> **raw @ 1024 rows**.


## 14. BuildingDetective (RECS, annual)  (`bldg_recs`)

BuildingDetective, RECS homes from annual totals (12 features; the released 'annual' network).

Targets = floor area (log10 m2), year built (sim: exact; RECS: midpoint of the survey's range, as the model's own
evaluate.eval_recs), heating group (electric = heat pump or resistance, fuel = fuel or district, none). Dev = the
1,848 RECS dev respondents; exam = the 7,401 RECS exam respondents. Sim pool = bank f2 annual shards 0-59 (R9 bank);
frozen sim set = shards 900-904.

- Feature table: 12 columns. Dev rows: 1848. Exam rows: 7401 (7401 bootstrap units).
- Fitted targets: `logA` (reg), `year` (reg), `heatgrp` (multi).
- Scored items (metric): floor area (mape); heating fuel vs electric (acc); vintage (mae).
- Headline: mean of floor-area MdAPE (%, negated) and heating-group accuracy (%); vintage MAE reported, not in the headline.
- Bootstrap unit: per exam unit; 7401 units.
- Card baselines re-scored beside the arms: none.
- C-dev: yes (ridge / logistic, section 4).
- LC k values: [5, 10, 20, 40, 80, 160, 320].

**Tuning results (dev and frozen simulated set only):**

- P on dev: headline 26.9982. P on the 400 frozen-sim rows: 27.2161.
- G-dev cross-validation inside dev: raw 31.2935; norm 31.3584 -> **norm**.
- C-dev cross-validation inside dev: headline 30.9072.
- G-sim on the frozen simulated set: raw@1024 25.8341; norm@1024 25.7067 -> **raw @ 1024 rows**.


## Amendment A4 (2026-11-18 21:25, after the GridPulse, BasinDetective and RECS freezes, before their exam scores)

BasinDetective's exam_predict failed at its last main-arm fit: the G-sim+dev classifier for `reg`, with 1,024 simulated
rows, 52 dev rows and 151 columns. It failed even at a 16-row test batch with an allocation error under the 2 GiB cap.
This was a memory failure on exam inputs only; no exam label or score had been looked at. Rule, for every model from
now on:

- If a fit fails at a 16-row batch, the simulated part of its context is cut to 7/8 by a seeded draw (seed + 100 +
  attempt) and the fit is retried, up to 5 times. Dev rows are always kept.
- For a context with no simulated part, the whole context is cut instead. This applies to G-dev and LC, which are far
  smaller than R_max, so it is not expected to trigger.
- Every cut is printed in the model's exam_predict log and reported in RESULTS.md.

Already-saved predictions are not recomputed.

## 15. BuildingDetective (BDG2, monthly)  (`bldg_bdg2`)

BuildingDetective, BDG2 buildings from monthly meter input (112 features; the released 'monthly' network).

Targets = floor area (log10 m2), year built, primary use (12 simulator classes; residential_* read as lodging, as in
the model's own evaluate.eval_bdg2). Dev = the 52 BDG2 dev buildings; exam = the 1,428 BDG2 exam buildings with a usable
2017 year. Sim pool = bank f2 shards 0-59 (the release R9 bank); frozen sim set = shards 900-904.

- Feature table: 112 columns. Dev rows: 52. Exam rows: 1428 (1428 bootstrap units).
- Fitted targets: `logA` (reg), `year` (reg), `use` (multi).
- Scored items (metric): floor area (mape); primary use (acc); year built (mae).
- Headline: mean of floor-area MdAPE (%, negated) and primary-use accuracy (%); year built reported, not in the headline.
- Bootstrap unit: per exam unit; 1428 units.
- Card baselines re-scored beside the arms: none.
- C-dev: yes (ridge / logistic, section 4).
- LC k values: [5, 10, 20, 40].

**Tuning results (dev and frozen simulated set only):**

- P on dev: headline -17.3249. P on the 400 frozen-sim rows: 19.3870.
- G-dev cross-validation inside dev: raw -3.8493; norm -5.7840 -> **raw**.
- C-dev cross-validation inside dev: headline -15.5499.
- G-sim on the frozen simulated set: raw@1024 15.0248; norm@1024 15.1446 -> **norm @ 1024 rows**.


## 16. PlantDetective  (`plantdetective`)

PlantDetective (grid platform) adapter. Read-only use of /home/claude/grid/plantdetective (release/training code).

Rows = (unit, calendar-year window) read in the model's 'out' mode (hourly output only; the exam's second mode), built
exactly as training/exam.py builds its 'year' windows. The 'ctx' mode cannot be rebuilt here: its EIA-860 generator
table (PUDL out_eia__yearly_generators.parquet) is no longer on this machine. Features = the model's 171 inputs.
Targets = the model's own transformed targets for min stable load (logit), ramp up p99 (log), starts per year (log1p),
full-load heat rate (log), and the technology class (4). Truth = the unit's whole-record CAMPD truth (exam: 2018-2025;
dev: dev years 2019 and 2021, as the model's dev gate).
Dev = the 40 dev units' year windows; exam = the exam units' year windows (the exam's 'from one calendar year' rows),
a fixed subsample of at most 2 windows per unit (seed 20261118) for the CPU budget; P is scored on the same rows.
Sim pool = the release bank's training windows (set_v2.npz, V = False) of a full year (W = 365 days); frozen simulated
set = set_v2.npz validation windows (V = True), W = 365.

- Feature table: 171 columns. Dev rows: 60. Exam rows: 781 (395 bootstrap units).
- Fitted targets: `pmin_frac` (reg), `ramp_up` (reg), `starts_yr` (reg), `hr_full` (reg), `tech` (multi).
- Scored items (metric): min stable load (mae_med); ramp up p99 (mape); starts per year (mape); full-load heat rate (mape); technology (acc).
- Headline: mean over min stable load (median abs error, points of pmax), ramp-up p99, starts per year and full-load heat rate (median relative error, %; starts as |(p+1)/(t+1) - 1|, as the card) - all negated - and technology accuracy (%).
- Bootstrap unit: per unit; 395 units.
- Card baselines re-scored beside the arms: .
- C-dev: no (the card has no dev-fitted classical learner).
- LC k values: [5, 10, 20, 40].

**Tuning results (dev and frozen simulated set only):**

- P on dev: headline 5.6352. P on the 400 frozen-sim rows: 13.3864.
- G-dev cross-validation inside dev: raw 12.9242; norm 13.3013 -> **norm**.
- G-sim on the frozen simulated set: raw@1024 17.9283; norm@1024 17.9623 -> **norm @ 1024 rows**.


## 17. EnergyNet  (`energynet`)

EnergyNet (vehicle platform) adapter. Read-only use of /home/claude/road/energynet.

Rows = trips. Features = the model's own assembled input vector (dataset.assemble with the record's tiers).
Target = the model's own: log(true energy / physics estimate E_phys).
Exam = VED exam trips (the headline rows; EPA/heavy-duty exam sets are not used here), a FIXED subsample of at most
8 trips per exam car (seed 20261118) to fit the CPU budget; P is scored on the same subsample.
Dev = the 16 VED dev cars' 1,498 trips (VED dev protocol). Bootstrap and dev CV are grouped by car.

- Feature table: 149 columns. Dev rows: 1496. Exam rows: 2549 (350 bootstrap units).
- Fitted targets: `lr` (reg).
- Scored items (metric): ICE (mape); HEV (mape); PHEV (mape); EV (mape); all trips (mape); log ratio (rho, mdae).
- Headline: mean of the per-trip MdAPE (%) over the ICE, HEV and PHEV exam groups (negated: higher is better); EV (2 Leafs) reported, not in the headline.
- Bootstrap unit: per exam unit; 350 units.
- Card baselines re-scored beside the arms: label fuel economy (classical), VSP MOVES-like (classical), class median (dumb), physics estimate alone.
- C-dev: no (the card has no dev-fitted classical learner).
- LC k values: [5, 10, 20, 40, 80, 160, 320].

**Tuning results (dev and frozen simulated set only):**

- P on dev: headline -13.5604. P on the 400 frozen-sim rows: -14.2441.
- G-dev cross-validation inside dev: raw -14.8783; norm -14.1385 -> **norm**.
- G-sim on the frozen simulated set: raw@1024 -14.6829; norm@1024 -15.0614 -> **raw @ 1024 rows**.

