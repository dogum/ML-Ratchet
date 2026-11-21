# Prior Wars: results

ML Ratchet method experiment C1, run 18-19 Nov 2026. Built with PriorLabs-TabPFN.

Every ML Ratchet model learns only from a domain simulator. TabPFN also learned only from synthetic data, but from a
generic prior over random causal models, and it predicts by reading a table in context. On six of the collection's
real exams, we compared the released simulation-trained model (**P**) with TabPFN-v2 given:

- the model's real dev rows (**G-dev**);
- the model's own simulated rows (**G-sim**);
- both (**G-sim+dev**).

Everything was pre-registered and frozen per model before its exam was predicted (`PREREG.md`, `work/prereg_log.json`).

## Verdict

**Given the dev labels, the generic prior beats the domain simulator on 3 of the 6 exams, ties on 2 and loses on 1.**
This is the pre-registered C1 count. The pre-registered sentence for it is "the generic prior beats the domain
simulator", but the count is the result.

- **C1: P vs G-dev.** P wins on BasinDetective. It ties on GridPulse and EnergyNet. It loses on BuildingDetective
  RECS, BuildingDetective BDG2 and PlantDetective.
- **C2: P vs G-sim** (the same simulator, 0 real labels each). P wins on 4 of 6: GridPulse, BDG2, PlantDetective and
  EnergyNet. It ties on BasinDetective and loses on RECS. With equal (zero) real labels, the purpose-built network
  usually reads its own simulator better than a generic in-context learner given 1,024 of its rows.
- **C3: G-sim+dev vs G-dev** (does the simulator, as data, add to the real labels?). It helps on BasinDetective only,
  ties on 3 and hurts on 2 (BDG2 and PlantDetective). Where the simulator's mismatch to reality is large, mixing its
  rows into the context passes the mismatch on.
- **k\*: how many real labels G-dev needs to reach P.** About 20 (BDG2), 40 (RECS, PlantDetective) and 1,152
  (EnergyNet, the full capped dev set). It is out of reach on GridPulse (8 dev rows) and BasinDetective (52 dev
  basins).

In one line: **a simulator is worth tens of real labels, not hundreds.** With 0 labels the physics models are the
best 0-label arm on 5 of 6 exams. But on 3 of the 6, TabPFN with the few dozen dev labels already used for each
model's dev gate beats them clearly.

## What this does and does not show

- **It shows** that on these six exams, a generic in-context learner given the dev set beats or matches the released
  physics-trained model on 5 of 6. The dev set is the handful of real units each model already uses for its gate. So
  the dev gate's labels are worth far more as training context than as a gate.
- **It shows** that the simulator is not wasted. P is the best zero-label arm on 5 of 6 exams. It wins BasinDetective
  outright, and it stays level on GridPulse and EnergyNet against TabPFN with 8 and 1,152 real labels.
- **It does not show** that simulation-only training is worse in general. C1 is unequal by design: k real labels
  against 0. The repository's rule (no real label trains a model) is what makes P's exam numbers clean. G-dev has no
  such claim: it has seen dev, and dev is drawn from the same population as the exam.
- **It does not test TabPFN at its best.** It used TabPFN-v2 (not 2.5 or later), n_estimators = 2, one CPU core,
  contexts of at most 1,024-2,560 rows, and a 2 GiB cap. Read every G arm as a lower bound.
- **It does not compare equal data in C2.** G-sim gets 1,024 simulated rows, and P trained on 12,000-29,000 or more
  rows of the same simulator.
- **Scope:** six tabular exams across four platforms (grid, water, buildings, vehicles). The signal models are not
  in it.

## Cross-collection summary

Headlines are per model and in that model's own units (errors negated, so higher is better). A C1/C2 verdict is stated for P; C3 for adding simulated rows to dev.

| Model | Dev k | P | G-dev | G-sim | G-sim+dev | C1 (P vs G-dev) | C2 (P vs G-sim) | C3 (G-sim+dev vs G-dev) | k* |
|---|---|---|---|---|---|---|---|---|---|
| GridPulse | 8 | -18.965 | -20.095 | -21.527 | -20.854 | P tie | P win | G-sim+dev tie | > 8 |
| BasinDetective | 52 | 0.452 | 0.367 | 0.439 | 0.447 | P win | P tie | G-sim+dev win | > 52 |
| BuildingDetective (RECS, annual) | 1848 | 26.830 | 32.121 | 27.266 | 31.896 | P loss | P loss | G-sim+dev tie | 40 |
| BuildingDetective (BDG2, monthly) | 52 | -17.638 | -4.965 | -20.396 | -8.063 | P loss | P win | G-sim+dev loss | 20 |
| PlantDetective | 60 | -1.843 | 4.026 | -5.168 | 1.996 | P loss | P win | G-sim+dev loss | 40 |
| EnergyNet | 1152 | -14.369 | -13.219 | -17.754 | -13.806 | P tie | P win | G-sim+dev tie | 1152 |

## Where each arm wins

| Model | Best arm (headline) | Best with 0 real labels | Best with k real labels | Items where each arm is best (P / G-dev / G-sim / G-sim+dev) |
|---|---|---|---|---|
| GridPulse | P | P | G-dev | 3 / 2 / 0 / 1 |
| BasinDetective | P | P | G-sim+dev | 4 / 0 / 1 / 2 |
| BuildingDetective (RECS, annual) | G-dev | G-sim | G-dev | 0 / 3 / 0 / 0 |
| BuildingDetective (BDG2, monthly) | G-dev | P | G-dev | 1 / 2 / 0 / 0 |
| PlantDetective | G-dev | P | G-dev | 0 / 2 / 0 / 3 |
| EnergyNet | G-dev | P | G-dev | 2 / 2 / 0 / 2 |

## Per model

The primary metric of every scored item, with the card's own baselines rebuilt on the same rows. Contrasts are paired over the same bootstrap resamples; "items won / tied / lost" counts per-item verdicts for the first arm. The learning curve gives the G-dev headline at k random dev rows (mean +/- SD over 3 subsets); the last k is the full G-dev context.

### GridPulse

Only 8 dev rows (4 events x 2 windows), so there is no learning curve. P reproduces the card (inertia MdAPE 14.5 % at N = 10 s, 16.4 % at N = 60 s). G-dev cannot learn inertia from 4 events, but it does well on MW lost at medium confidence. G-sim reads the same simulated events worse than the purpose-built network.

Exam: 824 rows, 412 bootstrap units. Dev: 8 rows. Headline: mean of MdAPE (%) over GB inertia at N=60 s, inertia at N=10 s and MW lost (medium-confidence labels, N=60) (negated).

| Arm | Real labels | inertia N=10 (mape) | inertia N=60 (mape) | MW lost medium N=60 (mape) | MW lost medium N=10 (mape) | MW lost low N=60 (mape) | MW lost low N=10 (mape) | Headline [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| **P** | 0 | 14.5 | 16.4 | 25.9 | 30.8 | 34.8 | 38.4 | -18.965 [-21.838, -17.373] |
| **G-dev** | 8 | 20.4 | 20.5 | 19.4 | 19.4 | 75.7 | 75.2 | -20.095 [-24.166, -17.511] |
| **G-sim** | 0 | 17.3 | 17.7 | 29.6 | 31.1 | 40.3 | 40.1 | -21.527 [-24.507, -19.622] |
| **G-sim+dev** | 8 | 17.2 | 17.9 | 27.5 | 28.6 | 35.1 | 35.9 | -20.854 [-23.976, -19.030] |
| constant 230 GW s / dev-median loss (dumb) | - | 20.4 | 20.4 | 32.7 | 32.7 | 112.0 | 112.0 | -24.508 [-29.608, -20.358] |
| k x demand / RoCoF with 230 GW s (classical) | - | 15.1 | 15.1 | 27.2 | 27.2 | 35.6 | 35.6 | -19.120 [-22.092, -17.167] |

| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |
|---|---|---|---|
| C1: G-dev - P | -1.130 [-4.733, +2.381] | tie | 1 / 1 / 4 |
| C2: G-sim - P | -2.561 [-4.659, -0.475] | loss | 0 / 5 / 1 |
| C3: G-sim+dev - G-dev | -0.758 [-4.442, +2.776] | tie | 3 / 3 / 0 |
| C4: G-sim+dev - P | -1.888 [-4.120, +0.209] | tie | 0 / 5 / 1 |

| k (real dev labels) | 8 | P (0 labels) |
|---|---|---|
| G-dev headline (mean of 3 subsets) | -20.095 | -18.965 |

k\* (fewest labels for G-dev to reach P): **> 8**.

### BasinDetective

Seven targets: the six out-of-period signature changes (Spearman rho) and regulated (AUC). P reproduces the card (rho 0.76 / 0.45 / 0.20 / 0.44 / 0.71 / 0.63). With 52 dev basins, TabPFN on dev alone loses clearly. Adding the 52 dev basins to 1,024 simulated basins recovers almost all of the gap, and it is the only model where the simulator rows help G-dev.

Exam: 619 rows, 619 bootstrap units. Dev: 52 rows. Headline: mean over the 6 change targets of Spearman rho and (2 AUC - 1) for regulated (= the model's dev composite).

| Arm | Real labels | d_logq (rho) | d_rr (rho) | d_bfi (rho) | d_logq5 (rho) | d_logq95 (rho) | d_hfd (rho) | reg (auc) | Headline [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|
| **P** | 0 | 0.757 | 0.453 | 0.203 | 0.440 | 0.709 | 0.630 | 0.485 | 0.452 [0.408, 0.490] |
| **G-dev** | 52 | 0.689 | 0.436 | 0.146 | 0.427 | 0.521 | 0.375 | 0.488 | 0.367 [0.327, 0.412] |
| **G-sim** | 0 | 0.717 | 0.445 | 0.208 | 0.428 | 0.685 | 0.560 | 0.516 | 0.439 [0.400, 0.477] |
| **G-sim+dev** | 52 | 0.735 | 0.452 | 0.243 | 0.503 | 0.615 | 0.580 | 0.500 | 0.447 [0.405, 0.488] |
| GR4J model-implied (classical) | - | 0.793 | 0.499 | 0.232 | 0.421 | 0.722 | 0.565 | 0.433 | 0.443 [0.404, 0.478] |
| no change (dumb) | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 [0.000, 0.000] |

| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |
|---|---|---|---|
| C1: G-dev - P | -0.085 [-0.124, -0.042] | loss | 0 / 4 / 3 |
| C2: G-sim - P | -0.013 [-0.037, +0.013] | tie | 0 / 5 / 2 |
| C3: G-sim+dev - G-dev | +0.080 [+0.044, +0.112] | win | 5 / 2 / 0 |
| C4: G-sim+dev - P | -0.005 [-0.035, +0.025] | tie | 1 / 4 / 2 |

| k (real dev labels) | 5 | 10 | 20 | 40 | 52 | P (0 labels) |
|---|---|---|---|---|---|---|
| G-dev headline (mean of 3 subsets) | 0.105 +/- 0.075 | 0.183 +/- 0.071 | 0.269 +/- 0.020 | 0.339 +/- 0.008 | 0.367 | 0.452 |

k\* (fewest labels for G-dev to reach P): **> 52**.

### BuildingDetective (RECS, annual)

1,848 RECS dev homes is a large dev set for a 12-number input. P reproduces the card (floor area 33.6 %, heating group 87.3 %). Here the generic prior with dev labels wins clearly, and so does the card's own dev-fitted ridge/logistic baseline (C-dev). G-sim with 1,024 simulated homes slightly beats the released network.

Exam: 7,401 rows, 7,401 bootstrap units. Dev: 1,848 rows. Headline: mean of floor-area MdAPE (%, negated) and heating-group accuracy (%); vintage MAE reported, not in the headline.

| Arm | Real labels | floor area (mape) | heating fuel vs electric (acc) | vintage (mae) | Headline [95 % CI] |
|---|---|---|---|---|---|
| **P** | 0 | 33.6 | 87.3 | 22.1 | 26.830 [26.235, 27.301] |
| **G-dev** | 1848 | 27.9 | 92.1 | 21.3 | 32.121 [31.614, 32.531] |
| **G-sim** | 0 | 34.0 | 88.6 | 22.1 | 27.266 [26.754, 27.924] |
| **G-sim+dev** | 1848 | 28.0 | 91.8 | 21.5 | 31.896 [31.430, 32.469] |
| **C-dev** | 1848 | 29.7 | 91.9 | 21.8 | 31.124 [30.646, 31.602] |

| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |
|---|---|---|---|
| C1: G-dev - P | +5.291 [+4.842, +5.792] | win | 3 / 0 / 0 |
| C2: G-sim - P | +0.437 [+0.160, +0.851] | win | 1 / 2 / 0 |
| C3: G-sim+dev - G-dev | -0.225 [-0.350, +0.035] | tie | 0 / 1 / 2 |
| C4: G-sim+dev - P | +5.066 [+4.666, +5.658] | win | 3 / 0 / 0 |

| k (real dev labels) | 5 | 10 | 20 | 40 | 80 | 160 | 320 | 1848 | P (0 labels) |
|---|---|---|---|---|---|---|---|---|---|
| G-dev headline (mean of 3 subsets) | 18.811 +/- 5.463 | 19.834 +/- 4.565 | 26.461 +/- 1.482 | 28.267 +/- 0.840 | 28.933 +/- 0.370 | 30.027 +/- 0.060 | 31.210 +/- 0.142 | 32.121 | 26.830 |

k\* (fewest labels for G-dev to reach P): **40**.

### BuildingDetective (BDG2, monthly)

The BuildingDetective network reads use type badly on real buildings (14 %; card 14.4 %). With the 52 dev buildings, TabPFN doubles that (28.5 %) and cuts floor-area error from 49 % to 38 %. Mixing in simulated buildings hurts: the simulator's use-type mismatch is passed on to the in-context learner.

Exam: 1,428 rows, 1,428 bootstrap units. Dev: 52 rows. Headline: mean of floor-area MdAPE (%, negated) and primary-use accuracy (%); year built reported, not in the headline.

| Arm | Real labels | floor area (mape) | primary use (acc) | year built (mae) | Headline [95 % CI] |
|---|---|---|---|---|---|
| **P** | 0 | 49.4 | 14.1 | 25.2 | -17.638 [-19.754, -15.975] |
| **G-dev** | 52 | 38.4 | 28.5 | 26.5 | -4.965 [-6.454, -3.372] |
| **G-sim** | 0 | 47.7 | 6.9 | 25.8 | -20.396 [-21.617, -18.662] |
| **G-sim+dev** | 52 | 42.9 | 26.8 | 25.7 | -8.063 [-9.591, -6.186] |
| **C-dev** | 52 | 70.4 | 28.1 | 36.3 | -21.159 [-24.308, -17.859] |

| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |
|---|---|---|---|
| C1: G-dev - P | +12.673 [+10.609, +15.156] | win | 2 / 0 / 1 |
| C2: G-sim - P | -2.758 [-4.346, -0.916] | loss | 0 / 2 / 1 |
| C3: G-sim+dev - G-dev | -3.098 [-4.619, -1.384] | loss | 0 / 2 / 1 |
| C4: G-sim+dev - P | +9.575 [+7.650, +11.702] | win | 2 / 1 / 0 |

| k (real dev labels) | 5 | 10 | 20 | 40 | 52 | P (0 labels) |
|---|---|---|---|---|---|---|
| G-dev headline (mean of 3 subsets) | -32.836 +/- 14.868 | -27.718 +/- 14.413 | -4.837 +/- 2.952 | -8.603 +/- 2.174 | -4.965 | -17.638 |

k\* (fewest labels for G-dev to reach P): **20**.

### PlantDetective

Read from a year of hourly output alone (the card's 'out' mode; the 'ctx' mode's EIA-860 table is not on this machine), on 2 year-windows per exam unit. With 60 dev windows from 40 units, TabPFN halves P's heat-rate error (9.8 % vs 21.3 %) and lifts technology accuracy from 69 % to 76 %. G-sim reads simulated plants into the real ones badly for technology (47 %). Adding the simulated rows to dev helps min load, ramp and starts, and hurts heat rate and technology.

Exam: 781 rows, 395 bootstrap units. Dev: 60 rows. Headline: mean over min stable load (median abs error, points of pmax), ramp-up p99, starts per year and full-load heat rate (median relative error, %; starts as |(p+1)/(t+1) - 1|, as the card) - all negated - and technology accuracy (%).

| Arm | Real labels | min stable load (mae_med) | ramp up p99 (mape) | starts per year (mape) | full-load heat rate (mape) | technology (acc) | Headline [95 % CI] |
|---|---|---|---|---|---|---|---|
| **P** | 0 | 6.9 | 20.8 | 28.9 | 21.3 | 68.6 | -1.843 [-2.991, -0.878] |
| **G-dev** | 60 | 4.3 | 15.9 | 26.4 | 9.8 | 76.4 | 4.026 [3.034, 5.067] |
| **G-sim** | 0 | 4.4 | 14.6 | 29.4 | 24.7 | 47.2 | -5.168 [-6.849, -3.795] |
| **G-sim+dev** | 60 | 3.6 | 14.5 | 25.0 | 15.1 | 68.2 | 1.996 [0.883, 3.129] |

| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |
|---|---|---|---|
| C1: G-dev - P | +5.870 [+4.931, +7.113] | win | 4 / 1 / 0 |
| C2: G-sim - P | -3.324 [-4.525, -1.959] | loss | 2 / 1 / 2 |
| C3: G-sim+dev - G-dev | -2.031 [-3.003, -1.090] | loss | 1 / 2 / 2 |
| C4: G-sim+dev - P | +3.839 [+2.958, +4.896] | win | 4 / 1 / 0 |

| k (real dev labels) | 5 | 10 | 20 | 40 | 60 | P (0 labels) |
|---|---|---|---|---|---|---|
| G-dev headline (mean of 3 subsets) | -16.302 +/- 8.749 | -5.258 +/- 2.592 | -1.860 +/- 1.427 | 3.974 +/- 0.152 | 4.026 | -1.843 |

k\* (fewest labels for G-dev to reach P): **40**.

### EnergyNet

VED exam trips, 8 per car (2,549 trips, 350 cars); P is scored on the same trips. P's ICE error here is 9.0 % (card, all 14,469 ICE trips: 8.6 %). With 1,152 dev trips from 16 cars, TabPFN ties P overall. It is better on hybrids (12.3 % vs 16.8 %), level on PHEVs and slightly worse on petrol cars. It takes the full dev context to draw level: at k = 320 it is still below P. Every learned arm beats the card's classical baselines, whose mean group MdAPE is 25.8 % (label economy) and 24.3 % (MOVES-like).

Exam: 2,549 rows, 350 bootstrap units. Dev: 1,496 rows. Headline: mean of the per-trip MdAPE (%) over the ICE, HEV and PHEV exam groups (negated: higher is better); EV (2 Leafs) reported, not in the headline.

| Arm | Real labels | ICE (mape) | HEV (mape) | PHEV (mape) | EV (mape) | all trips (mape) | log ratio (rho) | Headline [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| **P** | 0 | 9.0 | 16.8 | 17.3 | 22.5 | 10.9 | 0.621 | -14.369 [-16.568, -12.517] |
| **G-dev** | 1152 | 10.0 | 12.3 | 17.3 | 17.4 | 10.9 | 0.648 | -13.219 [-15.297, -11.826] |
| **G-sim** | 0 | 9.1 | 17.9 | 26.3 | 23.7 | 11.1 | 0.593 | -17.754 [-20.818, -16.025] |
| **G-sim+dev** | 1152 | 9.5 | 13.3 | 18.7 | 18.6 | 10.6 | 0.686 | -13.806 [-15.641, -11.979] |
| label fuel economy (classical) | - | 12.4 | 17.5 | 47.4 | 14.0 | 14.6 | 0.483 | -25.801 [-28.646, -23.298] |
| VSP MOVES-like (classical) | - | 10.6 | 16.2 | 46.2 | 12.9 | 12.8 | 0.563 | -24.345 [-25.698, -22.166] |
| class median (dumb) | - | 23.9 | 18.2 | 45.6 | 16.5 | 22.8 | 0.158 | -29.238 [-32.034, -25.710] |
| physics estimate alone | - | 29.0 | 15.5 | 20.0 | 12.2 | 25.5 | 0.000 | -21.528 [-24.061, -20.064] |

| Contrast | Headline difference [95 % CI] | Verdict (for the first arm) | Items won / tied / lost |
|---|---|---|---|
| C1: G-dev - P | +1.150 [-0.196, +2.506] | tie | 2 / 3 / 1 |
| C2: G-sim - P | -3.385 [-6.150, -1.499] | loss | 0 / 4 / 2 |
| C3: G-sim+dev - G-dev | -0.587 [-1.373, +0.615] | tie | 1 / 5 / 0 |
| C4: G-sim+dev - P | +0.563 [-0.509, +2.030] | tie | 3 / 3 / 0 |

| k (real dev labels) | 5 | 10 | 20 | 40 | 80 | 160 | 320 | 1152 | P (0 labels) |
|---|---|---|---|---|---|---|---|---|---|
| G-dev headline (mean of 3 subsets) | -20.578 +/- 2.139 | -21.083 +/- 4.456 | -19.128 +/- 1.082 | -17.779 +/- 1.831 | -16.145 +/- 0.847 | -15.303 +/- 0.084 | -14.589 +/- 1.145 | -13.219 | -14.369 |

k\* (fewest labels for G-dev to reach P): **1152**.

## Scope: models attempted and skipped

| Model | Status | Why |
|---|---|---|
| GridPulse (GB events) | **done** | Features, dev events and banks are local. Only 8 dev rows, so there is no learning curve. |
| BasinDetective (Watershed) | **done** | All data local. |
| BuildingDetective, RECS (annual) | **done** | 1,848 dev homes and a C-dev baseline. |
| BuildingDetective, BDG2 (monthly) | **done** | 52 dev buildings and a C-dev baseline. |
| PlantDetective ('out' mode) | **done** | Run on a fixed exam subsample of 2 year-windows per unit. The 'ctx' mode was skipped because its EIA-860 generator table is not on this machine. |
| EnergyNet (VED trips) | **done** | Exam subsampled to 8 trips per car; G-dev context capped at 1,152 of 1,496 trips. EPA and heavy-duty exams not used. |
| DemandCast, TripDetective, LoadCast, FaultFinder, CellSense, TransitNet vetting, BeatNet, BearingNet | skipped | Their real data and simulated banks are not on this machine. Re-fetching and regenerating the banks did not fit the CPU, memory and disk budget of a machine shared with two other agents, after the first hour's memory incident. |

## Protocol in brief

- **Arms** (PREREG section 2):
  - **P** is the released model, rerun from its shipped weights. P reproduces each card's exam numbers. For example: GridPulse inertia 14.5 % and 16.4 %; BasinDetective rho 0.76 / 0.45 / 0.20 / 0.44 / 0.71 / 0.63; RECS floor area 33.6 % and heating group 87.3 %; BDG2 monthly use 14.1 % (card 14.4 %).
  - **G-dev** is TabPFN-v2 with the model's real dev rows in context.
  - **G-sim** is TabPFN-v2 with 1,024 simulated rows from the model's own training bank.
  - **G-sim+dev** is G-sim's simulated rows plus every dev row, with an `is_real` column.
  - **C-dev** is a ridge or logistic regression on dev, only where the card already had a dev-fitted classical baseline (the BuildingDetective adapters).
  - **LC** is G-dev with k random dev rows: 3 subsets per k, k in {5, 10, 20, 40, 80, 160, 320} below the dev size.
- **TabPFN:**
  - `tabpfn` 10.2.0 with the TabPFN-v2 checkpoints, on CPU with one thread, n_estimators = 2 and random_state = 0.
  - One fit per target. Regression predicts the mean of the predictive distribution.
- **Settings, chosen without the exam** (PREREG section 4):
  - G-dev preprocessing (raw or the model's own normalisation): 5-fold CV inside dev.
  - G-sim preprocessing: 400 rows of the model's frozen simulated validation set.
  - The rule each time is "raw unless norm wins by more than 0.01".
- **Scoring:**
  - Each model's own exam metrics, on its own exam rows.
  - A paired bootstrap over the exam's units: B = 1,000, seed 0, the same resamples for every arm.
  - A verdict is a **win** or **loss** when the 95 % interval of the headline difference excludes zero, and a **tie** otherwise.
  - Every contrast is paired on the same resamples. C3 uses the saved bootstrap replicates.
- **Freezing:**
  - Each model's PREREG section was appended and the whole file hashed (`work/prereg_log.json`) before its exam predictions.
  - `harness.py exam_predict` and `score` refuse to run if that hash no longer matches.
  - `score` refuses a second run.
  - Freeze times (EST): GridPulse 18 Nov 19:18; BasinDetective 20:03; RECS 21:13; BDG2 21:30; PlantDetective 19 Nov 01:00; EnergyNet 04:03.

## Amendments (all in PREREG.md, all fixed by memory or CPU cost, none by a label)

- **A1 (before any freeze).** The first runs had several TabPFN processes on the shared 6.27 GB machine at once and starved two other agents. The fix:
  - one TabPFN process at a time, under an address-space cap;
  - context rows capped at R_max(F) = min(2,560, floor((210,000 / (F + 8) - 128) / 64) x 64);
  - EnergyNet's G-dev context capped at 1,152 of its 1,496 dev trips;
  - PlantDetective's exam subsampled to at most 2 year-windows per unit (781 rows, all 395 units; P scored on the same rows);
  - the GridPulse frozen-sim scoring fix.
- **A2 (before any freeze).** The cap became 2 GiB, and the test batch now halves on an allocation failure.
- **A3 (before any freeze).** The 2,048-row G-sim step was dropped everywhere: under the cap it was about 10 times slower per prediction. G-sim+dev takes at most 1,024 simulated rows.
- **A4 (after the GridPulse, BasinDetective, RECS and BDG2 freezes; before any of their scores).** If a fit fails even at a 16-row batch, its simulated rows are cut to 7/8 and the fit is retried. **It never triggered.**
  - The failure that prompted it was address space creeping up across fits in one long-lived process.
  - That was fixed in code with no change in predictions: each fit now runs in a forked child that exits afterwards.

Implementation fixes with no protocol change, all made before the affected model was scored:

- The GridPulse `k x demand` baseline used per-event demand on per-row data. This raised a shape error at score time, before anything was written.
- EnergyNet's frozen simulated set lacked the columns its group items need. This raised a KeyError in tuning.
- An LC subset with fewer than 2 known labels for a target predicts that label as a constant. This happened for BDG2 year built at k = 5, which is not in the headline.

## Licences

- **`tabpfn` 10.2.0 (code):** Apache-2.0.
- **TabPFN-v2 weights** (`Prior-Labs/TabPFN-v2-reg`, `Prior-Labs/TabPFN-v2-clf`, downloaded without a login): Prior Labs License 1.1. This is Apache 2.0 plus paragraph 10, which says that anything made available that uses the weights must keep a copy of the licence and prominently display "Built with PriorLabs-TabPFN".
  - The weights are not redistributed here.
  - The write-up and the page carry the attribution line.
  - No model was trained or distilled from TabPFN outputs, so the "TabPFN" model-name clause does not apply.
- **TabPFN 2.5, 3 and 3.5 (the package default):** not used. Their weights are non-commercial, and the client requires a Prior Labs account token and licence acceptance in a browser before downloading. Per the brief, this was recorded and not worked around.
- **ML Ratchet models, banks and real data:** under their existing licences (`DATA_LICENSES.md`). Nothing new was fetched.

## Rule notes

- **The exam chose nothing.**
  - Every TabPFN setting came from dev CV or the frozen simulated set.
  - The exam was predicted only after the model's PREREG section was hashed, and scored once.
  - P's exam numbers were public before this started; that is section 1 of PREREG.
- **No real label trained any released model.** TabPFN is not trained here, so its arms predict in context. G-dev and G-sim+dev see k dev labels, and every table says so.
- **The shared machine:**
  - After the first hour, one TabPFN process ran at a time, under a 2 GiB address-space cap, at `nice -n 10`, on one thread.
  - Each launch waited on the shared group's memory, and processes were stopped only by their own PIDs.
  - Disk use was about 0.63 GB. Nothing was pushed, and nothing was written to the claude.ai project.
