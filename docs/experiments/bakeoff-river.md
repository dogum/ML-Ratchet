# River Bake-off: one problem, every learner family

*Method experiment, 22-23 Nov 2026. Scripts, arms and raw results: [`experiments/bakeoff-river/`](../../experiments/bakeoff-river/).
Exam run `20261123T074907`, run once by `harness/exam.py` for all 21 registered arms together.*

The Model families page rates which kind of learner fits which problem, and until now each rating was a judgement
call. The River Bake-off runs one problem through nine learner families on equal terms: the same simulated bank, the
same inputs, the same targets, the same dev gate, at most about 4 single-thread CPU-hours and 3 rungs each, zero real
labels in training. Then it grades all of them once, on the same real exam. Only the built-in assumption differs.

**The problem** is FlowCast's: daily streamflow for ungauged CAMELS-US basins (forcing and static attributes only),
plus a gauged nowcast track (observed flow up to `lead` days before, leads 1, 3 and 7). Exam: 619 basins, main
WY1990-99 (Caravan Q) and 603 basins, recent WY2016-23 (NWIS Q). FlowCast v1.0 (an LSTM) is the incumbent; the
harness re-scored it and reproduced its published exam exactly (0.341 main, 0.378 recent; nowcast 0.569 / 0.474 / 0.426).

## Verdict

**Ungauged: the built-in assumption moved median exam NSE by at most 0.05, and five families cannot be separated.**
Gradient-boosted trees (0.345), FlowCast v1.0 (0.341), the same LSTM retrained on the shared bank (0.340), a dilated
TCN (0.338) and an S4D state space (0.336) tie on main; every 95 % interval against FlowCast covers zero. Attention
(0.323), physics in the loop (0.313), formulas (0.306) and the sparse GP (0.297) lose. On recent the retrained LSTM
(0.391) and the trees (0.388) beat FlowCast v1.0 (0.378) by small, clear margins.

**No learner reaches the regionalisation baseline.** All nine lose to U-dev (donor basins' calibrated GR4J; 0.372 /
0.433) by 0.03-0.09, and to NOAA's NWM (0.550 / 0.687) by 0.20-0.34. Every arm learned from the same simulated bank
and read the same ERA5-Land forcing: the ceiling is shared, and it is not the learner.

**Gauged: a Kalman filter on a physical model, never trained on flow, wins at one day.** A 32-member EnKF on WaterSim's
HBV scored 0.599 against FlowCast's nowcast at 0.569 on main (a win; CRPS/Q 0.207 against 0.248) and 0.615 against
0.591 on recent (a tie). It ties at 3 days on main and loses on recent (−0.024), and trails at 7 days (−0.021 tie,
−0.044 loss): its correction sits in the stores and fades as the model runs on alone.

**Conformal calibration equalised coverage, not sharpness.** A Mondrian layer fitted on 52 dev basins brought every
ungauged arm's 90 % band to 0.87-0.89 mean exam coverage (raw 0.87-0.95) and the 50 % band to 0.46-0.49 (raw
0.50-0.67), but CRPS rose in 14 of 18 arm-periods (ties in the other 4). For the EnKF at 7 days it helped both.

## Ungauged leaderboard

Median over basins; difference from FlowCast v1.0 by paired bootstrap over basins (10,000 resamples); win/loss only
when the 95 % interval excludes 0. Coverage is the mean over basins. Main = WY1990-99, 619 basins; recent =
WY2016-23, 603 basins.

| Row (family) | NSE main | vs FlowCast | NSE recent | vs FlowCast | KGE main | \|FHV\| % main | cov90 raw → conformal (main) | CRPS/Q main |
|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees (gbm) | **0.345** | +0.004 tie | 0.388 | +0.010 win | 0.383 | 36.0 | 0.87 → 0.88 | 0.398 |
| FlowCast v1.0, LSTM as released (ssm) | 0.341 | | 0.378 | | 0.367 | 39.1 | 0.95 → 0.88 | 0.392 |
| LSTM, FlowCast net retrained (ssm) | 0.340 | −0.001 tie | **0.391** | +0.013 win | 0.381 | 38.0 | 0.88 → 0.88 | 0.390 |
| TCN, dilated 1-D CNN (cnn1d) | 0.338 | −0.003 tie | 0.382 | +0.004 tie | 0.379 | 37.8 | 0.90 → 0.88 | 0.392 |
| S4D state space + encoder (ssm) | 0.336 | −0.005 tie | 0.386 | +0.008 tie | 0.373 | 38.2 | 0.90 → 0.88 | 0.397 |
| Transformer, daily tokens (attn) | 0.323 | −0.018 loss | 0.360 | −0.018 loss | 0.362 | 36.5 | 0.88 → 0.87 | 0.400 |
| Physics in the loop, HBV (diffsim) | 0.313 | −0.029 loss | 0.366 | −0.012 loss | 0.361 | 36.9 | 0.91 → 0.87 | 0.413 |
| 17 formulas in HBV (symbolic) | 0.306 | −0.035 loss | 0.363 | −0.015 loss | 0.367 | 36.4 | 0.91 → 0.87 | 0.420 |
| Sparse Gaussian process (gp) | 0.297 | −0.044 loss | 0.344 | −0.034 loss | 0.310 | 44.4 | 0.94 → 0.88 | 0.433 |
| Climatology, Budyko-Fu (dumb) | −0.029 | loss | −0.030 | loss | −0.452 | 88.5 | 0.89 | 0.601 |
| *U-dev regionalisation* | 0.372 | +0.031 | 0.433 | +0.054 | 0.433 | 34.3 | point | 0.613 (MAE) |
| *U-k12 regionalisation* | 0.385 | +0.044 | 0.443 | +0.064 | 0.462 | 31.3 | point | 0.612 |
| *GR4J calibrated per basin (gauged calibration)* | 0.473 | +0.132 | 0.517 | +0.139 | 0.602 | 20.3 | point | 0.532 |
| *NOAA NWM v3.0 (operational; 600 basins)* | 0.550 | +0.205 | 0.687 | +0.309 | 0.596 | 23.8 | point | 0.511 |

Against the in-bake-off control (the retrained LSTM, same bank and pipeline): trees +0.005 / −0.003, TCN −0.002 /
−0.009, S4D −0.004 / −0.005 (ties); transformer −0.017 / −0.031, physics in the loop −0.027 / −0.025, formulas
−0.034 / −0.028, GP −0.043 / −0.048 (losses). Formulas against their teacher (physics in the loop): −0.006 / −0.003, ties.

## Gauged leaderboard (NSE main / recent; CRPS/Q main)

| Row | Lead 1 d | Lead 3 d | Lead 7 d |
|---|---|---|---|
| EnKF on WaterSim HBV (assim) | **0.599** / **0.615**; 0.207 | 0.465 / 0.487; 0.277 | 0.405 / 0.422; 0.320 |
| FlowCast v1.0 nowcast (ssm) | 0.569 / 0.591; 0.248 | **0.474** / **0.511**; 0.277 | **0.426** / **0.466**; 0.313 |
| EnKF minus FlowCast, NSE | +0.030 win / +0.023 tie | −0.009 tie / −0.024 loss | −0.021 tie / −0.044 loss |
| EnKF minus FlowCast, CRPS/Q | −0.041 win / −0.041 win | 0.000 tie / −0.010 tie | +0.007 tie / +0.008 tie |
| Persistence | 0.475 / 0.480 | −0.190 / −0.210 | −0.465 / −0.458 |
| *GR4J calibrated (no gauge input)* | 0.473 / 0.517 | same | same |
| *NWM v3.0 (no gauge input)* | 0.550 / 0.687 | same | same |

Conformal on the EnKF: 90 % coverage 0.94 / 0.90 / 0.85 → 0.89 / 0.89 / 0.89 at leads 1 / 3 / 7 (main); CRPS
unchanged at 1 and 3 days, −0.006 (win) at 7. NWM is not like for like here: it does not read the gauge but runs on
better forcing with its own calibration.

## Where each assumption wins (ungauged, median NSE, best learner per class; main / recent)

| Class (basins) | Main: best learner | Recent: best learner | Clear differences (paired, both periods unless noted) |
|---|---|---|---|
| all (619 / 603) | trees 0.345 (4 tied) | LSTM retrained 0.391 (3 tied) | transformer, GP and both HBV arms lose to the LSTM |
| snow (118 / 115) | formulas 0.462 (8 tied) | trees 0.493 (6 tied) | none among learners; GR4J calibrated 0.675 / 0.697 |
| arid (174 / 171) | LSTM retrained 0.217 | LSTM retrained 0.232 | LSTM beats the TCN (−0.049 / −0.030) and the GP (−0.120 / −0.118) |
| humid (128 / 127) | S4D 0.528 | S4D 0.533 | both HBV arms lose to the LSTM by 0.07-0.10 |
| regulated (37 / 35) | trees 0.353 (8 tied) | TCN 0.437 (6 tied) | none (few basins) |
| large (48 / 48) | FlowCast v1.0 0.226 | LSTM retrained 0.349 (8 tied) | S4D loses to the LSTM on main (−0.099) |

The best reference beat the best learner in every class and both periods (NWM or calibrated GR4J), except large basins
on main (a tie with NWM). Gauged, at 1 day the EnKF is best in arid, humid and regulated basins; persistence is best
in snow and large basins (NSE 0.93 / 0.87 on main), where daily flow changes slowly.

## Before the exam (from the ratchet ladders)

- **The state-space model needed an input-side nonlinearity.** Plain S4D gained nothing from twice the steps (sim val
  0.398 → 0.399); a pointwise MLP before the linear filters lifted it to 0.422 (dev 0.435). Mamba-lite reached 0.415.
- **The transformer needed 4× the steps.** 3,000 steps: sim val 0.364 (LSTM 0.410). It took 12,000 steps to reach 0.420.
  Weekly patch tokens were 10× cheaper per step and lost (0.341).
- **Monotone constraints cost the trees and gained nothing.** Forcing 21 water features to push flow up: sim val 0.367;
  without: 0.398 (dev 0.418 → 0.421). The sign holds one feature at a time, not jointly for 21 correlated memories.
- **17 formulas matched their teacher network.** Laws of 7-15 symbols explain 54-92 % of the teacher's parameter
  variance yet score the same: sim val 0.415 vs 0.414, dev 0.386 vs 0.383, exam ties.
- **FlowCast's recipe reproduces only 0.410 on the new bank.** Its exact network and 3,000-step recipe gave sim val
  0.410 (3 seeds, 0.409-0.410) against 0.426 for the released weights on the same frozen set; 6,000 steps closed it
  (0.4255). Candidate causes, unresolved: the newer repo WaterSim (6 files differ), initialisation and dropout streams.

## Compute (single thread, shared 2-core machine)

| Arm | Train CPU-h (all rungs) | Exam predict CPU-s per basin-period | Stored numbers |
|---|---|---|---|
| LSTM retrained | 0.87 | 0.05 | 24,710 |
| S4D state space | 3.69 | 0.26 | 30,438 |
| TCN | 1.25 | 0.21 | 26,266 |
| Transformer | 3.55 | 0.54 | 26,854 |
| Gradient-boosted trees | 0.43 | 1.20 (numpy features) | 87,876 nodes in 1,172 trees |
| Sparse GP | 0.94 | 0.12 | 260,585 |
| Physics in the loop | 2.25 | 0.01 (numba WaterSim) | 6,481 |
| 17 formulas | 0.25 (+ the teacher) | 0.01 | 17 formulas |
| EnKF (gauged, 3 leads) | 0.98 (tuning on sim) | 0.29 | no learned weights |
| Conformal wrappers | < 0.04 each | base arm's | 2 shifts per stratum and job |

The cheapest learners (LSTM, trees, formulas) are among the best or tied with their teacher; the two that used the
whole budget (S4D, transformer) did not gain from it on the exam. The exam ran in about 3 hours on one core
(2.3 CPU-hours of predict time for 21 arms + references).

## What this does and does not show

- **It shows** that on this problem, with this bank, the choice among sequence learners and trees barely matters
  (five tie within 0.01), while a few assumptions cost measurably: attention under a CPU budget, smoothness (GP),
  and one lumped HBV structure per basin (both physics arms, in humid basins).
- **It shows** that the shared simulated bank caps ungauged skill below a simple donor regionalisation baseline (nearest-neighbour basins' calibrated GR4J). The
  next gain is in the bank (physics p2), not in the learner.
- **It shows** that given a gauge, classical data assimilation on a physical model matches or beats a trained
  network at 1-3 days, with sharper bands at 1 day.
- **It does not** rank these families in general: one problem, networks of about 26k weights, ≤ 4 CPU-hours each.
- **It does not** measure seed noise on the exam (one seed per arm; the LSTM's dev seed spread was about 0.02 sd).
- **It does not** score an ensemble of the tied learners: that would be a new arm needing its own dev gate.
- Most per-class "winners" are inside ties; only the differences marked win/loss are real.

## Rules and notes

- Zero real labels in training; dev used for rung gates and conformal calibration only. Each arm ≤ 3 rungs after the
  base and ≤ about 4 CPU-hours; the S4D (3.69) and transformer (3.55) came closest.
- One exam: `exam.py --run` once (07:49-10:45), no failures, no resume. Each arm re-hashed before and after scoring;
  all 21 unchanged. The exam chose nothing. Analysis (`phase3/analyze.py`) reads only the exam outputs; the extra
  paired comparisons above were run after the exam and are descriptive.
- The page's showcase rain bars read ERA5-Land basin precipitation for the 10 showcase basins after the exam, for
  display only, as the seeded storm draw did before it.
- GR4J calibrated and NWM are not ungauged rows; they are shown, as in FlowCast's exam, as the bar to clear.
- The 806 MB bank is not in the repo; `harness/bank/MANIFEST.json` and `build_bank.py` rebuild it (7.5 min, one core).

## Round 2 (24-25 Nov 2026, in the repository): a better bank, the tied learners retrained

The next rung the verdict pointed to: change the bank, not the learner. Pre-registered in
`experiments/bakeoff-river/round2/PREREG.md` before any training; exam 2 run once (run 20261125T013911).

**The bank.** Three dev-calibrated changes were each judged with the LSTM recipe trained three times (dev seed spread
is about 0.02). Only one cleared the 0.01 margin: a refit water-yield prior (b2b: log m = -0.026 + N(0, 0.153),
chosen by leave-one-basin-out robust RMSE; dev +0.012, at the margin). More reservoirs (+0.002) and two-structure
runoff mixtures (+0.008) were rejected. Bank b2 = p1 + that prior.

**Retrained families** (FlowCast's recipe for each; median exam NSE, paired bootstrap over basins against the same
family's v1.0 arm):

| Arm | Main WY1990-99 | vs v1.0 | Recent WY2016-23 | vs v1.0 |
|---|---|---|---|---|
| LSTM b2 | 0.364 | +0.024 win | 0.391 | -0.000 tie |
| Trees b2 | 0.361 | +0.016 win | 0.394 | +0.006 tie |
| S4D b2 | 0.358 | +0.022 win | 0.394 | +0.008 tie |
| TCN b2 | 0.354 | +0.016 win | 0.388 | +0.006 tie |
| E2: equal-weight ensemble of the four b2 arms + FlowCast v1.0 | **0.376** | U-dev +0.003 tie | **0.401** | U-dev -0.032 loss |
| E1: the same ensemble of the five v1.0 arms | 0.350 | FlowCast +0.009 win | 0.399 | FlowCast +0.021 win |

- **The bank moved every family together** on the main period, by about 0.02 NSE, as round 1's verdict predicted.
  On the recent period the gain does not hold.
- **E2 is the first learner row to tie the regionalisation baseline U-dev** (main: 0.376 vs 0.372). It still loses
  to U-dev on recent, and every row loses to GR4J calibrated per basin and to NWM in both periods.
- **The label-free gap detector fails.** Between-family spread does not predict a dev basin's error
  (Spearman 0.21, 95 % interval -0.10..0.48).
- On dev, b2 trades classes: snow basins gain (LSTM 0.54 to 0.59) and arid basins lose (0.18 to -0.10).

Full log: `experiments/bakeoff-river/round2/STATUS.md`.
