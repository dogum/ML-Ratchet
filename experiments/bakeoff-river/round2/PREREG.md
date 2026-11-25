# River Bake-off, round 2: pre-registration

Written 2026-11-24 (local), before any round-2 model was trained and before any round-2 bank variant was built or
scored. Only the p1 bank was being rebuilt (a reproduction of the frozen v1.0 bank). Changes after this commit are
appended as dated amendments at the end, each written before the results it affects. Nothing above them is rewritten.

**Question.** In round 1, five families tied on the ungauged exam (trees 0.345, FlowCast v1.0 0.341, LSTM 0.340,
TCN 0.338, S4D 0.336 median NSE, main) and all lost to regionalisation (U-dev 0.372) and NWM (0.550). All of them
learned from one simulated bank. Round 2 asks: does a better bank lift them, does an equal-weight ensemble of the tied
families help, and does the families' disagreement flag the basins where they fail?

**Names.** The new bank is **b2** ("river bank 2"). FlowCast already uses p2, p3 and p4 for its rejected physics rungs
(`models/flowcast/training/phys/`), so round 2 does not use those names. Bank variants are b2a, b2b, b2c (one change
each) and b2 (the final bank). p1 is FlowCast's released physics and the v1.0 bank.

## 0. Rules (unchanged from round 1)

- No real labelled example is ever a training sample. Dev data (the 52 dev basins, WY2001-2014, warm-up from
  1999-10-01, Caravan Q) is used only to calibrate bank physics and to gate rungs. The exam basins and exam periods
  are never read outside the exam driver. The exam chooses nothing.
- Frozen splits are unchanged (`splits.json` sha256 33f88ca0...). Every rung is logged, kept or rejected.
- Training is single-thread CPU, as in round 1 (`torch.set_num_threads(1)`; LightGBM `num_threads` 1). At most 6
  processes at a time on this machine.
- Banks and checkpoints live under `round2/work/` (git-ignored). They are never committed.

## 1. The incumbent bank (p1) and how it is reproduced

`round2/bank2.py build p1` calls `harness/build_bank.make_sample` unchanged with FlowCast's seeds (val 7, train 1),
400-run shards run in parallel processes, then merges them.

The rebuild finished before this file was committed, so I report it here rather than pretend a pass mark came first.
My draft pass mark was "every array's sha256 equals `harness/bank/MANIFEST.json`". Result: 31 of 36 array files are
byte-identical (every input: F, st, attrs, theta, all labels and seeds; and the val target qo). Five differ: q and
states in both splits, and train qo. q and states are WaterSim's float output, and this is a Windows build with a
different numba/LLVM than the cloud's Linux one, so last-bit differences are expected. qo is q through the gauge-noise
model rounded to 0.001 mm/day, so a last-bit change in q flips qo only where it sits on a rounding edge: val qo
(1.17 M values) did not flip at all; train qo (17.5 M) did somewhere.

The pass mark I use instead (fixed now, before any training): (1) all inputs and the val target are byte-identical
(met); (2) the released v1.0 `seq_lstm` arm scores sim-val median NSE 0.4255 on the rebuilt p1 val (its stored
number), and FlowCast v1.0 0.426; (3) the p1 control (section 3) is compared with round 1's LSTM numbers and the gap
is reported. If (2) fails, I stop and report.

## 2. Candidate bank changes (each one rung against p1)

Each variant is p1 plus one change. The change draws only from auxiliary random generators seeded by the run's own
simulation seed, so every variant has p1's basins, forcing windows, runoff models, snow routines, routing choices and
noise seeds, run for run. `bank2.py selftest` checks that the b2 code with every option off gives p1's arrays exactly.

- **b2a, reservoir operations.**
  1. *Consistency.* In p1 the camels prior copies the donor's reservoir storage attribute but places dams with
     WaterSim's own 30 % coin. So most runs whose attribute says "regulated" have no reservoir. (Only 5 of the 52 dev
     donors have storage > 0, so this affects about 3 % of camels-prior runs.) In b2a every run whose
     `reservoir_storage_frac` attribute is > 0 gets reservoirs (WaterSim's `_random_reservoirs`).
  2. *Operations.* Reservoirs are run by ReservoirPilot's simulated historical operator
     (`models/reservoirpilot/training/gen.py`, `operator`: follow, rule curve, SOP or Hanasaki; seasonal demand
     0.12-1.02 x mean inflow; release cap 1.5-12 x mean inflow; AR(1) operator noise), whose ranges were fitted to
     ResOpsUS dev and aux dams. It replaces WaterSim's default (demand 0.6 x mean inflow, no cap). Capacity is the
     basin's storage attribute in years of mean inflow; the upstream share and the consumptive-withdrawal flag come
     from WaterSim's reservoir draw.
- **b2b, precipitation prior** (the water-yield multiplier m, reconciling FlowCast and BasinDetective;
  `round2/calib_prior.py`, dev only):
  1. Recompute m* for each dev basin exactly as `models/flowcast/training/src/calib_water.py` does (attribute-prior
     WaterSim, 4 models x degree-day, P x m on a 15-point grid, m* matching the observed mean flow WY2001-2014).
     Check: the 4-coefficient fit reproduces p1's coefficients (within 0.01 each).
  2. Three candidate mean functions for log m*: M0 constant; M1 a + b frac_snow (BasinDetective's form);
     M2 a + b elev_km + c frac_snow + d log(P/3) (FlowCast's form). Each fitted with calib_water.py's robust
     least squares.
  3. Choose the form with the smallest leave-one-basin-out robust RMSE of log m* (1.4826 x median |LOO residual|).
     Refit it on all 52 basins. Its scatter sd is that LOO robust RMSE: the out-of-sample spread, not the in-sample
     0.146. Clip 0.4-2.5 as in p1.
  4. If the choice is M2 and its sd differs from 0.146 by less than 0.01, b2b equals p1: the rung is logged as
     "no change found" and not trained.
- **b2c, more model structures.** p1 already draws all four WaterSim runoff models (HBV, GR4J, SAC-SMA, VIC-lite)
  with equal odds, both snow routines, and two routings. So the variety missing is *within* a basin: every p1 run is
  one lumped structure. In b2c, half of the runs (share 0.5) are a mosaic of two sub-areas run by two different
  runoff models (fraction f ~ U(0.2, 0.8) and 1 - f; same basin, forcing, snow routine, routing and parameter draws;
  the second model drawn uniformly from the other three). Outlet flow and states are the area-weighted mix, and the
  gauge-noise model is applied to the mixed flow.

## 3. How each bank change is judged

- **Control learner.** The LSTM arm's released recipe (`arms_seq/train/configs/lstm_r1.json`: FlowCast's network,
  H 64, dropout 0.1, 6,000 steps x 32 windows, lr 0.002 cosine, the shared pipeline `arms_seq/common/seqlib.py`),
  retrained on each bank by a path-only copy of `arms_seq/train/train.py` (`round2/train_seq.py`).
- **Seeds.** Round 1 measured this net's dev seed noise at about 0.02 sd (dev 0.391 / 0.431 / 0.404 for three seeds
  of r0). One seed cannot resolve a 0.01 margin. So the control is trained with **seeds 0, 1 and 2** on every bank,
  p1 included, and "the control's dev median NSE" is the **mean over the 3 seeds** of the dev median NSE (ungauged,
  52 basins, harness `evaluate.py`).
- **Pass mark.** A change is kept if control dev (3-seed mean) on the variant > control dev on p1 + 0.01.
- **Also reported, not gating:** sim-val median NSE (800 runs, harness protocol) on the variant's own frozen val
  set and on p1's frozen val set; per-seed numbers; dev KGE, |FHV|, per-class dev NSE; a paired bootstrap over dev
  basins of the per-basin NSE averaged over the 3 seeds (10,000 resamples, `default_rng(20261118)`).
- **Order and combination.** b2a, b2b and b2c are each judged against p1 (in parallel). Then:
  - none kept: b2 = p1. Step 4 is not run, the round-2 entries are the v1.0 arms, and E2 = E1.
  - one kept: b2 = that variant.
  - two or more kept: b2 = p1 + every kept change. It is built and judged against p1 by the same pass mark. If it
    fails, b2 = the single kept change with the largest control dev gain.

## 4. Arms retrained on the final b2

Tab trees (`tab_gbm`, r2 recipe: `arms_tab/src/build_tab.py` K 100, 1 window per run, seeds 11/12; `train_gbm.py`
with monotone constraints off, all else default), LSTM (`seq_lstm`, `lstm_r1.json`, seed 0), TCN (`seq_tcn`,
`tcn_r1.json`), S4D (`seq_ssm`, `ssm_r2.json`). Each with its released v1.0 recipe and hyperparameters, nothing tuned.
FlowCast v1.0 is the released reference and is not retrained.

- **Rung rule (round 1's):** a family's b2 model is one rung against that family's v1.0 arm. Kept if its sim-val
  median NSE beats the v1.0 arm's by more than 0.01, **both scored on b2's frozen val set**, AND its dev median NSE
  is at least 0.95 x the v1.0 arm's dev median NSE.
- Every rung is logged in `round2/arms/<arm>/ratchet.json`, kept or rejected. A family whose b2 rung fails keeps
  its v1.0 arm as its round-2 entry (no new arm is registered for it).

## 5. Ensemble arms (new arms, equal weights)

- **E1** (`ens_e1`) = the five v1.0 tied arms: `tab_gbm`, `flowcast_v1` (ungauged weights), `seq_lstm`, `seq_tcn`,
  `seq_ssm`.
- **E2** (`ens_e2`) = the five round-2 entries: for each family its b2 arm if kept, else its v1.0 arm; FlowCast v1.0.
  If E2's members equal E1's, E2 is not registered.
- **Combination**, per target day, weights 1/5 each: point `mean` = the mean of the members' means. Each quantile
  (5, 25, 50, 75, 95 %) = the mean of the members' same quantile (quantile averaging), then sorted. No dev
  re-weighting, no widening.
- **Dev gate (a new arm, base rung r0):** registered if its dev median NSE >= 0.95 x the best member's dev median
  NSE. The claim "ensembling helps on dev" is made only if its dev median NSE exceeds the best member's by > 0.01.
  Also reported: sim-val NSE on p1's val (E1) and b2's val (E2), dev KGE, coverage, CRPS, and paired bootstraps on
  dev against each member and U-dev.
- **Conformal wrappers** (if time allows): the round-1 Mondrian layer (`arms_filt/conformal/lib/cf_mondrian.py`,
  unchanged), fitted on dev residuals, for each new round-2 arm, registered as `<arm>_conformal`. No pass mark (a
  calibration layer). If not made, recorded as not done.

## 6. The disagreement test (dev only; never touches the exam)

- Members: the five round-2 entries (E2's members). For each dev basin b (52):
  - spread_b = mean over target days of the standard deviation across the five members' daily mean predictions,
    divided by the mean over target days of the ensemble mean (relative spread; no labels);
  - error_b = 1 - NSE_b of the E2 ensemble mean against observed Caravan Q.
- Statistic: Spearman rho(spread_b, error_b) over the 52 basins; 95 % percentile CI from 10,000 bootstrap resamples
  of basins (`default_rng(20261118)`).
- **Pass mark:** "spread predicts error" if rho > 0 and the 95 % CI excludes 0. Reported either way, with the
  per-basin table. Descriptive extras (no pass mark): the same with error = |volume bias| and with E1's members.

## 7. The round-2 exam (written now, run once, later, by the owner)

- `round2/exam2.py`: a copy of `harness/exam.py`'s logic with its own registry (`round2/registry/`), its own lock and
  done files (`round2/results/EXAM2_RUNNING.json`, `EXAM2_DONE.json`), and results in `round2/results/`. The v1.0
  registry, lock and results are untouched.
- It scores **every round-2 registered arm** on the same exam sets as v1.0: main (WY1990-99, Caravan Q) and recent
  (WY2016-23, NWIS Q), ungauged, every exam basin with >= 365 observed days. No gauged arm changes in round 2, so
  there is no gauged track.
- Exam data passes only through `bake`'s guard (lock redirected to `EXAM2_RUNNING.json`). Each arm is re-hashed
  before and after scoring and refused if it changed since registration.
- **Reference rows are reused, not re-scored:** U-dev, U-k12, GR4J, NWM, climatology and every v1.0 arm row
  (FlowCast v1.0 and the five tied arms, raw and conformal) come from
  `results/exam_20261123T074907/scores.parquet`. NWM cannot be re-scored here (NOAA's daily table is absent).
- Before the exam: `exam2.py --check` (reads no exam data) and `exam2.py --rehearsal` (the same pipeline on dev,
  with dev reference rows) only. Then I stop.
- **Comparisons fixed now** (paired bootstrap over basins of the difference in medians, 10,000 resamples,
  `default_rng(20261118)`; win/loss only when the 95 % interval excludes 0):
  - each family's b2 arm vs its own v1.0 arm (does b2 lift the family?);
  - E1 vs FlowCast v1.0, vs U-dev, and vs its best member on dev (chosen now: the member with the highest dev
    median NSE in the round-2 evaluation, recorded in STATUS.md before the exam);
  - E2 vs E1, vs FlowCast v1.0 and vs U-dev;
  - every round-2 arm vs FlowCast v1.0, U-dev, NWM and GR4J (as round 1).
- The exam writes `round2/results/exam2_<run id>.json` (boards, paired bootstraps, per class) and
  `round2/results/exam2_<run id>/scores.parquet`, dated by its run id.

## 8. Budget

About 8 hours of wall time for steps 1-7 (without the exam). If a step is clearly too slow, I scale it down and
append an amendment before its results.

## Amendments

**A1 (2026-11-24 23:05, before any control result).** Two factual corrections, no change of plan.
(1) Section 2, b2a: the p1 train bank has 3.3 % of *all* runs (not "about 3 % of camels-prior runs") with a storage
attribute > 0 and no reservoir; b2a raises the runs with reservoirs from 16.6 % to 19.9 %.
(2) The single-thread LSTM recipe takes about 24 minutes per run on this machine with 5 jobs in parallel, so the
12 control runs and the 4 family retrains fit the budget; no scaling down.
