# ReservoirPilot: lab notebook

Historical record. Add dated entries; do not rewrite earlier ones.

## 18 Nov 2026 — build notes (reservoirpilot-status.md, as written)

ReservoirPilot is a daily release policy for one reservoir. It only knows what an operator knows today: storage,
capacity, today's inflow, recent inflow, the demand calendar, Qmax and the calibration-period inflow climatology. It
learned to operate dams in a simulator (WaterSim inflows plus simulated historical operators). It was graded once on
WaterData's frozen exam of 244 real ResOpsUS dams and 15 CDEC reservoirs, with WaterData's cost J.

### Recipe as built
- **Task:** WaterData's reservoir exam, reproduced by importing its own functions (`doy_curve`, `cost`, `simulate`,
  `dp`): dev baselines match WaterData's per-dam table to 1.6e-9; exam medians match exactly (one historical value
  differs by 3e-5).
- **Simulator.** Inflow: WaterSim. A generated basin (half camels prior with donor forcing, half default prior with the
  weather generator) runs through HBV, GR4J, SAC-SMA or VIC-lite with degree-day or energy-balance snow, on non-exam
  ERA5-Land forcing. Half the inflows get extra attenuation. Dam: random capacity (log-normal, median 0.8 years of
  inflow). Operator: a random simulated historical operator (follow, rule, SOP, Hanasaki, with AR(1) noise) runs the
  first 10 years. Task: D, Qmax, climatology and S0 are built from that operator's record exactly like WaterData builds
  them from a real one. 3,000 training tasks plus 300 frozen and 200 selection tasks per generator version.
- **Generator calibration** (dev + 6 aux dams with full records; WaterData allows aux for calibration): g2
  back-calculated inflow noise (negative-inflow days median 2 %, up to 35 %); g3 upstream regulation plus data spikes;
  g4 non-stationary halves, scored-decade inflow × exp(N(0, 0.3)) (dev scored/calibration inflow ratio 0.81 median, aux
  1.17 on 29 dams). On g4's frozen set SOP / rule / DP medians are 0.105 / 0.165 / 0.0069; WaterData's real exam
  0.108 / 0.143 / 0.0063 (not tuned to).
- **Method:** direct policy optimisation through the differentiable simulator: torch BPTT through WaterData's mass
  balance and cost, over 270- to 1,000-day windows plus a 365-day SOP tail so leftover storage is valued; loss mean of
  log(J + 0.03). Behaviour cloning of DP was tried and lost.
- **Network:** 15 → 64 → 64 → 2 tanh (5,314 weights). Output R = h·D + 1.25·e·(Qmax − D)+.
- **Equivalence:** numpy vs WaterData's operating loop 8e-17 (J); numpy vs torch 1.4e-8 (J); JS vs Python 5e-15
  (release / Qmax); release module vs exam 4e-15.
- **Ratchet:** frozen sim set S_J = 1 − median J / median J_SOP must improve by more than 0.01; dev gate median J on
  the 15 dev dams no more than 5 % worse than the incumbent's; generator switch g1 → g4 re-scored everything and
  re-took decisions in order; checkpoints chosen on a separate simulated selection set.

### Rungs (final re-scoring on frozen g4)

| Rung | Change | S_J g1 | S_J g4 | dev J | kept? |
|---|---|---|---|---|---|
| r1b | BC of perfect-foresight DP | 0.247 | 0.089 | 0.0960 | kept (beats SOP 0.0964) |
| c1 | classical: SOP + flood pool + hedging, 3 params tuned on sim g1 | 0.845 | 0.623 | 0.0422 | kept |
| r2 | DPO, network sets c1's 3 parameters daily | 0.862 | 0.756 | 0.0409 | kept |
| r3 | DPO, free-form h/e head | 0.876 | 0.659 | 0.0361 | kept on g1, rejected on g4 |
| r4 | + inflow climatology features | 0.896 | 0.704 | 0.0367 | kept on g1, rejected on g4 |
| c2 | c1 re-tuned on g4 | 0.794 | 0.717 | 0.0475 | rejected (sim, dev) |
| r5 | r4 fine-tuned on g4 | 0.843 | 0.767 | 0.0363 | kept |
| r6 | rule head + clim from scratch on g4 | – | – | – | rejected: diverged (grad norm 1e53) |
| r7 | 540-day windows | 0.878 | 0.828 | 0.0332 | kept |
| r8 | 1,000-day windows (**final**) | 0.872 | 0.846 | 0.0335 | kept |

### Exam (once, after closing the ratchet; medians over dams)

| | ResOpsUS 244 J | shortage ratio | days/yr > Qmax | worst excess / Qmax | mean storage | end − start | mass-balance-OK 151 J | CDEC 15 J |
|---|---|---|---|---|---|---|---|---|
| **ReservoirPilot** | **0.0154** (90 % CI 0.0097–0.0202) | 0.066 | 2.9 | 0.04 | 0.38 | +0.01 | 0.016 | 0.019 |
| tuned classical rule (sim g4) | 0.0285 | 0.037 | 0.0 | 0 | 0.40 | +0.01 | 0.030 | 0.035 |
| SOP | 0.108 | 0.010 | 2.2 | 0.96 | 0.59 | +0.30 | 0.099 | 0.095 |
| rule curve | 0.143 | 0.186 | 0.0 | 0 | 0.50 | +0.03 | 0.151 | 0.240 |
| historical | 0.183 | 0.285 | 1.9 | 0.27 | 0.44 | +0.00 | 0.200 | 0.108 |
| perfect-foresight DP | 0.0063 | 0.030 | 0.0 | 0 | 0.68 | +0.30 | 0.007 | 0.012 |

The pilot beats SOP on 84 % of dams, historical operations on 95 % and the tuned classical rule on 63 %. DP beats the
pilot on 70 %.

### Where it fails
- **Exploits what J ignores.** J does not value storage, so the pilot holds flood space: mean storage 38 % of capacity,
  against 59 % for SOP. It ends at least 20 % of capacity lower than it started on 34 % of dams (SOP 21 %). On CDEC the
  median end − start is −0.10.
- **Drier decades on large dams.** It loses to SOP on 37 dams (15 %), mostly large irrigation or supply reservoirs (2.0
  years of inflow) whose scored decade was drier than the calibration one.
- **Many small floods.** It exceeds Qmax slightly on more days (2.9 against 2.2 days a year) but by much less (4 %
  against 96 % of Qmax).

### Next rungs worth trying
1. **MPC or a learned inflow forecaster input.** DP is still 2.5× lower, and most of that gap is future-inflow
   information.
2. **A storage-aware variant.** Report and train under J plus a carry-over or head term, to show the price of the
   low-storage behaviour. Alternatively, condition on an operator-chosen storage floor.
3. **Better non-stationarity.** The drier-decade failures suggest letting the policy adapt demand and climatology online
   from recent years (e.g. a 1-3 year inflow EMA feature), and widening g4's spread.

### Upstream candidates for WaterSim
- `realdata.random_forcing` can draw a window that ends a few days inside the exam period. `span*365` days does not
  equal `DateOffset(years=span)`, so 20-year draws cross 1989-10-01. The guard raises `PermissionError`; ReservoirPilot
  redraws (`training/gen.py`). The fix belongs in the window arithmetic.
- Reservoir generator: add back-calculated inflow noise, a non-stationary decade factor and Qmax/demand from a simulated
  operator (as in `training/gen.py`) if other projects need realistic reservoir tasks.

### Sandbox gotchas
- `train.py` initially lacked `opt.step()` (two void runs). Check that weights move before trusting a flat validation
  curve.
- A hard end-storage penalty inside short windows punishes natural seasonal drawdown. Use a differentiable SOP tail
  instead.
- One g4 bank task had Qmax = 0 (NaN gradients). Training now skips tasks with Qmax ≤ 0 and non-finite gradient steps;
  the policy guards Qmax at 1e-9.
- Sigmoid heads initialised deep in saturation (bias ±6-8) barely learn. Start near SOP with moderate biases (4.5,
  −5.5).
- Training speed on the shared machine: 0.5–2.5 s per iteration (96 windows), depending on window length and load.

## 18 Nov 2026 — rule notes from the Watershed handoff (section 6)
- Dev data used to fit physics and also to gate rungs (partly in-sample): for ReservoirPilot the dev dams set part of
  the generator's g2-g4 ranges and gate every rung. The exam was untouched.
- Free-window forcing from exam basins was allowed in training banks. No exam Q, SWE or reservoir data was used for
  training.
- ReservoirPilot used aux-dam statistics (6 complete dams, plus inflow ratios on 29 aux dams) to set the generator's
  ranges. Its climatology features use each exam dam's calibration-half inflow, as WaterData's rule curve does. Two void
  training runs (the optimiser step was missing) were removed and logged. J charges nothing for storage.

## 18 Nov 2026: imported into ML Ratchet

What changed on import:
- **Renames (contract):** `weights.npz` → `reservoirpilot_weights.npz` (bytes unchanged; `Pilot.load()` now defaults to
  the file next to the module, and `pipeline.py --weights` defaults to it); `demo_data/exam_dams.json` →
  `examples/exam_dams.json`; `results/*.json` → `training/` (ratchet.json, exam_results.json, checks_equiv.json,
  check_js.json, twin_r1_dpo_hd_g1_g1.json, as shipped). The release README's content went into model.yaml.
- **Inference additions that do not change a number:** `day_of_year` instead of the deprecated `dayofyear` (pandas
  printed a Pandas4Warning); `tuned_rule()` (the exam's classical comparator, rung c2's f = 0.6, tau = 3, H = 10, ported
  from `training/policy.TunedSOP` with the exam rollout's mass balance) so the demo and the instrument show the fairest
  baseline; a `seen=` option on `Pilot.run` and `tuned_rule` (the inflow the policy reads, for the instrument's
  gauge-noise what-if; the mass balance always uses the true inflow); `cost_components` now also returns
  `shortage_days_frac`, `max_excess_ratio` and days near empty / full, with the exam's definitions. The released policy's
  output is unchanged: the demo's six dams reproduce the exam's per-dam J to 1e-4 relative, as the shipped assert
  requires.
- **demo.py** no longer writes a PNG into the folder: `--plot` writes `work/demo_reservoirpilot.png`. It now prints the
  tuned rule's J and the storage columns (mean storage and end − start, pilot / SOP) beside J.
- **Training paths:** every sandbox path (the old home-folder workspace) now goes through `training/_paths.py` (WORK = `models/reservoirpilot/work`
  or `$ML_RATCHET_WORK/reservoirpilot`; WaterSim, WaterData and WaterData's baseline scripts from `shared/water`; the data
  root `$ML_RATCHET_WATER`). `run_bank.sh` and `run_train.sh` cd into WORK. Nothing was re-run.
- **Instrument and lab:** `web/export_instrument.py`, `site/src/instruments/reservoirpilot/`, `web/build_lab.mjs`,
  `lab/`. The browser port started from the shipped page's `training/engine.js`.

Checks against the shipped files (what reproduced):
- Every exam median in the status doc and the shipped card matches `training/exam_results.json` (J and every component,
  the 151-dam subset, the CDEC exam, the 90 % intervals, the win rates 84 / 95 / 63 / 95 % and DP's 70 %).
- The baseline medians equal WaterData's frozen per-dam table on disk (`shared/water/data/proc/baselines/
  reservoir_scores.parquet`: SOP 0.108372, rule 0.142997, historical 0.183264, DP 0.006254) and the six demo dams' SOP,
  rule-curve, historical and DP J match it per dam.
- WaterData's `dp` re-run on the six demo excerpts reproduces their exam DP J to 6 significant digits on 5 dams;
  grand_148 (Oroville) gives 0.04029-0.04031 against 0.04029 (the excerpt rounds inflow, and DP is sensitive to it).
- The ratchet table in the status doc matches `training/ratchet.json` to the digits shown, and every decision follows
  the rule (r5's margin over r2 is +0.011, barely above 0.01; r8's dev J is 0.8 % above r7's, inside the 5 % gate).
- The twin test's 0.0361 → 0.0359 matches `twin_r1_dpo_hd_g1_g1.json` (the file is for rung r3's weights, which were
  first run as "r1").

What did not reproduce from the shipped files (not wrong, just not checkable here):
- "Ends at least 20 % of capacity lower on 34 % of dams (SOP 21 %)" and "loses to SOP on 37 dams, mostly large
  irrigation or supply reservoirs (median 2.0 years of inflow) in a drier decade": the 37 follows from the shipped
  pilot_worse_frac (0.1516 × 244), but the per-dam table (`exam/exam_per_dam.parquet`) was not shipped, so the 34 %,
  21 % and 2.0 years cannot be re-checked.
- "On g4's frozen set the rule curve's median is 0.165": ratchet.json carries only SOP and DP medians.
- "Dev scored/calibration inflow ratio 0.81, aux 1.17 on 29 dams" and the negative-inflow-day statistics:
  `data/dam_stats.parquet` was not shipped.
- "Release module vs exam 4e-15" and "dev baselines match to 1.6e-9": no shipped JSON holds these.
- The shipped page's JavaScript DP (`training/engine.js`) does not reproduce WaterData's DP on the full 10-year demo
  dams: on grand_1023 it gives J 0.00027 against 0.00129, on grand_470 0.0886 against 0.0901 (it builds the storage grid
  as cap·i/200 and interpolates differently from numpy, which changes DP's tie-breaks). `check_js.json` only tested
  1,100-day dev windows, where it matched. The site instrument's DP ports numpy's linspace / unique / interp arithmetic
  and matches WaterData's `dp` to 1e-14 on all 30 reference cases. So the "DP bound" is itself approximate: rounding
  changes can move it by several times on a dam whose J is near zero.
- Timing: the shipped JS DP took 0.35-3.1 s per 10-year dam in Node; the instrument's port takes 0.5-0.7 s.

Instrument: the browser runs the released network (float32 weights: float16 moved a dam's J by up to 0.4 % through the
10-year closed loop, and the page shows the exam J beside its own), SOP, the g4-tuned rule and WaterData's DP live on the
six exam dams. What-ifs change the input only: the decade's inflow ×0.6-1.4, a 31-day flood pulse of up to 3 × Qmax, the
capacity ×0.5-2 (same starting fill), a storage-gauge error the policies read (σ up to 0.5 % of capacity, back-calculated
into inflow noise), demand ×0.6-1.6. On the recorded decade the live pilot reproduces the exam J to 3e-6 relative (the excerpts store inflow to 6 significant digits; float32 weights add about 1e-8).

Open: a classical rule tuned on the 15 dev dams (the fairest baseline) and paired per-dam intervals both need the
ResOpsUS daily series, which are not on disk (`mlratchet data water`); not run.

## 18 Nov 2026: reproduction re-score (seen exam) and the dev-tuned rule

Approved by the owner after the import. A reproduction, not a rung: nothing in the release changed, and nothing was
chosen from the result. The exam had already been opened in the sandbox, so every number below is on a seen exam.

**Data.** ResOpsUS v2 (678 dams) and CDEC (18 reservoirs) rebuilt into a git-ignored data root (`ML_RATCHET_WATER`).
`python -I shared/water/data/build.py --only verify`: 1,170 of 1,170 shipped files identical; splits.json file sha256
33f88ca0... and field ed8109e5... both OK. Scored dams: exactly WaterData's 244 scored exam dams (the 6 other exam dams,
grand_1254, 338, 468, 508, 827 and 919, have no scorable second half, as in WaterData's table) and its 15 scored CDEC
reservoirs (PAR is in the crosswalk but not scorable, as in WaterData's table). Dev tasks rebuilt with
`evaluate.prep('dev')` match WaterData's per-dam table to 1.6e-9, as the status doc said.

**Path.** `training/rescore_exam.py`: `evaluate.prep('exam')` (WaterData's `doy_curve`, `cost`, `simulate`, `dp`
imported), the sim-tuned rules rebuilt from their logged constants (c1 f 0.75 / tau 3 / H 10, c2 f 0.6 / 3 / 10), then
the shipped `final_exam.main` on `reservoirpilot_weights.npz`, unchanged. Outputs under `work/exam/`; committed:
`training/rescore_per_dam.csv` (J and every component per dam and policy, 5 significant digits, no time series) and
`training/rescore_exam.json`.

**Reproduction.** Every median of the shipped `exam_results.json` (all components, all three splits), every win
fraction and every shipped 90 % interval reproduces to better than 1e-5; the pilot's median J is 0.015427 (claim
0.0154), SOP 0.10837 (0.108), rule curve 0.14300 (0.143), historical 0.18326 (0.183), DP 0.006254 (0.0063), sim-tuned
rule g4 0.02852 (0.0285). Baselines per dam against WaterData's shipped `reservoir_scores.parquet`: SOP max |ΔJ| 1.3e-15
and rule curve 1.1e-13 on all 244; historical differs on one dam by 3.1e-5 and DP on one dam by 1.2e-4 (all other
dams < 1e-6). CDEC against `reservoir_recent_scores.parquet`: SOP, rule curve, DP < 1.2e-10, historical one reservoir by
1.1e-6. So the rebuilt data match the sandbox's.

**The card's failure claims, now checkable, all hold.** The pilot ends ≥ 20 % of capacity lower than it started on
34.4 % of dams (SOP 20.9 %, sim-tuned g4 rule 33.2 %, dev-tuned rule 18.0 %, historical 6.1 %, DP 0.4 %). It loses to SOP
on 37 dams: 18 irrigation, 6 supply, 8 flood control, 4 hydropower, 1 recreation; median capacity 2.05 years of inflow
(all exam dams 0.79); 86 % of them had a scored decade drier than the calibration decade (all exam dams 73 %).

**Dev-tuned rule** (`training/tune_rule_dev.py`, `training/dev_tuned_rule.json`). The same rule as c1, the same grid
(f 0.4-1.01, tau 3/10/30 d, H 1/10/30/90 d) and objective (median over tasks of mean daily J, first minimum) as
`classical.py`, on WaterData's 15 dev tasks only. Check first: the c1 and c2 constants re-score to their logged dev J
(0.0422, 0.04747). Optimum f 0.75, tau 3 d, H 90 d (the grid's largest hedging horizon), dev median J 0.0369 (pilot r8:
0.0335). Frozen, then scored once on the exam: median J 0.0322 (mass-balance subset 0.0376, CDEC 0.0291), worse than the
simulator-tuned rule (0.0285); it keeps more water (mean storage 0.62, end − start +0.18) and rations on 11 % of days.

**Paired bootstrap** (10,000 resamples over dams, numpy `default_rng(20261118)`, 95 % percentile interval of
median(pilot) − median(baseline)):

| split | baseline | ΔJ | 95 % | verdict |
|---|---|---|---|---|
| 244 | sim-tuned g4 | −0.0131 | −0.0201 to −0.0073 | win |
| 244 | sim-tuned g1 | −0.0140 | −0.0250 to −0.0069 | win |
| 244 | dev-tuned | −0.0168 | −0.0257 to −0.0105 | win |
| 244 | SOP | −0.093 | −0.123 to −0.069 | win |
| 244 | rule curve | −0.128 | −0.143 to −0.108 | win |
| 244 | historical | −0.168 | −0.191 to −0.149 | win |
| 244 | DP (bound) | +0.0092 | +0.0035 to +0.0139 | loss (info) |
| 151 | sim-tuned g4 / dev-tuned / SOP | −0.0138 / −0.0214 / −0.083 | −0.0253 to −0.0080 / −0.0315 to −0.0109 / −0.116 to −0.061 | win |
| 15 CDEC | sim-tuned g4 and g1 | −0.016 | −0.039 to +0.0005 | tie |
| 15 CDEC | dev-tuned | −0.0098 | −0.0262 to +0.0020 | tie |
| 15 CDEC | SOP | −0.075 | −0.146 to −0.024 | win |

Components on the 244 dams: shortage ratio worse than SOP (+0.056, +0.043 to +0.069), the g4 rule (+0.029, +0.019 to
+0.041) and the g1 rule, level with the dev-tuned rule (+0.000, −0.019 to +0.019); days above Qmax worse than SOP (+0.65,
+0.05 to +1.24) and every tuned rule (+2.9, +2.1 to +3.5); mean storage lower than SOP (−0.21, −0.31 to −0.16), the
dev-tuned rule (−0.23) and DP (−0.30), level with the g4 rule (−0.02, −0.04 to +0.01); end − start lower than SOP
(−0.29) and the dev-tuned rule (−0.17), level with the g4 rule. On the 151-dam subset the days above Qmax against SOP
are level (+0.37, −0.20 to +0.90). The Clopper-Pearson win-fraction intervals agree (dev-tuned rule: pilot better on
57.8 %, 51.3-64.1 %) and stay in model.yaml as a cross-check.

What did not reproduce: nothing in the shipped exam log. Not reproduced by design: the shipped page's JS DP (see the
import entry). Model.yaml, the card and the lab tables now carry the dev-tuned rule and the bootstrap intervals; the
verdicts are unchanged (244: win; 151: win; CDEC: tie; shortage ratio and days above Qmax: loss; storage rows: info).

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). The exam takes capacity as the largest storage observed, including the scored window. make_task(causal_capacity=True) now offers the calibration-only version. Changing the exam's own definition would re-score every arm on the shared board; that is queued for the owner.

## 27 Nov 2026 — causal re-score of every arm; v1.0.1 makes the causal task the default

Owner-approved follow-up to the audit. A dated supplementary exam on a seen exam: nothing was trained, tuned or chosen.

**What changed in the task.** `training/task.py` and WaterData's `run_baselines_reservoir.py` gained a switch
(`RESERVOIRPILOT_TASK=causal`, `--causal <out dir>`). Three changes, nothing else:
1. capacity = max(GRanD capacity, highest storage in the calibration half). The exam definition also took the
   maximum over the scored half.
2. The record is split first and each half is interpolated (gaps up to 5 days) on its own.
3. The split day belongs to the scored half only (it was the last calibration day and the first scored day).

On the 244 scored exam dams this changes capacity on 37 (median ratio 0.977; grand_1585 1,900 → 128 hm³, grand_190
861 → 260, grand_1595 262 → 154), Qmax slightly on 169, the start day on 3, and nothing else. The same 244 dams stay
scorable. The CDEC recent task already took capacity, demand and Qmax from the 1980-2020 ResOpsUS record before its
window: unchanged. Intermediates are in `work/causal/reservoirpilot/` (git-ignored).

**Path.** `run_baselines_reservoir.py --causal` wrote the causal baseline table (SOP, rule curve, historical, DP), then
`RESERVOIRPILOT_TASK=causal python training/rescore_exam.py` (the 18 Nov script, unchanged except output names and the
table it compares with) re-scored the pilot, the sim-tuned rules (g4, g1), the dev-tuned rule and the four baselines,
with the same 10,000-resample paired bootstrap (seed 20261118). Committed: `training/rescore_exam_causal.json`,
`training/rescore_per_dam_causal.csv`. Cross-check: the re-scored baselines match the `--causal` table exactly for SOP,
rule curve and DP on all 244 dams, historical on 243 (one dam by 1.9e-4, as in the 18 Nov reproduction).

**Old vs new board** (medians over dams; old = the exam definition, reproduced 18 Nov 2026):

| split | arm | J old | J new | shortage ratio old → new | mean storage old → new | end − start old → new |
|---|---|---|---|---|---|---|
| 244 dams | pilot | 0.01543 | 0.01544 | 0.066 → 0.066 | 0.380 → 0.381 | +0.008 → −0.003 |
| 244 dams | sim-tuned rule g4 | 0.02852 | 0.02886 | 0.037 → 0.038 | 0.399 → 0.399 | +0.009 → +0.007 |
| 244 dams | sim-tuned rule g1 | 0.02939 | 0.02944 | 0.019 → 0.019 | 0.485 → 0.489 | +0.118 → +0.111 |
| 244 dams | dev-tuned rule | 0.03222 | 0.03222 | 0.065 → 0.065 | 0.615 → 0.617 | +0.176 → +0.172 |
| 244 dams | SOP | 0.10837 | 0.10905 | 0.010 → 0.010 | 0.587 → 0.585 | +0.300 → +0.296 |
| 244 dams | rule curve | 0.14300 | 0.14408 | 0.186 → 0.186 | 0.500 → 0.507 | +0.028 → +0.028 |
| 244 dams | historical | 0.18326 | 0.18327 | 0.285 → 0.285 | 0.440 → 0.441 | +0.002 → +0.002 |
| 244 dams | DP (bound) | 0.00625 | 0.00609 | 0.030 → 0.030 | 0.685 → 0.685 | +0.300 → +0.296 |
| 151 mass-balance dams | pilot | 0.01619 | 0.01618 | | | |
| 151 mass-balance dams | g4 rule | 0.03003 | 0.03041 | | | |
| 151 mass-balance dams | SOP | 0.09918 | 0.09925 | | | |
| 15 CDEC | every arm | identical | identical | | | |

Paired bootstrap of median(pilot) − median(arm), 244 dams, old → new: g4 rule −0.0131 (−0.0201 to −0.0073) → −0.0134
(−0.0211 to −0.0074); g1 rule −0.0140 → −0.0140 (−0.0256 to −0.0069); dev-tuned rule −0.0168 (−0.0257 to −0.0105) →
−0.0168 (−0.0260 to −0.0112); SOP −0.093 → −0.094; rule curve −0.128 → −0.129; historical −0.168 → −0.168; DP +0.0092
(+0.0035 to +0.0139) → +0.0093 (+0.0040 to +0.0139). Win fractions against the g4 rule 62.7 % → 62.3 % (55.9-68.4),
SOP 84.4 % → 85.2 %. The pilot loses to SOP on 35 dams (was 37). **Every verdict is unchanged** on all three splits
and every component row, with one exception in the storage row: the pilot's mean storage against the real operators
was level (−0.06, −0.12 to +0.00) and is now lower (−0.06, −0.13 to −0.005). The largest single move is the
recreation group (4 dams): pilot 0.017 → 0.030, because grand_1595's capacity fell from 262 to 154 hm³.

What failed or stays open: nothing failed. The six packaged demo dams (`examples/exam_dams.json`) are v1.0 tasks; two
change capacity under the causal task (grand_148 4,414 → 4,367 hm³, grand_470 39.09 → 39.04 hm³) and three Qmax by
< 0.03 %. They were not regenerated, so the demo, its golden output and the site instrument still show v1.0 tasks.
WaterData's frozen tables in `shared/water/data/proc/baselines` are untouched (they are the exam definition).

**v1.0.1.** `make_task(..., causal_capacity=True)` is now the default; `pipeline.py` splits before interpolating and
puts the split day in the scored half (`--exam-definition` restores v1.0). The weights are unchanged. `model.yaml`
carries the causal numbers, with the exam-definition values in the notes.
