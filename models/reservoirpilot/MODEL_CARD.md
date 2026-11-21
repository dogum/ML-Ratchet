# ReservoirPilot v1.0: model card

Part of ML Ratchet, Watershed Phase 2 ("Trained on physics. Graded on reality."). Lab page: `lab/index.html`; live
instrument on the model's site page.

**The headline, with its caveat.** On WaterData's frozen exam of 244 real ResOpsUS dams the pilot's median cost J is
**0.0154** (90 % interval 0.0097-0.0202), against **0.0285** for a classical flood-pool and hedging rule tuned on the
same simulator (the fairest baseline available), 0.108 for SOP and 0.183 for the real operators. It beats the tuned
rule on 63 % of dams (exact 95 % interval 56-69 %). A re-score on the rebuilt data reproduces every number, and the
paired bootstrap of the median difference excludes zero against every causal baseline, including the same rule tuned on
the 15 real dev dams (0.0322, added after the exam was seen). **J charges nothing for stored water, and the pilot uses that:** it
keeps a median 38 % of capacity full where SOP keeps 59 % and the perfect-foresight DP bound 68 %. Read every J below
next to the storage rows.

## What it does
It is a daily release policy for a single reservoir. Each day it picks a release (m³/s) using only what an operator
knows that day. Its goal is to keep WaterData's reservoir cost low:

J = mean over days of (shortage / demand)² + 10 × (excess over Qmax / Qmax)²

Here demand D(doy) is the day-of-year median release of the calibration half of the record. Qmax is that half's
99th-percentile release, the safe channel capacity. The cost is WaterData's, unchanged; this project did not choose it.

## Inputs (causal only)
Day t, all normalised by Qmax or capacity:
- storage S_t
- capacity
- today's inflow I_t
- exponential means of past inflow (5, 20 and 60 days)
- today's demand, and the demand calendar's means over the next 30 and 90 days
- the calibration half's inflow climatology over the next 30 and 90 days
- three derived quantities: headroom, the SOP spill pressure, and storage in days of demand; and two expected-overflow
  terms from the climatology

That makes 15 numbers. The policy never sees future inflow, future storage or the observed release.

## Output
Two sigmoids, h and e. The requested release is R = h·D + 1.25·e·max(Qmax − D, 0):
- h < 1 means hedging (rationing now).
- e > 0 means an extra release to make flood space.

WaterData's mass balance then caps the release by the water stored and spills anything above capacity; spill counts as
release. Evaporation is ignored, as in the benchmark.

## Network
15 → 64 → 64 → 2, tanh, 5,314 weights (plus 30 feature-scaling numbers), numpy only (`reservoirpilot.py`, weights
`reservoirpilot_weights.npz`). Checks:
- The numpy policy equals WaterData's own operating loop to 8e-17 in J (`training/checks_equiv.json`).
- The torch training rollout matches it to 1.4e-8 in J (same file).
- The shipped page's JavaScript matched it to 5e-15 (release relative to Qmax) on 5 dev dams (`training/check_js.json`).
- The release module matched the exam run to 4e-15 (status doc; not in a shipped file). On import, the demo's six exam
  dams reproduce the exam's per-dam J to 1e-4 relative (the excerpts store inflow to 6 significant digits).
- The site instrument (TypeScript, float32 weights) matches the Python module to 1.5e-14 on 30 cases, WaterData's DP
  included (`site/public/instruments/reservoirpilot/reference.json`).

## Training (simulation only)
- **Method.** Direct policy optimisation through the differentiable reservoir simulator, an analytic policy gradient.
  - Each training window runs the policy for 270 to 1,000 days with WaterData's exact mass balance and cost, then lets
    SOP run for 365 more days, so leftover storage is valued.
  - The cost is back-propagated through every day's storage.
  - The batch loss is the mean of log(J_window + 0.03), so dams count by relative improvement; the exam statistic is a
    median.
  - Adam, gradient clip 1, cosine learning rate. 96 windows per batch. The checkpoint was chosen on a separate
    simulated selection set, never on the frozen set or on real data.
- **Simulated tasks.** Inflows come from WaterSim (`shared/water/watersim`).
  - A generated basin (camels or default prior) runs through HBV, GR4J, SAC-SMA or VIC-lite with degree-day or
    energy-balance snow, on real ERA5-Land forcing from non-exam years or the stochastic weather generator.
  - Half the inflows get extra channel attenuation.
  - Each dam gets a random capacity: log-normal, median 0.8 years of mean inflow, 0.005 to 40 years.
  - A random simulated historical operator (follow-inflow, rule curve, SOP or Hanasaki, with AR(1) release noise) runs
    the first 10 years.
  - The task (D, Qmax, climatology, start storage) is then built from that operator's record exactly as WaterData
    builds it from a real record. The next 10 years are the scored window.
  - Banks hold 3,000 tasks per generator. No exam dam's inflow was ever used.
- **Generator versions** (the ratchet's physics axis). Ranges come from the 15 dev dams plus the 6 auxiliary dams with
  complete records (WaterData: aux dams may be used for calibration and are never scored).
  - g1: WaterSim inflow and the random operator.
  - g2: back-calculated inflow noise, (e_t − e_{t−1})/K with e ~ N(0, σ·cap). Dev and aux dams have negative inflow on
    a median 2 % of days, up to 35 %.
  - g3: an upstream regulating reservoir on 30 % of tasks, plus gross data spikes on 10 %.
  - g4: a non-stationary scored decade, inflow × exp(N(0, 0.3)). On dev dams the scored half carried 0.81 of the
    calibration half's mean inflow; on 29 aux dams the ratio was 1.17; a stationary simulator gives about 1.
  - On g4's frozen set, SOP and DP have median J 0.105 and 0.0069 (`training/ratchet.json`); the rule curve's 0.165
    is in the status doc, not in a shipped file. WaterData's real exam medians are 0.108, 0.143 and 0.0063. Those were
    known from the brief, but nothing was tuned to them.

## Ratchet (`ratchet.yaml`; raw log `training/ratchet.json`)
- **Primary sim metric:** S_J = 1 − median J / median J_SOP, on a frozen simulated set of 300 tasks per generator
  version. A rung needs more than +0.01 over the incumbent.
- **Dev gate:** median J on the 15 dev dams, at most 1.05 × the incumbent's (the first rung must beat SOP's 0.0964).
- **Re-scoring:** when the generator changed to g4, every rung was re-scored and every decision re-taken in order.

| rung | change | S_J g1 | S_J g4 | dev J | decision (re-taken on g4) |
|---|---|---|---|---|---|
| void ×2 | first DPO runs (free-form head, then rule head) | – | – | – | void: `opt.step()` missing, weights never moved; removed and re-run |
| r1b | behaviour cloning of perfect-foresight DP (400 sim tasks) | 0.247 | 0.089 | 0.0960 | kept vs SOP |
| c1 | classical rule, no network: SOP + flood pool + linear hedging, f=0.75, tau=3 d, H=10 d grid-tuned on sim g1 | 0.845 | 0.623 | 0.0422 | kept |
| r2 | DPO; network sets the c1 rule's 3 parameters daily | 0.862 | 0.756 | 0.0409 | kept |
| r3 | DPO, free-form h/e head | 0.876 | 0.659 | 0.0361 | rejected on g4 (kept on g1) |
| r4 | r3 + inflow-climatology features | 0.896 | 0.704 | 0.0367 | rejected on g4 (kept on g1) |
| c2 | c1 re-tuned on g4 (f=0.6) | 0.794 | 0.717 | 0.0475 | rejected (sim and dev) |
| r5 | r4 fine-tuned on the g4 generator | 0.843 | 0.767 | 0.0363 | kept (+0.011, just over the margin) |
| r6 | rule head + climatology from scratch on g4 | – | – | – | rejected: diverged (gradient norm 1e53 through 1/tau) |
| r7 | r5 with 540-day policy windows | 0.878 | 0.828 | 0.0332 | kept |
| r8 | r7 with 1,000-day windows (**released**) | 0.872 | 0.846 | 0.0335 | kept |

Not a rung: a first objective had a hard end-storage penalty inside one-year windows. It punished natural seasonal
drawdown and was replaced by the SOP tail. Every rung, the final one included, ends the simulated windows lower than it
started (r8: median −0.23 of capacity on frozen g4, −0.35 on dev); the ratchet never charged for it, because J does not.

**Twin test (dev dams, rung r3's weights).** Replacing each dev dam's inflow with a 5-day mean (removing gauge noise)
barely changed the pilot's J (0.0361 → 0.0359; `training/twin_r1_dpo_hd_g1_g1.json`). Gauge noise was therefore not the
main sim-to-real gap. The dev dams where the pilot lost to SOP were large reservoirs whose scored decade was drier than
their calibration decade. That finding led to g4.

## Exam (run once, after the ratchet was closed)
WaterData's frozen reservoir exam: 244 scored ResOpsUS exam dams. Every baseline was recomputed with WaterData's own
code; the medians match its table exactly (re-checked on import against
`shared/water/data/proc/baselines/reservoir_scores.parquet`). Medians over dams; storage columns are fractions of
capacity (`training/exam_results.json`).

| policy | J | shortage part | flood part | shortage ratio | days short | days/yr > Qmax | worst excess / Qmax | mean storage | end − start |
|---|---|---|---|---|---|---|---|---|---|
| **ReservoirPilot** | **0.0154** | 0.0094 | 0.00004 | 0.066 | 67 % | 2.9 | 0.04 | **0.38** | +0.01 |
| tuned flood-pool rule (classical, sim g4, c2) | 0.0285 | 0.0152 | 0 | 0.037 | 5 % | 0.0 | 0.00 | 0.40 | +0.01 |
| same rule tuned on sim g1 (c1) | 0.0294 | 0.0084 | 0 | 0.019 | 3 % | 0.0 | 0.00 | 0.48 | +0.12 |
| SOP | 0.108 | 0.0045 | 0.0109 | 0.010 | 1 % | 2.2 | 0.96 | 0.59 | +0.30 |
| rule curve (WaterData) | 0.143 | 0.122 | 0 | 0.186 | 44 % | 0.0 | 0.00 | 0.50 | +0.03 |
| historical operations | 0.183 | 0.168 | 0.001 | 0.285 | 55 % | 1.9 | 0.27 | 0.44 | +0.00 |
| perfect-foresight DP (bound) | 0.0063 | 0.0057 | 0 | 0.030 | 14 % | 0.0 | 0.00 | 0.68 | +0.30 |

### Reproduction re-score and the dev-tuned rule (18 Nov 2026; the exam had already been seen)
The released weights were re-scored on the rebuilt ResOpsUS and CDEC data (`mlratchet data water`; `build.py --only
verify` passed, splits sha256 33f88ca0...) through the shipped exam path (`training/final_exam.py`, WaterData's own
`doy_curve`, `cost`, `simulate`, `dp`), driven by `training/rescore_exam.py`. The dams are exactly WaterData's 244 scored
exam dams (6 of the 250 have no scorable second half) and its 15 scored CDEC reservoirs (PAR, in the crosswalk, is not
scorable). **Every median, win fraction and shipped bootstrap interval reproduces to better than 1e-5.** The re-scored
baselines match WaterData's shipped per-dam table: SOP and rule curve to 1e-12 on all 244 dams; historical differs on
one dam by 3.1e-5 and DP on one dam by 1.2e-4 (the status doc already noted the historical one).

Added on the seen exam: the same classical rule as c1 (SOP + flood pool + hedging), the same grid and objective as
c1's simulator tuning (`training/classical.py`), tuned instead on the 15 real dev dams (`training/tune_rule_dev.py`):
f = 0.75, tau = 3 d, **H = 90 d**, dev median J 0.0369 (the c1 and c2 constants re-score to their logged dev J,
0.0422 and 0.0475). Frozen in `training/dev_tuned_rule.json`, then scored once: **exam median J 0.0322**, worse than the
simulator-tuned rule (0.0285). It hedges over 90 days of demand and keeps far more water: mean storage 0.62, end − start
+0.18. A baseline, not a rung. Per-dam scores: `training/rescore_per_dam.csv`; intervals: `training/rescore_exam.json`.

### Verdicts, with intervals
Methods: (1) a paired bootstrap over dams (10,000 resamples, numpy `default_rng(20261118)`) of median(pilot) −
median(baseline); a verdict is win or loss only if its 95 % interval excludes zero, else tie; (2) as a cross-check, the
exact Clopper-Pearson 95 % interval of each paired win fraction (ties count against the pilot); (3) the shipped 90 %
bootstrap interval of each median.

| row | pilot | baseline | median(pilot) − median(baseline), 95 % paired bootstrap | pilot better on (exact 95 %) | verdict |
|---|---|---|---|---|---|
| ResOpsUS 244, J | 0.0154 | sim-tuned rule g4 0.0285 | −0.0131 (−0.0201 to −0.0073) | 62.7 % (56.3-68.8) | |
| | | sim-tuned rule g1 0.0294 | −0.0140 (−0.0250 to −0.0069) | 60.2 % (53.8-66.4) | |
| | | dev-tuned rule (added on import, seen exam) 0.0322 | −0.0168 (−0.0257 to −0.0105) | 57.8 % (51.3-64.1) | |
| | | SOP 0.108 | −0.093 (−0.123 to −0.069) | 84.4 % (79.3-88.7) | |
| | | rule curve 0.143 | −0.128 (−0.143 to −0.108) | 95.1 % (91.6-97.4) | |
| | | historical 0.183 | −0.168 (−0.191 to −0.149) | 94.7 % (91.1-97.1) | **win** |
| | | DP (a bound) 0.0063 | +0.0092 (+0.0035 to +0.0139) | 29.9 % (24.2-36.1) | info |
| ResOpsUS 244, shortage ratio | 0.066 | SOP 0.010 / g4 rule 0.037 / g1 rule 0.019 / dev rule 0.065 | +0.056 (+0.043 to +0.069) / +0.029 (+0.019 to +0.041) / +0.047 (+0.034 to +0.055) / +0.000 (−0.019 to +0.019) | – | loss |
| ResOpsUS 244, days/yr > Qmax | 2.9 | SOP 2.2 / historical 1.9 / tuned rules 0 | +0.65 (+0.05 to +1.24) / +1.0 (+0.2 to +1.45) / +2.9 (+2.1 to +3.5) | – | loss |
| ResOpsUS 244, worst excess / Qmax | 0.040 | tuned rules 0, SOP 0.96, historical 0.27 | not bootstrapped | – | mixed |
| ResOpsUS 244, mean storage | 0.38 | SOP 0.59 / g4 rule 0.40 / dev rule 0.62 / DP 0.68 / historical 0.44 | −0.21 (−0.31 to −0.16) / −0.02 (−0.04 to +0.01) / −0.23 (−0.26 to −0.20) / −0.30 (−0.34 to −0.26) / −0.06 (−0.12 to +0.00) | – | info (not charged by J) |
| ResOpsUS 244, end − start | +0.008 | SOP +0.30 / g4 rule +0.009 / dev rule +0.18 | −0.29 (−0.38 to −0.23) / −0.002 (−0.063 to +0.026) / −0.17 (−0.23 to −0.13) | – | info |
| Mass-balance-OK 151, J | 0.0162 | g4 rule 0.0300 / g1 0.0354 / dev rule 0.0376 / SOP 0.0992 | −0.0138 (−0.0253 to −0.0080) / −0.0192 (−0.0319 to −0.0084) / −0.0214 (−0.0315 to −0.0109) / −0.083 (−0.116 to −0.061) | g4 64.9 % (56.7-72.5), dev 60.3 % (52.0-68.1), SOP 82.8 % (75.8-88.4) | **win** |
| CDEC 15 (WY2021-2025), J | 0.0193 | g4 and g1 rules 0.0354 / dev rule 0.0291 / SOP 0.0947 | −0.016 (−0.039 to +0.0005) / −0.0098 (−0.0262 to +0.0020) / −0.075 (−0.146 to −0.024) | g4 53 % (27-79), dev 47 % (21-73), SOP 93 % (68-99.8) | **tie** |

Before the re-score (shipped files only: Clopper-Pearson and the shipped 90 % intervals):

| row | pilot | fairest baseline | others | pilot better on (95 % interval) | verdict |
|---|---|---|---|---|---|
| ResOpsUS 244, median J | 0.0154 (0.0097-0.0202) | tuned rule g4 0.0285 (0.0207-0.0374) | g1 rule 0.0294, SOP 0.108 (0.089-0.134), rule curve 0.143, historical 0.183 | g4 rule 62.7 % (56.3-68.8), g1 rule 60.2 % (53.8-66.4), SOP 84.4 % (79.3-88.7), historical 94.7 % (91.1-97.1), rule curve 95.1 % (91.6-97.4) | **win** |
| ResOpsUS 244, mean storage | 0.38 | tuned rule g4 0.40 | SOP 0.59, DP 0.68, historical 0.44 | – | info (not charged by J) |
| ResOpsUS 244, end − start | +0.008 | tuned rule g4 +0.009 | SOP +0.30, DP +0.30 | – | info |
| ResOpsUS 244, shortage ratio | 0.066 | tuned rule g4 0.037 | SOP 0.010, g1 rule 0.019 | – | loss |
| ResOpsUS 244, days/yr > Qmax | 2.9 | tuned rule g4 0.0 | SOP 2.2, historical 1.9 | – | loss |
| ResOpsUS 244, worst excess / Qmax | 0.040 | tuned rule g4 0 | SOP 0.96, historical 0.27 | – | mixed |
| Mass-balance-OK 151, median J | 0.0162 (0.0131-0.0255) | tuned rule g4 0.0300 (0.0248-0.0460) | SOP 0.099 (0.085-0.124), rule curve 0.151, historical 0.200 | g4 rule 64.9 % (56.7-72.5), SOP 82.8 % (75.8-88.4), historical 97.4 % (93.4-99.3) | **win** |
| CDEC 15 (WY2021-2025), median J | 0.0193 (0.0002-0.0546) | tuned rule g4 0.0354 (0.000-0.0756) | SOP 0.095 (0.054-0.158), rule curve 0.240, historical 0.108 | g4 rule 53 % (27-79), g1 rule 60 % (32-84), SOP 93 % (68-99.8), historical 100 % (78-100) | **tie** |
| DP bound, 244, median J | 0.0154 | DP 0.0063 (0.0049-0.0082) | – | 29.9 % (24.2-36.1) | info (a bound) |

By main use (median J, pilot / g4 rule / dev rule / SOP; paired bootstrap against the g4 rule): flood control (97
dams) 0.0053 / 0.0081 / 0.0252 / 0.186, −0.0029 (−0.0121 to +0.0006), level; irrigation (88) 0.0186 / 0.0345 / 0.0338 /
0.0517, −0.0159 (−0.0280 to −0.0088); hydropower (25) 0.0522 / 0.0718 / 0.0503 / 0.108, −0.0195 (−0.0415 to −0.0048),
but level with the dev rule; supply (20) 0.0062 / 0.0202 / 0.0127 / 0.0522, −0.0139 (−0.0299 to +0.0019), level.
Navigation (7) 0.125, 0.148, 0.155 (pilot, g4 rule, SOP); recreation (4) 0.017, 0.020, 0.691; other (3) 0.052, 0.072,
0.056.

The fairest baseline is still the classical rule tuned on the simulator (0.0285): the rule tuned on the real dev dams
scores worse on the exam (0.0322). Both were beaten on J; neither was beaten on stored water by the pilot.

## Where it fails / limits
1. **It exploits what J ignores.** J charges nothing for storage, so the pilot keeps reservoirs lower: median storage
   38 % of capacity, against SOP's 59 % and the dev-tuned rule's 62 %. On 34.4 % of dams it ends the window at least
   20 % of capacity lower than it started (SOP 20.9 %, dev-tuned rule 18.0 %; confirmed by the re-score). On CDEC the
   median end − start is −0.10.
2. **It loses to SOP on 15 % of dams** (37 of 244): 24 of them irrigation or supply reservoirs, large (median 2.05
   years of inflow, against 0.79 for all exam dams), and 86 % of them in a scored decade drier than the calibration
   decade (73 % of all exam dams). Confirmed by the re-score.
3. **It rations often and floods slightly more often, by much less.** Release is below demand on a median 67 % of days;
   it goes above Qmax more often than SOP (2.9 against 2.2 days a year) but by far less (median worst excess 4 % of
   Qmax, against 96 %). Under J's squared penalty that is cheaper; downstream it might not be.
4. **Foresight still wins.** DP is 2.5× lower. The pilot has no inflow forecast.
5. **Scope.** Single reservoir only: no cascades, no evaporation (the benchmark ignores it), no minimum environmental
   flow, demand is a fixed calendar.
6. **Data.** Back-calculated inflows; 93 of 244 exam dams fail the mass-balance check.
7. **Weather data.** The simulated training inflows are driven by ERA5-Land forcing, which under-catches mountain and
   Pacific Northwest precipitation (50 CAMELS basins have a runoff ratio above 1) and drifts about 6 % between decades.
   This caps every Watershed method. The exam itself runs on recorded ResOpsUS inflow, so the caveat reaches
   ReservoirPilot only through its training inflows. Literature numbers (Daymet- or NLDAS-forced LSTMs, NSE about
   0.69-0.76) are streamflow scores and are not comparable.

## Rule notes
From the Watershed handoff (section 6), as written for this model:
- Used aux-dam statistics (6 complete dams, plus inflow ratios on 29 aux dams) to set the generator's ranges.
- Its climatology features use each exam dam's calibration-half inflow, as WaterData's rule curve does.
- Two void training runs (the optimiser step was missing) were removed and logged.
- J charges nothing for storage.
- Added on import (18 Nov 2026), after the exam had been seen: the dev-tuned rule (tuned on the 15 dev dams only, then
  scored once) and the paired bootstrap intervals. Neither changed the release.

Also from the handoff, for the whole platform: dev data were used both to fit physics and to gate rungs (partly
in-sample; here the dev dams helped set the generator's g2-g4 ranges and gate every rung); free-window forcing from exam
basins was allowed in training banks, and no exam reservoir data was used for training. The exam was untouched until the
ratchet closed.

## Use
    pip install numpy pandas
    python demo.py                                     # 6 real exam dams, about 2 s
    python pipeline.py record.csv CAPACITY_HM3 [--out result.csv]   # any daily record: date, storage_hm3, inflow_cms, release_cms

    import reservoirpilot as rp
    pol = rp.Pilot.load()                              # reservoirpilot_weights.npz next to the module
    task = rp.make_task(cal_df, eval_df, capacity_hm3)
    R, S = pol.run(task); rp.cost_components(R, S, task)
    rp.sop(task); rp.tuned_rule(task)                  # the comparators

## Data and credits
- ResOpsUS v2 (Steyaert et al. 2022, Sci. Data 9:34, CC BY 4.0).
- GRanD (Lehner et al. 2011, CC BY 4.0), via ResOpsUS+CARS.
- California DWR CDEC (public data, no explicit licence text).
- Caravan and ERA5-Land forcing (Kratzert et al. 2023; Muñoz-Sabater et al. 2021; CC BY 4.0 and Copernicus licence).
- All of the above through WaterData (Watershed Phase 1).
- Simulator: WaterSim (Watershed Phase 1).
- Hanasaki et al. (2006) for one simulated operator type.

None of the sources is share-alike, so the weights are MIT like the code.
