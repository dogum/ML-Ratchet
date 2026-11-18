# ReservoirPilot v1.0: model card

Part of ML Ratchet, Watershed Phase 2 ("Trained on physics. Graded on reality."). Lab page: `lab/index.html`; live
instrument on the model's site page.

**The headline, with its caveat.** On WaterData's frozen exam of 244 real ResOpsUS dams the pilot's median cost J is
**0.0154** (90 % interval 0.0097-0.0202), against **0.0285** for a classical flood-pool and hedging rule tuned on the
same simulator (the fairest baseline available), 0.108 for SOP and 0.183 for the real operators. It beats the tuned
rule on 63 % of dams (exact 95 % interval 56-69 %). **J charges nothing for stored water, and the pilot uses that:** it
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

### Verdicts, with intervals
Methods: (1) the shipped 90 % bootstrap interval of the median J (dams resampled, 2,000 draws); (2) the exact
Clopper-Pearson 95 % interval of each shipped paired win fraction (scipy.stats.binomtest; ties count against the
pilot). Per-dam scores were not shipped, so there are no paired intervals on differences; component rows are judged from
the median only where the gap is large.

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

By main use (median J; no interval): flood control (97 dams) pilot 0.0053, tuned rule 0.0081, SOP 0.186; irrigation
(88) 0.0186, 0.0345, 0.0517; hydropower (25) 0.0522, 0.0718, 0.108; supply (20) 0.0062, 0.0202, 0.0522; navigation (7)
0.125, 0.148, 0.155; recreation (4) 0.017, 0.020, 0.691; other (3) 0.052, 0.072, 0.056.

The fairest baseline available is the classical rule tuned on the simulator. A rule tuned on the 15 dev dams would be
fairer still; it needs the ResOpsUS daily series, which are not on disk. That is an open item, not run.

## Where it fails / limits
1. **It exploits what J ignores.** J charges nothing for storage, so the pilot keeps reservoirs lower: median storage
   38 % of capacity, against SOP's 59 %. The shipped card says that on 34 % of dams it ends the window at least 20 % of
   capacity lower than it started (SOP: 21 %); the per-dam table behind that was not shipped, so it was not re-checked.
   On CDEC the median end − start is −0.10.
2. **It loses to SOP on 15 % of dams** (37 of 244). By the shipped card these are mostly large irrigation or supply
   reservoirs (median 2.0 years of inflow) in a scored decade drier than the calibration decade (not re-checked).
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
