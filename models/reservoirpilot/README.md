<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# ReservoirPilot <sub>v1.0</sub>

> **A daily dam release policy learned in a simulator, graded on 244 real US reservoirs**

> [!WARNING]
> Research benchmark, not an operating tool. J, WaterData's cost, charges for shortage against a calibrated demand and for releases above a safe channel capacity, and nothing else: not for stored water, evaporation, environmental flows, hydropower value or dam safety. The policy exploits that: it holds reservoirs lower than SOP does.

*How much water should this dam release today, knowing only what its operator knows today?*

ReservoirPilot picks each day's release for one reservoir from what an operator knows that day: storage, capacity, today's inflow, recent inflow, the demand calendar, the safe channel capacity Qmax and the calibration-period inflow normal. It never sees future inflow. A 5,314-weight network learned it only in simulation, by back-propagating WaterData's cost J through the reservoir mass balance on WaterSim inflows with simulated historical operators. It was graded once on WaterData's frozen exam: on 244 real ResOpsUS dams its median J is 0.0154 (90 % interval 0.0097-0.0202), against 0.0285 for a classical flood-pool and hedging rule tuned on the same simulator, 0.108 for SOP and 0.183 for the real operators; it beats the tuned rule on 63 % of dams (95 % interval 56-69 %). Perfect-foresight DP, a bound, reaches 0.0063. The caveat belongs beside the headline: J charges nothing for stored water, so the pilot keeps a median 38 % of capacity full where SOP keeps 59 %.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/reservoirpilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/reservoirpilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Daily, for one reservoir: storage S (hm³), capacity (hm³), today's inflow I (m³/s, as recorded, which for ResOpsUS is usually back-calculated from storage changes), and the task WaterData builds from the calibration half of the record: the demand calendar D(doy) (median release by day of year, 31-day smoothed), Qmax (99th-percentile release) and the inflow climatology by day of year. Fifteen features: S/cap, asinh(I/Qmax), 5-, 20- and 60-day inflow means, D and the next-30-day mean demand over Qmax, log capacity in days of Qmax, headroom, SOP spill pressure, storage in days of demand, the next 30- and 90-day inflow normal and two expected-overflow terms. Never future inflow, future storage or the observed release. |
| **Returns** | Today's release (m³/s), applied through WaterData's mass balance: capped by the water stored, anything above capacity spills and counts as release<br>The two decisions behind it: h, the share of demand released (h < 1 hedges, rationing now to keep water for later), and e, an extra release toward 1.25 × (Qmax − D) that makes flood space<br>No error bars: it is a policy, scored by WaterData's cost J = mean daily (shortage / demand)² + 10 (excess over Qmax / Qmax)² |
| **Learned from** | Reservoir water balance, WaterData's exam benchmark unchanged: S(t+1) = S(t) + (I − R)·K with K = 0.0864 hm³ per m³/s·day, release capped by the water stored, spill above capacity counted as release, no evaporation. The training inflows come from WaterSim, the Watershed simulator: generated basins run through HBV, GR4J, SAC-SMA or VIC-lite with degree-day or energy-balance snow and channel routing. A simulated historical operator runs each dam's first decade, and the task (demand, Qmax, climatologies, start storage) is then built from that record exactly as WaterData builds it from a real one. |
| **Noise** | Simulated, with ranges set on real dams: back-calculated inflow noise I_obs = I + (e_t − e_t−1)/K, e ~ N(0, σ·cap), which gives negative-inflow days as in ResOpsUS (dev and aux dams: median 2 % of days, up to 35 %); an upstream regulating reservoir on 30 % of tasks and gross data spikes on 10 %; a non-stationary scored decade (× exp(N(0, 0.3))); and the simulated operator's AR(1) release noise, which shapes demand and Qmax as real operators' records do. |
| **Network** | MLP 15 → 64 → 64 → 2, tanh, 5,314 weights (plus 30 feature-scaling numbers). Two sigmoid outputs h and e give the requested release R = h·D + 1.25·e·max(Qmax − D, 0); output biases start near SOP (4.5, −5.5). Trained by direct policy optimisation: torch back-propagation through WaterData's mass balance and cost over 270- to 1,000-day windows, then a 365-day SOP tail so leftover storage is valued by what it does next; loss mean log(J + 0.03); Adam, gradient clip 1, cosine learning rate, 96 windows per batch; checkpoints chosen on the simulated selection set. numpy inference, about 0.15 s per 10-year dam. |
| **Physics also enters** | inputs: Mass-balance features: headroom (cap − S), SOP spill pressure (S + I·K − D·K − cap), storage in days of demand, and expected overflow from the calibration-half inflow normal over the next 30 and 90 days<br>output: SOP: the head R = h·D + 1.25·e·(Qmax − D)+ starts at release-the-demand (h ≈ 1, e ≈ 0), and every release passes through WaterData's mass balance, which caps it by the water stored and spills above capacity; training differentiates J through that balance |
| **Real data's jobs** | backdrop, calibration, gate, exam (never training labels) |
| **Dev set** | WaterData's frozen split (shared/water/data/proc/splits.json, seed 20261118, frozen 18 Nov 2026): 15 ResOpsUS dev dams, one or more per main use (flood control, irrigation, hydropower, supply, other) × capacity above / below the median; free to use whole. 413 aux dams with incomplete records: calibration of single processes only, never scored |
| **Exam** | The other 250 core dams (at least 10 years of simultaneous storage, inflow and release, known capacity); 244 have a scorable second half. Per dam: the first half of the record from 1980 calibrates demand, Qmax and the climatologies; the second half (longest gap-free block, at most 10 years) is scored from the observed start storage with the observed inflow. Recent exam: 15 CDEC reservoirs, WY2021-2025, demand and Qmax from their ResOpsUS record 1980-2020. Mass-balance-consistent subset: the 151 exam dams whose record closes the daily balance (correlation ≥ 0.9) |

## Run it

```bash
cd models/reservoirpilot
pip install numpy pandas
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo reservoirpilot`.

```python
import reservoirpilot as rp
pol = rp.Pilot.load()                       # reservoirpilot_weights.npz next to this file
task = rp.make_task(cal_df, eval_df, capacity_hm3)   # WaterData's task from a daily record
R, S = pol.run(task)                        # release (m3/s), end-of-day storage (hm3)
rp.cost_components(R, S, task)              # J, its parts, and the storage J ignores
rp.sop(task), rp.tuned_rule(task)           # the comparators
# any daily record: python pipeline.py record.csv CAPACITY_HM3 [--out result.csv]
```

## The exam

WaterData's frozen reservoir exam, run once after the ratchet was closed (training/final_exam.py, training/exam_results.json). Medians over dams of the cost J and of each of its components. Baselines on the same dams and decades: the classical flood-pool and hedging rule grid-tuned on the simulator (on g4, rung c2: the fairest baseline; and on g1, rung c1), WaterData's storage-target rule curve, the real operators' record, SOP and the perfect-foresight DP bound. Intervals: the 90 % bootstrap interval of the median J as shipped (dams resampled); the exact (Clopper-Pearson) 95 % interval of the shipped paired win fractions (ties count against the pilot). Per-dam scores were not shipped, so there are no paired intervals on differences. J charges nothing for storage: the storage rows sit right after the headline. A rule tuned on the 15 dev dams would be a fairer baseline still; it needs the ResOpsUS series (open). Every number here is on real data that never chose anything.

| Metric | ReservoirPilot | Baselines | |
|---|---|---|---|
| ResOpsUS exam (244 dams): median cost J<br><sub>90 % bootstrap interval of the median (shipped): pilot 0.0097-0.0202, g4 rule 0.0207-0.0374, SOP 0.089-0.134. Pilot better on 62.7 % of dams than the g4 rule (exact 95 % interval 56.3-68.8 %, n = 244), 60.2 % than the g1 rule (53.8-66.4), 84.4 % than SOP (79.3-88.7), 94.7 % than historical (91.1-97.1), 95.1 % than the rule curve (91.6-97.4): every interval is above 50 %. It loses to SOP on 37 dams (15 %). J charges nothing for storage: see the next two rows. Perfect-foresight DP, a bound, 0.0063 (own row). A dev-tuned rule would be fairer still (not run: needs the ResOpsUS series).</sub> | **0.0154** | Flood-pool + hedging rule tuned on the simulator, g4 (rung c2) (classical): 0.0285<br>Same rule tuned on the simulator, g1 (rung c1) (classical): 0.0294<br>WaterData rule curve (storage target from the calibration half) (classical): 0.143<br>Historical operations (the record) (classical): 0.183<br>SOP: release the demand when the water is there (dumb baseline): 0.108 | ✅ wins |
| ResOpsUS exam (244 dams): median mean storage, share of capacity (not charged by J)<br><sub>Not graded, because J does not value stored water; reported because the pilot uses that. It keeps the reservoir lower than every comparator, and lower storage is part of how it wins J. Median lowest storage 1.2 % of capacity (SOP 0 %, historical 13.9 %). No interval: per-unit scores not shipped; needs the data rebuild (`mlratchet data water`).</sub> | **0.38** | Flood-pool + hedging rule, g4 (classical): 0.4<br>Flood-pool + hedging rule, g1 (classical): 0.48<br>WaterData rule curve (classical): 0.5<br>Historical operations (classical): 0.44<br>SOP (dumb baseline): 0.59<br>Perfect-foresight DP (bound) (classical): 0.68 | ℹ️ |
| ResOpsUS exam (244 dams): median end − start storage, share of capacity (not charged by J)<br><sub>Not graded. The status doc and shipped card say the pilot ends at least 20 % of capacity lower than it started on 34 % of dams (SOP 21 %); that does not reproduce from the shipped files (the per-dam table exam_per_dam.parquet was not shipped). On the mass-balance subset the median is −0.031 (SOP +0.261); on CDEC −0.098 (SOP +0.223).</sub> | **0.008** | Flood-pool + hedging rule, g4 (classical): 0.009<br>Flood-pool + hedging rule, g1 (classical): 0.118<br>WaterData rule curve (classical): 0.028<br>Historical operations (classical): 0.002<br>SOP (dumb baseline): 0.3<br>Perfect-foresight DP (bound; required to end at least as full) (classical): 0.3 | ℹ️ |
| ResOpsUS exam (244 dams): median shortage ratio (water short of demand ÷ demand)<br><sub>The pilot hedges: it releases less than 0.999 × demand on a median 67 % of days (g4 rule 5 %, SOP 1 %, DP 14 %, historical 55 %), small cuts that J's squared shortage term charges little for. J's shortage part: pilot 0.0094, g4 rule 0.0152, g1 rule 0.0084, SOP 0.0045, rule curve 0.122, historical 0.168, DP 0.0057. No interval: per-unit scores not shipped; the gap to SOP and both tuned rules is large.</sub> | **0.066** | Flood-pool + hedging rule, g4 (classical): 0.037<br>Flood-pool + hedging rule, g1 (classical): 0.019<br>WaterData rule curve (classical): 0.186<br>Historical operations (classical): 0.285<br>SOP (dumb baseline): 0.01 | ❌ loses |
| ResOpsUS exam (244 dams): median days per year above Qmax<br><sub>It goes above the safe channel capacity on more days than any comparator, but by far less (next row): many small floods instead of a few large ones, which J's squared penalty prefers; downstream it might not. J's flood part: pilot 0.00004, both tuned rules and the rule curve 0, SOP 0.0109, historical 0.0012. No interval: per-unit scores not shipped.</sub> | **2.9 days/yr** | Flood-pool + hedging rule, g4 (classical): 0 days/yr<br>WaterData rule curve (classical): 0 days/yr<br>Historical operations (classical): 1.9 days/yr<br>SOP (dumb baseline): 2.2 days/yr | ❌ loses |
| ResOpsUS exam (244 dams): median worst release above Qmax, ÷ Qmax<br><sub>Far below SOP (96 % of Qmax) and the real operators (27 %), above the tuned rule and the rule curve, which never exceed Qmax at the median. Days near empty (storage < 1 % of capacity), median: pilot 0, SOP 1.6 %; days full: pilot 0, SOP 10.0 %, DP 6.7 %. No interval: per-unit scores not shipped.</sub> | **0.04** | Flood-pool + hedging rule, g4 (classical): 0<br>WaterData rule curve (classical): 0<br>Historical operations (classical): 0.268<br>SOP (dumb baseline): 0.958 | ↔️ mixed |
| Mass-balance-consistent subset (151 exam dams, record closes the daily balance): median J<br><sub>90 % bootstrap interval of the median (shipped): pilot 0.0131-0.0255, g4 rule 0.0248-0.0460, SOP 0.0848-0.1237. Pilot better on 64.9 % of dams than the g4 rule (exact 95 % interval 56.7-72.5 %, n = 151), 60.9 % than the g1 rule (52.7-68.8), 82.8 % than SOP (75.8-88.4), 97.4 % than historical (93.4-99.3), 96.7 % than the rule curve (92.4-98.9). Mean storage 0.37 (SOP 0.54, g4 rule 0.35); end − start −0.031 (SOP +0.261, g4 rule −0.017). DP 0.0071.</sub> | **0.0162** | Flood-pool + hedging rule tuned on the simulator, g4 (classical): 0.03<br>Same rule, g1 (classical): 0.0354<br>WaterData rule curve (classical): 0.151<br>Historical operations (classical): 0.2<br>SOP (dumb baseline): 0.0992 | ✅ wins |
| CDEC recent exam (15 California reservoirs, WY2021-2025): median J<br><sub>With 15 reservoirs the tuned rule is a tie: pilot better on 8 of 15 (53 %, exact 95 % interval 27-79 %); g1 rule 9 of 15 (60 %, 32-84 %). It beats SOP on 14 of 15 (68-99.8 %) and the record on 15 of 15 (78-100 %). 90 % bootstrap interval of the median (shipped): pilot 0.0002-0.0546, g4 rule 0.000-0.0756, SOP 0.054-0.158. End − start −0.098 of capacity (g4 rule −0.143, SOP +0.223); mean storage 0.39 (SOP 0.54). DP 0.0124. The two tuned rules share the median J 0.0354 in the shipped table.</sub> | **0.0193** | Flood-pool + hedging rule tuned on the simulator, g4 (classical): 0.0354<br>Same rule, g1 (classical): 0.0354<br>WaterData rule curve (classical): 0.24<br>Historical operations (classical): 0.108<br>SOP (dumb baseline): 0.0947 | ➖ tie |
| Perfect-foresight DP (knows the whole decade's inflow), ResOpsUS 244 dams: median J<br><sub>Not a policy anyone can run. DP is better on 69.7 % of dams (pilot better on 29.9 %, exact 95 % interval 24.2-36.1 %); 90 % interval of DP's median 0.0049-0.0082. The pilot closes 91 % of the median gap from SOP to DP; most of the rest is future-inflow information. DP is itself approximate: on import, the shipped page's JavaScript DP gave J 0.00027 instead of 0.00129 on one demo dam (grid rounding changes its tie-breaks); the site instrument ports numpy's arithmetic and matches WaterData's DP to 1e-14.</sub> | **0.0154** | Perfect-foresight DP, WaterData (201 storage levels; a bound, not a competitor) (classical): 0.0063 | ℹ️ |
| By main use, flood control (97 dams): median J<br><sub>Medians per main use from training/exam_results.json (by_main_use); per-dam scores not shipped, so no interval. DP 0.0055.</sub> | **0.0053** | Flood-pool + hedging rule, g4 (classical): 0.0081<br>Historical operations (classical): 0.249<br>SOP (dumb baseline): 0.186 | ℹ️ |
| By main use, irrigation (88 dams): median J<br><sub>No interval (per-dam scores not shipped). DP 0.0064.</sub> | **0.0186** | Flood-pool + hedging rule, g4 (classical): 0.0345<br>Historical operations (classical): 0.141<br>SOP (dumb baseline): 0.0517 | ℹ️ |
| By main use, hydropower (25 dams): median J<br><sub>No interval (per-dam scores not shipped). DP 0.0208.</sub> | **0.0522** | Flood-pool + hedging rule, g4 (classical): 0.0718<br>Historical operations (classical): 0.153<br>SOP (dumb baseline): 0.108 | ℹ️ |
| By main use, water supply (20 dams): median J<br><sub>No interval (per-dam scores not shipped). DP 0.0033. Small groups: navigation (7 dams) pilot 0.125, g4 rule 0.148, SOP 0.155; recreation (4) 0.017, 0.020, 0.691; other (3) 0.052, 0.072, 0.056.</sub> | **0.0062** | Flood-pool + hedging rule, g4 (classical): 0.0202<br>Historical operations (classical): 0.181<br>SOP (dumb baseline): 0.0522 | ℹ️ |

### Where it wins

- The best causal policy on WaterData's reservoir exam: median J 0.0154 on 244 real dams against 0.0285 for the classical flood-pool and hedging rule tuned on the same simulator, 0.108 for SOP and 0.183 for the real operators; it beats the tuned rule on 63 % of dams (95 % interval 56-69 %), SOP on 84 % (79-89 %) and the record on 95 % (91-97 %).
- It holds on the 151 dams whose records close the mass balance (0.0162 vs 0.0300 for the tuned rule) and on flood-control, irrigation, hydropower and supply dams alike.
- Floods: its worst release above Qmax is a median 4 % of Qmax, against 96 % for SOP and 27 % for the real operators.
- Trained only in simulation: no real reservoir record was a training sample; the exam was opened once, after the ratchet closed.

### Where it loses

- It buys part of its J with stored water, which J does not charge for: median storage 38 % of capacity against SOP's 59 % and DP's 68 %. The shipped card says it ends at least 20 % of capacity lower than it started on 34 % of dams (SOP 21 %; not reproducible from the shipped files); on the 15 CDEC reservoirs the median end − start is −0.10 of capacity.
- It loses to SOP on 37 of 244 dams (15 %), by the shipped card mostly large irrigation or supply reservoirs (median 2.0 years of inflow) whose scored decade was drier than the calibration decade (per-dam table not shipped, so not re-checked).
- It rations often: release below demand on a median 67 % of days, and its shortage ratio (6.6 %) is worse than SOP's (1.0 %) and the tuned rule's (3.7 %).
- It exceeds Qmax on more days a year than any comparator (2.9 against 2.2 for SOP and 0 for the tuned rule), by small amounts.
- On the 15 CDEC reservoirs (WY2021-2025) it only ties the tuned rule (better on 8 of 15).
- Perfect-foresight DP is still 2.5 × lower (0.0063): the pilot has no inflow forecast.

## Limits

- J is the whole objective. It ignores stored water, evaporation, minimum environmental flows, hydropower head, water-quality and dam-safety rules; a policy tuned to J is not a policy to operate a dam with.
- Single reservoir, daily step: no cascades, no evaporation (the benchmark ignores it), no minimum flow, demand a fixed calendar from the calibration half.
- The calibration half sets demand, Qmax and the inflow normal; when the scored decade is drier, the policy keeps releasing for a normal that no longer comes (its known failure on large supply dams).
- ResOpsUS inflow is mostly back-calculated from storage changes: 189 of the 678 ResOpsUS dams have negative inflows somewhere, and 93 of the 244 exam dams fail the mass-balance check.
- Weather data: the simulated training inflows are driven by ERA5-Land, which under-catches mountain and Pacific Northwest precipitation (50 CAMELS basins have a runoff ratio above 1) and drifts about 6 % between decades; this caps every Watershed method. The exam runs on recorded ResOpsUS inflow, so it enters ReservoirPilot only through the training inflows. Literature streamflow numbers (Daymet- or NLDAS-forced LSTMs, NSE about 0.69-0.76) are not comparable.
- Rule notes: aux-dam statistics (6 complete dams, inflow ratios on 29) set the generator's ranges; the climatology features use each exam dam's calibration-half inflow, as WaterData's rule curve does; the dev dams both set generator ranges and gate rungs (partly in-sample); two void training runs (missing optimiser step) were removed and logged.

## The ratchet

**Objective:** S_J = 1 − median J(policy) / median J(SOP) on the frozen simulated set of the current generator version (300 tasks per version, seeds 9000+; WaterData's cost J and mass balance, the whole scored window from the task's start storage). Final generator g4; the g1-g3 columns are re-scores on the earlier frozen sets.  
**Rule:** Keep if S_J on the frozen set beats the incumbent's by more than 0.01 AND the median J on the 15 real dev dams is at most 1.05 × the incumbent's (the first rung: at most SOP's dev median, 0.0964). Checkpoints are chosen on a separate simulated selection set (200 tasks per version, seeds 7000+), never on the frozen set or real data.

| Rung | Change | Frozen-set score | Decision | Why | S_J g1 | S_J g2 | S_J g3 | S_J g4 | Dev median J | Dev vs incumbent | On g1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| void-1 | DPO through the differentiable simulator, free-form head (first attempt) | — | ○ rejected | Void: train.py had no opt.step(), so the weights never moved and the flat validation curve meant nothing. Removed from the log and re-run (r3 is the free-form re-run). | – | – | – | – | – | – | void |
| void-2 | DPO, rule head (first attempt) | — | ○ rejected | Void for the same missing optimiser step. Removed from the log and re-run (r2 is the rule-head re-run). | – | – | – | – | – | – | void |
| r1b | Behaviour cloning of WaterData's perfect-foresight DP on 400 simulated g1 tasks (DP release along DP's own storage path), free-form h/e head, 11 base features | 0.089 | ● kept | First rung: S_J 0.089 > 0.01 over SOP on g4; dev 0.0960 ≤ SOP's 0.0964. Cloning a policy that knows the future taught little: the network cannot see what DP saw. | 0.247 | 0.047 | 0.164 | 0.089 | 0.096 | −0.4 % vs SOP | kept |
| c1 | Classical rule, no network: SOP + flood pool (draw down toward f × cap at up to Qmax over tau days) + linear hedging over H days of demand; f = 0.75, tau = 3 d, H = 10 d grid-tuned on 200 simulated g1 tasks (training/classical.py) | 0.623 | ● kept | S_J 0.623 vs 0.089 (+0.534); dev 0.0422 vs 0.0960. The sim-tuned classical rule beats cloned DP by a wide margin. | 0.845 | 0.566 | 0.77 | 0.623 | 0.0422 | −56 % | kept |
| r2 | DPO: back-propagate J through the differentiable mass balance (torch BPTT); the network sets c1's three parameters (f, tau, H) every day, initialised at the tuned constants; base features, g1 bank, 1,200 iterations | 0.756 | ● kept | S_J 0.756 vs 0.623 (+0.134); dev 0.0409 vs 0.0422 (−3 %). | 0.862 | 0.649 | 0.77 | 0.756 | 0.0409 | −3.1 % | kept |
| r3 | DPO, free-form head: R = h·D + 1.25·e·(Qmax − D)+, two sigmoids; base features, g1 bank, 1,200 iterations | 0.659 | ○ rejected | On g4: S_J 0.659 vs r2's 0.756 (fails); dev 0.0361 (passes). Kept on g1 (0.876 vs 0.862), rejected when re-taken on g4. | 0.876 | 0.518 | 0.683 | 0.659 | 0.0361 | −11.8 % | kept |
| r4 | r3 + calibration-half inflow climatology features (next 30 and 90 days, expected overflow): 15 inputs; g1 bank, 1,200 iterations, checkpoint on the selection set | 0.704 | ○ rejected | On g4: S_J 0.704 vs r2's 0.756 (fails); dev 0.0367 (passes). Kept on g1 (0.896 vs 0.876), rejected when re-taken on g4. | 0.896 | 0.599 | 0.717 | 0.704 | 0.0367 | −10.2 % | kept |
| c2 | c1 re-tuned on the g4 generator: f = 0.6, tau = 3 d, H = 10 d (no network) | 0.717 | ○ rejected | S_J 0.717 vs r2's 0.756 (fails); dev 0.0475 vs 0.0409 (+16 %, fails). It is still the exam's fairest baseline: the tuned classical rule. | 0.794 | 0.561 | 0.735 | 0.717 | 0.0475 | +16.1 % | – |
| r5 | Generator g4 (back-calculated inflow noise, upstream regulation, data spikes, non-stationary halves): r4 fine-tuned on the g4 bank, 1,200 iterations, checkpoint on the g4 selection set | 0.767 | ● kept | S_J 0.767 vs r2's 0.756 (+0.011, just above the margin); dev 0.0363 vs 0.0409 (−11 %). | 0.843 | 0.679 | 0.811 | 0.767 | 0.0363 | −11.1 % | – |
| r6 | Rule head + climatology features, trained from scratch on g4 | — | ○ rejected | Diverged and stopped at 470 of 1,200 iterations: gradient norm 1e53 through the rule head's 1/tau term; batch median J rose from 0.000 to 0.09-0.15. No checkpoint scored. | – | – | – | – | – | – | – |
| r7 | r5 continued with 540-day policy windows (+ 365-day SOP tail), lr 1e-3, 1,000 iterations, g4 bank | 0.828 | ● kept | S_J 0.828 vs 0.767 (+0.061); dev 0.0332 vs 0.0363 (−9 %). | 0.878 | 0.757 | 0.85 | 0.828 | 0.0332 | −8.6 % | – |
| r8 | r7 continued with 1,000-day policy windows, lr 7e-4, 600 iterations (checkpoint 400 chosen on the g4 selection set) | 0.846 | **◆ released** | S_J 0.846 vs 0.828 (+0.018); dev 0.0335 vs 0.0332 (+0.8 %, within 5 %). The ratchet was closed before the exam; released as v1.0. | 0.872 | 0.753 | 0.852 | 0.846 | 0.0335 | +0.8 % | – |

<sub>Columns after *Why*: S_J on the four frozen simulated sets (re-scores; g4 decides), then the 15 dev dams: median J, its change against the incumbent at the time of the g4 decision, and the decision taken originally on g1.</sub>

Generator rungs (the physics axis): g1 WaterSim inflow and a random simulated historical operator; g2 adds back-calculated inflow noise (e_t − e_t−1)/K, e ~ N(0, σ·cap), which gives negative-inflow days as in ResOpsUS (dev and aux median 2 %, up to 35 %); g3 adds an upstream regulating reservoir on 30 % of tasks and gross data spikes on 10 %; g4 makes the halves non-stationary (scored-decade inflow × exp(N(0, 0.3))). g2-g4 ranges were set from the 15 dev dams and the aux dams with complete records (6 complete, inflow ratios on 29); the exam was never read. When the frozen set switched from g1 to g4, every rung was re-scored and every decision re-taken in order; the decisions below are the re-taken ones, with the g1 decision in the last column. r3 and r4 were kept on g1 and are rejected on g4, so r5 (r4 fine-tuned on g4) was judged against r2. Not rungs, logged in training/ratchet.json notes: a first objective with a hard end-storage penalty (1000 × deficit / cap) inside one-year windows punished natural seasonal drawdown and was replaced by a differentiable 365-day SOP tail; two first DPO runs (free-form head, then rule head) never applied the optimiser step (train.py lacked opt.step()), so their weights never moved; they were removed and re-run (the rows void-1 and void-2 below). The DPO free-form rung r3 was first run as "r1" (runs/r1_dpo_hd_g1; the twin test file keeps that name). r5's margin over r2 (+0.011) is barely above the 0.01 rule. Every rung, the final one included, ends the simulated windows with less water than it started with (median end − start on frozen g4: −0.23 of capacity for r8) and the ratchet never charged for it, because J does not.

Raw per-rung logs are in [`training/`](training/).

## Physics

Reservoir water balance, WaterData's exam benchmark unchanged: S(t+1) = S(t) + (I − R)·K with K = 0.0864 hm³ per m³/s·day, release capped by the water stored, spill above capacity counted as release, no evaporation. The training inflows come from WaterSim, the Watershed simulator: generated basins run through HBV, GR4J, SAC-SMA or VIC-lite with degree-day or energy-balance snow and channel routing. A simulated historical operator runs each dam's first decade, and the task (demand, Qmax, climatologies, start storage) is then built from that record exactly as WaterData builds it from a real one.

**Simulator**

- WaterSim v1 (shared/water/watersim): half the basins from the CAMELS prior with a donor basin's real ERA5-Land forcing from non-exam years (Caravan), half from the default prior with the stochastic weather generator; one of four runoff models and two snow routines; half the inflows get extra channel attenuation
- Dam: capacity log-normal, median 0.8 years of mean inflow (0.005 to 40 years); a random simulated operator (follow-inflow, rule curve, SOP or Hanasaki 2006, with AR(1) release noise) runs the first 10 years; the next 10 are scored
- Generator rungs, ranges from the 15 dev dams and the aux dams (never exam): g2 back-calculated inflow noise; g3 upstream regulation (30 % of tasks) and gross data spikes (10 %); g4 a non-stationary scored decade, inflow × exp(N(0, 0.3)) (dev scored/calibration inflow ratio median 0.81, 29 aux dams 1.17; a stationary simulator gives about 1)
- Banks: 3,000 training tasks, 300 frozen and 200 selection tasks per generator version (training/bank.py, selset.py; seeds 1000+, 9000+, 7000+)

**Checked against independent references**

- WaterSim (shared/water/checks): GR4J vs airGR 1.7.9 to 1.0e-7 mm (NSE 0.79882 vs the published 0.7988); HBV vs TUWmodel 1.1-1 identical to ≤ 1e-12 mm; water balance closes in 121/121 runs, worst 1.4e-15 relative; replaying observed inflow and release on the 15 dev dams ends within a median 5 % of capacity (reservoir_dev.json)
- Task reproduction: WaterData's own functions (doy_curve, cost, simulate, dp) are imported, not copied; dev baselines match WaterData's per-dam table to 1.6e-9 and the exam medians exactly (one historical value differs by 3e-5; status doc, not in a shipped file). Re-checked on import: the shipped baseline medians equal shared/water/data/proc/baselines/reservoir_scores.parquet (SOP 0.108372, rule 0.142997, historical 0.183264, DP 0.006254)
- Policy equivalence (training/checks_equiv.json, check_js.json): numpy policy vs WaterData's operating loop 8.3e-17 in J; vs the torch training rollout 1.4e-8 in J; the shipped page's JavaScript vs Python 5.0e-15 (release / Qmax) on 5 dev dams, 1,100 days. The site instrument (TypeScript) vs the Python module: 1.5e-14 on 30 cases, WaterData's DP included
- Generator vs reality: on g4's frozen set SOP and DP have median J 0.105 and 0.0069; on the real exam 0.108 and 0.0063 (known from the brief, not tuned to). The status doc's g4 rule-curve median 0.165 (exam 0.143) is not in a shipped file
- Twin test (dev dams, training/twin_r1_dpo_hd_g1_g1.json; weights of rung r3): replacing each dev dam's inflow with its 5-day mean moved the pilot's median J only from 0.0361 to 0.0359 (SOP 0.0964 to 0.0683), so gauge noise was not the main sim-to-real gap; that pointed at non-stationary decades and led to g4

## Help wanted

- The ResOpsUS and CDEC daily records are not in the repository (`mlratchet data water` rebuilds them). With them: tune the classical rule on the 15 dev dams (the fairest baseline), recompute per-dam exam scores for paired intervals, and re-check the 34 % and 37-dam failure claims (the sandbox's exam_per_dam.parquet was not shipped).
- Reservoir evaporation and operating rules (minimum flows, flood-control curves) for the exam dams, to make J closer to a real operator's objective.

## Next rungs

- A storage-aware objective: report and train under J plus a carry-over or head term to price the low-storage behaviour, or condition the policy on an operator-chosen storage floor.
- MPC, or an inflow-forecast input (FlowCast's nowcast is a natural partner): DP is still 2.5 × lower, mostly from future-inflow information.
- Adaptation to a drier or wetter decade: a 1-3 year inflow moving-average feature that lets demand and the inflow normal drift online, and a wider g4 spread.
- A fairer baseline: the flood-pool and hedging rule tuned on the 15 dev dams, and paired intervals from per-dam exam scores, once the ResOpsUS series are rebuilt.
- Cascades, evaporation, minimum flows and an hourly flood mode.
- Upstream to WaterSim: the reservoir task generator (operator-derived demand and Qmax, back-calculated inflow noise, upstream regulation and spikes, the non-stationary decade factor), and the realdata.random_forcing window arithmetic fix (span × 365 days is shorter than DateOffset(years=span), so 20-year draws can end a few days inside the exam period; the guard raises and ReservoirPilot redraws).

## Files

| Path | What |
|---|---|
| `reservoirpilot.py`, `pipeline.py` | Inference — numpy/scipy only |
| `reservoirpilot_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [ResOpsUS v2: daily storage, inflow and release of US reservoirs (Steyaert, Condon, Turner & Voisin 2022, Scientific Data 9:34)](https://doi.org/10.5281/zenodo.6612040) | dev (15 dev dams: generator ranges, the dev gate, the twin test), physics (aux dams, never scored: 6 complete records and inflow ratios on 29 set the g2-g4 ranges), exam (244 scored dams; examples/exam_dams.json holds six of them) | CC BY 4.0 |
| [GRanD dam attributes: capacity, main use (Lehner et al. 2011, Front. Ecol. Environ. 9:494), via ResOpsUS+CARS v1.0 (Casado-Rodriguez et al.)](https://doi.org/10.5281/zenodo.15978041) | dev, exam (capacity, raised to the observed maximum storage where larger; main-use strata of the split) | CC BY 4.0 |
| [California Data Exchange Center (CDEC), California Department of Water Resources: daily storage, inflow and outflow](https://cdec.water.ca.gov) | exam (recent exam: 15 California reservoirs, WY2021-2025) | Public data from California DWR; no explicit licence text (credit DWR). Not share-alike. |
| [Caravan v1.6 (Kratzert et al. 2023, Scientific Data 10:61) with CAMELS-US (Newman et al. 2015; Addor et al. 2017) attributes and ERA5-Land forcing (Muñoz-Sabater et al. 2021)](https://doi.org/10.5281/zenodo.15529786) | physics (donor-basin forcing for half the simulated inflows, non-exam years only; the camels prior of WaterSim's basin generator) | CC BY 4.0 (Caravan, CAMELS); ERA5-Land under the Copernicus licence (free, attribution) |
| [WaterSim v1 (Watershed Phase 1): HBV, GR4J, SAC-SMA, VIC-lite, degree-day and energy-balance snow, routing; Hanasaki et al. (2006) for one simulated operator type](https://github.com/dogum/ML-Ratchet/tree/main/shared/water/watersim) | physics (training inflows) | MIT (code in this repository) |
| [WaterData reservoir exam (Watershed Phase 1): task definition, cost J, frozen split and baselines (SOP, rule curve, historical, perfect-foresight DP)](https://github.com/dogum/ML-Ratchet/tree/main/shared/water/waterdata) | exam (baselines: shared/water/data/proc/baselines/reservoir_scores.parquet, reservoir_recent_scores.parquet) | Code MIT; scores derived from ResOpsUS / GRanD (CC BY 4.0) and CDEC |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
