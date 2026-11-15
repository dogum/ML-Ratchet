<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# PlantDetective <sub>v1.1</sub>

> **A power unit's technology, heat rate, minimum load, ramps and starts from its output.**

*What kind of fossil unit is this, how efficient is it, and how flexibly does it really run — read from its hourly output?*

PlantDetective reads a fossil power unit's hourly output (MW) and infers its technology (coal steam, gas combined cycle, gas turbine, gas steam), full- and part-load heat rate, minimum stable load, ramps, minimum up and down times, starts per year and their hot/warm/cold shares, capacity factor and an operating-flexibility class, each with a 68% and 95% range. Prices and EIA-860/923 facts are optional inputs. It learned only from 1,829 simulated units (GridSim systems plus a self-commitment dynamic programme and a CAMPD measurement emulator) and was examined once on 395 real Texas CAMPD units. v1.1 predicts starts per year, minimum up time, median run and minimum down time as learned corrections to the statistic counted in the window itself; it was retrained on the simulated bank rebuilt with GridSim physics 1.1 and examined once more on the same 395 units.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/plantdetective/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/plantdetective/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Hourly unit output in MW (gross, at least 30 days; NaN = missing) with UTC timestamps; optionally the hourly electricity price ($/MWh) and gas price ($/MMBtu), and EIA-860/923 facts (prime mover, CHP flag, nameplate, minimum load, cold-start time, online year, annual heat rate). Several years are read window by window and pooled. |
| **Returns** | Technology probabilities: coal steam, gas combined cycle, gas turbine, gas steam<br>Full-load and part-load heat rate (MMBtu/MWh, CAMPD gross basis; part load = max(min stable load, 40%)) and the heat-rate curve<br>Minimum stable load (fraction of capacity), ramp up and down (99th percentile hourly change while running)<br>Starts per year and hot (< 8 h off) / warm (8-48 h) / cold (>= 48 h) shares; minimum up, median run and minimum down time (h)<br>Capacity factor and an operating-flexibility class (baseload, load-following, cycling, peaking; a rule on the estimates)<br>Every estimate with 68% and 95% ranges, widened by dev-fitted factors<br>v1.1: starts per year, minimum up, median run and minimum down time are the window's own counted statistic plus the network's correction (where the window has one) |
| **Learned from** | Two layers of unit physics on the shared grid platform (shared/grid). GridSim generates power systems and dispatches them with a unit-level MILP driven by real EIA-930 demand, wind and solar shapes and Henry Hub gas of the train-input years. Each simulated unit then commits itself against a price background with an exact dynamic programme: on/off and time in state, 7 output levels, a convex fuel curve, hot/warm/cold start costs, minimum up and down times, ramp and start-up/shut-down limits, forced and planned outages. A measurement emulator turns the schedule into a CAMPD-like hourly record and an EIA-923-like annual heat rate. |
| **Noise** | No real labelled unit is a training sample. The CAMPD emulator above supplies the measurement effects, with its start/stop shapes, trip rates, offer slope, start-cost multipliers, CHP steam credit and EIA-923 gross ratio calibrated on the 10 dev plants (dev years 2019 and 2021). |
| **Network** | Ensemble of 3 MLPs (171 inputs: 170 window features + a net-output flag -> 256 -> 256 -> 128 -> 4 technology logits + 13 means + 13 log sigma; leaky ReLU), 440,797 weights; Gaussian NLL + cross-entropy; variance = mean member variance + spread of member means; windows pooled by summed log-probabilities and precision-weighted means. v1.1: the means of starts per year, minimum up, median run and minimum down time are the counted statistic of the window plus the network output (count skips), in the transformed target space |
| **Physics also enters** | inputs: On-probability and load level against the market implied heat rate (price / gas, MMBtu/MWh, 12 bins): when a unit runs compared with the fuel it would burn<br>output: v1.1: the run statistics counted in the window (starts per year, minimum up, median run, minimum down) are the backbone; the network outputs a correction to each |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 10 dev plants (40 CAMPD units: 2 coal steam, 2 gas steam, 3 combined cycle, 3 gas-turbine plants), dev years 2019 and 2021 only — GridData's frozen plant split (shared/grid/data/proc/splits.json, sha256_of_content 895b4e9a…, seed 20261113) |
| **Exam** | 129 exam plants, 407 CAMPD units (395 with enough full-load hours for a truth), all years 2018-2025, each scored against its whole-record CAMPD truth; 2026 (reserve year) not used |

## Run it

```bash
cd models/plantdetective
pip install numpy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo plantdetective`.

```python
import json, numpy as np
from plantdetective import Detective
det = Detective()                                  # v1.1: 3-network ensemble in plantdetective_weights.npz
z = np.load("examples/6179_3.npz")                 # a real exam coal unit, EPA CAMPD 2018-2025
meta = json.loads(str(z["meta"]))
r = det.infer(z["x"].astype(float), z["t"], z["price"], z["gas"], meta["context"])
print(r["tech"], r["estimates"]["pmin_frac"], r["flex_class"])
```

## The exam

Every exam unit's hourly output was cut into windows (one month, one quarter, one calendar year) and also read whole (all years pooled); each reading is scored against the unit's whole-record CAMPD truth. Median errors: fractions in absolute points, the rest relative. Mode 'ctx' (output + ERCOT hub price + Henry Hub gas + EIA-860/923 facts) unless a row says otherwise. Baselines: simple statistics of the same output, technology defaults (given the true technology), the EIA-923 annual heat rate, and the dev median per true technology. Run once per version, after the release choice was committed. The rows are v1.1's (run once on 15 Nov 2026); each note keeps the v1.0 value. The exam fed 2023-2025 ERCOT prices as optional context; no choices were made from them. Every number here is on real data that never chose anything.

| Metric | PlantDetective | Baselines | |
|---|---|---|---|
| Minimum stable load, all years pooled (395 units)<br><sub>v1.0: 5.7 pt. One calendar year: 6.0 pt (v1.0: 6.8; 2,835 windows).</sub> | **4.7 pt** | Output 1st percentile while running (classical): 37.9 pt<br>Technology default (true technology) (classical): 14 pt<br>Dev median per true technology (dumb baseline): 8 pt | ✅ wins |
| Minimum stable load, from one month (11,463 windows)<br><sub>v1.0: 10.8 pt. One quarter: 8.9 pt (v1.0: 9.2), also worse than the dev median.</sub> | **10.1 pt** | Output 1st percentile while running (classical): 29.4 pt<br>Technology default (true technology) (classical): 13.2 pt<br>Dev median per true technology (dumb baseline): 8 pt | ❌ loses |
| Ramp up (99th percentile hourly change), all years pooled<br><sub>v1.0: 13.7% (v1.1 is worse). One year 20.4% (v1.0: 17.7%), one month 26.4% (v1.0: 24.8%); dev median 34.9% / 35.5%.</sub> | **14.9%** | Output 99th percentile hourly change (classical): 69.6%<br>Dev median per true technology (dumb baseline): 38.9% | ✅ wins |
| Starts per year, from one month<br><sub>Error on (starts + 1). v1.0: 44.9%.</sub> | **44.1%** | Starts counted in the month, scaled to a year (classical): 56.9%<br>Dev median per true technology (dumb baseline): 69.9% | ✅ wins |
| Starts per year, from one calendar year<br><sub>v1.0: 31.0% (a loss). v1.1 predicts a correction to the count and wins by 0.2 points (27.62% vs 27.80%), a thin margin. All years pooled: 8.5% (v1.0: 23.3%); the count over all years is the truth's own definition (0.1%). The error bars are now far too narrow: 17% of year windows fall inside the 68% band (v1.0: 69%).</sub> | **27.6%** | Starts counted in the year (classical): 27.8%<br>Dev median per true technology (dumb baseline): 69.9% | ✅ wins |
| Median run length, from one month<br><sub>v1.0: 42.2% (v1.1 is worse on one month). One quarter 32.3% vs 25.0% (v1.0: 35.7%); one year 19.4% vs 18.2% (v1.0: 33.9%). Plain counts still win at every length.</sub> | **45.1%** | Median run counted in the month (classical): 23.1% | ❌ loses |
| Full-load heat rate, all units, all years pooled<br><sub>v1.0: 13.7%. Output only (no prices, no EIA facts): 14.1% (v1.0: 22.0%).</sub> | **11.6%** | Technology default (true technology) (classical): 10.3%<br>EIA-923 annual heat rate (classical): 14.1%<br>Dev median per true technology (dumb baseline): 18.7% | ❌ loses |
| Full-load heat rate, coal steam (26 units, all years)<br><sub>v1.0: 4.4% (v1.1 is worse: 5.01% vs 5.18% for the dev median, a thin margin).</sub> | **5%** | EIA-923 annual heat rate (classical): 10.3%<br>Technology default (classical): 7.5%<br>Dev median (dumb baseline): 5.2% | ✅ wins |
| Full-load heat rate, gas steam (61 units, all years)<br><sub>v1.0: 4.7% (a win). v1.1 lost this row: 6.3% vs 5.8% for the dev median. The count skips do not touch the heat-rate heads; the change comes from retraining on the physics-1.1 bank (new windows).</sub> | **6.3%** | EIA-923 annual heat rate (classical): 22.2%<br>Technology default (classical): 11%<br>Dev median (dumb baseline): 5.8% | ❌ loses |
| Full-load heat rate, gas combined cycle (154 units, all years)<br><sub>v1.0: 20.7%. Biased high. All 3 dev combined-cycle plants are CHP in EIA-860: two have truths of 10-13 MMBtu/MWh (process steam or gas-turbine-only reporting), one (55327) reports the whole block (truth 7.0, read as 9.9-10.7 by the network); most exam units report the whole block (about 7.3). One whole-block dev plant cannot support a CC-specific fit, so none was tried.</sub> | **23.5%** | Technology default (classical): 7%<br>EIA-923 annual heat rate (classical): 21.8%<br>Dev median (dumb baseline): 36.7% | ❌ loses |
| Full-load heat rate, gas turbine (154 units, all years)<br><sub>v1.0: 16.2%. The dev gas turbines did not show the bias (median +1%; 6.0% error vs 5.1% for EIA-923 on 8 dev units), so no turbine-specific change was made. The gain comes from retraining on the physics-1.1 bank.</sub> | **8.8%** | EIA-923 annual heat rate (classical): 6.5%<br>Technology default (classical): 12.4%<br>Dev median (dumb baseline): 18.2% | ❌ loses |
| Part-load heat rate, all units, all years pooled<br><sub>v1.0: 14.5% (14.53% both).</sub> | **14.5%** | EIA-923 annual heat rate (classical): 11.8%<br>Technology default (true technology) (classical): 13.3%<br>Dev median per true technology (dumb baseline): 40.4% | ❌ loses |
| Capacity factor, from one month<br><sub>v1.0: 5.3 pt. All years pooled: 1.6 pt (v1.0: 2.1), against 0 for the record's own capacity factor (the truth's definition).</sub> | **5.4 pt** | Capacity factor of the month itself (classical): 10.6 pt<br>Dev median per true technology (dumb baseline): 17.7 pt | ✅ wins |
| Annual heat input per unit-year, coal steam (178 unit-years)<br><sub>GridData's baseline metric; the model reads that year's window. v1.0: 5.7%.</sub> | **6.4%** | EIA-923 annual heat rate x output (classical): 8.8% | ✅ wins |
| Annual heat input per unit-year, gas steam (432 unit-years)<br><sub>v1.0: 10.1% (v1.1 is worse: 11.5% vs 11.8%, a thin margin).</sub> | **11.5%** | EIA-923 annual heat rate x output (classical): 11.8% | ✅ wins |
| Annual heat input per unit-year, gas turbine (680 unit-years)<br><sub>v1.0: 17.3%.</sub> | **11.3%** | EIA-923 annual heat rate x output (classical): 5.1% | ❌ loses |
| Annual heat input per unit-year, combined cycle (1,209 unit-years)<br><sub>v1.0: 22.5%. All 2,499 unit-years: 16.7% (v1.0: 17.1%); on the 2,255 with an EIA-923 figure 15.2% vs 13.7% (v1.0: 16.1%).</sub> | **26.5%** | EIA-923 annual heat rate x output (classical): 22% | ❌ loses |
| Technology from the output alone (one month / all years pooled)<br><sub>v1.0: 66% / 76.5%. With the EIA-860 prime mover (network + rule): 95% from one month, 95% from one year, 91% pooled (v1.0: 95%, 96%, 93%).</sub> | **61% / 74.7%** | — | ℹ️ |

Coverage of the 68% error bars on the exam: Min stable load (year windows) **73%**, Ramp up **61%**, Ramp down **59%**, Starts per year **17%**, Min up time **40%**, Median run **33%**, Min down time **31%**, Hot-start share **58%**, Warm-start share **50%**, Cold-start share **53%**, Full-load heat rate **81%**, Part-load heat rate **79%**, Capacity factor **65%**.

### Where it wins

- Minimum stable load from all years: 4.7 pt (v1.0: 5.7) against 8.0 for the dev median and 37.9 for the output's own 1st percentile.
- Ramp limit: 14.9% (v1.0: 13.7%) against 38.9% for the dev median and 69.6% for the output's own 99th percentile.
- Coal heat rate: 5.0% (v1.0: 4.4%) against 10.3% for the EIA-923 annual figure and 5.2% for the dev median.
- Starts per year: from one month 44.1% (counting 56.9%); from one calendar year 27.6% against 27.8% for the count (v1.0 lost this row at 31.0%; the v1.1 margin is 0.2 points).
- Short records: from one month, capacity factor 5.4 pt (the month's own 10.6 pt).
- Annual heat input of coal and gas-steam units beats EIA-923 (6.4% vs 8.8%, 11.5% vs 11.8%).

### Where it loses

- Gas-steam heat rate: 6.3% against 5.8% for the dev median. v1.0 won this row (4.7%); retraining on the physics-1.1 bank lost it.
- Combined-cycle heat rate 23.5% (v1.0: 20.7%), biased high, against 7.0% for a plain technology default.
- Gas-turbine heat rate 8.8% (v1.0: 16.2%) against 6.5% for EIA-923; turbine heat input 11.3% (v1.0: 17.3%) against 5.1%.
- All units together, full-load heat rate 11.6% (v1.0: 13.7%) loses to the technology default (10.3%), and part load 14.5% to EIA-923 (11.8%).
- From one month, minimum stable load (10.1 pt) is worse than the dev median (8.0 pt).
- Median run: plain counts still win at every window length (one month 45.1% vs 23.1%, one year 19.4% vs 18.2%).
- v1.1's run-statistic error bars are too narrow on the exam: 17% (starts per year), 33% (median run), 31% (min down) and 40% (min up) of year windows fall inside the 68% band. The dev-fitted factors (1.05-1.56) did not catch it: dev truth covers 2 years, exam truth 8.

## Limits

- Combined-cycle heat rate is biased high (v1.1: 23.5% error): the 3 dev combined cycles are all CHP in EIA-860; two have truths of 10-13 MMBtu/MWh, and on the one that reports the whole block the network reads 9.9-10.7 against a truth of 7.0. Three plants cannot support a CC-specific fit.
- Gas-turbine heat rate: v1.0 was biased about +15% on the exam. The 8 dev gas turbines do not show it (median +1%), so it cannot be fixed from dev; v1.1 reads 8.8% after retraining, still behind EIA-923 (6.5%).
- Minimum up and down times (5th percentile of spells) are noisy targets. With the count skips, run-statistic error bars are too narrow (17-40% coverage of the 68% band in year windows).
- The count skips help long windows and pooled records (starts per year from all years 23.3% -> 8.5%, median run 26% -> 9.8%) but not one month (median run 42.2% -> 45.1%).
- Calibrated on 10 dev plants; ERCOT hub prices are used for every unit, including the SPP, MISO and El Paso ones; ramp truth is the observed 99th-percentile change, a lower bound on capability.
- The exam fed 2023-2025 ERCOT hub prices (exam years) as optional context at test time; no choices were made from them. Training used only 2018/2020/2022 prices.
- Bank v2 was built with GridSim physics 1.0. Its coal, CC, CT and gas-steam outage defaults came from 2024/2025 Texas RE data (exam period). They set the 69 GridSim-dispatched units and the prices behind 37 of 110 self-commitment shards (about 29% of frozen-set windows). Rebuilt with physics 1.1: frozen-set skill of R4 0.492 -> 0.501, R4's lead over a single network +0.025 -> +0.029 (margin 0.01), dev gate unchanged (0.215). No rung decision changes and the release stands.
- Self-committed units' outage ranges (EFOR 2-15%, planned 0-12% by technology) have no recorded source. Real units' EIA-860 facts come from the latest report (up to 2025); limiting them to reports up to 2022 moves the dev gate by 1e-6.

## The ratchet

### PlantDetective v1 (13-14 Nov 2026)

**Objective:** Skill in [0, 1] on the frozen simulated validation set of the current bank version (13% of simulated units, held out by unit): mean of technology accuracy and, per target, max(0, 1 - MAE / MAE of the per-technology median), in the transformed target space  
**Rule:** Keep if the frozen-val skill beats the incumbent by more than 0.01 AND the dev score (the same skill on the 40 dev units, dev years 2019 and 2021, mean over month, quarter and year windows) falls by no more than 0.005. When the simulator (bank version) changes, the incumbent is re-scored on the new frozen set first.

| Rung | Change | Frozen-set score | Decision | Why | Val tech acc | Dev skill | Dev month | Dev quarter | Dev year | Dev tech acc |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | GridSim unit-level MILP units only (3 generated systems x 92 days, train-year inputs), emulator v1 | 0.021 | ● kept | first rung (incumbent by definition) | 0.294 | 0.015 | 0.019 | 0.013 | 0.012 | 0.167 |
| R1 | + self-committed (price-taker DP) and cogeneration units on GridSim and ERCOT train-year price backgrounds | 0.491 | ● kept | val 0.491 vs 0.021 (d=+0.470, margin 0.01); dev 0.117 vs 0.015 (d=+0.102, tol 0.005) | 0.919 | 0.117 | 0.099 | 0.109 | 0.143 | 0.8 |
| R2 | bank v2: dev twin calibration (start/stop trajectories measured on dev starts, trips and aborted starts, offer slope, start-cost multipliers per technology, CHP steam credit, EIA-923 gross ratio from dev unit-years); GridSim units once each. Twin mismatch 0.211 -> 0.128 | 0.475 | ● kept | val 0.475 vs 0.266 for R1 re-scored on the v2 set (d=+0.210, margin 0.01); dev 0.200 vs 0.117 (d=+0.083, tol 0.005) | 0.959 | 0.2 | 0.122 | 0.157 | 0.321 | 0.867 |
| R3 | heat-rate skip connection: when an EIA-923 heat rate is given, the network predicts the full-load heat rate as a correction to it | 0.470 | ○ rejected | val 0.470 vs 0.475 (d=-0.005, margin 0.01); dev 0.191 vs 0.200 (d=-0.009, tol 0.005) | 0.957 | 0.191 | 0.119 | 0.158 | 0.296 | 0.867 |
| R4 | ensemble of 3 networks (R2 recipe, seeds 0-2): mean of predictions, variance = mean member variance + spread of member means | 0.501 | ● kept | val 0.501 vs 0.475 (d=+0.026, margin 0.01); dev 0.215 vs 0.200 (d=+0.015, tol 0.005); released as v1.0, replaced by v1.1 (next ladder) | 0.962 | 0.215 | 0.134 | 0.177 | 0.333 | 0.867 |
| R4-bars | error bars: widen-only factor per output so that at least 68.3% of the dev windows fall within 1 sigma (report.fit_factors; training/sigma_factors.json, factors 1.00-1.90) | — | ● kept | calibration on dev only; changes no estimate. Exam year-window coverage of the 68% band afterwards: 48-79% (median about 65%) |  |  |  |  |  |  |

<sub>Columns after *Why*: training/ratchet.json: frozen-val technology accuracy, dev skill (mean and by window length) and dev technology accuracy (year windows); higher is better.</sub>

R0 and R1 were scored on the v1 frozen set, R2-R4 on v2 (bank v2, dev twin calibration). R1 re-scored on v2: 0.266. Banks were built with GridSim physics 1.0 (outage defaults from exam-period data). Check of 14 Nov 2026: bank v2 rebuilt under 1.0 and 1.1, R4 frozen-set skill 0.492 -> 0.501, R4 minus a single R2-recipe network +0.025 -> +0.029, dev gate 0.215 both; no decision changes (NOTEBOOK).

### PlantDetective v1.1 (15 Nov 2026)

**Objective:** As v1: skill in [0, 1] on the frozen simulated validation set (mean of technology accuracy and, per target, max(0, 1 - MAE / MAE of the per-technology median), transformed target space)  
**Rule:** Unchanged: keep if the frozen-val skill beats the incumbent by more than 0.01 AND the dev score (40 dev units, 2019 + 2021, mean over month, quarter and year windows) falls by no more than 0.005. The incumbent (v1.0, R4) is re-scored on the new frozen set first.

| Rung | Change | Frozen-set score | Decision | Why | Val tech acc | Dev skill | Dev month | Dev quarter | Dev year | Dev tech acc |
|---|---|---|---|---|---|---|---|---|---|---|
| R4 | v1.0 as released: 3-network ensemble trained on bank v2 (physics 1.0), re-scored on v2p11 | 0.501 | ● kept | incumbent (v1.0) | 0.967 | 0.215 | 0.134 | 0.177 | 0.333 | 0.867 |
| R5 | R4 recipe (3 networks, seeds 0-2) retrained on the training split of bank v2 rebuilt with physics 1.1 | 0.514 | ○ rejected | val 0.514 vs 0.501 (d=+0.013, margin 0.01) but dev 0.209 vs 0.215 (d=-0.006, tol 0.005) | 0.965 | 0.209 | 0.129 | 0.168 | 0.329 | 0.867 |
| R6 | R5 + count skips: starts per year, min up, median run and min down are the statistic counted in the window plus the network's correction (where the window has one); 3 networks, seeds 0-2. Error bars refitted on dev (training/sigma_factors_v1_1.json, 1.00-2.07) | 0.532 | **◆ released** | val 0.532 vs R4 0.501 (d=+0.031, margin 0.01); dev 0.225 vs 0.215 (d=+0.010, tol 0.005) -> kept; released as v1.1 | 0.968 | 0.225 | 0.128 | 0.171 | 0.376 | 0.867 |

<sub>Columns after *Why*: training/ratchet_v1_1.json: frozen-val technology accuracy, dev skill (mean and by window length) and dev technology accuracy (year windows); higher is better.</sub>

Frozen set v2p11 = bank v2 rebuilt with GridSim physics 1.1 by the HANDOFF P0.5 check (same seeds and scripts; 1,678 usable units, 20,136 windows, 2,700 frozen). The dev evidence for R6: on the 40 dev units, the count in the window beats v1.0 on one-year windows for starts per year (17% vs 33% median error), median run (13% vs 44%), min up (35% vs 44%) and min down (25% vs 37%). Log: training/ratchet_v1_1.json (training/ladder_v1_1.py).

Raw per-rung logs are in [`training/`](training/).

## Physics

Two layers of unit physics on the shared grid platform (shared/grid). GridSim generates power systems and dispatches them with a unit-level MILP driven by real EIA-930 demand, wind and solar shapes and Henry Hub gas of the train-input years. Each simulated unit then commits itself against a price background with an exact dynamic programme: on/off and time in state, 7 output levels, a convex fuel curve, hot/warm/cold start costs, minimum up and down times, ramp and start-up/shut-down limits, forced and planned outages. A measurement emulator turns the schedule into a CAMPD-like hourly record and an EIA-923-like annual heat rate.

**Simulator**

- GridSim generate_fleet + unit-level MILP dispatch (cluster=False; physics 1.0, v1.1: 1.1), 5 generated systems x 92 days, train-input years 2018, 2020, 2022 (12-21 min per system on one CPU)
- Self-commitment DP per unit; fuel curve F(P) = P x HR_full x (1 + beta(1/f - 1) + c(f - 1)^2); price backgrounds: GridSim system prices (30%) or real ERCOT day-ahead hub prices of 2018/2020/2022 (70%); cogeneration units run near flat with trips
- CAMPD measurement emulator: partial start and stop hours, warm-up ramps measured on the dev starts (coal: first hour ~1% of capacity, then 2-8%/h), AR(1) load noise, summer gas-turbine derate, trips and aborted starts, CEMS heat-input noise, combined cycles that report only the gas turbine, cogeneration steam fuel
- EIA-923 emulator: linked-generator accounting, CHP fuel allocation, gross/net ratio fitted on dev unit-years
- Bank v2: 1,829 simulated units, 20,580 windows of 30-365 days, 13% frozen validation by unit. v1.1 trains on bank v2 rebuilt with GridSim physics 1.1 (HANDOFF P0.5; 1,678 usable units, 20,136 windows, frozen set v2p11)

**Checked against independent references**

- Self-commitment DP vs an independent MILP (scipy / HiGHS) on 20 random 96-hour single-unit cases with ERCOT 2022 prices: max relative profit difference 1.0e-14 (training/checks/check_dp.json; re-run in this repository: 1.0058963487522703e-14, identical)
- Dev twin test, 40 dev units in 2019 and 2021 against their simulated twins: pattern mismatch 0.211 with bank v1, 0.128 with bank v2 (training/twin_v1.json, twin_v2.json)
- GridSim's own physics checks are in shared/grid/checks/

## Help wanted

- EPA CAMPD hourly data for states beyond Texas: only the Texas bulk files were fetched, so the exam covers ERCOT-area units only.
- Price history at each unit's own ERCOT settlement point, and SPP / MISO / El Paso prices for the non-ERCOT Texas units (the hub average stands in for all).
- Unit-level combined-cycle configuration and gas-turbine model data that do not come from CAMPD, to re-centre the heat-rate priors.

## Next rungs

- Re-centre the combined-cycle and gas-turbine heat-rate priors with non-CAMPD evidence (EIA-860 combined-cycle configuration, turbine vintage and model), and make trust in EIA-923 learnable per technology (a skip that passes the dev gate).
- Unit-specific prices (the ERCOT settlement point of the unit's node) and coal delivered prices from EIA-923 fuel receipts.
- Widen the run-statistic error bars honestly: the dev truth covers 2 years, the exam truth 8; fit the factors on dev windows scored against a truth that spans more than the window (for example 2019 windows against 2019+2021).
- Decide on the exam's price context: keep the 2023-2025 ERCOT prices as optional inputs, or restrict the context to pre-exam years and re-score.
- Upstream to GridSim: soft history constraints so dispatch(outages='random') stays feasible when an outage starts while a unit is on; the self-commitment DP and CAMPD emulator as a 'unit behaviour' layer.

## Files

| Path | What |
|---|---|
| `plantdetective.py`, `pdcore.py`, `pipeline.py` | Inference — numpy/scipy only |
| `plantdetective_weights.npz`, `plantdetective_weights_v1_0.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [EPA Clean Air Markets Program Data (CAMPD), hourly unit emissions — Texas bulk files 2018-2025](https://campd.epa.gov) | truth (gross load, operating time, measured heat input), dev, exam | Public domain (U.S. Government) |
| [EIA-860 / EIA-923 via Catalyst Cooperative PUDL (and its EPA-EIA crosswalk)](https://catalyst.coop/pudl/) | optional inputs (prime mover, CHP, nameplate, minimum load, cold-start time, online year, annual heat rate) and the EIA-923 baseline | PUDL data CC-BY 4.0; EIA data public domain |
| [EIA-930 Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/) | physics (GridSim demand, wind and solar shapes of train-input years) | Public domain (U.S. Government) |
| [ERCOT day-ahead settlement point prices (hub average HB_HUBAVG)](https://www.ercot.com) | physics (price backgrounds 2018/2020/2022), dev, optional input at the exam (2018-2025) | Public ERCOT market data (cite) |
| [Henry Hub natural gas spot price, daily (EIA)](https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm) | physics, optional input | Public domain (U.S. Government) |
| [RTS-GMLC generator defaults (Barrows et al., IEEE Trans. Power Syst. 35(1), 2020)](https://github.com/GridMod/RTS-GMLC) | physics (GridSim technology defaults) | BSD-3-Clause |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
