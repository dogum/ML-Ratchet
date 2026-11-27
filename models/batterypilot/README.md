<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BatteryPilot <sub>v1.3</sub>

> **When should a grid battery charge and discharge, knowing only the past?**

*Before each hour, what is one stored MWh worth to this battery — and so should it charge, hold or sell at the real-time price?*

BatteryPilot is a causal dispatch policy for a 1 MW grid battery doing energy arbitrage. Before each hour it returns a value curve: the worth of one stored MWh at 0, 25, 50, 75 and 100 % full. The battery uses that curve as its offer for the hour, cleared against the real-time price, with 85 % round trip and rainflow wear. v1.3 (shipped) is a hybrid: the rolling day-ahead optimiser's plan, written as a value curve, plus a small network's residual on it. It was trained only on simulated price paths (GridSim fleets, a scarcity model calibrated on dev years, and from v1.3 GridSim's real-time scarcity that the day-ahead market cannot see) and examined once per version on real ERCOT 2023-25 and CAISO 2024-25 prices. v1.3 earns 67 % (ERCOT) and 59 % (CAISO) of perfect hindsight net of wear: better than the daily-spread and threshold rules, worse than the rolling day-ahead optimiser (69 % and 68 %), and worse than v1.1 (68 % and 61 %) although it was better on the real dev years. It wins only with 4 h ERCOT batteries.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/batterypilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/batterypilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Hourly and causal: real-time prices of the last 24 h, day-ahead prices for the rest of today (and tomorrow after 14:00 local; 36 h, masked where not yet published), the RT-DA spread of the last 6 h, the balancing authority's day-ahead demand forecast for 24 h (EIA-930, relative to its 30-day max), the last temperature, hour, day of year, weekend, and the battery (duration, wear cost / price scale, price scale). Prices are scaled by the 7-day median DA price and squashed with asinh; 135 features. v1.3 also reads the day-ahead backbone curve at the 5 knots (a backward dynamic programme over the DA prices known before the hour; 140 inputs). |
| **Returns** | Value curve λ(SOC): $/MWh of stored energy at 5 state-of-charge knots, strictly decreasing<br>Executed by the offer rule: discharge down to the SOC where λ = η_d (p − c_wear), charge up to where λ = p / η_c, within power and energy limits; cleared against the hour's RT price<br>Settled: gross revenue, rainflow wear cost, net revenue ($) and cycles; no error bar on the curve (the exam's intervals are on revenue, from a weekly block bootstrap) |
| **Learned from** | Battery: 1 MW grid side, usable energy = P × duration (5-95 % of nameplate), 85 % round trip (√ each way), wear by rainflow cycle counting (ASTM E1049) on a Wöhler depth curve (depth^1.3 / 6,000 per cycle) with LiFePO4 temperature and C-rate dependence (Wang et al. 2011) in an HVAC container, at $130/kWh nameplate. Prices: the shared GridSim fleet dispatch (physics 1.2 from BatteryPilot v1.3: 1.1 plus gridsim.market's real-time scarcity; 1.1 for v1.1) for the real ERCOT and CISO fleets of 2018, 2020 and 2022, wrapped by BatteryPilot's price simulator (training/pricesim.py) with a merit stack, scarcity and day-ahead prices. |
| **Noise** | Simulated real-time noise (Student-t, AR(1), scaled by net-load ramps) and real EIA-930 day-ahead demand-forecast errors laid on the simulated day-ahead; scenario randomisation of demand, renewables, storage, gas price, retirements and scarcity parameters |
| **Network** | v1.3: the day-ahead backbone curve (asinh of value / price scale at the 5 knots) plus an MLP residual, 140 → 128 → 128 → 5 (SiLU), 35,205 weights; each knot kept at least 0.01 below the one before (a strictly decreasing curve). v1.0/v1.1: MLP 135 → 128 → 128 → 5, 34,565 weights, top knot and 4 softplus steps. Trained by direct policy search: the battery physics (offer execution, limits, losses, linear wear) runs inside the loss in torch, maximising revenue + value of energy left − value at the start over 14-21-day windows, each divided by its own 2 h perfect-foresight value. |
| **Physics also enters** | inputs: The battery's own wear cost (temperature-dependent linear proxy from the rainflow model) and duration; prices scaled by a causal 7-day median<br>output: v1.3: the network outputs a residual on the classical day-ahead backbone (the rolling optimiser's plan as a value curve). The curve feeds a physics executor: an offer rule with round-trip losses, power and energy limits, and rainflow wear; the battery physics sits inside the training loss |
| **Real data's jobs** | calibration, gate, exam (never training labels) |
| **Dev set** | GridData's frozen time split (data/proc/splits.json, sha256 895b4e9a…, seed 20261113): dev = ERCOT 2019 and 2021 at HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST; CAISO Aug-Dec 2023 at TH_NP15 and TH_SP15 (OASIS starts 2023-08). Simulator inputs only from the train-input years 2018, 2020, 2022. |
| **Exam** | ERCOT 2023, 2024, 2025 at 4 hubs + 8 load zones (36 node-years); CAISO 2024, 2025 at TH_NP15, TH_SP15, TH_ZP26 and DLAP PG&E, SCE, SDG&E (12 node-years); 1, 2 and 4 h batteries. Never used for choosing anything; 2026 is held in reserve. |

## Run it

```bash
cd models/batterypilot
pip install numpy scipy
python demo.py        # about 25 s
```

or from anywhere in the repo: `mlratchet demo batterypilot`.

```python
import csv, numpy as np
import batterypilot as bp
net = bp.PolicyNet()                                   # v1.3 (batterypilot_weights.npz); bp.WEIGHTS_V1_1, bp.WEIGHTS_V1_0
bat = bp.Battery(P=1.0, duration_h=2.0)                # 1 MW / 2 MWh usable, 85 % round trip
rows = list(csv.DictReader(open("examples/ERCOT_HB_WEST_2024.csv")))
path = {k: np.array([float(r[k]) for r in rows]) for k in ("rt", "da", "df", "T")}
path |= {k: np.array([int(r[k]) for r in rows]) for k in ("hour", "day", "dow", "doy")}
ch, dis, soc, V = bp.run_policy(net, path, bat, bat.c_lin(path["T"]))
res = bp.settle(bat, path["rt"], ch, dis, 0.0, path["T"])
print(res["net"] / 1000, "$/kW net of wear;", V[0], "$/MWh at 0/25/50/75/100 % full")
```

## The exam

Share of perfect-hindsight revenue net of wear (each node-year and battery's net revenue divided by an annual LP on the same RT prices, then averaged), on real prices, run once per version with the released network. Every strategy settles at the hourly RT price with rainflow wear at the BA's temperature. DA optimiser = a rolling LP that re-plans at 14:00 with day-ahead prices as the forecast; spread rule and threshold rule are GridData's baselines (the threshold uses the node's dev-period RT quartiles). Revenue intervals come from a weekly block bootstrap (2,000 draws). The rows below are v1.3's (run once after its release choice was committed); each note keeps v1.0's first look and v1.1's second look (v1.1 re-scored on the same run: identical). Every number here is on real data that never chose anything.

| Metric | BatteryPilot | Baselines | |
|---|---|---|---|
| Share of perfect hindsight, ERCOT 2023-25, all batteries (36 node-years × 1/2/4 h)<br><sub>v1.3 beats the spread rule in 82 % of node-year-battery cases, the threshold rule in all, the DA optimiser in 56 %; 222 cycles a year vs 301 for the optimiser. v1.0: 66.7 % (beats the optimiser in 57 %); v1.1: 67.9 % (63 %). v1.3 is lower than v1.1 here although it was higher on dev (0.660 vs 0.638).</sub> | **67.1%** | Rolling day-ahead LP optimiser (classical): 68.5%<br>Daily spread rule (GridData) (classical): 56.1%<br>Threshold rule, dev RT quartiles (GridData) (dumb baseline): 11.4% | ❌ loses |
| Share of perfect hindsight, CAISO 2024-25, all batteries (12 node-years × 1/2/4 h)<br><sub>v1.3 beats the spread rule in 50 % of cases, the DA optimiser in 19 %. v1.0: 64.2 % (spread 81 %, optimiser 19 %); v1.1: 61.4 % (58 %, 11 %). Worse each version: the CAISO-like simulator family is hand-set and v1.3's GridSim scarcity has no CAISO fit.</sub> | **59.4%** | Rolling day-ahead LP optimiser (classical): 68.1%<br>Daily spread rule (GridData) (classical): 58.9%<br>Threshold rule, dev RT quartiles (GridData) (dumb baseline): 15.8% | ❌ loses |
| Share of perfect hindsight, ERCOT, 1 h battery (36 node-years)<br><sub>v1.3 beats the optimiser in 7 of 36 node-years; 2025 1 h −$3.7/kW-yr, 95 % CI [−7.7, −0.4]. v1.0: 55.9 % (25 %); v1.1: 57.8 % (36 %).</sub> | **54.8%** | Rolling day-ahead LP optimiser (classical): 62.9%<br>Daily spread rule (GridData) (classical): 53.1%<br>Threshold rule (GridData) (dumb baseline): 1.8% | ❌ loses |
| Share of perfect hindsight, ERCOT, 2 h battery (36 node-years)<br><sub>v1.3 beats the optimiser in 24 of 36 node-years but loses on the mean by 0.3 points: 2024 +$2.2/kW-yr [+0.7, +3.6], 2025 −$3.9 [−8.1, −0.3], 2023 +$7.5 [−3.5, +18.1]. v1.0: 67.1 %; v1.1: 68.0 %.</sub> | **68.1%** | Rolling day-ahead LP optimiser (classical): 68.4%<br>Daily spread rule (GridData) (classical): 57%<br>Threshold rule (GridData) (dumb baseline): 10.9% | ❌ loses |
| Share of perfect hindsight, ERCOT, 4 h battery (36 node-years)<br><sub>v1.3 beats the optimiser in 30 of 36 node-years: 2023 +$12.9/kW-yr [+3.9, +24.7], 2024 +$4.9 [+3.0, +6.6], 2025 −$0.7 [−5.5, +2.9]. v1.0: 76.9 % (81 %); v1.1: 78.0 % (86 %).</sub> | **78.5%** | Rolling day-ahead LP optimiser (classical): 74.2%<br>Daily spread rule (GridData) (classical): 58.3%<br>Threshold rule (GridData) (dumb baseline): 21.6% | ✅ wins |
| Share of perfect hindsight, CAISO, 1 h battery (12 node-years)<br><sub>v1.3 loses to the optimiser in all 12 node-years; 2024 −$3.2/kW-yr [−5.5, −1.7]. v1.0: 56.3 %; v1.1: 52.1 %.</sub> | **48.3%** | Rolling day-ahead LP optimiser (classical): 64.1%<br>Daily spread rule (GridData) (classical): 55.2%<br>Threshold rule (GridData) (dumb baseline): 10.4% | ❌ loses |
| Share of perfect hindsight, CAISO, 2 h battery (12 node-years)<br><sub>v1.3 beats the optimiser in 1 of 12 node-years. v1.0: 63.4 % (1 of 12); v1.1: 60.4 % (0 of 12).</sub> | **57.9%** | Rolling day-ahead LP optimiser (classical): 68%<br>Daily spread rule (GridData) (classical): 59.2%<br>Threshold rule (GridData) (dumb baseline): 16.6% | ❌ loses |
| Share of perfect hindsight, CAISO, 4 h battery (12 node-years)<br><sub>v1.3 beats the optimiser in 6 of 12 node-years; 2024 −$2.4/kW-yr [−6.2, +0.3], 2025 +$1.2 [−0.3, +2.6]. v1.0 won this row thinly (73.0 %); v1.1 71.9 % and v1.3 71.8 % lose it.</sub> | **71.8%** | Rolling day-ahead LP optimiser (classical): 72.3%<br>Daily spread rule (GridData) (classical): 62.3%<br>Threshold rule (GridData) (dumb baseline): 20.3% | ❌ loses |
| Comparison only: the same network trained on real ERCOT 2018/2020/2022 node prices (breaks the physics-only rule; not released)<br><sub>What the simulator cost v1.0 in ERCOT (−2.3 points) and gained it in CAISO (+7.8 points), where no real training prices exist. v1.3: ERCOT 67.1, CAISO 59.4.</sub> | **ERCOT 69.0, CAISO 56.4%** | BatteryPilot v1.0 (simulator-trained) (classical): ERCOT 66.7, CAISO 64.2% | ℹ️ |

### Where it wins

- 4 h ERCOT batteries: 0.785 vs 0.742 for the rolling day-ahead optimiser (v1.3; beats it in 30 of 36 node-years; 2023 +$12.9/kW-yr, CI [+3.9, +24.7]; 2024 +$4.9, CI [+3.0, +6.6]).
- Beats GridData's daily-spread rule (ERCOT 0.671 vs 0.561 of perfect hindsight) in 82 % of ERCOT node-year-battery cases, and the threshold rule in every case.
- Cycles less: about 222 (ERCOT) and 215 (CAISO) cycles a year vs 301 and 282 for the optimiser; its worst month at the median node was −$0.07/kW (any node: −$0.09/kW, CAISO ZP26 December).
- Trained without a single real price path: real dev years only calibrated the simulator and gated rungs; GridSim's RT scarcity was fitted on train-input-year prices.

### Where it loses

- Loses to the classical rolling day-ahead optimiser on both pooled rows in every version: v1.3 ERCOT 0.671 vs 0.685, CAISO 0.594 vs 0.681 (v1.0 0.667 / 0.642, v1.1 0.679 / 0.614).
- v1.3 is worse than v1.1 on the exam (ERCOT 0.671 vs 0.679, CAISO 0.594 vs 0.614) although it was better on the real dev years (0.660 vs 0.638) and on the new frozen set (0.772 vs 0.735). The dev gain came from 2021 alone (Uri; 0.48-0.50 vs 0.37-0.38); on dev 2019 (0.77-0.78 vs 0.81-0.82) and CAISO 2023 (0.73 / 0.81 vs 0.78 / 0.84) it was already worse, as it is on the exam. The pooled dev gate let the Uri year decide.
- 1 h batteries lose clearly: ERCOT 0.548 vs 0.629 (7 of 36 node-years won); CAISO 0.483 vs 0.641, every node-year.
- CAISO 4 h, a thin win for v1.0 (0.730 vs 0.723), is a loss for v1.1 (0.719) and v1.3 (0.718).
- The simulator fix did not make the frozen set agree with dev in general: over 10 fixed policies the rank correlation fell from 0.67 (v4) to 0.43 (v6). On twins of dev 2019 the rolling DA optimiser captures 55-56 % vs 79 % real: physics 1.2 under-rates following the DA plan, where 1.1 over-rated it (90 %).

## Limits

- Energy arbitrage only: hourly settlement (mean of the 15-min RT), no ancillary services, no 5-min prices, price taker; copper-plate simulator (no congestion, nodal differences untrained).
- The CAISO-like family is hand-set from 5 dev months (Aug-Dec 2023) and is more volatile than real CAISO. GridSim's RT scarcity (physics 1.2) has no CAISO fit (no CAISO prices before Aug 2023), so CAISO paths have none of it.
- Calibration hit grid edges again in v1.3 (σ 3,500 MW, the low end; cold f_max 0.55, the high end); dev-2019 twins carry $57/kW-yr of 2 h perfect-foresight value vs $103 real.
- Wear parameters are assumptions from the literature; calendar ageing is not charged.
- Exam leak in v1.0, fixed in v1.1: GridSim physics 1.0 took its coal, CC, CT and gas-steam outage shares from Texas RE's 2024/2025 reports (exam-period data). On the ERCOT 2019 fleet they took 14.8 GW of 74.1 GW of thermal capacity out on average, vs 10.8 GW with the pre-2023 rates of physics 1.1. v1.0's scarcity calibration absorbed most of the difference (ORDC μ −4,000 MW; on physics 1.1 the fit moves to μ 0 MW). For v1.1 the calibration, banks and frozen set were rebuilt and every rung retrained and re-decided.
- The forced-outage rates of physics 1.1 and 1.2 are PJM class averages (demand-period EFORd), not ERCOT's own pre-2023 rates.
- The exams have been seen: v1.3's exam was run once after its release choice was committed, and the dev-chosen v1.3 lost ground on it.
- No error bar on the value curve itself; the exam's intervals are on revenue (weekly block bootstrap), so the recipe's coverage step does not apply.
- Information boundary (audit, 25 Nov 2026): the day-ahead planner's wear cost uses realized temperatures for the coming hours, a perfect temperature forecast. causal_wear(bat, T) (24 h persistence, opt-in since 27 Nov 2026) moves the three packaged node-years by -$0.06 to +$0.01/kW-year; the exam used the realized form. At the first hour of a path the price history clipped to that hour's own RT price; fixed 27 Nov 2026 (changes the packaged node-years by at most $0.002/kW-year).

## The ratchet

### v1.0 — GridSim physics 1.0 (as released)

**Objective:** Capture = net revenue / perfect-foresight net revenue (net of wear), summed over the 1, 2 and 4 h batteries of each simulated path-year and averaged over the path-years of the frozen simulated set (each path-year counts the same)  
**Rule:** Keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its capture on the 10 real dev node-years (ERCOT 2019 and 2021 at 4 hubs, CAISO Aug-Dec 2023 at NP15 and SP15; all 3 durations) is not more than 0.01 below the incumbent's. When the simulator changed, a new frozen set was built and every logged rung re-scored and re-decided in order.

| Rung | Change | Frozen-set score | Decision | Why | v1 frozen | v2 frozen | v3 frozen | Dev capture |
|---|---|---|---|---|---|---|---|---|
| R0 | Classical rolling-horizon LP on day-ahead prices (re-plans at 14:00, follows the plan) | 0.102 | ● kept | Starting incumbent. | 0.1018 | 0.4894 | 0.4472 | 0.5927 |
| R1 | MLP value-curve policy trained by policy search on sim v1 (GridSim merit-order prices, no scarcity curve, no outage weather, no RT noise); 45 path-years | 0.593 | ○ rejected | Frozen +0.491 but the dev gate failed: dev capture 0.483 vs the incumbent's 0.593 (−0.110). The v1 prices have no scarcity, so the policy cycles on small spreads and is empty when spikes come. Still rejected on v2 (+0.043 frozen, dev −0.110) and v3. | 0.5926 | 0.5326 | 0.5696 | 0.4828 |
| R2 | Same network and policy search, trained on sim v2 (80 ERCOT path-years from 2018/2020/2022 inputs with dev-calibrated scarcity) | 0.767 | ● kept | Frozen 0.767 vs R0 0.489 on v2 (+0.278); dev 0.598 vs 0.593 → new incumbent. Re-scored on v3: 0.7465 vs R0 0.4472, still kept. | — | 0.7671 | 0.7465 | 0.5979 |
| R3 | Train on all 120 ERCOT path-years + 45 CAISO-like path-years (CISO fleet, duck-curve solar, carbon price, $1,000-2,000 caps), 2,500 steps | 0.749 | ○ rejected | Frozen 0.7489 vs R2 0.7465 on v3 (+0.0024, below the 0.01 margin), although dev improved (0.623 vs 0.598). | — | — | 0.7489 | 0.6228 |
| R4 | Fine-tune R3 on the same 165 path-years with longer windows (21 days), batch 64, lower learning rate (900 steps) | 0.758 | ● kept | Frozen 0.7580 vs R2 0.7465 on v3 (+0.0115 > 0.01); dev 0.637 vs 0.598. Released as v1.0 (weights kept as batterypilot_weights_v1_0.npz). Its simulator used GridSim physics 1.0, whose outage defaults leaked exam-period data; replaced by v1.1 (next ladder). | — | — | 0.758 | 0.6372 |

<sub>Columns after *Why*: Frozen capture on each version of the frozen set (re-scores from training/ratchet.json) and dev capture on the real dev node-years.</sub>

v1 = GridSim merit-order prices only; v2 = + dev-calibrated ORDC scarcity, cold outages, gas-turbine derate, RT noise and DA expectation (24 ERCOT path-years); v3 = v2 + 9 CAISO-like path-years on the real CISO fleet. Each rung's sim score is on the set it was judged on. A comparison network trained on real ERCOT 2018/2020/2022 prices breaks the physics-only rule; it was never a rung and is reported only in the exam table.

### v1.1 — GridSim physics 1.1 (leak fix, 14 Nov 2026)

**Objective:** As v1.0: capture = net revenue / perfect-foresight net revenue (net of wear), summed over the 1, 2 and 4 h batteries of each simulated path-year and averaged over the path-years of the frozen simulated set  
**Rule:** Unchanged from v1.0: keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its capture on the 10 real dev node-years is not more than 0.01 below the incumbent's. Every v1.0 decision re-taken in order, every network retrained.

| Rung | Change | Frozen-set score | Decision | Why | v4 frozen | Dev capture |
|---|---|---|---|---|---|---|
| R0 | Classical rolling-horizon LP on day-ahead prices (re-plans at 14:00, follows the plan) | 0.504 | ● kept | Starting incumbent. | 0.5042 | 0.5927 |
| R1 | MLP value-curve policy trained by policy search on sim v1 (GridSim merit-order prices only); 45 path-years, 1,500 steps | 0.607 | ○ rejected | Frozen +0.103 but the dev gate failed: dev 0.489 vs 0.593 (−0.104). | 0.6074 | 0.4888 |
| R2 | Same network, trained on sim v2 with the v1.1 dev calibration: 80 ERCOT path-years, 1,500 steps | 0.779 | **◆ released** | Frozen 0.7792 vs R0 0.5042 (+0.275); dev 0.638 vs 0.593 → kept. No later rung beat it by the margin: released as v1.1. | 0.7792 | 0.6382 |
| R3 | Train on all 120 ERCOT path-years + 45 CAISO-like path-years, 2,500 steps | 0.788 | ○ rejected | Frozen +0.0085 over R2, below the 0.01 margin (dev 0.644 vs 0.638). | 0.7877 | 0.6437 |
| R4 | Fine-tune R3 on the same 165 path-years with 21-day windows, batch 64, learning rate 5e-4 (900 steps) | 0.789 | ○ rejected | Frozen 0.7890 vs R2 0.7792 (+0.0098, 0.0002 short of the 0.01 margin), although dev rose to 0.653 vs 0.638. Rejected as the rule is written. | 0.789 | 0.653 |

<sub>Columns after *Why*: Frozen capture on v4_frozen and dev capture on the real dev node-years (training/ratchet_v1_1.json).</sub>

GridSim physics 1.0 took its coal, CC, CT and gas-steam forced and planned outage shares from Texas RE's 2024/2025 assessments (exam-period data). Physics 1.1 uses pre-2023 sources (PJM 2017-2021 class EFORd; Texas RE 2017) and a coal offer adder re-fitted on dev 2019 (-5.73 $/MWh, was -6.62). Everything downstream was redone on 1.1: the scarcity calibration on dev 2019, 2021 and Uri (ORDC mu -4,000 -> 0 MW, sigma 5,000 MW, cold-outage f_max 0.55 -> 0.40), the training banks (120 ERCOT + 45 CAISO-like + 45 sim-v1 path-years) and the frozen set v4 (24 ERCOT + 9 CAISO-like). For reference (not a rung), the v1.0 network scores 0.7778 frozen on v4 and 0.6373 dev. R4 missed the margin by 0.0002 and is rejected as the rule says; R2 is released as v1.1.

### v1.2 attempt — day-ahead value-curve backbone + learned residual (15 Nov 2026, not released)

**Objective:** As v1.1: capture = net revenue / perfect-foresight net revenue (net of wear), summed over the 1, 2 and 4 h batteries of each simulated path-year and averaged over the path-years of the frozen simulated set  
**Rule:** Unchanged: keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its dev capture is not more than 0.01 below the incumbent's. Incumbent: v1.1 (R2).

| Rung | Change | Frozen-set score | Decision | Why | v4 frozen | Dev capture |
|---|---|---|---|---|---|---|
| R2 | v1.1 as released: MLP value curve, 80 ERCOT path-years, 1,500 steps | 0.779 | ● kept | Incumbent (v1.1). | 0.7792 | 0.6382 |
| R5 | Reference, not a candidate: the day-ahead value-curve backbone alone, executed by the offer rule (no network) | 0.684 | ○ rejected | A classical reference. Dev 0.630 (LP 0.593): gains in 2021 (0.51-0.52 vs LP 0.28-0.32), loses in 2019 (0.67 vs 0.78-0.79) and with 1 h ERCOT batteries (0.35 vs 0.49). | 0.6836 | 0.6295 |
| R6 | Hybrid: the network reads the backbone curve too and outputs a residual on it (kept non-increasing by a running minimum; zero at the start); 80 ERCOT path-years, 1,500 steps | 0.702 | ○ rejected | Frozen 0.7024 vs 0.7792 (-0.077); dev 0.661 vs 0.638 (+0.023). Best on dev, worst on the frozen set. Flat pieces of the curve made the offer rule's gradient blow up; training capture fell from 0.33 to 0.26. | 0.7024 | 0.6614 |
| R7 | R6 on 120 ERCOT + 45 CAISO-like path-years, 2,500 steps | — | ○ rejected | Stopped at step 775: training capture 0.21 vs 0.37 for R3 on the same windows (the R6 flaw). Not scored. | — | — |
| R8 | R6 with a strictly decreasing curve (each knot at least 0.01 below the last, the v1.1 floor); 80 ERCOT path-years, 1,500 steps | 0.777 | ○ rejected | Frozen 0.7769 vs 0.7792 (-0.002); dev 0.615 vs 0.638 (-0.023, gate failed). | 0.7769 | 0.615 |
| R9 | R8 on 120 ERCOT + 45 CAISO-like path-years, 2,500 steps | 0.787 | ○ rejected | Frozen 0.7869 vs 0.7792 (+0.0077, below the 0.01 margin); dev 0.633 vs 0.638 (-0.006). About R3's frozen score (0.7877) with a lower dev score (R3 0.644). | 0.7869 | 0.6327 |

<sub>Columns after *Why*: Frozen capture on v4_frozen and dev capture on the real dev node-years (training/ratchet_v1_2.json).</sub>

The backbone (training/backbone.py) is the rolling day-ahead LP's plan written as a value curve: a backward dynamic programme over the known DA horizon (re-planned at 14:00 local, as the LP) gives the marginal value of stored energy at every SOC, and the offer rule trades it against the RT price. Hybrid rungs read the v1.1 features plus that curve and output a residual on it. R5 is a classical reference, not a candidate. No rung beat v1.1 by the rule: nothing released, no exam run. R6 (closest to the backbone) is the best policy on real dev and the worst on the frozen set: the simulator rewards DA-predictable spikes more than real prices do (the twin-test gap). Log: training/ratchet_v1_2.json (training/ladder_v1_2.py).

### v1.3 — GridSim physics 1.2 (real-time scarcity the day-ahead market cannot see, 15 Nov 2026)

**Objective:** As v1.1: capture = net revenue / perfect-foresight net revenue (net of wear), summed over the 1, 2 and 4 h batteries of each simulated path-year and averaged over the path-years of the frozen simulated set  
**Rule:** Unchanged: keep a rung if its frozen capture beats the incumbent's by more than 0.01 AND its dev capture is not more than 0.01 below the incumbent's. A physics change re-takes the ladder: every v1.1 and v1.2-attempt decision re-taken in order from R0, then new rungs. Release (fixed before the ladder): the last incumbent ships only if its dev capture is above v1.1's (0.6382) and it beats v1.1 re-scored on the new frozen set by the same rule.

| Rung | Change | Frozen-set score | Decision | Why | v6 frozen | Dev capture |
|---|---|---|---|---|---|---|
| R0 | Classical rolling-horizon LP on day-ahead prices (re-plans at 14:00, follows the plan) | 0.384 | ● kept | Starting incumbent. | 0.3843 | 0.5927 |
| R1 | MLP trained on sim v1 (GridSim merit-order prices only); its bank is unchanged by physics 1.2, so the v1.1 network is re-scored | 0.617 | ○ rejected | Frozen +0.233 but the dev gate failed: 0.489 vs 0.593 (−0.104). | 0.6173 | 0.4888 |
| R2 | MLP value curve on sim v2 with physics 1.2 and the v1.3 calibration: 80 ERCOT path-years, 1,500 steps | 0.755 | ● kept | Frozen 0.7546 vs R0 0.3843 (+0.370); dev 0.628 vs 0.593 → kept. | 0.7546 | 0.6277 |
| R3 | All 120 ERCOT + 45 CAISO-like path-years, 2,500 steps | 0.768 | ○ rejected | Frozen +0.0139 over R2 but the dev gate failed: 0.596 vs 0.628 (−0.032). | 0.7685 | 0.5961 |
| R4 | Fine-tune R3 with 21-day windows, batch 64, learning rate 5e-4 (900 steps) | 0.775 | ○ rejected | Frozen +0.0209 over R2 but the dev gate failed: 0.599 vs 0.628 (−0.029). | 0.7755 | 0.599 |
| R5 | Reference, not a candidate: the day-ahead value-curve backbone alone, executed by the offer rule | 0.673 | ○ rejected | A classical reference (dev 0.630). | 0.6729 | 0.6295 |
| R6 | Hybrid: backbone curve + learned residual, kept non-increasing by a running minimum; 80 ERCOT path-years, 1,500 steps | 0.732 | ○ rejected | Frozen −0.023 vs R2; dev 0.581 vs 0.628 (gate failed). | 0.7321 | 0.5809 |
| R8 | Hybrid with a strictly decreasing curve (each knot at least 0.01 below the last); 80 ERCOT path-years, 1,500 steps | 0.772 | **◆ released** | Frozen 0.7719 vs R2 0.7546 (+0.0173); dev 0.660 vs 0.628 → kept. Seeds 0-2: frozen sd 0.003, all three clear R2 by more than 0.01. No later rung beat it by the margin; it beats v1.1 re-scored on v6 (0.7346, dev 0.638): released as v1.3. | 0.7719 | 0.6601 |
| R9 | R8 on all 120 ERCOT + 45 CAISO-like path-years, 2,500 steps | 0.778 | ○ rejected | Frozen 0.7779 vs R8 0.7719 (+0.0060, below the 0.01 margin), although dev rose to 0.685 (the best dev score so far). | 0.7779 | 0.6854 |
| R10 | The R3 recipe trained on the superseded v5 banks (GridSim's first fit); put on the ladder after its dev score (0.658) was seen among the v5 probes | 0.769 | ○ rejected | Frozen 0.7693 vs R8 0.7719 (−0.003). | 0.7693 | 0.6579 |
| R11 | R9 fine-tuned with the R4 recipe (21-day windows, batch 64, learning rate 5e-4, 900 steps) | 0.780 | ○ rejected | Frozen 0.7803 vs R8 0.7719 (+0.0084, below the margin); dev 0.681. No further rung was tried. | 0.7803 | 0.6811 |

<sub>Columns after *Why*: Frozen capture on v6_frozen and dev capture on the real dev node-years (training/ratchet_v1_3.json).</sub>

GridSim physics 1.2 adds gridsim.market: unforecast RT scarcity events fitted on ERCOT 2018/2020/2022 prices (start rate by the DA price's rank and level, month and demand-forecast miss; size grows with the DA level). The dev calibration was re-fitted with it (ORDC mu -1,000 MW, sigma 3,500 MW, a grid edge; it also matches the share of RT spike value the DA sees) and the banks and frozen set rebuilt with the v4 seeds. v5_frozen was built on GridSim's first fit (clock-hour profile); the dev twin test sent that fit back, so the v5 scores are probes (training/ratchet_v1_3.json, probes_v5) and the ladder is decided on v6. ERCOT only: the CAISO-like family is unchanged (no CAISO fit). R8 cleared the margin by 0.007: seeds 0-2 give frozen 0.772 / 0.772 / 0.767 (sd 0.003) and dev 0.660 / 0.667 / 0.644 (sd 0.012). v1.1 re-scored on v6: 0.7346 frozen, 0.6382 dev. Log: training/ratchet_v1_3.json (training/ladder_v1_3.py).

Raw per-rung logs are in [`training/`](training/).

## Physics

Battery: 1 MW grid side, usable energy = P × duration (5-95 % of nameplate), 85 % round trip (√ each way), wear by rainflow cycle counting (ASTM E1049) on a Wöhler depth curve (depth^1.3 / 6,000 per cycle) with LiFePO4 temperature and C-rate dependence (Wang et al. 2011) in an HVAC container, at $130/kWh nameplate. Prices: the shared GridSim fleet dispatch (physics 1.2 from BatteryPilot v1.3: 1.1 plus gridsim.market's real-time scarcity; 1.1 for v1.1) for the real ERCOT and CISO fleets of 2018, 2020 and 2022, wrapped by BatteryPilot's price simulator (training/pricesim.py) with a merit stack, scarcity and day-ahead prices.

**Simulator**

- GridSim (shared/grid, unmodified): real ERCOT / CISO fleets of 2018, 2020, 2022 (heat-rate curves, EIA-923 fuel prices + Henry Hub daily, Markov forced and planned outages, the dev-2019 coal offer adder). v1.1: GridSim physics 1.1, outage shares from pre-2023 sources (forced / planned: coal 13.6 / 8.3 %, CC 3.6 / 8.3 %, CT 5.8 / 8.3 %, gas steam 13.6 / 8.3 %), coal adder −5.73 $/MWh. v1.0: physics 1.0, shares from Texas RE's 2024/2025 reports (coal 15 / 11 %, CC 8 / 11 %, CT 8 / 6 %, gas steam 13 / 19 %), coal adder −6.62 $/MWh
- Merit-order price from pmin blocks and 3 heat-rate segments per cluster; slow units needed for the day's peak stay at pmin (80 %); below that, wind and solar set a negative price
- Gas-turbine ambient derate (0.6 %/K CT, 0.4 %/K CC around the 35 °C rating); cold-weather outages f(T) = 0.40 Φ((−3 °C − T)/3 K) persisting with a 36 h time constant, fitted on Uri (dev 2021; v1.0: 0.55)
- v1.3: GridSim physics 1.2's real-time scarcity the day-ahead market cannot see (gridsim.market, fitted on ERCOT 2018/2020/2022 prices only): unforecast events starting by the DA price's rank within the day and level against the past week, the month and the real EIA-930 demand-forecast miss; geometric duration (mean 2 h); generalised Pareto size growing with the DA level; added to RT, its expectation to DA. Rate scaled ×0.6-1.5 per path. ERCOT paths only (no CAISO fit)
- ERCOT-style ORDC adder (VOLL − λ) × LOLP(R − X), fitted on dev 2019 + 2021 without Uri: v1.3 μ −1,000 MW, σ 3,500 MW (a grid edge), cold-outage f_max 0.55, and the fit also matches how much of the RT value of hours above $300 the DA sees (31 % real 2019, 13 % 2021 without Uri); v1.1 μ 0 MW, σ 5,000 MW; v1.0 μ −4,000 MW. A storage fleet flattens the daily net load
- Real-time noise (Student-t, AR(1), scaled by net-load ramps); day-ahead price = forecast inputs (real EIA-930 demand-forecast errors, AR(1) wind/solar errors, outages known at noon the day before) + the expected ORDC adder
- Scenario draws: demand ×0.97-1.22, wind ×0.8-1.6, solar up to 45 % of peak, storage 0-20 % of peak, gas ×0.6-1.8, retirements 0-8 %, caps $5,000/$9,000; a CAISO-like family on the CISO fleet (solar 35-80 % of peak, imports ×0.4-0.95, carbon $10-40/t, caps $1,000/$2,000)
- Banks: 120 ERCOT + 45 CAISO-like simulated path-years for training; 24 + 9 frozen. Rebuilt on physics 1.2 for v1.3 with the v4 seeds (frozen set v6); the shipped v1.3 network (rung R8) was trained on 80 of the ERCOT path-years. v1.1: physics 1.1, frozen set v4

**Checked against independent references**

- GridData's perfect-foresight, daily-spread and threshold baselines reproduced on 89 dev days at two hubs to 1e-13 $ (one spread day differs by $1.65 at HB_WEST: a price tie) (training/checks/battery_checks.json)
- Rainflow counting matches the rainflow package exactly on 200 random series; executor energy balance 1.1e-16 MWh; rainflow wear of a summer-2019 PF schedule $2,414 vs the linear proxy $2,416
- Uri 2021 (dev), v1.1: 158 and 150 simulated hours above $1,000 for two outage draws vs 153 real; 89-90 % of the real ones covered (training/checks/calibration_v1_1.json). v1.0: 180 and 166, 92-95 % (training/checks/calibration.json)
- CAISO-like family vs real Aug-Dec 2023 (dev): hourly medians $18-25 midday and $59-61 evening vs real $21-25 and $66-72 (checked with physics 1.0; not re-checked on 1.1)
- Day-ahead foresight on dev (training/checks/predictability_*.json): the DA price covers 31 % of the RT value of hours above $300 in real 2019 (13 % in 2021 without Uri); simulated twins 69-72 % (80 / 71 %) with physics 1.1, 60 % (14-37 %) with physics 1.2
- Twin test with physics 1.2 (v1.1 network, training/checks/twin_r2_v4_p1_2.json): on twins of dev 2019 BatteryPilot captures 76-85 % (real 82 %, was 94 % on 1.1 twins), the backbone alone 64-78 % (real 67 %), but the rolling DA optimiser only 55-56 % (real 79 %, was 90 % on 1.1 twins): the simulator now under-rates following the DA plan
- Does the frozen set rank policies like dev? (training/checks/agreement_v1_3*.json) Over each ladder's own rungs: v6 Spearman 0.77 with the same top two (R9, R11) as dev; v4 0.68 with a different best. Over 10 fixed policies (the v1.1 and v1.2-attempt networks, backbone, LP): v4 0.67, v6 0.43: worse
- Twin test, v1.1: the policy captures 94 % of perfect foresight on simulated twins of dev 2019 but 82 % on real 2019; 2021 (with Uri) twins 71-75 % vs real 38 % (training/checks/twin_r2_v4.json). v1.0: 91-94 % vs 80 %; 64-68 % vs 38 % (training/checks/twin_r4_v3.json)

## Help wanted

- CAISO OASIS serves only about 3 years (data from 2023-08): an archive of CAISO hub and DLAP prices for 2018-2022 would give the CAISO family real dev years, a GridSim RT scarcity fit for CAISO, and a longer exam.
- Pre-2023 ERCOT forced-outage rates by technology (NERC GADS or Texas RE): GridSim physics 1.1 and 1.2 use PJM 2017-2021 class averages and one ERCOT-wide planned-outage factor from 2017.
- ERCOT and CAISO ancillary-service clearing prices, and ERCOT resource-node prices in West Texas, to add ancillary services and a congestion basis.
- Historical ERCOT day-ahead wind and solar forecasts (the RT scarcity process can only use the demand-forecast miss from EIA-930).

## Next rungs

- Fix what the dev twin test still shows: the rolling DA optimiser captures 55-56 % on 1.2 twins of dev 2019 vs 79 % real. Real unforecast spikes put 86 % of their value in the day's 3 dearest DA hours (2019), the simulator 57 %; and the simulated DA carries a scarcity premium in those hours that real RT often undercuts by more. Calibrate the size and placement until the twin captures of the LP, the backbone and the network all match real dev.
- A dev gate that one year cannot carry (for example: no dev year worse by more than 0.01), decided before the next ladder: v1.3 passed the pooled gate on 2021 (Uri) alone and lost ground on the exam.
- A 1 h specialist or a 15-minute execution head: every version loses to the day-ahead optimiser on 1 h batteries and in CAISO.
- R9 (the hybrid on all 165 path-years) had the best dev score of any rung (0.685) but missed the margin on the frozen set by 0.004: a frozen set that agrees with dev would settle it.
- Nodal basis: a West Texas congestion model; a real CAISO import and hydro supply curve; ancillary services.
- Upstream to GridSim behind flags: the ORDC adder, the cold-outage curve, the gas-turbine ambient derate, the day-ahead price generator and the single-LP perfect-foresight baseline.
- Queued (audit): a dated re-score of the exam with the planner on causal_wear (24 h persistence or a day-ahead temperature forecast), on the owner's approval.

## Files

| Path | What |
|---|---|
| `batterypilot.py` | Inference — numpy/scipy only |
| `batterypilot_weights.npz`, `batterypilot_weights_v1_1.npz`, `batterypilot_weights_v1_0.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [ERCOT historical DAM / RTM settlement point prices (report types 13060, 13061), via GridData](https://www.ercot.com/mp/data-products) | dev (calibration and gate, 2019 and 2021), exam (2023-2025, 4 hubs + 8 load zones); physics (GridSim 1.2's RT scarcity fit, HB_HUBAVG of the train-input years 2018, 2020, 2022 only) | Public ERCOT market data (cite ERCOT) |
| [CAISO OASIS LMPs (PRC_LMP DAM, PRC_RTPD_LMP), via GridData](http://oasis.caiso.com) | dev (Aug-Dec 2023, CAISO-like family check and gate), exam (2024-2025, 3 trading hubs + 3 DLAPs) | Public CAISO data (cite CAISO) |
| [EIA-930 Hourly Electric Grid Monitor (demand, day-ahead demand forecast, generation by fuel)](https://www.eia.gov/electricity/gridmonitor/) | physics (simulator inputs and forecast errors, train-input years), model input (demand forecast) | Public domain (U.S. Government) |
| [NASA POWER hourly temperature (BA population-weighted)](https://power.larc.nasa.gov) | physics (simulator weather, wear temperature), model input | Free use; cite NASA LaRC POWER Project |
| [EIA-860 / EIA-923 generator fleets and fuel costs, via PUDL (through GridSim)](https://catalyst.coop/pudl/) | physics (fleets, heat rates, fuel prices) | CC BY 4.0 (PUDL); EIA public domain |
| [GridSim outage-rate defaults, physics 1.1: PJM 2017-2021 class average EFORd (May 2022) and Texas RE 2017 Assessment of Reliability Performance (2018)](https://www.texasre.org) | physics (forced and planned outage shares) in v1.1. v1.0 used physics 1.0's Texas RE 2024/2025 aggregates, published during the exam years: an exam leak, fixed in v1.1 (limits) | Public reports (cited by GridSim) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
