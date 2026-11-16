<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# HybridPilot <sub>v2.2</sub>

> **A learned hybrid controller, raced against the dynamic-programming optimum.**

> [!WARNING]
> Graded inside the simulator. With HybridPilot driving the calibrated twins, simulated hybrid fuel is within 3.5 % of VED's logged fuel (exam trip median), but plug-ins read 20–31 % low, and the strategies differ by only 0.5–1.5 %, so the ranking holds only inside RoadSim's physics.

*Second by second, should a hybrid run its engine, and how hard — to use the least fuel the physics allows?*

HybridPilot is a learned, causal energy-management policy for hybrids: every second it decides engine on/off and engine power, holding charge in hybrids and depleting-then-sustaining in plug-ins. It learned by imitating the dynamic-programming (DP) optimum inside the RoadSim simulator. Version 2.1 runs on RoadSim physics v3.1 (chosen on dev: closer engine starts and fuel on the dev twins), with a dead band on its SOC feedback and a fixed equivalence factor for buses and mild hybrids. Version 2.2 adds a charge-sustaining rule for plug-ins at their CS level and a weak SOC feedback for buses. It was examined on 432 trips — real VED routes driven by calibrated twins of the exam cars, bus legs and cycles — against rule-based control, ECMS, adaptive ECMS and MPC.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/hybridpilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/hybridpilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | The vehicle's parameters and, each second, the state (SOC, coolant temperature, timers, power limits), current demand (speed, acceleration, grade, wheel power), trip history and optionally a navigation preview. |
| **Returns** | Engine on/off and engine power (series, power-split) or torque share (P2) every second<br>The resulting fuel, battery use and engine starts over the trip, compared against DP and the classical strategies |
| **Learned from** | The shared RoadSim simulator (shared/vehicle), physics v3.1 pinned (roadsim-physics-v3.1; v2.0 used v2): EPA ALPHA engine and motor maps, Thevenin battery, eCVT / P2 (with torque converter and start energy) / series (with direct drive) powertrains, two-node cabin HVAC, tyre model, production hybrid rules (8 s / 3 s minimum on/off, Toyota-type engine-on speed, warm-up holds, 2 s engine run-on into braking, plug-in charge-depleting rules), the v3 generator hybrid hardware, an energy ledger checked on every run. Teacher: DP over engine on/off × 121 SOC points; reference: DP with the same 8 s / 3 s on/off timers. |
| **Noise** | Real routes, grades and weather; random start SOC, temperatures from −20 to 36 °C, warm and cold starts |
| **Network** | MLP 66 → 96 → 96 → 2 (15,938 parameters) predicting the equivalence factor s (buses and mild hybrids use a fixed s of 2.6–2.8 instead; plug-ins at their CS level use at least the HEV costate, 2.42 power-split / 2.70 series); the decision minimises fuel + s × battery power among RoadSim's candidate controls, with A-ECMS-style SOC feedback (Kp 12 beyond a ±0.03 dead band, Ki 0.02; plug-ins below CS Kp 24 without the band; buses Kp 3, no integral) |
| **Physics also enters** | inputs: Wheel power demand from RoadSim's road-load model<br>output: Its output, the equivalence factor s, sets an ECMS cost over candidate controls scored by the powertrain model |
| **Real data's jobs** | backdrop, calibration, gate, exam (never training labels) |
| **Dev set** | 76 trips — routes of the VED dev cars (calibrated twins of the 7 dev hybrids + 18 dev generated vehicles) and full-ZTBus dev legs (by-month split) |
| **Exam** | 432 trips — 216 real VED exam routes driven by calibrated twins of the 108 exam hybrids, 216 by 54 held-out generated hybrids, full-ZTBus exam legs and HD cycles |

## Run it

```bash
cd models/hybridpilot
pip install numpy scipy
python demo.py        # about 5 s
```

or from anywhere in the repo: `mlratchet demo hybridpilot`.

```python
from pipeline import compare, load_example     # needs RoadSim: found in shared/vehicle automatically
spec, trip = load_example("VED_428_465")
for r in compare(spec, trip, init_soc=None, warm=True):
    print(r["strategy"], r["fuel_g"], r["batt_kWh_out"])
```

## The exam

432 exam trips (430 trip ids, since two routes were drawn twice for the same generated vehicle), graded as % of the DP optimum (DP with the same 8 s / 3 s on/off timers) at matched final charge, on RoadSim physics v3.1. Real routes, simulated cars. Every baseline, v2.1 and v2.0 were re-scored on the same physics. v2.2 was run once, after its release choice was committed (48e9985). Earlier versions' numbers are in the notes. Every number here is on real data that never chose anything.

| Metric | HybridPilot | Baselines | |
|---|---|---|---|
| All 432 trips, % of DP optimum<br><sub>Beats every causal baseline: +0.49 points over A-ECMS, 95% CI [−0.07, +0.98] (bootstrap by vehicle); +0.38 [+0.13, +0.64] over constant-s ECMS, which holds charge on only 75% of trips. Level with MPC, which sees 15 s of the future: 98.686 vs 98.665, +0.02 [−0.58, +0.54], so a tie, not a win. Holds charge on 93% of trips (A-ECMS 97%). Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 98.66 on 430, verdict mixed with MPC 0.01 higher). v2.1: 98.58 (published 98.57); v2.0 on physics v2 98.86 (published 98.84), a win; v2.0 re-scored on v3.1: 98.45.</sub> | **98.69%** | Adaptive ECMS (classical): 98.2%<br>ECMS (constant s) (classical): 98.31%<br>MPC, 15 s of the future (classical): 98.67%<br>Rule-based (classical): 97% | ➖ tie |
| Power-split hybrids (148 trips), % of DP<br><sub>+0.59 points [+0.42, +0.78] over A-ECMS, +0.20 [−0.02, +0.42] over constant-s ECMS. Holds charge on 93% (A-ECMS 97%, rule 93%, ECMS 72%). Same as v2.1 (v2.2 changes only plug-ins and buses). v2.0: 98.41 on physics v2.</sub> | **97.64%** | Adaptive ECMS (classical): 97.05%<br>ECMS (constant s) (classical): 97.44%<br>Rule-based (classical): 97.44% | ✅ wins |
| P2 and series hybrids (64 trips), % of DP<br><sub>P2 +1.39 [+0.31, +3.29], series +0.88 [+0.57, +1.18] over A-ECMS. Holds charge on 94% (A-ECMS 92%; constant-s ECMS 59%). Same as v2.1. Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 99.63 on 63 trips, A-ECMS 98.42). v2.0: 99.85 on physics v2 (published 99.84).</sub> | **99.64%** | Adaptive ECMS (classical): 98.44%<br>ECMS (constant s) (classical): 99.29%<br>Rule-based (classical): 95.35% | ✅ wins |
| Mild P1 hybrids, CR-Z-type twins (56 trips), % of DP<br><sub>+1.08 [+0.79, +1.51] over A-ECMS, but holds charge on only 79% of trips (A-ECMS 96%; v2.0 re-scored on v3.1: 91%). Constant-s ECMS scores 0.21 higher (CI of the difference [−0.53, +0.10]) and holds charge on 34%. Unchanged from v2.1: a mild rule without the dead band was rejected on dev (R20). v2.0: 98.10 on physics v2 (holds 95%).</sub> | **98.36%** | Adaptive ECMS (classical): 97.28%<br>ECMS (constant s) (classical): 98.57%<br>Rule-based (classical): 93.45% | ↔️ mixed |
| Plug-in hybrids (104 trips), % of DP<br><sub>Power-split plug-ins −0.91 [−4.51, +2.16], series plug-ins −0.73 [−1.78, −0.07] against A-ECMS; the charge-depleting rule 99.35. The v2.2 CS rule raised plug-in charge holding from 98 to 102 of 104 trips, not their fuel score. Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 99.22 on 103 trips, A-ECMS 100.17). v2.1: 99.26 (published 99.25); v2.0: 99.33 on physics v2 (published 99.27; A-ECMS 99.56, published 99.59).</sub> | **99.29%** | Adaptive ECMS (classical): 100.13%<br>Charge-depleting then sustaining (classical): 99.35% | ❌ loses |
| Hybrid buses (24 trips), % of DP<br><sub>Against constant-s ECMS: series buses 99.36 vs 98.30 (+1.06 [+0.86, +1.23]), P2 buses 99.66 vs 99.48 (+0.18 [−0.02, +0.42]). Against A-ECMS: series +3.87 [+2.00, +7.04], P2 +1.13 [+0.62, +1.47]. The cost: series buses hold charge on 9 of 12 trips (v2.1 12, ECMS 11); P2 buses 12 of 12. v2.1: 97.66, a loss to ECMS; v2.0: 97.50 on physics v2.</sub> | **99.51%** | Adaptive ECMS (classical): 97.01%<br>ECMS (constant s) (classical): 98.89%<br>Rule-based (classical): 96.07% | ✅ wins |
| Charge held within its window (all trips)<br><sub>401 of 432 trips (A-ECMS 420). v2.1: 400 (93%): v2.2 gains 4 plug-in trips and loses 3 series-bus trips; mild hybrids still 79%. Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 399 of 430; A-ECMS 418). v2.0 on physics v2: 95% (A-ECMS 96%); v2.0 re-scored on v3.1: 96%.</sub> | **93%** | Adaptive ECMS (classical): 97% | ❌ loses |
| Engine starts per km (all trips)<br><sub>v2.1: 3.0. v2.0: 4.0 vs 4.5 on physics v2 (v3.1's 2 s run-on into braking lowers starts for every strategy).</sub> | **3 /km** | Adaptive ECMS (classical): 3.2 /km | ✅ wins |
| Against v2.1 re-scored on the same exam, % of DP<br><sub>+0.11 points [+0.01, +0.24] (series buses +2.88, P2 buses +0.83, plug-ins about +0.04; every other trip identical). Charge held on 401 vs 400 trips. Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 98.66 vs 98.55; the gap between v2.1's published 98.57 and that re-score came from the same bug, not from RoadSim).</sub> | **98.69%** | HybridPilot v2.1 (published): 98.58% | ✅ wins |
| Against v2.0 re-scored on the same exam (v2.0 network and settings on physics v3.1), % of DP<br><sub>+0.23 points [+0.09, +0.41] (mild +0.44, buses +1.6 to +3.2), but holds charge on 93% of trips against v2.0's 96%. v2.1: +0.14. Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 98.66 vs 98.43).</sub> | **98.69%** | HybridPilot v2.0 (published): 98.45% | ↔️ mixed |
| Against the v1 release re-scored on the same exam (v1 network and settings on physics v3.1), % of DP<br><sub>+0.12 points [−0.02, +0.31]; v1 holds charge on 91% of trips (v2.2 93%). v2.1: +0.03. v2.0 on physics v2: 98.86 vs 99.02 (published 98.84 vs 99.00), a loss. Corrected 15 Nov 2026: two duplicate trip ids were graded against each other's baselines; now all 432 trips are scored (was 98.66 vs 98.54).</sub> | **98.69%** | HybridPilot v1 (published): 98.56% | ➖ tie |
| Simulated vs logged fuel, calibrated twins driven by HybridPilot (exam, trip median)<br><sub>v2.2, the same as v2.1 on these cars: power-split HEVs +2.2% (168 trips, 56 cars), mild +2.3%, P2 +3.4% (5 trips); plug-in power-split −20.1%, plug-in series −30.9% (fuel + battery energy). Engine starts on power-splits 2.42 per km vs 1.94 real (v2.0 on physics v2: 3.65). Under the 8% target for hybrids, not for plug-ins.</sub> | **+2.2% HEV, −20% / −31% plug-ins** | — | ℹ️ |

### Where it wins

- Beats every causal baseline on all 432 exam trips: 98.69% of the DP optimum against 98.20% for adaptive ECMS (+0.49 points, CI [−0.07, +0.98]), 98.31% for constant-s ECMS and 97.00% for rule-based control, and level with MPC, which sees 15 s of the future (98.67%).
- Buses (v2.2): 99.51% of DP against 98.89% for constant-s ECMS and 97.01% for A-ECMS; series buses +1.06 points over ECMS.
- Clear gains against A-ECMS on power-split hybrids (+0.59 points), P2 (+1.39), series hybrids (+0.88), mild hybrids (+1.08), series buses (+3.87) and P2 buses (+1.13).
- Fewer engine starts than A-ECMS (3.0 vs 3.2 per km).
- With HybridPilot driving them, the calibrated twins' fuel is within 3.5% of logged fuel for exam hybrids (trip median), and their engine starts are much closer to the real cars' than on physics v2 (2.4 vs 1.9 per km real; was 3.7).

### Where it loses

- No clear margin over MPC, which sees 15 s of the future: 98.69% vs 98.67% overall, +0.02 [−0.58, +0.54].
- Plug-ins: −0.91 points (power-split) and −0.73 (series) against A-ECMS.
- Holds charge on 93% of trips against A-ECMS's 97% (v2.0 on the same exam: 96%); mild hybrids only 79%, series buses 75% (v2.2's weak bus feedback).
- Mild hybrids: constant-s ECMS scores 98.57% against 98.36%, though it holds charge on only 34% of those trips.

## Limits

- Graded in simulation; plug-ins on real roads read 20–31% low in RoadSim, and simulated power-split hybrids still start their engine more often than the real cars (2.4 vs 1.9 starts per km).
- For buses and mild hybrids the network's s is not used: a fixed s (series bus 2.8, P2 bus 2.6, mild 2.8, fitted on 30 training-distribution simulations) with SOC feedback, which makes it an adaptive ECMS there (for buses a weak one, Kp 3).
- RoadSim never runs the engine of a P2 or P1 hybrid at a stop, so on long stops the auxiliaries drain a mild hybrid's 0.6 kWh pack below its window whatever the controller does (a dev mild trip falls from 0.64 to 0.11–0.16).
- The policy outputs no uncertainty; intervals are cluster-bootstrap intervals of paired differences.
- DP ignores RoadSim's forced-on production rules (Toyota-type engine-on above 45 km/h, warm-up holds), so causal strategies cannot reach 100% on those cars.
- Preview, a timed-DP teacher, a start penalty, a distance-to-go plug-in plan, a larger network, a re-teach on physics v3.1, a mild rule without the dead band and a stronger HEV feedback did not pass the ratchet.

## The ratchet

### v1.0 — RoadSim physics v1 (12 Nov 2026)

**Objective:** Frozen skill = 1 − Σ gap(policy) / Σ gap(rule strategies) over strata, gap = 1 − min(% of DP, 1), on a frozen set of 106 trips (36 held-out generated vehicles on held-out routes, synthetic drives and cycles)  
**Rule:** Keep if the frozen skill beats the incumbent by more than 0.01 AND dev % of DP is no more than 0.2 points worse AND dev charge-holding is no more than 3 points worse.

| Rung | Change | Frozen-set score | Decision | Why | Frozen % of DP | Frozen holds | Dev % of DP | Dev holds |
|---|---|---|---|---|---|---|---|---|
| R0 | RoadSim adaptive ECMS (the incumbent) | 0.564 | ● kept | Starting incumbent. | 98.88 | 98% | 98.99 | 100% |
| R1 | Behaviour cloning: s and on/off from the network | 0.592 | ○ rejected | Dev charge holding fell to 74%. | 98.7 | 69% | 99.64 | 74% |
| R2 | DAgger round 1 + SOC feedback (Kp 6) | 0.596 | ● kept | +0.032; dev gate passed. | 98.88 | 94% | 99.2 | 99% |
| R3 | On/off by the cost with the learned s (on/off head unused) | 0.702 | ● kept | +0.106; dev gate passed. Released as v1.0. | 99.31 | 94% | 99.26 | 99% |
| R4 | R3 + 30 s preview | 0.709 | ○ rejected | +0.007, short of the margin. | 99.34 | 94% | 99.26 | 99% |
| R5 | Policy-search fine-tune of head knobs on 60 training trips | 0.671 | ○ rejected | Score fell. | 99.19 | 96% | 99.13 | 99% |
| R6 | DAgger round 2 (+600 trips driven by R3) | 0.674 | ○ rejected | Score fell. | 99.28 | 94% | 99.22 | 99% |
| R7 | R6 + 30 s preview | 0.677 | ○ rejected | Score fell. | 99.23 | 92% | 99.21 | 99% |

<sub>Columns after *Why*: Frozen set (% of DP, charge holding) and dev set (% of DP, charge holding).</sub>

R0 is RoadSim's adaptive ECMS, the incumbent to beat. Grading re-solves DP to each strategy's own final charge (matched SOC).

### v2.0 — RoadSim physics v2 (Phase D, 14 Nov 2026; calibrated twins)

**Objective:** Frozen skill = 1 − Σ gap(policy) / Σ gap(rule strategies) over strata, gap = 1 − min(% of DP, 1), on a new frozen set of 106 trips built on physics v2 (36 held-out generated vehicles on held-out routes, synthetic drives and cycles); DP reference with the physics' 8 s / 3 s minimum on/off times  
**Rule:** Keep if the frozen skill beats the incumbent by more than 0.01 AND dev % of DP is no more than 0.2 points worse AND dev charge-holding is no more than 3 points worse.

| Rung | Change | Frozen-set score | Decision | Why | Frozen % of DP | Frozen holds | Dev % of DP | Dev holds | Frozen starts/km |
|---|---|---|---|---|---|---|---|---|---|
| R0 | RoadSim adaptive ECMS (the incumbent) | 0.255 | ● kept | Starting incumbent. | 98.4 | 96% | 98.2 | 97% | 3.81 |
| R1 | Behaviour cloning of DP (800 teacher trips): s and on/off from the network | 0.691 | ○ rejected | Dev charge holding 70 % (gate 94 %). | 99.27 | 63% | 99.33 | 70% | 3.26 |
| R2 | DAgger round 1 (+600 trips driven by R1) + SOC feedback Kp 6 | 0.620 | ○ rejected | Dev charge holding 88 %. | 99.2 | 78% | 99.27 | 88% | 3.2 |
| R3 | On/off by the cost with the learned s (v1's release recipe) | 0.719 | ○ rejected | Dev charge holding 87 %. | 99.51 | 82% | 99.54 | 87% | 3.01 |
| R4 | R3 + 30 s preview | 0.730 | ○ rejected | Dev charge holding 91 %. | 99.56 | 85% | 99.34 | 91% | 3.02 |
| R5 | Policy-search fine-tune of head knobs on 60 training trips | 0.734 | ○ rejected | Dev charge holding 88 %. | 99.55 | 88% | 99.71 | 88% | 2.91 |
| R6 | DAgger round 2 (+600 trips driven by R3) | 0.646 | ○ rejected | Dev charge holding 88 %. | 99.38 | 85% | 99.54 | 88% | 3.06 |
| R7 | R6 + 30 s preview | 0.677 | ○ rejected | Dev charge holding 91 %. | 99.48 | 89% | 99.54 | 91% | 3.02 |
| R8 | R3 with RoadSim A-ECMS's own SOC feedback gains (Kp 12, Ki 0.02 1/s) | 0.659 | **◆ released** | +0.404 over A-ECMS; dev % of DP 99.19 vs 98.20, dev charge holding 97 % = A-ECMS. | 99.33 | 93% | 99.19 | 97% | 3.1 |
| R9 | Re-teach with the calibrated twins' hardware: 40 mild P1 (CR-Z-type) training vehicles, +600 DAgger trips | 0.657 | ○ rejected | −0.002. Dev mild-hybrid charge holding 83 → 92 %. | 99.32 | 95% | 99.15 | 97% | 3.13 |
| R10 | Causal proxy of DP with minimum on/off times: teacher = DP with the 8 s / 3 s timers (+1,400 trips) | 0.624 | ○ rejected | −0.036. | 99.27 | 95% | 99.12 | 95% | 3.13 |
| R11 | Drivability term: 20 kJ virtual start cost in the decision | 0.560 | ○ rejected | −0.100; starts/km 3.10 → 2.90. | 99.0 | 91% | 99.01 | 97% | 2.9 |
| R12 | Distance-to-go plug-in plan: with a known trip length the SOC reference falls linearly to the CS level at the trip end | 0.659 | ○ rejected | −0.001. | 99.33 | 92% | 99.18 | 97% | 3.1 |
| R13 | Larger network on the GPU: 3 × 256 (~150k parameters), 40 epochs | 0.671 | ○ rejected | +0.012 but dev charge holding 93 % (gate 94 %). | 99.36 | 95% | 99.47 | 93% | 3.13 |
| R14 | R8 + 30 s preview | 0.667 | ○ rejected | +0.007, short of the margin. | 99.38 | 97% | 99.1 | 99% | 3.1 |
| R15 | Policy-search re-tune of the head knobs incl. Ki, from R8 | 0.682 | ○ rejected | +0.022 but dev charge holding 88 %. | 99.4 | 83% | 99.28 | 88% | 2.89 |

<sub>Columns after *Why*: Frozen set (% of DP, charge holding), dev set (% of DP, charge holding), frozen engine starts per km.</sub>

Physics changed (v1 → v2), so the sets were rebuilt and every v1 decision re-taken from R0. R1–R7 are v1's rungs re-run on v2: all beat A-ECMS on the frozen set by +0.37 to +0.48 but failed the dev charge-holding gate (70–91 % vs 97 %; R1 held 70 %). R8 (A-ECMS's own SOC feedback gains) passed and is released. The ladder was closed before the exam.

### v2.1 — RoadSim physics v3.1 (15 Nov 2026)

**Objective:** Frozen skill as v2.0 (share of the rule strategies' gap to DP closed, strata weigh the same) on a frozen set of 106 trips rebuilt on physics v3.1 (same draws and seeds; generated hybrids carry the v3 generator hardware)  
**Rule:** Keep if the frozen skill beats the incumbent by more than 0.01 AND dev % of DP is no more than 0.2 points worse AND dev charge-holding is no more than 3 points worse.

| Rung | Change | Frozen-set score | Decision | Why | Frozen % of DP | Frozen holds | Dev % of DP | Dev holds | Frozen starts/km |
|---|---|---|---|---|---|---|---|---|---|
| R0 | RoadSim adaptive ECMS (the incumbent) | 0.144 | ● kept | Starting incumbent. | 97.5 | 95% | 98.14 | 97% | 2.44 |
| R1 | v2 R1 re-scored: behaviour cloning of DP | 0.692 | ○ rejected | +0.548 vs R0; dev charge holding 67 % (gate 94 %). | 99.08 | 61% | 99.21 | 67% | 2.21 |
| R2 | v2 R2 re-scored: DAgger round 1 + SOC feedback Kp 6 | 0.644 | ○ rejected | +0.500 vs R0; dev charge holding 88 % (gate 94 %). | 99.03 | 77% | 98.94 | 88% | 2.22 |
| R3 | v2 R3 re-scored: on/off by the cost (v1's recipe) | 0.717 | ○ rejected | +0.573 vs R0; dev charge holding 88 % (gate 94 %). | 99.25 | 79% | 99.37 | 88% | 2.19 |
| R4 | v2 R4 re-scored: R3 + 30 s preview | 0.725 | ○ rejected | +0.581 vs R0; dev charge holding 89 % (gate 94 %). | 99.32 | 83% | 99.11 | 89% | 2.19 |
| R5 | v2 R5 re-scored: policy-search tune of head knobs | 0.738 | ○ rejected | +0.595 vs R0; dev charge holding 88 % (gate 94 %). | 99.29 | 82% | 99.55 | 88% | 2.16 |
| R6 | v2 R6 re-scored: DAgger round 2 | 0.667 | ○ rejected | +0.523 vs R0; dev charge holding 88 % (gate 94 %). | 99.18 | 83% | 99.29 | 88% | 2.21 |
| R7 | v2 R7 re-scored: R6 + 30 s preview | 0.702 | ○ rejected | +0.558 vs R0; dev charge holding 91 % (gate 94 %). | 99.33 | 84% | 99.27 | 91% | 2.19 |
| R8 | v2 R8 re-scored (the v2.0 release): R3 + A-ECMS's SOC feedback gains (Kp 12, Ki 0.02) | 0.662 | ● kept | +0.518 vs R0; dev gate passed. | 99.06 | 90% | 98.94 | 96% | 2.22 |
| R9 | v2 R9 re-scored: mild P1 class in training | 0.630 | ○ rejected | -0.032 vs R8. | 98.94 | 90% | 98.86 | 97% | 2.24 |
| R10 | v2 R10 re-scored: timed-DP teacher | 0.625 | ○ rejected | -0.038 vs R8. | 98.97 | 92% | 98.86 | 93% | 2.25 |
| R11 | v2 R11 re-scored: 20 kJ start cost | 0.553 | ○ rejected | -0.109 vs R8. | 98.67 | 88% | 98.83 | 95% | 2.05 |
| R12 | v2 R12 re-scored: distance-to-go plug-in plan | 0.660 | ○ rejected | -0.002 vs R8. | 99.07 | 90% | 98.94 | 96% | 2.22 |
| R13 | v2 R13 re-scored: 3 x 256 network | 0.663 | ○ rejected | +0.001 vs R8, short of the margin; dev charge holding 92 % (gate 93 %). | 99.02 | 91% | 99.19 | 92% | 2.25 |
| R14 | v2 R14 re-scored: R8 + 30 s preview | 0.651 | ○ rejected | -0.011 vs R8. | 99.07 | 93% | 98.85 | 99% | 2.24 |
| R15 | v2 R15 re-scored: policy-search re-tune incl. Ki | 0.662 | ○ rejected | +0.000 vs R8, short of the margin; dev charge holding 92 % (gate 93 %). | 99.12 | 82% | 99.13 | 92% | 2.12 |
| R16 | Dead band +-0.03 on the proportional SOC feedback (charge-sustaining term acts on drift, not on short swings) | 0.689 | ● kept | +0.027 vs R8; dev gate passed. | 99.19 | 87% | 99.16 | 95% | 2.2 |
| R17 | R16 + a constant equivalence factor in place of the network's s for buses (series 2.8, P2 2.6) and mild P1 hybrids (2.8), fitted on training-distribution probe simulations | 0.704 | **◆ released** | +0.015 vs R16; dev gate passed. Released as v2.1. | 99.22 | 87% | 99.28 | 95% | 2.19 |
| R18 | Re-teach on physics v3.1 banks (v2.0 recipe: 800 teacher + 600 DAgger trips), R17's rule | 0.698 | ○ rejected | -0.006 vs R17. | 99.22 | 88% | 99.13 | 93% | 2.18 |

<sub>Columns after *Why*: Frozen set (% of DP, charge holding), dev set (% of DP, charge holding), frozen engine starts per km.</sub>

Physics rung first, decided on dev only: HybridPilot v2.0 driving the dev twins on v3.1 vs v2 (84 trips) moved power-split fuel from -4.6 to -2.4 % of the logs, the engine-on gap from -9.3 to -4.4 points and starts from 3.67 to 2.66 per km (real 2.32); mild +2.9 -> +3.4 %; plug-ins unchanged. Physics changed, so R0-R15 were re-scored in order with their v2 networks and settings: the same decisions as on v2 (R8, the v2.0 release, kept). R16-R17 settings were chosen on a 190-trip training-distribution probe set, never on dev or exam. The ladder was closed and the release choice committed (d9fd0fd) before the exam.

### v2.2 — round 2 on RoadSim physics v3.1 (15 Nov 2026): charge holding

**Objective:** Frozen skill as v2.1 (share of the rule strategies' gap to DP closed, strata weigh the same) on the same frozen set (frozen_106_v3_1), and charge holding (share of trips that hold charge within their window, the exam's holding definition) on the frozen and dev trips together (182 trips)  
**Rule:** Rule v2.2, fixed before R19 was scored. Keep if (A) the frozen skill beats the incumbent by more than 0.01 AND the dev gate passes (unchanged from v2.1), OR (B) charge holding over the frozen and dev trips together rises by at least 2 points AND neither set's holding falls AND the frozen skill is no more than 0.01 below the incumbent's AND dev % of DP is no more than 0.2 points worse.

| Rung | Change | Frozen-set score | Decision | Why | Frozen % of DP | Frozen holds | Dev % of DP | Dev holds | Frozen starts/km |
|---|---|---|---|---|---|---|---|---|---|
| R19 | Plug-ins at their CS level: s no lower than the DP median costate of the HEV of the same architecture (power-split 2.42, series 2.70); below CS, SOC feedback Kp 24 without the dead band | 0.712 | ● kept | Path B: frozen + dev holding 90.1 -> 94.0 % (frozen series plug-ins 7 -> 12 of 12, power-split 16 -> 18 of 18); frozen skill +0.008; dev % of DP 99.28 -> 99.30. | 99.27 | 93% | 99.3 | 95% | 2.21 |
| R20 | R19 + mild P1 hybrids without the dead band (s0 2.8 kept) | 0.712 | ○ rejected | Frozen has no mild trips (skill +0.000); dev mild holding 9 -> 10 of 12, pooled holding +0.5 points (needs +2); dev % of DP 99.30 -> 99.23. | 99.27 | 93% | 99.23 | 96% | 2.21 |
| R21 | R19 + HEVs (incl. P2 trucks) Kp 18 beyond the dead band | 0.691 | ○ rejected | Frozen skill -0.022 (limit -0.01), dev % of DP -0.15 points; holding would rise to 96 % frozen, 96 % dev. | 99.17 | 96% | 99.15 | 96% | 2.23 |
| R22 | R19 + buses: SOC feedback Kp 3 without the integral term around the fixed bus s | 0.774 | **◆ released** | Path A: frozen skill +0.062 (frozen P2 buses 97.40 -> 99.39 % of DP); dev gate passed (99.30 %, holds 95 %). Frozen holding 93.4 -> 90.6 % (buses end 0.68-0.70 like constant-s ECMS). Released as v2.2. | 99.38 | 91% | 99.3 | 95% | 2.21 |

<sub>Columns after *Why*: Frozen set (% of DP, charge holding), dev set (% of DP, charge holding), frozen engine starts per km.</sub>

Why a second path: v2.1 holds charge on 93 % of exam trips against 96 % for v2.0 and 97 % for A-ECMS, and the same drift shows before the exam: v2.1 holds 87 % of frozen trips (A-ECMS 95 %) and 95 % of dev trips (A-ECMS 97 %). The frozen-skill margin measures fuel only, so a rung that fixes holding at no fuel cost could never pass path A. Settings are chosen on training-distribution probe sets (probe, and a new 80-trip mild probe), never on dev or exam. The starting incumbent is v2.1 (R17 of the v2.1 ladder: frozen skill 0.704; frozen + dev holding 164 of 182 trips, 90.1 %).

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle), physics v3.1 pinned (roadsim-physics-v3.1; v2.0 used v2): EPA ALPHA engine and motor maps, Thevenin battery, eCVT / P2 (with torque converter and start energy) / series (with direct drive) powertrains, two-node cabin HVAC, tyre model, production hybrid rules (8 s / 3 s minimum on/off, Toyota-type engine-on speed, warm-up holds, 2 s engine run-on into braking, plug-in charge-depleting rules), the v3 generator hybrid hardware, an energy ledger checked on every run. Teacher: DP over engine on/off × 121 SOC points; reference: DP with the same 8 s / 3 s on/off timers.

**Simulator**

- 400 generated hybrids in 18 class × architecture combinations (v2 donor model: hybrid bodies from real hybrid records), cars to series buses and Class-8 P2 trucks; the network was trained on physics-v2 banks (a re-teach on v3.1 banks did not pass the ratchet)
- VED cars as RoadSim's calibrated twins (road load and engine level matched to each car's fueleconomy.gov city / highway values; CR-Z-type cars as mild P1 hybrids)
- Routes from 835 real VED trips (real speed, lidar grade, airport weather), the full ZTBus set (by-month split), EPA/HD cycles and synthetic drives
- DP teacher labels (engine on/off and the costate s*) plus a DAgger round on the learned policy's own trips

**Checked against independent references**

- Numpy inference module == training policy: identical fuel and final SOC on 44 frozen and dev trips incl. 22 plug-in and 8 bus trips (v2.2 rule); numpy == torch to 5e-6
- Physics v2 → v3.1 decided on dev: HybridPilot v2.0 driving the dev twins, power-split fuel −4.6 → −2.4 % of the logs, engine-on gap −9.3 → −4.4 points, starts 3.67 → 2.66 per km (real 2.32)
- Simulator vs logged fuel with HybridPilot v2.2 driving the calibrated twins (OBD label model, exam trip median): power-split HEV +2.2 %, mild +2.3 %, P2 +3.4 % (5 trips); plug-in power-split −20 %, plug-in series −31 %
- Grading re-solves DP to each strategy's own final SOC (matched SOC); ledger closes to < 0.5 % on every run

## Next rungs

- Mild charge holding: a frozen mild stratum (the frozen set has none, so no mild rung can pass) and RoadSim engine-at-stop charging for P1/P2 hybrids; then a mild rule judged on it.
- Charge holding on HEVs: Kp 18 reaches A-ECMS's frozen holding (96%) but costs 0.022 of frozen skill; a terminal term using the known trip length (nav) is untried.
- A plug-in teacher conditioned on the trip-length uncertainty (nav known or not), and a dedicated plug-in frozen stratum.

## Files

| Path | What |
|---|---|
| `hybridpilot.py`, `pipeline.py` | Inference — numpy/scipy only |
| `weights/hybridpilot_v2.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Vehicle Energy Dataset (VED) — Oh, LeBlanc, Peng, IEEE T-ITS 2020](https://github.com/gsoh/VED) | noise (routes), dev, exam (routes) | Apache-2.0 |
| [ZTBus — Widmer, Ritter, Onder, Scientific Data 10 (2023), ETH Zurich](https://doi.org/10.3929/ethz-b-000626723) | dev, exam (bus routes) | CC BY 4.0 |
| [US EPA drive cycles and test-car lists; fueleconomy.gov; 40 CFR 1037 heavy-duty cycles; USGS 3DEP lidar; IEM ASOS weather](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules) | physics, exam (cycles) | Public domain / free use with credit |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
