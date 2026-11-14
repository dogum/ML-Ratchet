<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# HybridPilot <sub>v2.0</sub>

> **A learned hybrid controller, raced against the dynamic-programming optimum.**

> [!WARNING]
> Graded inside the simulator. With HybridPilot driving the calibrated twins, simulated hybrid fuel is within 3 % of VED's logged fuel (exam trip median), but plug-ins read 22–31 % low, and the strategies differ by only 0.5–1.5 %, so the ranking holds only inside RoadSim's physics.

*Second by second, should a hybrid run its engine, and how hard — to use the least fuel the physics allows?*

HybridPilot is a learned, causal energy-management policy for hybrids: every second it decides engine on/off and engine power, holding charge in hybrids and depleting-then-sustaining in plug-ins. It learned by imitating the dynamic-programming (DP) optimum inside the RoadSim simulator. Version 2 was rebuilt on RoadSim physics v2 (calibrated on real dev data) and examined on 430 trips — real VED routes driven by calibrated twins of the exam cars, bus legs and cycles — against rule-based control, ECMS, adaptive ECMS and MPC.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/hybridpilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/hybridpilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | The vehicle's parameters and, each second, the state (SOC, coolant temperature, timers, power limits), current demand (speed, acceleration, grade, wheel power), trip history and optionally a navigation preview. |
| **Returns** | Engine on/off and engine power (series, power-split) or torque share (P2) every second<br>The resulting fuel, battery use and engine starts over the trip, compared against DP and the classical strategies |
| **Learned from** | The shared RoadSim simulator (shared/vehicle), physics v2 pinned (roadsim-physics-v2): EPA ALPHA engine and motor maps, Thevenin battery, eCVT / P2 (with torque converter and start energy) / series (with direct drive) powertrains, two-node cabin HVAC, tyre model, production hybrid rules (8 s / 3 s minimum on/off, Toyota-type engine-on speed, warm-up holds, plug-in charge-depleting rules), an energy ledger checked on every run. Teacher: DP over engine on/off × 121 SOC points; reference: DP with the same 8 s / 3 s on/off timers. |
| **Noise** | Real routes, grades and weather; random start SOC, temperatures from −20 to 36 °C, warm and cold starts |
| **Network** | MLP 66 → 96 → 96 → 2 (15,938 parameters) predicting the equivalence factor s; the decision minimises fuel + s × battery power among RoadSim's candidate controls, with A-ECMS-style SOC feedback (Kp 12, Ki 0.02) |
| **Dev set** | 76 trips — routes of the VED dev cars (calibrated twins of the 7 dev hybrids + 18 dev generated vehicles) and full-ZTBus dev legs (by-month split) |
| **Exam** | 430 trips — 216 real VED exam routes driven by calibrated twins of the 108 exam hybrids, 214 by 54 held-out generated hybrids, full-ZTBus exam legs and HD cycles |

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

430 exam trips, graded as % of the DP optimum (DP with the same 8 s / 3 s on/off timers) at matched final charge. Real routes, simulated cars. Run once, after the release choice was committed. Every number here is on real data that never chose anything.

| Metric | HybridPilot | Baselines | |
|---|---|---|---|
| All 430 trips, % of DP optimum<br><sub>+0.53 points over A-ECMS, 95% CI [+0.20, +0.82] (bootstrap by vehicle). ECMS holds charge on only 74% of trips; MPC sees the future.</sub> | **98.84%** | Adaptive ECMS (classical): 98.32%<br>ECMS (constant s) (classical): 98.53%<br>MPC, 15 s of the future (classical): 98.8%<br>Rule-based (classical): 97.39% | ✅ wins |
| Power-split hybrids (148 trips), % of DP<br><sub>+0.89 points [+0.75, +1.04].</sub> | **98.41%** | Adaptive ECMS (classical): 97.52%<br>Rule-based (classical): 97.92% | ✅ wins |
| P2 and series hybrids (63 trips), % of DP<br><sub>P2 +1.32 [+0.37, +2.85]; series +1.04 [+0.68, +1.39].</sub> | **99.84%** | Adaptive ECMS (classical): 98.63%<br>Rule-based (classical): 95.84% | ✅ wins |
| Mild P1 hybrids, CR-Z-type twins (56 trips), % of DP<br><sub>+0.60 [+0.29, +0.83].</sub> | **98.1%** | Adaptive ECMS (classical): 97.51%<br>Rule-based (classical): 94.02% | ✅ wins |
| Plug-in hybrids (103 trips), % of DP<br><sub>Power-split plug-ins −0.29 [−2.31, +1.53], series plug-ins −0.35 [−1.09, +0.12].</sub> | **99.27%** | Adaptive ECMS (classical): 99.59%<br>Charge-depleting then sustaining (classical): 99.43% | ❌ loses |
| Hybrid buses (24 trips), % of DP<br><sub>Series buses +0.70 [+0.21, +1.29]; P2 buses −0.48 [−0.63, −0.40].</sub> | **97.5%** | Adaptive ECMS (classical): 97.39%<br>Rule-based (classical): 96.6% | ↔️ mixed |
| Charge held within its window (all trips) | **95%** | Adaptive ECMS (classical): 96% | ❌ loses |
| Engine starts per km (all trips) | **4 /km** | Adaptive ECMS (classical): 4.5 /km | ✅ wins |
| Against the v1 release re-scored on the same exam (v1 network and settings on physics v2), % of DP<br><sub>−0.16 points [−0.26, −0.06], but v1 holds charge on only 89% of trips (v2: 95%); v1's settings failed the v2 dev charge-holding gate.</sub> | **98.84%** | HybridPilot v1 (published): 99% | ❌ loses |
| Simulated vs logged fuel, calibrated twins driven by HybridPilot (exam, trip median)<br><sub>Power-split HEVs +2.2% (168 trips, 56 cars), mild +0.3%, P2 +1.6%; plug-in power-split −22.4%, plug-in series −30.9% (fuel + battery energy). Under the 8% target for hybrids, not for plug-ins.</sub> | **+2.2% HEV, −22% / −31% plug-ins** | — | ℹ️ |

### Where it wins

- Reaches 98.84% of the DP optimum on 430 exam trips against 98.32% for adaptive ECMS (+0.53 points, CI [+0.20, +0.82]), 97.39% for rule-based control and 98.80% for MPC, which sees 15 s of the future.
- Clear gains on power-split hybrids (+0.89 points), P2 (+1.32), series hybrids (+1.04), mild hybrids (+0.60) and series buses (+0.70).
- Fewer engine starts than A-ECMS (4.0 vs 4.5 per km).
- With HybridPilot driving them, the calibrated twins' fuel is within 3% of logged fuel for exam hybrids (trip median).

### Where it loses

- Plug-ins: −0.29 points (power-split) and −0.35 (series) against A-ECMS, inside wide intervals.
- P2 buses: −0.48 points.
- The v1 release, re-scored on v2, burns 0.16 points less fuel, but fails to hold charge on 11% of trips.
- Holds charge on 95% of trips against A-ECMS's 96%.

## Limits

- Graded in simulation; plug-ins on real roads read 22–31% low in RoadSim, and simulated hybrids start their engine about twice as often as the real cars (3.7 vs 1.9 starts per km on exam power-splits).
- The policy outputs no uncertainty; intervals are cluster-bootstrap intervals of paired differences.
- DP ignores RoadSim's forced-on production rules (Toyota-type engine-on above 45 km/h, warm-up holds), so causal strategies cannot reach 100% on those cars.
- Preview, a timed-DP teacher, a start penalty, a distance-to-go plug-in plan and a larger network did not pass the ratchet.

## The ratchet

### v1 — RoadSim physics v1

**Objective:** Frozen skill = 1 − Σ gap(policy) / Σ gap(rule strategies) over strata, gap = 1 − min(% of DP, 1), on a frozen set of 106 trips (36 held-out generated vehicles on held-out routes, synthetic drives and cycles)  
**Rule:** Keep if the frozen skill beats the incumbent by more than 0.01 AND dev % of DP is no more than 0.2 points worse AND dev charge-holding is no more than 3 points worse.

| Rung | Change | Frozen-set score | Decision | Why | Frozen % of DP | Frozen holds | Dev % of DP | Dev holds |
|---|---|---|---|---|---|---|---|---|
| R0 | RoadSim adaptive ECMS (the incumbent) | 0.564 | ● kept | Starting incumbent. | 98.88 | 98% | 98.99 | 100% |
| R1 | Behaviour cloning: s and on/off from the network | 0.592 | ○ rejected | Dev charge holding fell to 74%. | 98.7 | 69% | 99.64 | 74% |
| R2 | DAgger round 1 + SOC feedback (Kp 6) | 0.596 | ● kept | +0.032; dev gate passed. | 98.88 | 94% | 99.2 | 99% |
| R3 | On/off by the cost with the learned s (on/off head unused) | 0.702 | ● kept | +0.106; dev gate passed. Released as v1. | 99.31 | 94% | 99.26 | 99% |
| R4 | R3 + 30 s preview | 0.709 | ○ rejected | +0.007, short of the margin. | 99.34 | 94% | 99.26 | 99% |
| R5 | Policy-search fine-tune of head knobs on 60 training trips | 0.671 | ○ rejected | Score fell. | 99.19 | 96% | 99.13 | 99% |
| R6 | DAgger round 2 (+600 trips driven by R3) | 0.674 | ○ rejected | Score fell. | 99.28 | 94% | 99.22 | 99% |
| R7 | R6 + 30 s preview | 0.677 | ○ rejected | Score fell. | 99.23 | 92% | 99.21 | 99% |

<sub>Columns after *Why*: Frozen set (% of DP, charge holding) and dev set (% of DP, charge holding).</sub>

R0 is RoadSim's adaptive ECMS, the incumbent to beat. Grading re-solves DP to each strategy's own final charge (matched SOC).

### v2 — RoadSim physics v2 (calibrated twins)

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

Physics changed (v1 → v2), so the sets were rebuilt and every v1 decision re-taken from R0. R1–R7 are v1's rungs re-run on v2: all beat A-ECMS on the frozen set by +0.37 to +0.48 but failed the dev charge-holding gate (87–91 % vs 97 %). R8 (A-ECMS's own SOC feedback gains) passed and is released. The ladder was closed before the exam.

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle), physics v2 pinned (roadsim-physics-v2): EPA ALPHA engine and motor maps, Thevenin battery, eCVT / P2 (with torque converter and start energy) / series (with direct drive) powertrains, two-node cabin HVAC, tyre model, production hybrid rules (8 s / 3 s minimum on/off, Toyota-type engine-on speed, warm-up holds, plug-in charge-depleting rules), an energy ledger checked on every run. Teacher: DP over engine on/off × 121 SOC points; reference: DP with the same 8 s / 3 s on/off timers.

**Simulator**

- 400 generated hybrids in 18 class × architecture combinations (v2 donor model: hybrid bodies from real hybrid records), cars to series buses and Class-8 P2 trucks
- VED cars as RoadSim's calibrated twins (road load and engine level matched to each car's fueleconomy.gov city / highway values; CR-Z-type cars as mild P1 hybrids)
- Routes from 835 real VED trips (real speed, lidar grade, airport weather), the full ZTBus set (by-month split), EPA/HD cycles and synthetic drives
- DP teacher labels (engine on/off and the costate s*) plus a DAgger round on the learned policy's own trips

**Checked against independent references**

- Numpy inference module == training policy: identical fuel on 23 frozen and dev trips; numpy == torch to 5e-6
- Simulator vs logged fuel with HybridPilot driving the calibrated twins (OBD label model, exam trip median): power-split HEV +2.2 %, mild +0.3 %, P2 +1.6 % (5 trips); plug-in power-split −22 %, plug-in series −31 %
- Grading re-solves DP to each strategy's own final SOC (matched SOC); ledger closes to < 0.5 % on every run

## Next rungs

- A charge-holding term in the teacher, so the network's own s holds charge without the strong A-ECMS feedback (R3-R7 lost 0.6 points of skill to charge holding).
- A plug-in teacher conditioned on the trip-length uncertainty (nav known or not), and a dedicated plug-in frozen stratum.
- Learn the real cars' engine-start behaviour from VED dev logs (simulated starts are twice the real rate).

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
| [ZTBus — Widmer, Ritter, Onder, Scientific Data 10 (2023), ETH Zurich](https://www.research-collection.ethz.ch/handle/20.500.11850/626723) | dev, exam (bus routes) | CC BY 4.0 |
| [US EPA drive cycles and test-car lists; fueleconomy.gov; 40 CFR 1037 heavy-duty cycles; USGS 3DEP lidar; IEM ASOS weather](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules) | physics, exam (cycles) | Public domain / free use with credit |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
