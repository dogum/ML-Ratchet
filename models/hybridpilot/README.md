<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# HybridPilot <sub>v1.0</sub>

> **A learned hybrid controller, raced against the dynamic-programming optimum.**

> [!WARNING]
> Graded inside the simulator. RoadSim burns about 20% less fuel than real hybrids, a gap far larger than the 1–3% between strategies, so the ranking holds only inside RoadSim's physics.

*Second by second, should a hybrid run its engine, and how hard — to use the least fuel the physics allows?*

HybridPilot is a learned, causal energy-management policy for hybrids: every second it decides engine on/off and engine power, holding charge in hybrids and depleting-then-sustaining in plug-ins. It learned by imitating the dynamic-programming (DP) optimum inside the RoadSim simulator, and was examined on 430 trips — real VED routes and bus legs driven by simulated twins — against rule-based control, ECMS, adaptive ECMS and MPC.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/hybridpilot/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/hybridpilot/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | The vehicle's parameters and, each second, the state (SOC, coolant temperature, timers, power limits), current demand (speed, acceleration, grade, wheel power), trip history and optionally a navigation preview. |
| **Returns** | Engine on/off and engine power (series, power-split) or torque share (P2) every second<br>The resulting fuel, battery use and engine starts over the trip, compared against DP and the classical strategies |
| **Learned from** | The shared RoadSim simulator (shared/vehicle): EPA ALPHA engine and motor maps, Thevenin battery, eCVT / P2 / series powertrains, HVAC, an energy ledger checked on every run; DP over engine on/off × 121 SOC points as the teacher. |
| **Noise** | Real routes, grades and weather; random start SOC, temperatures from −20 to 36 °C, warm and cold starts |
| **Network** | MLP 66 → 96 → 96 → 2 (15,938 parameters) predicting the equivalence factor s; the decision minimises fuel + s × battery power among RoadSim's candidate controls |
| **Dev set** | 76 trips — routes of the 16 VED dev cars (twins of 7 dev hybrids + 18 dev generated vehicles) and ZTBus dev legs |
| **Exam** | 430 trips — 216 real VED exam routes driven by twins of those cars, 214 by 54 held-out generated hybrids, ZTBus exam legs and HD cycles |

## Run it

```bash
cd models/hybridpilot
pip install numpy scipy
python demo.py        # about 11 s
```

or from anywhere in the repo: `mlratchet demo hybridpilot`.

```python
from pipeline import compare, load_example     # needs RoadSim: found in shared/vehicle automatically
spec, trip = load_example("VED_428_465")
for r in compare(spec, trip, init_soc=None, warm=True):
    print(r["strategy"], r["fuel_g"], r["batt_kWh_out"])
```

## The exam

430 exam trips, graded as % of the DP optimum at matched final charge. Real routes, simulated cars. Every number here is on real data that never chose anything.

| Metric | HybridPilot | Baselines | |
|---|---|---|---|
| All 430 trips, % of DP optimum<br><sub>+0.13 points over A-ECMS, 95% CI [−0.06, +0.29] (bootstrap by vehicle): too small to call.</sub> | **99.49%** | Adaptive ECMS (classical): 99.35%<br>Rule-based (classical): 98.28% | ➖ tie |
| Power-split hybrids (204 trips), % of DP<br><sub>+0.41 points [+0.29, +0.55].</sub> | **99.31%** | Adaptive ECMS (classical): 98.9%<br>Rule-based (classical): 98.59% | ✅ wins |
| Series hybrid buses (12 trips), % of DP | **99.78%** | Adaptive ECMS (classical): 99.12%<br>Rule-based (classical): 95.31% | ✅ wins |
| Series plug-ins (46 trips), % of DP<br><sub>−0.93 points [−2.43, −0.09]; DP is not a tight reference here.</sub> | **100%** | Adaptive ECMS (classical): 100.93%<br>Rule-based (classical): 100.23% | ❌ loses |
| Charge held within its window (all trips) | **97%** | Adaptive ECMS (classical): 99% | ❌ loses |
| Simulator fuel vs real logged fuel (exam power-split hybrids, 252 trips)<br><sub>The simulator–reality gap (~20%) dwarfs the 1–3% between strategies.</sub> | **0.80×** | — | ℹ️ |

### Where it wins

- Reaches 99.49% of the DP optimum against 99.35% for adaptive ECMS and 98.28% for rule-based control.
- Clear gains on Prius-type power-split hybrids (+0.41 points) and series hybrid buses (+0.66).
- Fewer engine starts per km than A-ECMS (3.8 vs 4.1).

### Where it loses

- Series plug-ins (−0.93 points against A-ECMS).
- Holds charge on 97% of trips against A-ECMS's 99%.
- The overall margin over A-ECMS is inside its confidence interval.

## Limits

- Graded in simulation; RoadSim's fuel on real hybrid trips is ~20% low and plug-in engine use too high, so the ranking holds only inside its physics.
- The policy outputs no uncertainty; intervals are cluster-bootstrap intervals of paired differences.
- Preview did not pass the ratchet; the on/off head is unused.
- DP assumes a warm engine and no catalyst rule (values above 100% mean DP was not tight).
- Built before the 13 Nov RoadData fixes (VED cars 136 kg heavy in the twins).

## The ratchet

**Objective:** Frozen skill = 1 − Σ gap(policy) / Σ gap(rule strategies) over strata, gap = 1 − min(% of DP, 1), on a frozen set of 106 trips (36 held-out generated vehicles on held-out routes, synthetic drives and cycles)  
**Rule:** Keep if the frozen skill beats the incumbent by more than 0.01 AND dev % of DP is no more than 0.2 points worse AND dev charge-holding is no more than 3 points worse.

| Rung | Change | Frozen-set score | Decision | Why | Frozen % of DP | Frozen holds | Dev % of DP | Dev holds |
|---|---|---|---|---|---|---|---|---|
| R0 | RoadSim adaptive ECMS (the incumbent) | 0.564 | ● kept | Starting incumbent. | 98.88 | 98% | 98.99 | 100% |
| R1 | Behaviour cloning: s and on/off from the network | 0.592 | ○ rejected | Dev charge holding fell to 74%. | 98.7 | 69% | 99.64 | 74% |
| R2 | DAgger round 1 + SOC feedback (Kp 6) | 0.596 | ● kept | +0.032; dev gate passed. | 98.88 | 94% | 99.2 | 99% |
| R3 | On/off by the cost with the learned s (on/off head unused) | 0.702 | **◆ released** | +0.106; dev gate passed. | 99.31 | 94% | 99.26 | 99% |
| R4 | R3 + 30 s preview | 0.709 | ○ rejected | +0.007, short of the margin. | 99.34 | 94% | 99.26 | 99% |
| R5 | Policy-search fine-tune of head knobs on 60 training trips | 0.671 | ○ rejected | Score fell. | 99.19 | 96% | 99.13 | 99% |
| R6 | DAgger round 2 (+600 trips driven by R3) | 0.674 | ○ rejected | Score fell. | 99.28 | 94% | 99.22 | 99% |
| R7 | R6 + 30 s preview | 0.677 | ○ rejected | Score fell. | 99.23 | 92% | 99.21 | 99% |

<sub>Columns after *Why*: Frozen set (% of DP, charge holding) and dev set (% of DP, charge holding).</sub>

R0 is RoadSim's adaptive ECMS, the incumbent to beat. Grading re-solves DP to each strategy's own final charge (matched SOC).

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle): EPA ALPHA engine and motor maps, Thevenin battery, eCVT / P2 / series powertrains, HVAC, an energy ledger checked on every run; DP over engine on/off × 121 SOC points as the teacher.

**Simulator**

- 400 generated hybrids in 18 class × architecture combinations, from cars to series buses and Class-8 P2 trucks
- Routes from 835 real VED trips (real speed, lidar grade, airport weather), EPA/HD cycles and synthetic drives
- DP teacher labels (engine on/off and the costate s*) plus a DAgger round on the learned policy's own trips

**Checked against independent references**

- RoadSim power-split HEV cycles −2.3% vs EPA; ledger closes to < 0.5% on every run; numpy == torch to 5e-6
- Grading re-solves DP to each strategy's own final SOC (matched SOC), which fixes the old factor's over-credit (ECMS 105.2% → 99.6%)

## Next rungs

- Rebuild after RoadSim's twin fixes (Toyota engine-on above ~72 km/h, electric cabin heat in plug-ins, engine-start penalty).
- A plug-in teacher using distance-to-go; RL fine-tuning on matched-SOC fuel.
- A drivability score (starts/km, on/off chatter) in the objective.

## Files

| Path | What |
|---|---|
| `hybridpilot.py`, `pipeline.py` | Inference — numpy/scipy only |
| `weights/hybridpilot_v1.npz` | Weights |
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
| [ZTBus — Widmer et al., ETH Zurich](https://www.research-collection.ethz.ch/handle/20.500.11850/626734) | dev, exam (bus routes) | CC BY 4.0 |
| [US EPA drive cycles; 40 CFR 1037 heavy-duty cycles; USGS 3DEP lidar; IEM ASOS weather](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules) | physics, exam (cycles) | Public domain / free use with credit |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
