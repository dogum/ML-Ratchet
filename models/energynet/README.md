<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# EnergyNet <sub>v2.1</sub>

> **How much fuel or electricity will this trip take, before you drive it?**

*How much energy will this vehicle use on this route, in this weather — and where does it go?*

EnergyNet predicts a trip's fuel or battery energy before the trip, with a 68% and 95% range and a split into air drag, rolling, braking, accessories and powertrain losses. It covers 7 vehicle classes from cars to Class-8 trucks and 7 powertrains. Version 2 learned only from RoadSim physics v2 driven over real Ann Arbor routes and weather (86,400 simulated trips), reads the vehicle's model year, and was examined on 25,793 real trips, EPA tests and 1,115 full Zurich bus missions. Version 2.1 runs the simulated A/C and cold tests under the EPA test conditions, drops idle-stop from the simulated petrol cars and reads two public specs (a hybrid's system, a bus's curb mass).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/energynet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/energynet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | A route at 1 Hz (speed, grade, temperature, wind, heading, air density, surface) and whatever is known about the vehicle (architecture, class, mass, road load, engine, battery, window-sticker label, start charge, model year); unknown fields are masked. |
| **Returns** | Trip energy (MJ) with 68% and 95% ranges — fuel chemical energy, battery energy, or both for plug-ins<br>Split into aero, rolling, friction braking, accessories/HVAC and powertrain losses |
| **Learned from** | The shared RoadSim simulator (shared/vehicle), physics v2 (pinned: roadsim-physics-v2): a forward-facing 1 Hz whole-vehicle model with real road loads, EPA ALPHA engine and motor maps, transmissions, a battery with thermal model, two-node HVAC, a tyre model and seven powertrains, calibrated on dev data only, driven over real Ann Arbor routes with lidar grades and airport weather. |
| **Noise** | Real routes and station weather; the simulated car drives slightly different conditions (temperature ± 1.5 °C, wind × 0.4–0.9) than the input reports |
| **Network** | Ensemble of 5 MLPs (151 → 256 → 256 → 256 → 128 → 7), predicting a correction to a physics energy estimate, its log σ and 5 energy components; Gaussian NLL; trained on a GPU |
| **Physics also enters** | inputs: Road-load work integrals (aero, rolling, grade, inertia) from the same physics as RoadSim<br>output: Predicts log(E / E_phys), a correction to a physics energy estimate |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 16 VED cars plus dev EPA vehicles, 294 ZTBus missions (by month), Mersin routes and one ICCT truck — RoadData's frozen splits (sha256 d917df78…; ZTBus by-month split 18e0fead…) |
| **Exam** | 25,793 VED trips from 350 cars, EPA exam families, 1,115 ZTBus missions, Mersin and ICCT exam sets |

## Run it

```bash
cd models/energynet
pip install numpy
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo energynet`.

```python
import json, numpy as np
from energynet import EnergyNet
net = EnergyNet()                                  # the 5-network v2 ensemble in weights/
ex = json.load(open("examples/examples.json"))[0]
z = np.load(f"examples/{ex['trip_id']}.npz")
trip = {k: z[k] for k in z.files if k != "valid"}
r = net.predict(trip, ex["vehicle"])
print(r["energy_MJ"])
```

## The exam

Real VED trips, EPA tests and ZTBus missions split once and frozen; median absolute % error. Run once, after the release choice was committed. The v1.0 release was re-scored on the same data (notes). Every number here is on real data that never chose anything.

| Metric | EnergyNet | Baselines | |
|---|---|---|---|
| Petrol cars, per trip (14,469 VED trips)<br><sub>Bias +0.3%. v2.0: 8.7%; v1.0 re-scored: 9.1%. Car totals 7.4% vs label 8.7%.</sub> | **8.6%** | MOVES-like (classical): 10.3%<br>Window label × distance (classical): 13%<br>Class median (dumb baseline): 22.8% | ✅ wins |
| Hybrids, per trip (8,241 trips, 86 cars)<br><sub>Bias +3.8%. v2.0: 13.5%; v1.0 re-scored: 15.5% (+8.4%). Car totals 7.2% vs label 8.9%. The 247 trips of P2 hybrids: 18.3% (v2.0 17.2%, v1.0 10.4%).</sub> | **13.4%** | MOVES-like (classical): 15.3%<br>Window label × distance (classical): 16.2%<br>Class median (dumb baseline): 17.2% | ✅ wins |
| Plug-in hybrids, per trip (2,916 trips)<br><sub>EnergyNet is given the logged start charge; the baselines are not. v2.0: 17.1%; v1.0 re-scored: 19.6%.</sub> | **16.8%** | MOVES-like (classical): 41.3%<br>Class median (dumb baseline): 42.7% | ✅ wins |
| Electric (2 Nissan Leafs), per trip (167 trips)<br><sub>Biased -20.3%; worse than v2.0 (22.6%, -19.5%); v1.0 24.5%. The one dev Leaf is +4.6%, so dev cannot show this bias.</sub> | **23.2%** | MOVES-like (classical): 19%<br>Class median (dumb baseline): 27.1% | ❌ loses |
| EPA petrol highway tests (20,426)<br><sub>Bias -0.8%. v2.0: 6.7% (0.0%); v1.0: 15.6% (+15.3%).</sub> | **6.8%** | Log-linear regression (fit on dev) (classical): 5%<br>Dev median (dumb baseline): 16.8% | ❌ loses |
| EPA petrol US06 tests (2,832)<br><sub>v2.0: 5.1%; v1.0 re-scored: 11.5%.</sub> | **5.2%** | Log-linear regression (fit on dev) (classical): 6.9%<br>Dev median (dumb baseline): 17.3% | ✅ wins |
| EPA petrol FTP tests (20,065)<br><sub>Bias -5.0%. v2.0: 8.9% (-6.5%); v1.0 re-scored: 9.6%.</sub> | **8.6%** | Log-linear regression (fit on dev) (classical): 7.1%<br>Dev median (dumb baseline): 18.7% | ❌ loses |
| EPA petrol 20 °F cold FTP tests (1,734)<br><sub>Bias -6.1%. v2.0: 9.5%; v1.0 re-scored: 7.5%.</sub> | **8.5%** | Log-linear regression (fit on dev) (classical): 9.7%<br>Dev median (dumb baseline): 24.5% | ✅ wins |
| EPA petrol SC03 (A/C) tests (1,819)<br><sub>Bias -6.2%. v2.0: 18.9% (-18.4%), a loss; v1.0 re-scored: 15.6%. v2.1 simulates the test's sun, A/C setting and hot-soaked cabin.</sub> | **8.6%** | Log-linear regression (fit on dev) (classical): 9.3%<br>Dev median (dumb baseline): 21.7% | ✅ wins |
| EPA hybrid FTP tests (1,098)<br><sub>Bias -8.3%. The test car is told its hybrid system (P2 / power-split / series) from the model line. v2.0: 10.2%; v1.0 re-scored: 8.7%.</sub> | **8.7%** | Log-linear regression (fit on dev) (classical): 16.3%<br>Dev median (dumb baseline): 49% | ✅ wins |
| EPA hybrid highway tests (1,156)<br><sub>v2.0: 6.0%; v1.0 re-scored: 5.9%.</sub> | **5.6%** | Log-linear regression (fit on dev) (classical): 21.5%<br>Dev median (dumb baseline): 20.1% | ✅ wins |
| EPA BEV city tests (746)<br><sub>Bias -9.5%; worse than v2.0 (8.8%) and v1.0 (7.9%), still under the regression.</sub> | **9.5%** | Log-linear regression (fit on dev) (classical): 9.8%<br>Dev median (dumb baseline): 19% | ✅ wins |
| ZTBus trolleybus mission totals (1,115 missions)<br><sub>Sum of every leg and layover of the mission's full trace; bias -1.5%. The bus is told its published 19 t curb mass. v2.0: 9.5% (-8.9%); v1.0: 14.6% on the same traces, 16.0% with its route proxy.</sub> | **5.6%** | Dev kWh/km (294 dev missions) × distance (classical): 10.7% | ✅ wins |
| ZTBus trolleybus legs (5,847)<br><sub>Bias +0.3%. v2.0: 8.6%; v1.0 re-scored: 12.5%.</sub> | **7%** | Dev kWh/km × distance (classical): 14% | ✅ wins |
| Mersin diesel bus trips (652)<br><sub>Biased -16.7%. v2.0: 17.0%; v1.0: 14.8%. Dev (February, no A/C trip) is -3.1%, so dev does not show this bias. CNG buses: 19.2% (v2.0 18.0%) against 7.1%.</sub> | **17.2%** | Dev median on the same route (dumb baseline): 8.6% | ❌ loses |
| ICCT tractor-trailers (2 trucks × 3 cycles)<br><sub>v2.0: 10.0%; v1.0: 11.4%.</sub> | **10.7%** | Dev truck on the same cycle (classical): 7% | ❌ loses |

Coverage of the 68% error bars on the exam: VED all trips (pooled) **59%**, VED petrol **55%**, VED hybrid **65%**, VED electric **41%**, VED plug-in **64%**, EPA petrol FTP **64%**, EPA petrol highway **75%**, EPA petrol SC03 **58%**, ZTBus missions **91%**.

### Where it wins

- Petrol cars: 8.6% per trip against 10.3% for the best classical method; car totals 7.4% against 8.7%.
- Hybrids: 13.4% per trip against 15.3% for MOVES-like (v1.0 lost at 15.5%).
- Plug-ins: 16.8% against 41% (with the start charge as an input).
- Zurich trolleybus missions on full traces: 5.6% against 10.7% for the dev kWh/km baseline (v2.0: 9.5%, v1.0: 16.0%).
- EPA petrol SC03 (A/C) now wins: 8.6% against 9.3% for a regression on the test car's own spec (v2.0 lost at 18.9%).
- EPA petrol US06 (5.2% vs 6.9%) and 20 °F cold FTP (8.5% vs 9.7%) beat the same regression.

### Where it loses

- The two exam Leafs are 20.3% low and lose to MOVES-like (23.2% vs 19.0%; v2.0 22.6%). The one dev Leaf is +4.6%, so dev cannot show it.
- EPA petrol FTP (8.6% vs 7.1%) and highway (6.8% vs 5.0%) lose to a regression on the test car's own spec.
- Diesel and CNG buses (17.2%, 19.2%) and the trucks (10.7%) lose to the dev median on the same route or cycle, and all three are slightly worse than v2.0.
- P2 hybrids on VED (18.3%, v1.0 10.4%) and plug-in car totals (20.6%, v1.0 10.1%) are worse than v1.0.

## Limits

- Error bars: 59% of VED exam trips inside the 68% band overall, petrol 55% and the Leafs 41%; ZTBus missions too wide (91%).
- VED fuel is a MAF-based estimate, Ann Arbor 2017–18 only, mostly warm trip fragments.
- Hybrid and EV calibration rests on 5 dev hybrids and 1 dev Leaf; RoadSim v2's known hybrid gaps (old Toyota engines, TMED P2, plug-in series) are learned around only where dev data shows them.
- Heavy duty is thin and partly synthetic (Mersin speed profiles from stops and duration).
- The dev score D2 moves by about 0.005 between two sets of five networks trained on the same data (half the 0.01 margin), mostly through the small heavy-duty terms.

## The ratchet

### v2 — RoadSim physics v2 (Phase D: v2.0 14 Nov 2026, v2.1 15 Nov 2026)

**Objective:** Skill = 1 - median |log error| / median |log error| of the class-median guess, on the frozen simulated set val2 (400 unseen generated vehicles x 10 trips on RoadSim physics v2)  
**Rule:** Keep if the val2 skill beats the incumbent by more than 0.01 AND the dev score D2 (mean of 18 dev median errors: VED 4 fleets, EPA 9 tests, full-ZTBus dev legs and missions, Mersin, ICCT) gets no more than 0.01 worse. A rung that changes the simulated physics is judged on its own physics version of val2, the incumbent re-scored on it. Gate-only rungs (inputs only real data carries) must improve D2 by more than 0.01.

| Rung | Change | Frozen-set score | Decision | Why | Val2 MdAPE | Dev D2 |
|---|---|---|---|---|---|---|
| v2-r1 | MLP predicts log trip energy directly (physics integrals only as inputs); RoadSim physics v2, raw simulated energy | 0.614 | ● kept | first rung | 9.0% | 0.1376 |
| v2-r2 | predict a correction to the generic road-load physics estimate | 0.629 | ● kept | skill +0.016 on val2[a] (0.629 vs 0.614); dev D2 0.1327 vs 0.1376 (-0.0049) | 8.6% | 0.1327 |
| v2-r3 | RoadSim's OBD label model (roadsim.obd.label_fuel: MAF fuel-cut air, missed enrichment) for MAF-estimated fuel; flag input | 0.630 | ○ rejected | skill +0.003 on val2[b] (0.630 vs 0.628); dev D2 0.1234 vs 0.1327 (-0.0093); physics rung, judged on D2 (needs < -0.01); first rule: rejected | 8.6% | 0.1234 |
| v2-r4 | v1's dev-fitted on-road loads wrapper on top of v2 (v2 already carries its own on-road load) | 0.631 | ○ rejected | skill +0.097 on val2[c] (0.631 vs 0.534); dev D2 0.1378 vs 0.1327 (+0.0050); physics rung, judged on D2 (needs < -0.01); first rule: kept | 8.3% | 0.1378 |
| v2-r5 | v1's plug-in hardware patch on the generated plug-ins (pack power >= 1.2 x motor, resistance / 2.5, MG2 torque x 1.8); dev twin read: the network put the dev plug-ins 33% above their own RoadSim twins | 0.626 | ○ rejected | skill +0.005 on val2[f] (0.626 vs 0.622); dev D2 0.1228 vs 0.1327 (-0.0100); physics rung, judged on D2 (needs < -0.01); first rule: rejected | 8.7% | 0.1228 |
| v2-r5b | both near-misses together: RoadSim's OBD label model (r3) and the plug-in patch (r5), each of which improved D2 by just under 0.01 alone | 0.632 | ● kept | skill +0.013 on val2[e] (0.632 vs 0.619); dev D2 0.1225 vs 0.1327 (-0.0102); physics rung, judged on D2 (needs < -0.01); first rule: kept | 8.6% | 0.1225 |
| v2-r6 | ensemble of 5 networks (seeds); sigma includes their spread | 0.636 | ○ rejected | skill +0.004 on val2[e] (0.636 vs 0.632); dev D2 0.1259 vs 0.1225 (+0.0033) | 8.4% | 0.1259 |
| v2-r7 | model-year input (VED: fueleconomy.gov models matched to the car; EPA: test year); the fleet's engine efficiency follows the fueleconomy.gov dev era trend, so the network can tell a 2006 car from a 2022 one | 0.651 | ● kept | skill +0.019 on val2[e] (0.651 vs 0.632); dev D2 0.1260 vs 0.1225 (+0.0035) | 8.1% | 0.126 |
| v2-r8 | EPA test-car hybrids told their hybrid system (P2 / power-split / series, from the model line; RoadSim's public spec-sheet table) instead of 'power-split' for all; the same networks | 0.651 | ○ rejected | skill +0.000 on val2[e] (0.651 vs 0.651); dev D2 0.1195 vs 0.1260 (-0.0065); gate-only rung | 8.1% | 0.1195 |
| v2-r9 | ZTBus told the trolleybus's published curb mass (19 t, 'reported' mass) instead of the class prior (15 t); the same networks | 0.651 | ○ rejected | skill +0.000 on val2[e] (0.651 vs 0.651); dev D2 0.1239 vs 0.1260 (-0.0021); gate-only rung | 8.1% | 0.1239 |
| v2-r9b | both public-spec inputs together (r8 EPA hybrid system + r9 bus curb mass), each short of the margin alone | 0.651 | ○ rejected | skill +0.000 on val2[e] (0.651 vs 0.651); dev D2 0.1174 vs 0.1260 (-0.0086); gate-only rung | 8.1% | 0.1174 |
| v2-r10 | simulated window-sticker labels by RoadSim's EPA procedure emulation (FTP-75 with hot soak, hybrids preconditioned and corrected for net battery energy) instead of fleet.py's shortcut; physics rung (labels change) | 0.640 | ○ rejected | val2[e+L] skill 0.640 vs 0.649 (d -0.009); dev D2 0.1277 vs 0.1260 (d +0.0016); physics rung: dev D2 must improve by > 0.01; own-version skill 0.640 vs incumbent's 0.651 on its own version (may not drop by > 0.01); first rule: rejected; physics rung, judged on D2 (needs < -0.01); first rule: rejected | 8.3% | 0.1277 |
| v2-r11 | GPU: twice the simulated data (a second bank of 28,800 trips on the same fleet, new seed), wider networks (512 units, 0.73M weights each) and 100 epochs | 0.632 | ○ rejected | skill -0.019 on val2[e] (0.632 vs 0.651); dev D2 0.1219 vs 0.1260 (-0.0041) | 8.6% | 0.1219 |
| v2-r11b | twice the simulated data only (both banks, 57,600 trips), networks and training as r7 | 0.660 | ○ rejected | skill +0.009 on val2[e] (0.660 vs 0.651); dev D2 0.1195 vs 0.1260 (-0.0065) | 7.9% | 0.1195 |
| v2-r13 | three times the simulated data (three banks on the same fleet, 86,400 trips), networks and training as r7 | 0.670 | ● kept | skill +0.019 on val2[e] (0.670 vs 0.651); dev D2 0.1195 vs 0.1260 (-0.0065). Released as v2.0 (14 Nov 2026); superseded by v2.1 | 7.7% | 0.1195 |
| v2-r14 | SC03 and 20 F cold-FTP dynamometer draws run under the EPA test conditions (roadsim.epa: SC03 850 W/m2 sun, A/C maximum on recirculated air at 72 F, cabin soaked to 45 C after a preconditioning drive; cold FTP heat at 72 F on outside air); dev twin read: the network sat 17.5% below the dev SC03 tests while RoadSim's own SC03 twin of 47 dev vehicles is -0.3%; the old bank draw missed 12% of a conventional car's SC03 fuel (median of 13 training vehicles, check_dynoproc.py); physics rung | 0.669 | ○ rejected | skill +0.004 on val2[g] (0.669 vs 0.665); dev D2 0.1137 vs 0.1195 (-0.0058); physics rung, judged on D2 (needs < -0.01); first rule: rejected | 7.6% | 0.1137 |
| v2-r15 | generated conventional light-duty cars without engine idle-stop (RoadSim's generator gave 30% of them idle-stop at any model year, invisible to the network; RoadSim's dev-calibrated twins have none and match the dev FTP within 1.4%); their rows re-simulated on the same trips and start states, labels recomputed; dev twin read: the network put RoadSim's own twins of 60 dev EPA cars 5.5% low on the UDDS and 0.8% low on the HWFET, and the dev FTP 3.9% low while the highway is +0.4%; physics rung, on top of r14's banks | 0.680 | ○ rejected | skill +0.007 on val2[h] (0.680 vs 0.673); dev D2 0.1153 vs 0.1195 (-0.0042); physics rung, judged on D2 (needs < -0.01); first rule: rejected | 7.5% | 0.1153 |
| v2-r16 | the incumbent's networks told two public-spec inputs, each rejected alone on r7: the EPA test-car hybrids' system (r8) and the ZTBus curb mass (r9); gate-only rung | 0.670 | ○ rejected | skill +0.000 on val2[e] (0.670 vs 0.670); dev D2 0.1140 vs 0.1195 (-0.0055); gate-only rung | 7.7% | 0.114 |
| v2-r17 | r15's networks (SC03 / cold-FTP test conditions and no idle-stop) with r16's two public-spec inputs; physics rung against the incumbent (planned before r15 was scored) | 0.680 | **◆ released** | skill +0.007 on val2[h] (0.680 vs 0.673); dev D2 0.1077 vs 0.1195 (-0.0118); physics rung, judged on D2 (needs < -0.01); first rule: rejected | 7.5% | 0.1077 |
| v2-r12 | error bars: sigma x a per-architecture factor that brings the simulated val2 set to 68% (simulation only), then the dev factor per group | — | ● kept | leave-one-car-out on the 16 dev VED cars, mean \|held-out coverage - 68%\|: 0.064 vs 0.072 with v1 factors alone, 0.074 with a car-level term; no effect on the energy |  |  |

<sub>Columns after *Why*: Frozen-set median error and the dev score D2 (lower is better).</sub>

Versions of val2: a RoadSim v2 raw, b + OBD label model, c + v1's on-road wrapper, d c + v1's plug-in patch, e b + plug-in patch, f a + plug-in patch (rung v2-r5), eL e with labels from RoadSim's EPA procedure emulation (rung v2-r10, 'e+L'), g e with the SC03 / 20 F cold-FTP dyno draws under the EPA test conditions (rung v2-r14, 15 Nov 2026), h g with the generated conventional cars' idle-stop off (rung v2-r15). Each rung is scored on the version it was judged on.

### v1.0 — RoadSim physics v1 (12 Nov 2026)

**Objective:** Skill = 1 − median |log error| / median |log error| of the class-median guess, on the frozen simulated validation set (3,991 examples from 400 unseen vehicles)  
**Rule:** Keep if the skill beats the incumbent by more than 0.01 AND the dev score D (mean of 15 dev median errors across VED fleets, EPA tests, ZTBus, Mersin and ICCT) gets no more than 0.01 worse.

| Rung | Change | Frozen-set score | Decision | Why | Val MdAPE | Dev D |
|---|---|---|---|---|---|---|
| r1 | MLP predicts log energy directly | 0.544 | ● kept | First rung. | 11.2% | 0.152 |
| r2 | Predict a correction to the physics energy estimate | 0.556 | ● kept | +0.012; dev D 0.160 (within +0.01). | 11.0% | 0.16 |
| r3 | MAF label model + flag input (physics v2) | 0.580 | ● kept | +0.024; dev D 0.152. | 10.3% | 0.152 |
| r4 | On-road loads vs temperature, fitted on dev (physics v3) | 0.641 | ● kept | +0.061; dev D 0.142. | 9.0% | 0.142 |
| r5 | Plug-in hardware patch (physics v4) | 0.648 | ○ rejected | +0.007, short of the 0.01 margin. | 8.9% | 0.129 |
| r6 | Ensemble of 5 networks on v4 physics | 0.656 | ● kept | +0.015 over rung 4; dev D 0.129. Released as v1.0; superseded by v2. | 8.7% | 0.129 |

<sub>Columns after *Why*: Frozen-set median error and the dev score D (lower is better).</sub>

The frozen set was re-scored at each physics version (v2 MAF label model, v3 on-road loads, v4 plug-in patch).

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle), physics v2 (pinned: roadsim-physics-v2): a forward-facing 1 Hz whole-vehicle model with real road loads, EPA ALPHA engine and motor maps, transmissions, a battery with thermal model, two-node HVAC, a tyre model and seven powertrains, calibrated on dev data only, driven over real Ann Arbor routes with lidar grades and airport weather.

**Simulator**

- 2,400 training vehicles from RoadSim's generator (v2 donors), each light-duty one with a model year 2005-2026 and an engine efficiency from an era prior fitted on fueleconomy.gov dev families (a 2005 engine burns 21% more than the same engine in 2021)
- 86,400 training trips (three banks of 12 per vehicle): 1,092 unlabelled real VED routes, dynamometer cycles (now with cold starts and the 20 F cold FTP) and synthetic urban/highway/bus/truck drives
- v2.1: SC03 and 20 F cold-FTP draws under the EPA test conditions (roadsim.epa: 850 W/m2 sun, A/C maximum, cabin soaked to 45 C; heat at 72 F on outside air); generated petrol cars without idle-stop (RoadSim's dev-calibrated twins have none)
- Simulated window-sticker labels (FTP-75 and HWFET on a dynamometer spec, EPA 5-cycle formulas)
- RoadSim's own measurement layer for OBD MAF fuel labels (roadsim.obd) and v1's plug-in hardware patch, both kept by the v2 ratchet

**Checked against independent references**

- RoadSim v2 vs EPA on held-out vehicles: conventional FTP +0.4% / HWFET -1.9%, BEV UDDS -0.4% / HWY -1.8% (RoadSim LIMITS.md)
- Energy ledger closes in all 600 stress runs (max 5.6e-4); RoadSim bus twin on the full ZTBus exam legs -2.3%

## Help wanted

- NREL Fleet DNA truck and bus traces with fuel (nrel.gov times out for scripts): a real exam for trucks, which today rest on 2 ICCT trucks x 3 cycles.

## Next rungs

- Leaf: older BEVs use 10-17% more than their twins (RoadSim LIMITS); a BEV era or battery-age input needs dev BEVs beyond one Leaf.
- Buses: real Mersin speed profiles, A/C trips on dev, and a spark-ignition (CNG) heavy-duty engine in the generator.
- EPA petrol city: the network still reads RoadSim's own dev twins 4.5% low on the FTP (0.8% on the highway); the remaining generator difference is not found.
- Argonne D3 as a dev and exam set: RoadData now holds it (65 vehicles, 1,661 usable 10 Hz tests, split by vehicle), with more dev hybrids, plug-ins and BEVs than VED's 5 hybrids and 1 Leaf.
- A dev score weighted per data set, or more networks, so a one-group fix is not judged mostly by seed noise in other groups.

## Files

| Path | What |
|---|---|
| `energynet.py`, `features.py`, `model_np.py` | Inference — numpy/scipy only |
| `weights/r15_s0.npz`, `weights/r15_s1.npz`, `weights/r15_s2.npz`, `weights/r15_s3.npz`, `weights/r15_s4.npz`, `sigma_factors.json` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Vehicle Energy Dataset (VED) — Oh, LeBlanc, Peng, IEEE T-ITS 2020](https://github.com/gsoh/VED) | noise (routes), dev, exam | Apache-2.0 |
| [US EPA test-car lists, drive cycles and ALPHA maps; fueleconomy.gov](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy) | physics (fueleconomy.gov dev families -> model-era prior), dev, exam | Public domain (U.S. Government) |
| [USGS 3DEP lidar elevation](https://www.usgs.gov/3d-elevation-program) | physics (road grade) | Public domain (U.S. Government) |
| [Iowa Environmental Mesonet ASOS weather](https://mesonet.agron.iastate.edu/request/download.phtml) | physics (weather) | Free use with credit |
| [ZTBus — Widmer, Ritter, Onder, Scientific Data 2023 (ETH Zurich), all 1,409 missions](https://doi.org/10.3929/ethz-b-000626723) | dev (294 missions), exam (1,115 missions) — trolleybuses | CC BY 4.0 |
| [Mersin bus trips — Abut, Zenodo](https://zenodo.org) | dev, exam (diesel and CNG buses) | CC BY 4.0 |
| [ICCT 2018 tractor-trailer tests (Rodriguez, Delgado, Muncrief)](https://theicct.org) | dev, exam (trucks) | Cite the report |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
