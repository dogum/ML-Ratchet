<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# EnergyNet <sub>v2.0</sub>

> **How much fuel or electricity will this trip take, before you drive it?**

*How much energy will this vehicle use on this route, in this weather — and where does it go?*

EnergyNet predicts a trip's fuel or battery energy before the trip, with a 68% and 95% range and a split into air drag, rolling, braking, accessories and powertrain losses. It covers 7 vehicle classes from cars to Class-8 trucks and 7 powertrains. Version 2 learned only from RoadSim physics v2 driven over real Ann Arbor routes and weather (86,400 simulated trips), reads the vehicle's model year, and was examined on 25,793 real trips, EPA tests and 1,115 full Zurich bus missions.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/energynet/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/energynet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | A route at 1 Hz (speed, grade, temperature, wind, heading, air density, surface) and whatever is known about the vehicle (architecture, class, mass, road load, engine, battery, window-sticker label, start charge, model year); unknown fields are masked. |
| **Returns** | Trip energy (MJ) with 68% and 95% ranges — fuel chemical energy, battery energy, or both for plug-ins<br>Split into aero, rolling, friction braking, accessories/HVAC and powertrain losses |
| **Learned from** | The shared RoadSim simulator (shared/vehicle), physics v2 (pinned: roadsim-physics-v2): a forward-facing 1 Hz whole-vehicle model with real road loads, EPA ALPHA engine and motor maps, transmissions, a battery with thermal model, two-node HVAC, a tyre model and seven powertrains, calibrated on dev data only, driven over real Ann Arbor routes with lidar grades and airport weather. |
| **Noise** | Real routes and station weather; the simulated car drives slightly different conditions (temperature ± 1.5 °C, wind × 0.4–0.9) than the input reports |
| **Network** | Ensemble of 5 MLPs (151 → 256 → 256 → 256 → 128 → 7), predicting a correction to a physics energy estimate, its log σ and 5 energy components; Gaussian NLL; trained on a GPU |
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
| Petrol cars, per trip (14,469 VED trips)<br><sub>Bias -0.7%. v1.0 re-scored: 9.1%. Car totals 7.5% vs label 8.6%.</sub> | **8.7%** | MOVES-like (classical): 10.3%<br>Window label × distance (classical): 13%<br>Class median (dumb baseline): 22.8% | ✅ wins |
| Hybrids, per trip (8,241 trips, 86 cars)<br><sub>Bias +3.9% (v1.0: +8.4%, 15.5% re-scored). Car totals 7.5% vs label 8.9%. The 247 trips of P2 hybrids got worse: 17.2% (v1.0: 10.4%).</sub> | **13.5%** | MOVES-like (classical): 15.3%<br>Window label × distance (classical): 16.2%<br>Class median (dumb baseline): 17.2% | ✅ wins |
| Plug-in hybrids, per trip (2,916 trips)<br><sub>EnergyNet is given the logged start charge; the baselines are not. v1.0 re-scored: 19.6%.</sub> | **17.1%** | MOVES-like (classical): 41.3%<br>Class median (dumb baseline): 42.7% | ✅ wins |
| Electric (2 Nissan Leafs), per trip (167 trips)<br><sub>Still biased -19.5% (v1.0: -21.6%, 24.5%). The dev Leaf is +2%.</sub> | **22.6%** | MOVES-like (classical): 19%<br>Class median (dumb baseline): 27.1% | ❌ loses |
| EPA petrol highway tests (20,426)<br><sub>Bias 0.0% (v1.0: +15.3%, 15.6%): the model-year input fixed the old-fleet prior.</sub> | **6.7%** | Log-linear regression (fit on dev) (classical): 5%<br>Dev median (dumb baseline): 16.8% | ❌ loses |
| EPA petrol US06 tests (2,832)<br><sub>v1.0 re-scored: 11.5%.</sub> | **5.1%** | Log-linear regression (fit on dev) (classical): 6.9%<br>Dev median (dumb baseline): 17.3% | ✅ wins |
| EPA petrol FTP tests (20,065)<br><sub>Bias -6.5%. v1.0 re-scored: 9.6%.</sub> | **8.9%** | Log-linear regression (fit on dev) (classical): 7.1%<br>Dev median (dumb baseline): 18.7% | ❌ loses |
| EPA petrol 20 °F cold FTP tests (1,734)<br><sub>New in v2. v1.0 re-scored: 7.5%.</sub> | **9.5%** | Log-linear regression (fit on dev) (classical): 9.7%<br>Dev median (dumb baseline): 24.5% | ✅ wins |
| EPA petrol SC03 (A/C) tests (1,819)<br><sub>Biased -18%. New in v2. v1.0 re-scored: 15.6%.</sub> | **18.9%** | Log-linear regression (fit on dev) (classical): 9.3%<br>Dev median (dumb baseline): 21.7% | ❌ loses |
| EPA hybrid FTP tests (1,098)<br><sub>Bias -9.9%. v1.0 re-scored: 8.7%.</sub> | **10.2%** | Log-linear regression (fit on dev) (classical): 16.3%<br>Dev median (dumb baseline): 49% | ✅ wins |
| EPA hybrid highway tests (1,156)<br><sub>v1.0 re-scored: 5.9%.</sub> | **6%** | Log-linear regression (fit on dev) (classical): 21.5%<br>Dev median (dumb baseline): 20.1% | ✅ wins |
| EPA BEV city tests (746)<br><sub>v1.0 re-scored: 7.9%.</sub> | **8.8%** | Log-linear regression (fit on dev) (classical): 9.8%<br>Dev median (dumb baseline): 19% | ✅ wins |
| ZTBus trolleybus mission totals (1,115 missions)<br><sub>Sum of every leg and layover of the mission's full trace; bias -8.9%. v1.0: 14.6% on the same traces, 16.0% with its route proxy.</sub> | **9.5%** | Dev kWh/km (294 dev missions) × distance (classical): 10.7% | ✅ wins |
| ZTBus trolleybus legs (5,847)<br><sub>v1.0 re-scored: 12.5%.</sub> | **8.6%** | Dev kWh/km × distance (classical): 14% | ✅ wins |
| Mersin diesel bus trips (652)<br><sub>Biased -16.7%; worse than v1.0 (14.8%).</sub> | **17%** | Dev median on the same route (dumb baseline): 8.6% | ❌ loses |
| ICCT tractor-trailers (2 trucks × 3 cycles)<br><sub>v1.0: 11.4%.</sub> | **10%** | Dev truck on the same cycle (classical): 7% | ❌ loses |

Coverage of the 68% error bars on the exam: VED all trips (pooled) **61%**, VED petrol **57%**, VED hybrid **66%**, VED electric **43%**, VED plug-in **65%**, EPA petrol FTP **65%**, EPA petrol highway **76%**, ZTBus missions **88%**.

### Where it wins

- Petrol cars: 8.7% per trip against 10.3% for the best classical method; car totals 7.5% against 8.6%.
- Hybrids now win: 13.5% per trip against 15.3% for MOVES-like (v1.0 lost at 15.5%); bias +3.9% (v1.0 +8.4%).
- Plug-ins: 17.1% against 41% (with the start charge as an input).
- Zurich trolleybus missions on full traces: 9.5% against 10.7% for the dev kWh/km baseline (v1.0: 16.0%).
- EPA petrol highway bias gone (0.0%, v1.0 +15%); EPA US06 and 20 °F cold FTP beat the regression.

### Where it loses

- The two exam Leafs are still 19.5% low and lose to MOVES-like (22.6% vs 19.0%).
- EPA petrol FTP, highway and SC03 lose to a regression on the test car's own spec; SC03 (A/C) is 18% low.
- Diesel and CNG buses (17-18%) and the trucks (10%) lose to the dev median on the same route or cycle; the buses got worse than v1.0.
- P2 hybrids on VED got worse (17.2% vs v1.0's 10.4%), and so did plug-in car totals (20.1% vs 10.1%).

## Limits

- Error bars: 61% of VED exam trips inside the 68% band overall, but petrol 57% and the Leafs 43%; ZTBus missions too wide (88%).
- VED fuel is a MAF-based estimate, Ann Arbor 2017–18 only, mostly warm trip fragments.
- Hybrid and EV calibration rests on 5 dev hybrids and 1 dev Leaf; RoadSim v2's known hybrid gaps (old Toyota engines, TMED P2, plug-in series) are learned around only where dev data shows them.
- Heavy duty is thin and partly synthetic (Mersin speed profiles from stops and duration).

## The ratchet

### v2 (RoadSim physics v2, Phase D, 14 Nov 2026)

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
| v2-r13 | three times the simulated data (three banks on the same fleet, 86,400 trips), networks and training as r7 | 0.670 | **◆ released** | skill +0.019 on val2[e] (0.670 vs 0.651); dev D2 0.1195 vs 0.1260 (-0.0065) | 7.7% | 0.1195 |
| v2-r12 | error bars: sigma x a per-architecture factor that brings the simulated val2 set to 68% (simulation only), then the dev factor per group | — | ● kept | leave-one-car-out on the 16 dev VED cars, mean \|held-out coverage - 68%\|: 0.064 vs 0.072 with v1 factors alone, 0.074 with a car-level term; no effect on the energy |  |  |

<sub>Columns after *Why*: Frozen-set median error and the dev score D2 (lower is better).</sub>

Versions of val2: a RoadSim v2 raw, b + OBD label model, c + v1's on-road wrapper, d c + v1's plug-in patch, e b + plug-in patch. Each rung is scored on the version it was judged on.

### v1 (RoadSim physics v1, 12 Nov 2026)

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
- Simulated window-sticker labels (FTP-75 and HWFET on a dynamometer spec, EPA 5-cycle formulas)
- RoadSim's own measurement layer for OBD MAF fuel labels (roadsim.obd) and v1's plug-in hardware patch, both kept by the v2 ratchet

**Checked against independent references**

- RoadSim v2 vs EPA on held-out vehicles: conventional FTP +0.4% / HWFET -1.9%, BEV UDDS -0.4% / HWY -1.8% (RoadSim LIMITS.md)
- Energy ledger closes in all 600 stress runs (max 5.6e-4); RoadSim bus twin on the full ZTBus exam legs -2.3%

## Help wanted

- Argonne D3 10 Hz dynamometer tests (anl.gov answers HTTP 403 to scripts): more dev hybrids, plug-ins and BEVs; needs a browser download.

## Next rungs

- EPA hybrids told their hybrid system and the bus told its curb mass (rungs r8, r9: real gains on their dev terms, short of the D2 margin): a dev score weighted per data set would judge such rungs fairly.
- Leaf: older BEVs use 10-17% more than their twins (RoadSim LIMITS); a BEV era or battery-age input needs dev BEVs beyond one Leaf.
- Real Mersin speed profiles; A/C on conventional cars and hybrids (SC03 -18%).

## Files

| Path | What |
|---|---|
| `energynet.py`, `features.py`, `model_np.py` | Inference — numpy/scipy only |
| `weights/r13_s0.npz`, `weights/r13_s1.npz`, `weights/r13_s2.npz`, `weights/r13_s3.npz`, `weights/r13_s4.npz`, `sigma_factors.json` | Weights |
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
| [ZTBus — Widmer, Ritter, Onder, Scientific Data 2023 (ETH Zurich), all 1,409 missions](https://www.research-collection.ethz.ch/handle/20.500.11850/626734) | dev (294 missions), exam (1,115 missions) — trolleybuses | CC BY 4.0 |
| [Mersin bus trips — Abut, Zenodo](https://zenodo.org) | dev, exam (diesel and CNG buses) | CC BY 4.0 |
| [ICCT 2018 tractor-trailer tests (Rodriguez, Delgado, Muncrief)](https://theicct.org) | dev, exam (trucks) | Cite the report |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
