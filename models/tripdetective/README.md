<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# TripDetective <sub>v1.0</sub>

> **What car was this? Read it from one trip log.**

*From a car's trip log alone, what kind of powertrain is it, how heavy, what engine, how much drag — and where did the energy go?*

TripDetective reads one or more 1 Hz trip logs of a light-duty vehicle — speed plus whatever an OBD logger has — and infers its architecture, total mass, engine size, CdA, rolling resistance, hybrid strategy and each trip's energy ledger, every output with an error bar. It learned only from the RoadSim simulator with an emulated OBD logger, and was graded on 351 real cars it never saw.

[Project page](https://dogum.github.io/ML-Ratchet/models/tripdetective/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One or more 1 Hz trip logs (speed, and when logged fuel rate, rpm, HV battery current/voltage/SOC, A/C power), with route grade and weather. |
| **Returns** | Architecture (7 layouts) with probabilities<br>Total and curb mass, engine displacement, CdA, Crr — each with a 1σ range<br>Hybrid control strategy and engine-off share<br>Each trip's energy ledger in kJ/km (aero, rolling, grade, friction braking, powertrain losses, accessories) |
| **Learned from** | The shared RoadSim simulator (shared/vehicle) drives generated cars over real Ann Arbor routes; an OBD-logger emulator then polls each signal on its own clock, holds and quantises it, and estimates fuel from MAF the way real loggers do. |
| **Noise** | Real routes and weather; the emulated logger reproduces real OBD cadences, holds, quantisation and MAF-based fuel estimates |
| **Network** | 1-D CNN over each trip window plus 16 classical road-load estimates, pooled across a car's trips, with heads for architecture, strategy and 7 regressions (Gaussian NLL); 0.40 M parameters |
| **Dev set** | 16 VED dev cars — RoadData's frozen split (sha256 d917df78…) |
| **Exam** | 351 VED exam cars (at most 40 trips each, fixed seed), 25 EPA road-load cars, and 150 EPA exam vehicles rebuilt in RoadSim |

## Run it

```bash
cd models/tripdetective
pip install numpy scipy
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo tripdetective`.

```python
import tripdetective as TD
td = TD.TripDetective()
car = td.read_car(trips)            # trips: list of dicts of 1 Hz arrays (see demo.py)
one = td.read_trip(trips[0])
```

## The exam

Real VED exam cars; baselines fitted on dev only. Every number here is on real data that never chose anything.

| Metric | TripDetective | Baselines | |
|---|---|---|---|
| Curb mass per car, median \|error\| (335 cars) | **116 kg** | Physics least-squares fit (classical): 205 kg<br>Dev median (dumb baseline): 227 kg | ✅ wins |
| Road-load force vs EPA target at 40 mph (25 cars) | **17%** | Physics least-squares fit (classical): 35% | ✅ wins |
| Engine displacement per car, median \|error\|<br><sub>Only narrowly better than the dev median; biased −0.48 L.</sub> | **0.5 L** | Idle-fuel law (classical): 0.64 L<br>Dev median (dumb baseline): 0.6 L | ✅ wins |
| Powertrain type per car (351 cars) | **98.9%** | Simple rule (classical): 99.4%<br>Majority class (dumb baseline): 68.6% | ❌ loses |
| Mass response to +800 kg (simulated sweep)<br><sub>It mostly recognises the kind of car rather than reading the physics of each acceleration.</sub> | **≈ +200 kg** | — | ℹ️ |

Coverage of the 68% error bars on the exam: Mass, per car **73%**, Mass, per trip **79%**, Engine size, per car **61%**.

### Where it wins

- Mass within 116 kg per car against 205 kg for the physics fit and 227 kg for the dev median.
- Road load about 17% off EPA targets against 19–66% for the physics fit.
- On 150 EPA cars rebuilt in RoadSim: mass vs test weight 7.8%, road load at 60 mph 9.4%.

### Where it loses

- A simple rule beats it on powertrain type (99.4% vs 98.9%).
- Engine size only narrowly beats guessing the dev median.
- The two Leafs read 460 kg heavy (RoadSim's Leaf twin over-heats its cabin).

## Limits

- Mass, efficiency and drag trade off in 1 Hz MAF-based logs: +800 kg moves the estimate only ~200 kg and CdA barely registers. Pooling 32 trips helps little (153 → 126 kg).
- The energy ledger has only simulated truth; no skill on net grade work.
- Light duty only; RoadSim limits apply (P2 city optimistic, cold too mild, A/C too cheap).
- Built before the 13 Nov RoadData fixes (VED mass double-counted 136 kg).

## The ratchet

**Objective:** Mean simulated skill over architecture, mass, displacement, CdA, Crr, engine-off share, strategy and the ledger, on frozen val_v3 (600 cars × 8 trips)  
**Rule:** Keep if the skill beats the incumbent by more than 0.01 AND the mean of 4 dev error ratios is at most 1.05 AND dev VED-type accuracy drops by no more than 0.03.

| Rung | Change | Frozen-set score | Decision | Why | Dev mass error / car | Dev engine error / car | Dev type accuracy / trip |
|---|---|---|---|---|---|---|---|
| r0 | CNN + set pooling, generator as published, 1,200 cars | 0.305 | ● kept | First rung. | 140 kg | 0.31 L | 0.91 |
| r1 | Dev-twin calibrated priors v2, 1,200 cars | 0.362 | ○ rejected | Dev errors +23%. | 128 kg | 0.52 L | 0.9 |
| r2 | v2 priors, 2,000 cars | 0.366 | ○ rejected | Dev errors +15%. | 116 kg | 0.51 L | 0.93 |
| r3 | + road-load physics channels | 0.363 | ○ rejected | Dev errors +23% and type accuracy fell. | 129 kg | 0.53 L | 0.88 |
| r4 | + 16 classical per-trip estimates | 0.368 | ○ rejected | Dev errors +17%. | 136 kg | 0.49 L | 0.9 |
| r5 | Classical estimates on the v1 bank | 0.317 | ● kept | +0.012; dev +2%. | 139 kg | 0.34 L | 0.91 |
| r6 | v3 mass prior (VED-era fleet) + classical estimates, 2,000 cars | 0.402 | **◆ released** | +0.085; dev mass error 110 kg (bias −9). | 110 kg | 0.44 L | 0.92 |

<sub>Columns after *Why*: Dev cars (gate inputs).</sub>

A network twin test found the generator's fleet far heavier than VED-era cars (2,200 vs 1,720 kg), which became the v3 mass prior. Every v2-prior rung made dev engine size worse; the cause was not isolated.

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle) drives generated cars over real Ann Arbor routes; an OBD-logger emulator then polls each signal on its own clock, holds and quantises it, and estimates fuel from MAF the way real loggers do.

**Simulator**

- RoadSim road loads, ALPHA engine and motor maps, transmissions, battery, HVAC, 7 architectures, 14 strategies
- OBD emulator (training/obdlog.py): per-PID polling cadences measured on dev cars, held values, MAF fuel estimate, dropouts
- Dev-twin priors (training/priors.py): VED-era engine efficiency, idle, hybrid strategy spread, milder HVAC, and a VED-era mass prior (median 1,720 kg)

**Checked against independent references**

- RoadSim road-load force within 1.6 / 0.8 / 0.5% of 47 real EPA coast-down targets at 20 / 40 / 60 mph
- Energy ledger closes to max 5e-4 over 600 stress runs; numpy == torch to 8e-6

## Next rungs

- Find which v2 prior hurts engine size; rebuild on v1 plus the v3 mass prior.
- Make mass and drag identifiable from the dynamics (a physics-regression head, pooled multi-trip training).
- Fix BEV cabin heat; add buses (ZTBus mass vs passengers).

## Files

| Path | What |
|---|---|
| `tripdetective.py`, `pipeline.py` | Inference — numpy/scipy only |
| `tripdetective_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Vehicle Energy Dataset (VED) — Oh, LeBlanc, Peng, IEEE T-ITS 2020](https://github.com/gsoh/VED) | noise (routes), dev, exam | Apache-2.0 |
| [US EPA test-car lists, coast-down targets, ALPHA maps; fueleconomy.gov](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy) | physics, exam | Public domain (U.S. Government) |
| [USGS 3DEP lidar elevation; Iowa Environmental Mesonet ASOS weather](https://www.usgs.gov/3d-elevation-program) | physics | Public domain / free use with credit |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
