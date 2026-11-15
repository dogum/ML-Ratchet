<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# TripDetective <sub>v2.0</sub>

> **What car was this? Read it from one trip log.**

*From a car's trip log alone, what kind of powertrain is it, how heavy, what engine, how much drag — and where did the energy go?*

TripDetective reads one or more 1 Hz trip logs of a light-duty vehicle — speed plus whatever an OBD logger has — and infers its architecture, total mass, engine size, CdA, rolling resistance, hybrid strategy and each trip's energy ledger, every output with an error bar. It learned only from the RoadSim simulator (physics v2) with an emulated OBD logger, and was graded on 350 real cars it never saw. Version 2.0 adds a physics-regression head: the road-load fit of each trip is pooled over a car's trips and solved, so a heavier car now reads heavier (+524 kg for +800 kg in simulation; v1.0 +182 kg). A separate bus network reads city-bus mass from a power log (ZTBus).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/tripdetective/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/tripdetective/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One or more 1 Hz trip logs (speed, and when logged fuel rate, rpm, HV battery current/voltage/SOC, A/C power), with route grade and weather. |
| **Returns** | Architecture (7 layouts) with probabilities<br>Total and curb mass, engine displacement, CdA, Crr — each with a 1σ range<br>Hybrid control strategy and engine-off share<br>Each trip's energy ledger in kJ/km (aero, rolling, grade, friction braking, powertrain losses, accessories) |
| **Learned from** | The shared RoadSim simulator (shared/vehicle, physics v2, pinned) drives generated cars over real Ann Arbor routes; RoadSim's OBD-logger emulator (roadsim/obd.py, moved there from this model) then polls each signal on its own clock, holds and quantises it, and estimates fuel from MAF the way real loggers do. |
| **Noise** | Real routes and weather; the emulated logger reproduces real OBD cadences, holds, quantisation and MAF-based fuel estimates |
| **Network** | 1-D CNN over each trip window (23 channels, incl. a source-power channel) plus 16 classical estimates; trip embeddings pooled across a car's trips together with the road-load regression's normal equations, which are solved per car (physics-regression head: mass = fitted b0 x a learned efficiency correction); heads for architecture, strategy and 7 regressions (Gaussian NLL); 0.40 M parameters. A second, separately trained copy reads buses. |
| **Physics also enters** | inputs: 16 classical per-trip estimates and the road-load regression's normal equations<br>output: Mass = the fitted road-load coefficient b0 × a learned efficiency correction |
| **Real data's jobs** | backdrop, calibration, gate, exam (never training labels) |
| **Dev set** | 16 VED dev cars — RoadData's frozen split (sha256 d917df78…); 757 ZTBus legs of the dev missions (by-month split, the 'gate' half) |
| **Exam** | 350 VED exam cars (at most 40 trips each, fixed seed), 25 EPA road-load cars, 150 EPA exam vehicles rebuilt in RoadSim, and 5,847 ZTBus legs of the 1,115 exam missions |

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
bus = TD.TripDetectiveBus().read_leg(leg)   # leg: dict of 1 Hz v, power_W, grade, T_amb, rho_air
```

## The exam

Real VED exam cars and ZTBus exam legs; baselines fitted on dev only. v1.0 is re-scored on the same exam with the same code (in each note). Every number here is on real data that never chose anything.

| Metric | TripDetective | Baselines | |
|---|---|---|---|
| Curb mass per car, median \|error\| (335 cars)<br><sub>v1.0: 116 kg. Per trip v2.0 is a little better (112 vs 114 kg); per car worse, mostly on HEVs (181 vs 93 kg).</sub> | **127 kg** | Physics least-squares fit (classical): 205 kg<br>Dev median (dumb baseline): 227 kg | ✅ wins |
| Engine displacement per car, median \|error\|<br><sub>v1.0: 0.50 L (bias −0.48 L); v2.0 bias −0.31 L.</sub> | **0.38 L** | Idle-fuel law (classical): 0.64 L<br>Dev median (dumb baseline): 0.6 L | ✅ wins |
| Road-load force vs EPA target at 40 mph (25 cars)<br><sub>v1.0: 17%. At 20 mph v2.0 is worse (24% vs 16%).</sub> | **19%** | Physics least-squares fit (classical): 35% | ✅ wins |
| Powertrain type per car (350 cars)<br><sub>v1.0: 98.9%. v2.0 calls 10 of 86 HEVs conventional (8 of them small Prius-c-type cars).</sub> | **97.1%** | Simple rule (classical): 99.4%<br>Majority class (dumb baseline): 68.6% | ❌ loses |
| The two Leafs, curb mass \|error\|<br><sub>v1.0: +641 / +278 kg. Both Leafs sit in the 3,500 lb class, which is also the dev median, so the dumb guess is exact here.</sub> | **119 / 105 kg** | Physics least-squares fit (classical): 173 / 373 kg<br>Dev median (dumb baseline): 0 / 0 kg | ❌ loses |
| Mass response to +800 kg (64 simulated cars, 8 trips)<br><sub>v1.0 on the same sweep: +182 kg. BEVs +760, PHEVs +679, ICE +448, HEVs +384 kg. CdA ×1.5 still barely registers (×1.03).</sub> | **+524 kg** | — | ℹ️ |
| Bus mass per ZTBus exam leg, mean \|error\| (5,847 legs)<br><sub>Truth = 19 t + 70 kg per passenger. Reads 1.0 t heavy; with the dev-fitted level, 752 kg. Slope on passenger mass 0.70, correlation 0.35.</sub> | **1204 kg** | Physics least-squares fit (scaled on dev) (classical): 3547 kg<br>Dev median (dumb baseline): 453 kg | ❌ loses |
| EPA exam families rebuilt in RoadSim (150 vehicles, 4 trips), mass vs test weight, MdAPE<br><sub>v1.0 (on physics v1): 7.8%. Road-load force at 60 mph 10.3% (v1.0 9.4%).</sub> | **5%** | — | ℹ️ |

Coverage of the 68% error bars on the exam: Mass, per car **56%**, Mass, per trip **76%**, Engine size, per car **48%**, Bus mass, per leg **68%**.

### Where it wins

- Mass within 127 kg per car (112 kg per trip) against 205 kg for the physics fit and 227 kg for the dev median.
- Engine size 0.38 L against 0.60 L for the dev median (v1.0: 0.50 L).
- Mass now follows the physics: +800 kg moves the estimate +524 kg in simulation (v1.0: +182 kg).

### Where it loses

- A simple rule beats it on powertrain type (99.4% vs 97.1%); v2.0 is worse here than v1.0 (98.9%).
- Per car, mass is a little worse than v1.0 (127 vs 116 kg), driven by HEVs (181 vs 93 kg); 1σ coverage fell to 56%.
- Buses: the dev-median guess (453 kg) beats the bus network (1,204 kg; 752 kg with the dev-fitted level).
- The two Leafs: 119 / 105 kg off (v1.0: +641 / +278 kg; the V2_PLAN target of 150 kg is met), but the dev-median guess is exact for both, because they sit in its 3,500 lb class.

## Limits

- CdA and Crr are still weakly identified: CdA ×1.5 moves the estimate by 3%.
- Real cars turn power into acceleration differently from their twins (dev twin b0 ratio 0.83-1.37 per car); the dev Leaf's battery power fits a lighter car (0.83× its twin).
- ZTBus: the logged power and the logged traction force both regress to ~0.7 × (19 t + passengers), so the bus mass level is not verified; per leg the physics fit scatters ±15%.
- The energy ledger has only simulated truth; no skill on net grade work.
- Light duty is graded on 2017-18-era VED cars only. RoadSim v2 limits apply (older Toyota hybrids too frugal on the FTP, TMED P2 too thirsty, plug-in series engine-on too low).

## The ratchet

### v1.0 — RoadSim physics v1 (12-13 Nov 2026)

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
| r6 | v3 mass prior (VED-era fleet) + classical estimates, 2,000 cars | 0.402 | ● kept | +0.085; dev mass error 110 kg (bias −9). Released as v1.0. | 110 kg | 0.44 L | 0.92 |

<sub>Columns after *Why*: Dev cars (gate inputs).</sub>

A network twin test found the generator's fleet far heavier than VED-era cars (2,200 vs 1,720 kg), which became the v3 mass prior. Every v2-prior rung made dev engine size worse; the cause was not isolated.

### v2.0 — RoadSim physics v2 (Phase D, 14 Nov 2026)

**Objective:** Mean simulated skill over architecture, mass, displacement, CdA, Crr, engine-off behaviour, strategy and the ledger (single trip and whole car), on frozen sets rebuilt on physics v2 (600 cars × 8 trips, seed 99)  
**Rule:** Keep if the skill beats the incumbent by more than 0.01 AND the mean of 4 dev error ratios is at most 1.05 AND dev VED-type accuracy drops by no more than 0.03 (unchanged from v1.0).

| Rung | Change | Frozen-set score | Decision | Why | Dev mass error / car | Dev engine error / car | Dev type accuracy / trip | Mass response to +800 kg |
|---|---|---|---|---|---|---|---|---|
| p2r0 | CNN + set pooling, generator as published (prior v1), 1,200 cars | 0.378 | ● kept | First rung. | 156 kg | 0.39 L | 0.85 | +257 kg |
| p2r1 | v1.0's dev-twin priors (v2), 1,200 cars | 0.363 | ○ rejected | Sim −0.015; dev errors +14%. | 106 kg | 0.60 L | 0.91 | +202 kg |
| p2r2 | v2 priors, 2,000 cars | 0.377 | ○ rejected | Sim −0.001; dev errors +13%. | 123 kg | 0.55 L | 0.89 | +222 kg |
| p2r3 | + road-load physics channels | 0.374 | ○ rejected | Sim −0.004; dev errors +7%. | 103 kg | 0.56 L | 0.85 | +215 kg |
| p2r4 | + 16 classical per-trip estimates | 0.386 | ○ rejected | Sim +0.008; dev errors +17%. | 125 kg | 0.67 L | 0.91 | +275 kg |
| p2r5 | Classical estimates on the v1 bank | 0.397 | ● kept | +0.019; dev +0%. | 134 kg | 0.46 L | 0.93 | +331 kg |
| p2r6 | v3 mass prior + classical estimates, 2,000 cars (v1.0's release recipe) | 0.405 | ○ rejected | +0.008 (short of the margin); dev VED-type accuracy −0.043. | 122 kg | 0.51 L | 0.89 | +227 kg |
| p2r7 | Priors rebuilt: the v1 generator + the v3 VED-era mass prior (prior v4), 2,000 cars | 0.430 | ● kept | +0.033; dev errors −9%. | 126 kg | 0.43 L | 0.92 | +295 kg |
| p2r8 | BEV cabin heat matched to the dev Leaf (prior v5) | 0.430 | ○ rejected | +0.000; dev errors −10%. | 102 kg | 0.35 L | 0.91 | +226 kg |
| p2r9 | VED logger channel mix measured on the dev cars (prior v6) | 0.431 | ○ rejected | +0.001; dev errors −5%, dev type 16/16 cars. | 121 kg | 0.32 L | 0.93 | +360 kg |
| p2r10 | Sharper channels: a wheel-equivalent source-power channel (feature version 4) | 0.432 | ○ rejected | +0.002; dev errors −7%. | 116 kg | 0.37 L | 0.9 | +257 kg |
| p2r11 | Physics-regression head: the road-load fit's normal equations pooled over a car's trips and solved; mass = b0 × a learned efficiency correction | 0.450 | **◆ released** | +0.020; dev errors −10%. | 133 kg | 0.31 L | 0.91 | +524 kg |
| p2r12 | Pooled multi-trip training: 90% of samples pool 2-8 trips, 32 cars per batch | 0.452 | ○ rejected | +0.002; dev VED-type accuracy −0.031. | 122 kg | 0.27 L | 0.88 | +519 kg |
| p2r13 | GPU: wider CNN (48-192 channels, 160-d embedding), 8,000 steps | 0.491 | ○ rejected | +0.041 in simulation, but dev errors +6%. | 142 kg | 0.33 L | 0.91 | +571 kg |
| p2r14 | VED logger channel mix again, with the physics head | 0.444 | ○ rejected | −0.006. | 121 kg | 0.25 L | 0.91 | +503 kg |
| p2r15 | BEV cabin heat again, with the physics head | 0.450 | ○ rejected | +0.000; dev errors +3%. | 128 kg | 0.32 L | 0.9 | +485 kg |

<sub>Columns after *Why*: Dev cars (gate inputs) and the identifiability sweep.</sub>

Every v1.0 decision re-taken from r0 on physics v2, then rescored in order on each new frozen set; no decision changed between sets. The v1.0 dev-twin wrappers (priors v2) now over-correct: physics v2 already carries VED-era engine efficiency (twin fuel/km 0.95× real with the plain generator, 1.07× with the wrappers). Mass response = median change of the pooled estimate when 64 simulated cars (stratified by type) get +800 kg, 8 trips (sweep.py; info, not part of the rule). v1.0 on the same sweep: +182 kg.

### v2.0 bus — city-bus mass (Phase D, 14 Nov 2026)

**Objective:** Mean skill over log mass, log CdA and log Crr on a frozen set of 300 simulated electric buses × 6 windows (single window and pooled)  
**Rule:** Keep if the skill beats the incumbent by more than 0.01 AND the level-corrected dev mass error is at most 5% worse AND the slope of estimated mass on passenger mass drops by no more than 0.1.

| Rung | Change | Frozen-set score | Decision | Why | Dev mass error (raw) | Dev mass error (level-corrected) | Slope on passenger mass | Correlation with passengers |
|---|---|---|---|---|---|---|---|---|
| pb1 | Bus network: CNN + set pooling + classical estimates on 1,500 simulated electric buses (power ÷ 12) | 0.524 | **◆ released** | First rung. | 1,127 kg | 805 kg | 0.42 | 0.25 |
| pb2 | + physics-regression head and the source-power channel | 0.546 | ○ rejected | Reads 4.8 t light; level-corrected error +239%. | 4,832 kg | 2,731 kg | 1.25 | 0.29 |

<sub>Columns after *Why*: ZTBus dev gate legs.</sub>

Rule changed in the open after pb2 and before any other rung: the first rule gated on the raw error. The truth's level is a convention (19 t curb): on 40 dev legs both the logged power and the logged traction force regress to 0.67-0.73 × (19 t + passengers). pb2 fails under both rules.

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle, physics v2, pinned) drives generated cars over real Ann Arbor routes; RoadSim's OBD-logger emulator (roadsim/obd.py, moved there from this model) then polls each signal on its own clock, holds and quantises it, and estimates fuel from MAF the way real loggers do.

**Simulator**

- RoadSim physics v2 (dev-calibrated): road loads with a tyre model, ALPHA engine maps with per-map efficiency tilt, P2 converter and start energy, Toyota production rules, two-node HVAC, on-road loads; 7 architectures, 14 strategies
- OBD emulator (roadsim/obd.py, EMU_VERSION 1): per-PID polling cadences measured on dev cars, held values, MAF fuel estimate, dropouts
- Priors (training/priors.py, v4): the RoadSim generator as is plus a VED-era mass prior (median 1,750 kg total); v1.0's engine-efficiency wrappers now over-correct and were dropped
- Buses (training/busbank.py): battery-electric city buses of 14-24 t plus 0-110 passengers, driven on ZTBus dev routes

**Checked against independent references**

- RoadSim road-load force within 1.6 / 0.8 / 0.5% of 47 real EPA coast-down targets at 20 / 40 / 60 mph
- Energy ledger closes to max 5.6e-4 over 600 stress runs (physics v2); numpy == torch to 4.6e-6; browser == Python to 1.6e-5
- Dev twin test on physics v2: conventional twin fuel/km 0.95x real and HEV 1.03x with the plain generator (with v1.0's wrappers 1.07x / 1.11x); the dev Leaf's twin used 1.44x its real energy (cabin heat)

## Next rungs

- Powertrain type: the VED channel mix (rung p2r9: 16/16 dev cars) gained only +0.001 in simulation; try a type-weighted objective or the rule as an input.
- Check the small Prius-c-type HEVs on dev twins (engine-on share under physics v2).
- Buses: a measurement model for the ZTBus power log (why its road-load fit reads 0.7×), then the physics head again.

## Files

| Path | What |
|---|---|
| `tripdetective.py`, `pipeline.py` | Inference — numpy/scipy only |
| `tripdetective_weights.npz`, `tripdetective_bus_weights.npz` | Weights |
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
| [ZTBus — Widmer, Ritter, Onder, Scientific Data 2023 (full set, 1,409 missions)](https://doi.org/10.3929/ethz-b-000626723) | noise (bus routes from dev missions), dev, exam | CC-BY-4.0 |
| [US EPA test-car lists, coast-down targets, ALPHA maps; fueleconomy.gov](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy) | physics, exam | Public domain (U.S. Government) |
| [USGS 3DEP lidar elevation; Iowa Environmental Mesonet ASOS weather](https://www.usgs.gov/3d-elevation-program) | physics | Public domain / free use with credit |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
