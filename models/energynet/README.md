<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# EnergyNet <sub>v1.0</sub>

> **How much fuel or electricity will this trip take, before you drive it?**

*How much energy will this vehicle use on this route, in this weather — and where does it go?*

EnergyNet predicts a trip's fuel or battery energy before the trip, with a 68% and 95% range and a split into air drag, rolling, braking, accessories and powertrain losses. It covers 7 vehicle classes from cars to Class-8 trucks and 7 powertrains. It learned only from the RoadSim vehicle simulator driven over real Ann Arbor routes and weather, and was examined on 26,000 real trips, 47,000 EPA tests and real buses.

[Project page](https://dogum.github.io/ML-Ratchet/models/energynet/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | A route at 1 Hz (speed, grade, temperature, wind, heading, air density, surface) and whatever is known about the vehicle (architecture, class, mass, road load, engine, battery, window-sticker label, start charge); unknown fields are masked. |
| **Returns** | Trip energy (MJ) with 68% and 95% ranges — fuel chemical energy, battery energy, or both for plug-ins<br>Split into aero, rolling, friction braking, accessories/HVAC and powertrain losses |
| **Learned from** | The shared RoadSim simulator (shared/vehicle): a forward-facing 1 Hz whole-vehicle model with real road loads, EPA ALPHA engine and motor maps, transmissions, a battery with thermal model, HVAC and seven powertrains, driven over real Ann Arbor routes with lidar grades and airport weather. |
| **Noise** | Real routes and station weather; the simulated car drives slightly different conditions (temperature ± 1.5 °C, wind × 0.4–0.9) than the input reports |
| **Network** | Ensemble of 5 MLPs (149 → 256 → 256 → 256 → 128 → 7), predicting a correction to a physics energy estimate, its log σ and 5 energy components; Gaussian NLL |
| **Dev set** | 16 VED cars plus dev EPA vehicles, ZTBus legs, Mersin routes and one ICCT truck — RoadData's frozen split (sha256 d917df78…) |
| **Exam** | 25,793 VED trips from 350 cars, EPA exam families (47,718 tests), ZTBus, Mersin and ICCT exam sets |

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
net = EnergyNet()                                  # the 5-network ensemble in weights/
ex = json.load(open("examples/examples.json"))[0]
z = np.load(f"examples/{ex['trip_id']}.npz")
trip = {k: z[k] for k in z.files if k != "valid"}
r = net.predict(trip, ex["vehicle"])
print(r["energy_MJ"])
```

## The exam

Real VED trips and EPA tests split once and frozen; median absolute % error. Every number here is on real data that never chose anything.

| Metric | EnergyNet | Baselines | |
|---|---|---|---|
| Petrol cars, per trip (14,469 VED trips) | **8.9%** | MOVES-like (classical): 10.3%<br>Window label × distance (classical): 13%<br>Class median (dumb baseline): 22.8% | ✅ wins |
| Plug-in hybrids, per trip (2,916 trips)<br><sub>EnergyNet is given the logged start charge; the baselines are not.</sub> | **20.1%** | MOVES-like (classical): 41.3%<br>Class median (dumb baseline): 42.7% | ✅ wins |
| Hybrids, per trip (8,241 trips)<br><sub>Biased +12%.</sub> | **16.7%** | MOVES-like (classical): 15.3%<br>Class median (dumb baseline): 17.2% | ❌ loses |
| Electric (2 Nissan Leafs), per trip<br><sub>Biased −20%.</sub> | **22.8%** | MOVES-like (classical): 19%<br>Class median (dumb baseline): 27.1% | ❌ loses |
| EPA hybrid FTP tests (1,098) | **8.7%** | Log-linear regression (fit on dev) (classical): 16.3%<br>Dev median (dumb baseline): 49% | ✅ wins |
| EPA BEV city tests (746) | **7.9%** | Log-linear regression (classical): 9.8%<br>Dev median (dumb baseline): 19% | ✅ wins |
| EPA petrol highway tests (20,426)<br><sub>Biased +15% — the fleet prior was widened toward older engines to fit VED.</sub> | **15.6%** | Log-linear regression (classical): 5%<br>Dev median (dumb baseline): 16.8% | ❌ loses |
| ZTBus trolleybus legs (78) | **16.4%** | Dev kWh/km × distance (classical): 22.1% | ✅ wins |
| Mersin diesel bus trips (652) | **14.8%** | Dev median on the same route (dumb baseline): 8.6% | ❌ loses |
| ICCT tractor-trailers (2 trucks × 3 cycles) | **11.4%** | Dev truck on the same cycle (classical): 7% | ❌ loses |

Coverage of the 68% error bars on the exam: VED petrol **55%**, VED hybrid **65%**, VED electric **47%**, VED plug-in **65%**, EPA petrol FTP **77%**, EPA petrol highway **54%**.

### Where it wins

- Petrol cars: 8.9% per trip against 10.3% for the best classical method, in every temperature, speed, grade and length band.
- Plug-ins: 20% against 41% (with the start charge as an input).
- EPA hybrid and electric tests, and the Zurich trolleybus legs.

### Where it loses

- Real hybrids (+12% bias) and the two exam Leafs (−20%) lose to simple baselines.
- EPA petrol highway and US06 tests are biased +11–15%; a regression on the label wins easily.
- Diesel and CNG buses and the trucks lose to the dev median on the same route or cycle.

## Limits

- Error bars are too narrow on real VED trips (47–65% inside the 68% band) despite dev widening.
- VED fuel is a MAF-based estimate, Ann Arbor 2017–18 only, mostly warm trip fragments.
- Heavy duty is thin and partly synthetic (Mersin speed profiles, ZTBus mass unknown).
- Built before the 13 Nov RoadData fixes: its twins ran VED cars 136 kg heavy (EPA labels were graded corrected).
- Cold-start penalty and P2 city optimism are not modelled from dev.

## The ratchet

**Objective:** Skill = 1 − median |log error| / median |log error| of the class-median guess, on the frozen simulated validation set (3,991 examples from 400 unseen vehicles)  
**Rule:** Keep if the skill beats the incumbent by more than 0.01 AND the dev score D (mean of 15 dev median errors across VED fleets, EPA tests, ZTBus, Mersin and ICCT) gets no more than 0.01 worse.

| Rung | Change | Frozen-set score | Decision | Why | Val MdAPE | Dev D |
|---|---|---|---|---|---|---|
| r1 | MLP predicts log energy directly | 0.544 | ● kept | First rung. | 11.2% | 0.152 |
| r2 | Predict a correction to the physics energy estimate | 0.556 | ● kept | +0.012; dev D 0.160 (within +0.01). | 11.0% | 0.16 |
| r3 | MAF label model + flag input (physics v2) | 0.580 | ● kept | +0.024; dev D 0.152. | 10.3% | 0.152 |
| r4 | On-road loads vs temperature, fitted on dev (physics v3) | 0.641 | ● kept | +0.061; dev D 0.142. | 9.0% | 0.142 |
| r5 | Plug-in hardware patch (physics v4) | 0.648 | ○ rejected | +0.007, short of the 0.01 margin. | 8.9% | 0.129 |
| r6 | Ensemble of 5 networks on v4 physics | 0.656 | **◆ released** | +0.015 over rung 4; dev D 0.129. | 8.7% | 0.129 |

<sub>Columns after *Why*: Frozen-set median error and the dev score D (lower is better).</sub>

The frozen set was re-scored at each physics version (v2 MAF label model, v3 on-road loads, v4 plug-in patch).

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared RoadSim simulator (shared/vehicle): a forward-facing 1 Hz whole-vehicle model with real road loads, EPA ALPHA engine and motor maps, transmissions, a battery with thermal model, HVAC and seven powertrains, driven over real Ann Arbor routes with lidar grades and airport weather.

**Simulator**

- 2,400 training vehicles from RoadSim's generator, widened to cover 2005–2010-era engines and older automatics
- Routes from 1,092 unlabelled real VED trips, dynamometer cycles and synthetic urban/highway/bus/truck drives
- Simulated window-sticker labels (FTP-75 and HWFET, EPA 5-cycle formulas)
- Dev-calibrated wrappers (training/wrappers.py): OBD MAF fuel during fuel cut-off, on-road loads vs temperature, plug-in hardware patch

**Checked against independent references**

- RoadSim vs EPA on 33 held-out vehicles: conventional FTP +5.6% / HWFET −4.3%, power-split hybrids −2.3%, BEV −0.5% / −1.7%
- Energy ledger closes in all 600 stress runs (max 5e-4); FASTSim 3.1 cross-check within 3% (conventional), 4% (BEV)

## Next rungs

- Upstream the dev-calibrated wrappers into RoadSim (bump PHYSICS_VERSION), rebuild the frozen set, re-score every rung.
- Cold-start calibration from EPA dev FTP_cold; a model-era input; per-architecture σ.
- Real Mersin speed profiles; ZTBus bus mass.

## Files

| Path | What |
|---|---|
| `energynet.py`, `features.py`, `model_np.py` | Inference — numpy/scipy only |
| `weights/r5_phev.npz`, `weights/r6_s1.npz`, `weights/r6_s2.npz`, `weights/r6_s3.npz`, `weights/r6_s4.npz`, `sigma_factors.json` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Vehicle Energy Dataset (VED) — Oh, LeBlanc, Peng, IEEE T-ITS 2020](https://github.com/gsoh/VED) | noise (routes), dev, exam | Apache-2.0 |
| [US EPA test-car lists, drive cycles and ALPHA maps; fueleconomy.gov](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy) | physics, dev, exam | Public domain (U.S. Government) |
| [USGS 3DEP lidar elevation](https://www.usgs.gov/3d-elevation-program) | physics (road grade) | Public domain (U.S. Government) |
| [Iowa Environmental Mesonet ASOS weather](https://mesonet.agron.iastate.edu/request/download.phtml) | physics (weather) | Free use with credit |
| [ZTBus — Widmer et al., ETH Zurich](https://www.research-collection.ethz.ch/handle/20.500.11850/626734) | dev, exam (trolleybuses) | CC BY 4.0 |
| [Mersin bus trips — Abut, Zenodo](https://zenodo.org) | dev, exam (diesel and CNG buses) | CC BY 4.0 |
| [ICCT 2018 tractor-trailer tests (Rodriguez, Delgado, Muncrief)](https://theicct.org) | dev, exam (trucks) | Cite the report |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
