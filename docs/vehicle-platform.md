# The vehicle platform

Most models in the collection carry their own simulator. The vehicle models share one: **RoadSim**, a verified
whole-vehicle simulator, and **RoadData**, real driving data split once into dev and exam and frozen. Three models
sit on it today, and new vehicle problems start from it instead of from scratch.

| Model | The question |
|---|---|
| [EnergyNet](../models/energynet) | How much fuel or electricity will this trip take — and where does it go? |
| [TripDetective](../models/tripdetective) | From a trip log alone, what car is this: powertrain, mass, engine, drag? |
| [HybridPilot](../models/hybridpilot) | Second by second, how should a hybrid run its engine to use the least fuel? |

Code and data: [`shared/vehicle/`](https://github.com/dogum/ML-Ratchet/tree/main/shared/vehicle).

## RoadSim: the simulator

- **Seven powertrains** — conventional (gasoline, diesel), P2 / series / power-split hybrids, power-split and series
  plug-ins, battery-electric — across **seven classes**, from cars to Class-8 tractors.
- **Subsystem physics:** EPA ALPHA engine and motor maps with cold start; automatic, dual-clutch, CVT, automated
  manual and eCVT transmissions; a battery equivalent circuit with a thermal model; regenerative braking; wind along
  the heading; rolling resistance with temperature, speed and surface; real lidar grade; heating and A/C.
- **Every classic hybrid strategy:** rule-based, power-follower, thermostat, ECMS, adaptive ECMS, MPC, and the
  dynamic-programming optimum.
- **An energy ledger on every run:** fuel in, then every loss — engine, motor, transmission, battery, aero, rolling,
  grade, brakes, accessories — must add up. It closes in all 600 stress runs (max 5e-4).
- **Checked against reality:** on 33 EPA vehicles held out from tuning, conventional FTP +5.6% and HWFET −4.3%,
  power-split hybrids −2.3%, BEVs −0.5% / −1.7%; against FASTSim 3.1 with matched components within 3–4%.
- **Fast:** about 60,000× real time for conventional cars and EVs, 5,000× for hybrids, on one thread.

## RoadData: the real data

- 33,015 real trips from the Vehicle Energy Dataset (Ann Arbor, 2017–18), with USGS lidar road grades and airport
  weather on the day.
- 80,227 EPA dynamometer tests and fueleconomy.gov labels; 24 drive cycles.
- Heavy duty: Zurich trolleybuses (ZTBus), diesel and CNG buses from Mersin, and ICCT tractor-trailer tests.
- A frozen dev/exam split by vehicle (`splits.json`, sha256 `d917df78…`) that every vehicle exam depends on.

## What it gets wrong

The simulator's measured gaps are its next rungs, each with a size and a "done when" in
[`HANDOFF.md`](https://github.com/dogum/ML-Ratchet/blob/main/shared/vehicle/HANDOFF.md):

- P2 hybrids in the city read 8–31% optimistic; Honda's e:HEV −9% on the highway.
- Cold weather and A/C are under-predicted (+10–16% and +9–15% on EPA tests).
- Real Toyota hybrids keep the engine on far more (55% vs 35% of the time); real hybrids burn about 20% more fuel
  than their simulated twins.
- Heavy duty rests on a handful of real buses and trucks.

The two most valuable missing datasets — Argonne's dynamometer database and NREL's Fleet DNA truck and bus traces —
were blocked from the cloud sandbox where the platform was built.

## Using it

```bash
pip install -e ".[vehicle]"
cd shared/vehicle/roadsim && python demo.py
```

```python
from roadsim import simulate, load_vehicle, cycle
import realdata as rd
```

Every physics change bumps `PHYSICS_VERSION`, rebuilds each vehicle model's frozen simulated set, and re-takes their
ratchet decisions in order.
