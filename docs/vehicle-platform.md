# The vehicle platform

Most models in the collection carry their own simulator. The vehicle models share one: **RoadSim**, a verified
whole-vehicle simulator, and **RoadData**, real driving data split once into dev and exam and frozen. Three models
sit on it today, and new vehicle problems start from it instead of from scratch.

| Model | The question |
|---|---|
| [EnergyNet](../models/energynet) | How much fuel or electricity will this trip take — and where does it go? |
| [TripDetective](../models/tripdetective) | From a trip log alone, what car is this: powertrain, mass, engine, drag? |
| [HybridPilot](../models/hybridpilot) | Second by second, how should a hybrid run its engine to use the least fuel? |

All three run on the same trip in the [garage](https://dogum.github.io/ML-Ratchet/garage/): pick a vehicle and a real
route, and see EnergyNet's forecast, the RoadSim run, TripDetective reading the log back and HybridPilot driving it.

Code and data: [`shared/vehicle/`](https://github.com/dogum/ML-Ratchet/tree/main/shared/vehicle).

## RoadSim: the simulator

- **Seven powertrains** — conventional (gasoline, diesel), P2 / series / power-split hybrids, power-split and series
  plug-ins, battery-electric — across **seven classes**, from cars to Class-8 tractors.
- **Subsystem physics:** EPA ALPHA engine and motor maps with cold start; automatic, dual-clutch, CVT, automated
  manual and eCVT transmissions; a battery equivalent circuit with a thermal model; regenerative braking; a tyre
  model; wind along the heading; rolling resistance with temperature, speed and surface; real lidar grade; heating and
  A/C.
- **Every classic hybrid strategy:** rule-based, power-follower, thermostat, ECMS, adaptive ECMS, MPC, and the
  dynamic-programming optimum.
- **An energy ledger on every run:** fuel in, then every loss — engine, motor, transmission, battery, aero, rolling,
  grade, brakes, accessories — must add up. It closes in all 600 stress runs (max 5.6e-4).
- **Checked against reality:** the default physics is v2, calibrated on dev data only. On 33 EPA test vehicles held
  out from tuning, conventional cars read +0.4 % on the FTP (the EPA city test) and −1.9 % on the HWY (highway) test,
  battery-electric cars −0.4 % / −1.8 % (fuel economy; + = the simulator burns less). Against NREL's FASTSim 3.1 with
  matched components, conventional cars agree within −1.9 / −1.2 / −6.4 % on the city, highway and US06 cycles.
- **Fast:** about 30,000× real time for conventional cars (29,500-29,900×), 33,400× for EVs and 4,600-5,900× for
  hybrids, on one thread (UDDS city cycle, after the speed pass of 14 Nov 2026).

## RoadData: the real data

- 33,015 real trips from the Vehicle Energy Dataset (Ann Arbor, 2017–18), with USGS lidar road grades and airport
  weather on the day.
- 80,227 EPA dynamometer tests and fueleconomy.gov labels; 24 drive cycles.
- Argonne's Downloadable Dynamometer Database (D3): 65 vehicles, 1,661 usable 10 Hz chassis-dynamometer tests with
  fuel, battery current, engine speed and temperatures. RoadSim's dev and exam set, split by vehicle (32 dev, 33 exam).
- Heavy duty: the full Zurich trolleybus set (ZTBus, 1,409 missions, split by month: 294 dev, 1,115 exam), diesel and
  CNG buses from Mersin, and ICCT tractor-trailer tests.
- A frozen dev/exam split by vehicle (`splits.json`, sha256 `d917df78…`) that every vehicle exam depends on.

Still missing: NREL's Fleet DNA truck and bus traces. nrel.gov timed out or refused every scripted request, so
heavy duty has no real truck traces yet.

## Physics versions

Every physics change bumps `PHYSICS_VERSION`, is calibrated on dev data only and is judged on data it never saw.

- **v1** (12-13 Nov 2026): the first build, the one the models' v1.0 releases used.
- **v2** (13 Nov 2026): the default. A calibration pass on dev data only (EPA dev vehicles, D3 dev vehicles, VED dev
  twins), frozen before the held-out run. The models' v2.0 releases pin it.

Held-out results, v1 → v2 (median). EPA rows are fuel economy (+ = the simulator burns less); D3 and ZTBus rows are
energy (+ = the simulator uses more).

| check | v1 | v2 |
|---|---|---|
| EPA held-out conventional, FTP / HWY (n 15) | +5.6 / −4.3 % | +0.4 / −1.9 % |
| EPA held-out P2 hybrids, FTP (n 5) | +11.9 % | −4.0 % |
| EPA held-out Honda e:HEV, HWY (n 1) | −9.5 % | −3.8 % |
| EPA held-out power-split hybrids, FTP / HWY (n 4) | +1.0 / −2.4 % | −5.1 / −7.2 % (worse) |
| EPA held-out BEVs, UDDS / HWY (n 8) | −0.5 / −1.7 % | −0.4 / −1.8 % |
| D3 exam conventional, all tests (n 240) | −5.1 % | −3.2 % |
| ZTBus exam legs, bus twin (n 5,847) | −13.8 % | −2.3 % |
| FASTSim cross-check, conventional UDDS / HWY / US06 | −0.3 / −3.1 / −3.0 % | −1.9 / −1.2 / −6.4 % |
| Ledger, 600 stress runs | max 5.1e-4 | max 5.6e-4 |
| Speed, UDDS, one thread | conventional 50,000-55,000×, hybrids 3,800-4,800× | conventional 29,500-29,900×, hybrids 4,600-5,900× |

v2 made the Toyota power-splits worse on the EPA cycles: the 45 km/h engine-on rule that fixed them on real roads
costs fuel on the dynamometer.

### Opt-in presets

Three later passes are kept as named options, `simulate(..., options='roadsim-physics-v3')`. None is the default.
Each froze a fresh judge before any fitting, where one was left:

- **Exam-2:** a fresh hybrid exam frozen on 14 Nov 2026 before v2.1 was fitted: 409 EPA test-car records of hybrids
  and plug-ins no check had simulated, and 108 VED exam hybrid and plug-in cars (11,157 trips).
- **Exam-3:** 64 more EPA test-car records no check had simulated, frozen before v3 was fitted.
- **Second look:** a result on data seen before the version was fitted. It is reported, but it judges nothing.

| preset | what it changes | judged on | result | why not the default |
|---|---|---|---|---|
| v2.1 | a running hybrid engine keeps running 2 s into braking (measured on six D3 dev hybrids) | exam-2, frozen first | VED Toyota-type engine-on gap −10.1 → −3.8 points; EPA fuel errors move 0.1-2.5 points, both ways | no clear fuel gain on dev |
| v3 | older Toyota/Lexus engines by model year; the torque converter held open in 1st-2nd gear in P2 automatics | exam-3, clean | pre-TNGA Toyota/Lexus FTP +44 → +31 % (n 3); converter P2s +16.6 → +14.6 % (n 9) | the gain rests on three dev vehicles |
| v3.1 | an engine-generation scale for hybrids that use a conventional engine map (×0.875 / 0.942 / 0.975 for model years ≤2011 / 2012-15 / 2016-17) | no clean judge left | mild hybrids on exam-3 FTP +23 → +8 % (n 18), a second look | one narrow dev gain |

TNGA is Toyota's current engine generation; "pre-TNGA" means model year 2017 and older. Details:
[CALIBRATION.md](https://github.com/dogum/ML-Ratchet/blob/main/shared/vehicle/roadsim/CALIBRATION.md) (every rung, kept
or not) and [LIMITS.md](https://github.com/dogum/ML-Ratchet/blob/main/shared/vehicle/roadsim/LIMITS.md) (every table).

A worked example of the ratchet applied to the simulator itself, RoadSim v2 → v2.1 with exam-2 frozen before any
fitting, is on the [mental models](https://dogum.github.io/ML-Ratchet/mental-models/#worked-example) page.

## What it gets wrong

The simulator's measured gaps under the default v2 (FTP fuel economy, + = the simulator burns less). Each is a next
rung in [`LIMITS.md`](https://github.com/dogum/ML-Ratchet/blob/main/shared/vehicle/roadsim/LIMITS.md):

- **TMED P2 hybrids** (Hyundai/Kia, whose motor sits in the transmission) are 8-10 % too thirsty (exam-2 FTP −10.2 %,
  n 53).
- **P2 hybrids with a torque converter** (Ford, BMW, Mercedes, Toyota i-FORCE MAX) are too frugal in the city: FTP
  +18 % on exam-3 (n 9) and +28 % on exam-2 (n 19); +15 to +22 % with the opt-in v3.1.
- **Older Toyota/Lexus hybrids** (pre-TNGA) are too frugal on the FTP: +20 % on exam-2 (n 100), +44 % on exam-3 (n 3);
  +14 % and +31 % with the opt-in v3.
- **Plug-ins on VED roads** use 20-23 % less energy in the simulator than the real cars logged (exam-2 trip median),
  mostly in the cold.
- **Hybrid A/C** is under-predicted: on the SC03 A/C test, exam-2 Toyotas read +18 %, converter P2s +37 %, plug-in
  series hybrids +45 %.
- **Mild hybrids** (Honda IMA, GM BAS): exam-3 FTP +25 % under v2 and +23 % under v3; +8 % under the opt-in v3.1, a
  second look.
- **Heavy duty** rests on two trolleybuses, two Mersin bus types and six truck dynamometer numbers.

Fixed in v2: Honda's e:HEV on the highway (−9.5 → −3.8 %), the conventional city/highway tilt, conventional cold
(held-out 20 °F FTP −5.5 %, n 2; D3 exam cold −9.0 → −4.9 %) and A/C (held-out SC03 +11.4 → +1.9 %, n 2), and the
VED Toyota fuel gap (dev twins −20.3 → −0.9 %).

## How the models did on v2

All three were rebuilt on physics v2 on 14 Nov 2026 (v2.0), every ratchet decision re-taken, and examined once.

- **EnergyNet** ([card](../models/energynet/MODEL_CARD.md)): petrol cars 8.7 % median error per trip against 10.3 %
  for a MOVES-like model; hybrids 13.5 % against 15.3 % (v1.0 lost here); plug-ins 17.1 % against 41 %, given the
  logged start charge; Zurich bus missions 9.5 % against 10.7 %. It loses on the two exam Leafs (22.6 % against
  19.0 %, still 19.5 % low), on EPA petrol FTP, highway and A/C tests to a regression on the car's own spec (A/C 18 %
  low), and on Mersin buses and ICCT trucks to a same-route or same-cycle median.
- **TripDetective** ([card](../models/tripdetective/MODEL_CARD.md)): curb mass within 127 kg per car on 335 real cars
  (physics fit 205 kg, dev median 227 kg); engine size within 0.38 L (dev median 0.60 L); +800 kg now moves its mass
  estimate +524 kg in simulation (v1.0 +182 kg). It loses on powertrain type to a simple rule (97.1 % against 99.4 %),
  on the two Leafs to the dev median (which happens to be their weight class), and on bus mass (1,204 kg against the
  dev median's 453 kg).
- **HybridPilot** ([card](../models/hybridpilot/MODEL_CARD.md)), graded in simulation on real routes: 98.84 % of the
  dynamic-programming optimum on 430 exam trips against 98.32 % for adaptive ECMS and 98.80 % for MPC, which sees 15 s
  ahead. It loses on plug-ins (99.27 % against 99.59 %), on buses to constant-s ECMS (97.50 % against 98.98 %), and on
  charge holding (95 % of trips against 96 %).

## Using it

```bash
pip install -e ".[vehicle]"
mlratchet data vehicle            # rebuild RoadData from the raw downloads (resumable)
cd shared/vehicle/roadsim && python demo.py
```

```python
from roadsim import simulate, load_vehicle, cycle
import realdata as rd
```

Every physics change bumps `PHYSICS_VERSION`, rebuilds each vehicle model's frozen simulated set, and re-takes their
ratchet decisions in order.
