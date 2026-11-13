# EnergyNet v1.0: model card

## What it does
Trip energy before the trip, with a 68% / 95% interval and a split into air drag, rolling, friction braking,
accessories/HVAC and powertrain losses. Energy = fuel chemical energy (fuel g x 0.042 MJ/g) for conventional and
hybrid vehicles, battery DC net discharge for battery-electric, the sum for plug-in hybrids (RoadData's label definitions).
All 7 classes (car, SUV, pickup, van, bus, medium and heavy truck) and 7 powertrains (conventional, P2 / series /
power-split hybrid, series / power-split plug-in, battery-electric).

## Inputs
- Route (1 Hz): speed, grade, temperature, wind speed and direction, heading, air density, road surface.
  Summarised as 90 numbers: duration, distance, stops, a speed x acceleration histogram, grade histogram and climb, weather.
- Road-load physics: 8 integrals (positive/negative wheel energy, aero, rolling, grade up/down, inertia) computed with the
  same force model as RoadSim from the known or imputed mass, CdA and Crr; a generic energy estimate E_phys from them.
- Vehicle (47 numbers, each unknown field masked): architecture, class, mass (exact / reported / unknown), road-load
  A/B/C, engine size, power, diesel, transmission and gears, battery size, window-sticker label (fuel and electric),
  start charge, warm/cold start, OBD-MAF label flag, dynamometer flag.

## Network
Ensemble of 5 MLPs (149 -> 256 -> 256 -> 256 -> 128 -> 7, tanh-GELU, 204k weights each). Outputs: log(E / E_phys),
log sigma, and 5 log energy components. Loss: Gaussian NLL + 0.2 x component MSE. Ensemble sigma includes the spread of
the members. Numpy inference; the browser runs a float16 copy (median difference 8e-5, max 5e-4; JS == numpy to 1e-14).

## Simulator and training data
RoadSim (Phase 1, frozen). 2,400 training and 400 validation vehicles from `roadsim.generator`, widened in EnergyNet's
wrapper: engine efficiency 0.68-1.08 (dev twins of 2005-2010 VED cars needed 0.72-1.07), road load +/-10-12%, older
4-6 speed automatics, BEV motor loss scale x0.7-1.5, 35% articulated buses. Each light-duty vehicle got simulated
window-sticker labels (FTP-75 approx + HWFET, EPA derived 5-cycle formulas; EV wall energy /0.88, x0.7).
Routes: 1,092 VED train_noise routes (unlabelled cars; real speed, lidar grade, airport weather; weather redrawn 40% of
the time), dynamometer cycles (dyno conditions or on-road weather/grade), synthetic drives (urban, suburban, highway,
congested, hilly, bus, delivery, long-haul; stops, hills, heading, weather). Real-noise analogue: the network sees
station weather while the simulated car drives T + N(0, 1.5 C) and wind x U(0.4, 0.9).
Banks: 28,800 training examples (12 per vehicle), 3,991 frozen validation examples (400 unseen vehicles).
Spec masking: tiers redrawn every epoch, from EPA-like full specs to class + architecture only, with label noise 2-12%.

## Dev calibration (twin tests on the 16 dev cars, never the exam)
| stage | petrol bias | hybrid bias | electric | plug-in series / split |
|---|---|---|---|---|
| generic spec | -28% | -22% | -5% | +7% / +37% |
| label-matched twin (engine efficiency + road load fitted to the car's unadjusted lab mpg) | -17% | -20% | | |
| + MAF label model | -6% | -20% | | |
| + on-road loads (fitted here, in-sample) | 0% | 0% | 0% | |
Wrappers (in `training/wrappers.py`; RoadSim not edited, to be upstreamed by the coordinator):
- **maf**: OBD MAF fuel estimate during fuel cut-off: f = exp(-1.578) D^0.342 (rpm/1000)^0.226 g/s (10,273 dev seconds).
- **rw** (on-road loads): extra power c + k_ac clip(T-15, 0, 15) + k_cold clip(5-T, 0, 25): conv 1380 W, 133 W/C, 20 W/C
  (fuel power); hybrids 2831 W, 204 W/C, 25 W/C; BEV 246 W, 0, -43 W/C (battery); scaled by cabin UA/120; not on dyno tests.
- **phev_patch**: plug-in pack power >= 1.2 x traction motor, resistance /2.5, power-split MG2 torque x1.8 (dev plug-ins
  burned no fuel on 95% of trips above 30% charge; simulated ones took 31% of energy from fuel).

## Ratchet (frozen val set re-scored at the current physics version v4; margin 0.01; dev gate +0.01 on D)
Skill = 1 - median |log error| / median |log error| of the class-median guess. D = mean of 15 dev MdAPEs (VED 4 fleets,
EPA 7 tests, ZTBus dev legs, Mersin dev diesel/CNG, ICCT dev truck).
| rung | change | val skill | val MdAPE | dev D | decision |
|---|---|---|---|---|---|
| r1_abs | MLP predicts log energy directly | 0.544 | 11.2% | 0.152 | kept |
| r2_corr | predict a correction to the physics estimate | 0.556 | 11.0% | 0.160 | kept |
| r3_maf | MAF label model + flag input (physics v2) | 0.580 | 10.3% | 0.152 | kept |
| r4_rw | on-road loads (physics v3) | 0.641 | 9.0% | 0.142 | kept |
| r5_phev | plug-in hardware patch (physics v4) | 0.648 | 8.9% | 0.129 | rejected (+0.007 < 0.01) |
| r6_ens | 5-network ensemble on v4 physics | 0.656 | 8.7% | 0.129 | **kept, released** |

## Exam (real, never used for choosing; median absolute % error)
VED trips (RoadData baselines fitted on dev):
| fleet | trips (cars) | EnergyNet | label x distance | MOVES-like | class median | car totals: EnergyNet / label / MOVES-like |
|---|---|---|---|---|---|---|
| petrol | 14,469 (240) | **8.9%** | 13.0% | 10.3% | 22.8% | **6.8%** / 8.6% / 9.6% |
| hybrid (84 power-split + 2 P2) | 8,241 (86) | 16.7% (bias +12%) | 16.2% | **15.3%** | 17.2% | 16.2% / **8.9%** / 12.7% |
| electric (2 Leafs) | 167 (2) | 22.8% (bias -20%) | 21.5% | **19.0%** | 27.1% | 17.3% / **15.1%** / 16.6% |
| plug-in | 2,916 (22) | **20.1%** | 46.1% | 41.3% | 42.7% | **8.9%** / 14.4% / 35.7% |
P2 hybrids alone (247 trips): 9.0% vs 18.3% MOVES-like. Plug-in series 14.4%, power-split 23.9%. The plug-in result
uses the logged start charge as an input; the baselines do not use it.

EPA exam families (per vehicle from its spec; cycle labels corrected, see below; regression = RoadData's log-linear
model refitted on dev with the corrected labels):
| test | n | EnergyNet | bias | regression | dev median |
|---|---|---|---|---|---|
| petrol FTP | 20,065 | 9.6% | +2% | **7.1%** | 18.7% |
| petrol HWY | 20,426 | 15.6% | +15% | **5.0%** | 16.8% |
| petrol US06 | 2,832 | 11.5% | +11% | **6.9%** | 17.3% |
| hybrid FTP | 1,098 | **8.7%** | -4% | 16.3% | 49.0% |
| hybrid HWY | 1,156 | **5.9%** | +1% | 21.5% | 20.1% |
| hybrid US06 | 649 | **14.9%** | +15% | 15.8% | 29.2% |
| BEV city (UDDS, wall kWh/mi) | 746 | **7.9%** | -1% | 9.8% | 19.0% |
| BEV highway | 746 | **5.0%** | +2% | 6.5% | 11.3% |

Heavy duty:
| set | n | EnergyNet | baseline |
|---|---|---|---|
| ZTBus trolleybus legs | 78 | **16.4%** (bias -16%) | 22.1% (dev kWh/km x distance) |
| ZTBus missions (route proxy: traced legs of the same line at the mission's temperature) | 1,407 | 16.2% | **14.5%** |
| Mersin diesel bus trips (synthetic speed profiles from stops and duration) | 652 | 14.8% | **8.6%** (dev median, same route) |
| Mersin CNG bus trips | 672 | 15.5% | **7.1%** |
| ICCT tractor-trailers (2 trucks x 3 cycles) | 6 | 11.4% | **7.0%** (dev truck on the same cycle) |

Coverage after dev widening (factors: petrol 1.47, hybrid 1.48, EV 2.55, plug-in 2.55, EPA 1.03-1.17, Mersin 1.49,
ZTBus 1.0, ICCT 2.87): VED inside 68% band 55% / 65% / 47% / 65% (petrol/hybrid/EV/plug-in), inside 95% 85-89%;
EPA petrol FTP 77% / 98%, highway 54% / 94%; ZTBus 53% / 94%; Mersin diesel 78% / 100%. Bands are too narrow on VED.

Breakdowns (VED, EnergyNet vs MOVES-like): petrol wins in every temperature, speed, grade and length band (worst
below -5 C: 10.7% vs 12.1%). Hybrids lose below 25 km/h (27% vs 20%) and above 25 C (21% vs 14%). Plug-ins are worst
below -5 C (37.5%). Full tables in `training/exam.json`.

## Limits
- Hybrids and the two exam Leafs lose to simple baselines; calibration rests on 5 dev hybrids and 1 dev Leaf.
- Petrol EPA highway/US06 biased +11-15%: the fleet prior was widened toward older engines to fit VED.
- Intervals too narrow on real VED trips despite dev widening.
- Plug-in accuracy depends on the start-charge input.
- Heavy duty is thin and partly synthetic (Mersin speed profiles, ZTBus mass unknown, CNG engines out of the training range).
- VED fuel is a MAF-based estimate; Ann Arbor 2017-18 only; mostly warm fragments (cold starts barely tested).
- Not modelled from dev: cold-weather start penalty (RoadSim LIMITS #4), P2 city optimism (no dev P2 cars).

## Data issue found
RoadData's EPA wide table maps test procedures to cycles wrongly: 'HWFE' tests are labelled FTP and 'Federal fuel 2-day
exhaust (w/can load)' (the main FTP-75 procedure) is labelled HWY, so `FTP_mpg` / `HWY_mpg` are mostly swapped
(conv medians 35.6 / 24.5 mpg). `training/epa_fix.py` rebuilds them from procedure names (medians 24.6 / 37.7) without
touching data/. RoadData's published EPA baseline numbers (8.8% / 7.0%) refer to the swapped labels.

## Data and licences
VED (Oh, LeBlanc, Peng, IEEE T-ITS 2020; Apache-2.0). US EPA test-car lists, dynamometer schedules, ALPHA maps,
fueleconomy.gov, USGS 3DEP lidar, IEM ASOS weather, eCFR 40 CFR 1037 (public domain / free use). ZTBus (Widmer et al.,
ETH Zurich; CC-BY-4.0). Mersin 33M (Abut, Zenodo; CC-BY-4.0). ICCT 2018 tractor-trailer tests (Rodriguez, Delgado,
Muncrief). Code, weights, docs: MIT (ML Ratchet).
