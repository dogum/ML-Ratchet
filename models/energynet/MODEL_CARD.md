# EnergyNet v2.1: model card

## What it does
Trip energy before the trip, with a 68% / 95% interval and a split into air drag, rolling, friction braking,
accessories/HVAC and powertrain losses. Energy = fuel chemical energy (fuel g x 0.042 MJ/g) for conventional and
hybrid vehicles, battery DC net discharge for battery-electric, the sum for plug-in hybrids (RoadData's label definitions).
All 7 classes (car, SUV, pickup, van, bus, medium and heavy truck) and 7 powertrains (conventional, P2 / series /
power-split hybrid, series / power-split plug-in, battery-electric).

v2.0 (14 Nov 2026) is v1.0 rebuilt on RoadSim physics v2 with every ratchet decision re-taken. v2.1 (15 Nov 2026)
fixes two generator misses that dev twin reads showed (the A/C and cold tests ran without the EPA test conditions;
30% of the simulated petrol cars had idle-stop that RoadSim's dev-calibrated twins do not) and reads two public specs (an
EPA test hybrid's system, the trolleybus's curb mass). v2.0 and v1.0 re-scored on the same exam are in the tables below;
v1.0's own card numbers are in the history of this file and in `training/exam.json`, v2.0's run in `training/exam2.json`.

## Inputs
- Route (1 Hz): speed, grade, temperature, wind speed and direction, heading, air density, road surface.
  Summarised as 90 numbers: duration, distance, stops, a speed x acceleration histogram, grade histogram and climb, weather.
- Road-load physics: 8 integrals (positive/negative wheel energy, aero, rolling, grade up/down, inertia) computed with the
  same force model as RoadSim from the known or imputed mass, CdA and Crr; a generic energy estimate E_phys from them.
- Vehicle (49 numbers, each unknown field masked): architecture, class, mass (exact / reported / unknown), road-load
  A/B/C, engine size, power, diesel, transmission and gears, battery size, window-sticker label (fuel and electric),
  start charge, warm/cold start, OBD-MAF label flag, dynamometer flag, **model year** (new in v2).
- Public specs used at prediction time (v2.1): an EPA test-car hybrid is given its hybrid system from the model line
  (P2 / power-split / series, `training/epa_hybkind.py`); a ZTBus trolleybus its published 19 t curb mass as a reported mass.

## Network
Ensemble of 5 MLPs (151 -> 256 -> 256 -> 256 -> 128 -> 7, tanh-GELU, 204k weights each). Outputs: log(E / E_phys),
log sigma, and 5 log energy components. Loss: Gaussian NLL + 0.2 x component MSE, 60 epochs, trained on an RTX 3080.
Ensemble sigma includes the spread of the members. Numpy inference; the browser runs a float16 copy (energy within
1.3e-4 of float64; v2.1: energy within 5.6e-5, JS == numpy to 1.9e-7 on 32 cases).

## Simulator and training data
RoadSim **physics v2**, pinned (`training/physics.py`: `options='roadsim-physics-v2'`, generator donors 'v2'). v2 carries
what v1 kept in this folder: an on-road extra load per architecture and the OBD MAF label model (`roadsim.obd`).
- 2,400 training and 400 validation vehicles from `roadsim.generator`. Each light-duty vehicle gets a model year
  (2005-2026) and an engine efficiency from an era prior fitted on fueleconomy.gov **dev** families (`training/era.py`:
  fuel per mile at fixed engine size, class and transmission; MY2005 x1.21, MY2011 x1.10, MY2017 x1.005 vs MY2021;
  residual sd 0.09). It replaces v1's flat 0.68-1.08 widening; the dev twins' fitted efficiencies agree. Road load
  +/-10-12%, older 4-6 speed automatics, BEV motor loss x0.7-1.5, 35% articulated buses (as v1).
- Window-sticker labels: FTP-75 approximation + HWFET on a dynamometer spec (no on-road load, HVAC off), EPA derived
  5-cycle formulas.
- Routes: 1,092 VED train_noise routes, dynamometer cycles (cold starts now included: v1's bank never started a dyno
  cycle cold), synthetic drives. The network sees station weather; the car drives T + N(0, 1.5 C), wind x 0.4-0.9.
- Targets: RoadSim's OBD label model for MAF-estimated fuel (rung r5b), v1's plug-in hardware patch on the generated
  plug-ins (pack power >= 1.2 x motor, resistance / 2.5, MG2 torque x 1.8; rung r5b).
- Banks: 86,400 training trips (three banks of 12 per vehicle, seeds 101-103), 4,000 frozen validation trips (val2).
- v2.1 (rung r17 = r14 + r15 + the two public-spec inputs): the SC03 and 20 F cold-FTP dyno draws run under the EPA test
  conditions (`training/dynoproc.py`, from `roadsim.epa`: SC03 at 850 W/m2 with A/C maximum on recirculated air at 72 F,
  a preconditioning drive and the cabin soaked to 45 C; cold FTP heat at 72 F on outside air), and the generated
  conventional light-duty cars have no idle-stop (`training/bank_idlestop.py`). The affected rows (1,359 dyno rows; every
  row of 247 cars, 8,892 rows) were re-simulated on their own trips and start states (`bank_replay.py`); the other rows
  are unchanged. The old SC03 draw missed 12% of a conventional car's fuel (`check_dynoproc.py`); idle-stop takes 7%
  off a hot UDDS and 0.1% off a HWFET.

## Physics checks
From RoadSim's own checks (`shared/vehicle/roadsim/LIMITS.md`, physics v2; held-out data):
- EPA held-out vehicles: conventional FTP +0.4 % / HWFET -1.9 % (n 15), BEV UDDS -0.4 % / HWY -1.8 % (n 8) on fuel
  economy (+ = the simulator burns less). Known gaps: older Toyota/Lexus hybrids too frugal, TMED P2s 8-10 % too
  thirsty, plug-ins on VED roads 20-23 % low (Limits below).
- Energy ledger closes in all 600 stress runs (max 5.6e-4, none over 0.5 %).
- RoadSim's bus twin on the 5,847 full-ZTBus exam legs: -2.3 % median energy (nothing fitted).

Inference parity: the browser's float16 copy is within 1.3e-4 of float64 numpy, and the JS port equals numpy to
1.3e-7 (32 cases). No numpy-vs-torch parity number is recorded in `training/`.

## Twin reads on dev (what found the v2.1 changes)
RoadSim's own twins of 235 dev EPA petrol cars (`shared/vehicle/checks/epa_pool__roadsim-physics-v2.json`) against the
network on the same cars (`training/twin_read_epa.py`); fuel, + = more than the test:

| dev test | RoadSim twin | v2.0 network | v2.0 network / twin | v2.1 network / twin |
|---|---|---|---|---|
| FTP | +1.4% | -3.9% | -6.3% | -4.5% |
| HWY | +1.4% | +2.8% | +0.3% | -0.8% |
| 20 F cold FTP | -2.5% | -9.7% | -6.8% | -5.0% |
| SC03 (A/C) | +0.3% | -13.3% | -14.2% | -0.8% |

## Ratchet (frozen set val2; margin 0.01; dev score D2)
Skill = 1 - median |log error| / median |log error| of the class-median guess. D2 = mean of 18 dev MdAPEs (VED 4 fleets;
EPA conventional FTP / HWY / US06 / 20 F cold FTP / SC03, hybrids FTP / HWY, BEV city / HWY; full-ZTBus dev legs and
294 dev missions; Mersin diesel / CNG; ICCT). v1.0 scores D2 = 0.1307.
Rule: model rungs need +0.01 skill and D2 no worse than +0.01. **Changed in the open after r4:** a physics rung (new
simulated targets or labels) must improve D2 by more than 0.01 (judging it on its own frozen set favours it by
construction: r4 gained +0.097 skill while D2 got worse). Gate-only rungs (inputs only real data carries) must improve
D2 by more than 0.01.

| rung | change | judged on | val2 skill | val2 MdAPE | dev D2 | decision |
|---|---|---|---|---|---|---|
| r1 | MLP predicts log energy | a | 0.614 | 9.0% | 0.1376 | kept |
| r2 | correction to the physics estimate | a | 0.629 | 8.6% | 0.1327 | kept |
| r3 | RoadSim's OBD label model + flag | b | 0.630 | 8.6% | 0.1234 | rejected (D2 -0.0093) |
| r4 | v1's on-road wrapper on top of v2 | c | 0.631 | 8.3% | 0.1378 | rejected (first rule: kept) |
| r5 | v1's plug-in hardware patch | f | 0.626 | 8.7% | 0.1228 | rejected (D2 -0.0100) |
| r5b | r3 + r5 together | e | 0.632 | 8.6% | 0.1225 | kept |
| r6 | plain 5-network ensemble | e | 0.636 | 8.4% | 0.1259 | rejected (+0.004) |
| r7 | 5 networks reading the model year | e | 0.651 | 8.1% | 0.1260 | kept |
| r8 | EPA hybrids told their hybrid system (gate-only) | e | 0.651 | 8.1% | 0.1195 | rejected (D2 -0.0065) |
| r9 | ZTBus told its 19 t curb mass (gate-only) | e | 0.651 | 8.1% | 0.1239 | rejected |
| r9b | r8 + r9 (gate-only) | e | 0.651 | 8.1% | 0.1174 | rejected (D2 -0.0086) |
| r10 | labels from RoadSim's EPA procedure emulation | e+L | 0.640 | 8.3% | 0.1277 | rejected |
| r11 | 2x data, 512-wide networks, 100 epochs | e | 0.632 | 8.6% | 0.1219 | rejected |
| r11b | 2x data (57,600 trips) | e | 0.660 | 7.9% | 0.1195 | rejected (+0.009) |
| r12 | error bars: simulation factor per architecture, then dev factors | — | — | — | — | kept (leave-one-car-out 0.064 vs 0.072) |
| r13 | 3x data (86,400 trips) | e | 0.670 | 7.7% | 0.1195 | kept, released as v2.0 |
| r14 | SC03 / 20 F cold FTP under the EPA test conditions | g | 0.669 | 7.6% | 0.1137 | rejected (D2 -0.0058) |
| r15 | r14 + no idle-stop on generated petrol cars | h | 0.680 | 7.5% | 0.1153 | rejected (D2 -0.0042) |
| r16 | v2.0's networks + r8 + r9 inputs (gate-only) | e | 0.670 | 7.7% | 0.1140 | rejected (D2 -0.0055) |
| r17 | r15's networks + r8 + r9 inputs (planned before r15 was scored) | h | 0.680 | 7.5% | 0.1077 | **kept, released as v2.1** (D2 -0.0118) |

Versions of val2: a RoadSim v2 raw; b + OBD label model; c + v1's on-road wrapper; e b + plug-in patch; f a + plug-in
patch; e+L e with EPA-procedure labels; g e with the SC03 / cold-FTP rows under the EPA test conditions; h g without
idle-stop. Raw log: `training/ratchet2.json`. The D2 terms that a rung cannot touch move by 1-3 points between two sets of
five networks, about 0.005 on D2: half the margin. r17's dev gains are SC03 20.5 -> 9.6%, EPA hybrids FTP / HWY 18.5 /
13.4 -> 15.7 / 10.0%, ZTBus legs / missions 9.0 / 9.5 -> 7.0 / 5.6%; it is worse on dev for petrol FTP / HWY (8.8 / 6.6
-> 9.3 / 7.0%), BEV city (7.9 -> 8.9%), Mersin (12.3 / 15.9 -> 13.0 / 17.0%) and ICCT (18.1 -> 19.0%).

## Exam (real, never used for choosing; run once after the release choice was committed; median absolute % error)
v2.1 and v2.0 (re-scored in the same run, its own error bars; reproduces its card) on the same records
(`training/exam3.json`).

VED trips (RoadData baselines fitted on dev):
| fleet | trips (cars) | EnergyNet v2.1 | v2.0 | v1.0 | label x distance | MOVES-like | class median | car totals v2.1 / v2.0 / label |
|---|---|---|---|---|---|---|---|---|
| petrol | 14,469 (240) | **8.6%** (bias +0.3%) | 8.7% | 9.1% | 13.0% | 10.3% | 22.8% | **7.4%** / 7.5% / 8.7% |
| hybrid | 8,241 (86) | **13.4%** (+3.8%) | 13.5% | 15.5% | 16.2% | 15.3% | 17.2% | **7.2%** / 7.5% / 8.9% |
| electric (2 Leafs) | 167 (2) | 23.2% (-20.3%) | 22.6% | 24.5% | 21.5% | **19.0%** | 27.1% | 19.1% / 19.1% / **15.1%** |
| plug-in | 2,916 (22) | **16.8%** (-9.4%) | 17.1% | 19.6% | 46.1% | 41.3% | 42.7% | 20.6% / 20.1% / **14.4%** |
By architecture: power-split hybrids 13.3% (v2.0 13.3%, MOVES-like 15.1%); P2 hybrids (247 trips) 18.3%, biased -18%
(v2.0 17.2%, v1.0 10.4%, MOVES-like 18.3%); plug-in power-split 16.7% (v2.0 16.9%), plug-in series 17.1% (v2.0 17.7%).
The plug-in result uses the logged start charge as an input; the baselines do not use it.

EPA exam families (per vehicle from its spec; regression = RoadData's log-linear model refitted on dev):
| test | n | EnergyNet v2.1 | bias | v2.0 | v1.0 | regression | dev median |
|---|---|---|---|---|---|---|---|
| petrol FTP | 20,065 | 8.6% | -5.0% | 8.9% | 9.6% | **7.1%** | 18.7% |
| petrol HWY | 20,426 | 6.8% | -0.8% | 6.7% | 15.6% | **5.0%** | 16.8% |
| petrol US06 | 2,832 | **5.2%** | +2.0% | 5.1% | 11.5% | 6.9% | 17.3% |
| petrol 20 F cold FTP | 1,734 | **8.5%** | -6.1% | 9.5% | 7.5% | 9.7% | 24.5% |
| petrol SC03 (A/C) | 1,819 | **8.6%** | -6.2% | 18.9% | 15.6% | 9.3% | 21.7% |
| hybrid FTP | 1,098 | **8.7%** | -8.3% | 10.2% | 8.7% | 16.3% | 49.0% |
| hybrid HWY | 1,156 | **5.6%** | +4.2% | 6.0% | 5.9% | 21.5% | 20.1% |
| hybrid US06 | 649 | **10.9%** | +10.8% | 12.7% | 14.9% | 15.8% | 29.2% |
| hybrid SC03 | 633 | 11.4% | -11.2% | 18.6% | 23.1% | — | — |
| BEV city (wall kWh/mi) | 746 | **9.5%** | -9.5% | 8.8% | 7.9% | 9.8% | 19.0% |
| BEV highway | 746 | **4.7%** | -2.0% | 4.6% | 5.0% | 6.5% | 11.3% |

Heavy duty:
| set | n | EnergyNet v2.1 | v2.0 | v1.0 | baseline |
|---|---|---|---|---|---|
| ZTBus mission totals (full traces: every leg + layover) | 1,115 | **5.6%** (bias -1.5%) | 9.5% (-8.9%) | 14.6% (route proxy: 16.0%) | 10.7% (dev kWh/km x distance, 294 dev missions) |
| ZTBus legs | 5,847 | **7.0%** (+0.3%) | 8.6% | 12.5% | 14.0% |
| Mersin diesel bus trips | 652 | 17.2% (bias -16.7%) | 17.0% | 14.8% | **8.6%** (dev median, same route) |
| Mersin CNG bus trips | 672 | 19.2% (-18.8%) | 18.0% | 15.5% | **7.1%** |
| ICCT tractor-trailers (2 trucks x 3 cycles) | 6 | 10.7% | 10.0% | 11.4% | **7.0%** (dev truck, same cycle) |

Worse than v2.0: Leafs (22.6 -> 23.2%), petrol HWY (6.7 -> 6.8%) and US06 (5.1 -> 5.2%), BEV city (8.8 -> 9.5%), P2
hybrids on VED (17.2 -> 18.3%), plug-in car totals (20.1 -> 20.6%), Mersin (17.0 -> 17.2%, 18.0 -> 19.2%), ICCT (10.0 ->
10.7%). Row verdicts that changed: petrol SC03 is now a win (8.6% vs 9.3%; v2.0 lost at 18.9%).

Coverage (68% / 95% band): VED pooled 59% / 86% (v2.0 61% / 87%); petrol 55 / 84, hybrid 65 / 90, Leafs 41 / 71,
plug-in 64 / 84; EPA petrol FTP 64%, highway 75%, SC03 58% (v2.0 20%); ZTBus missions 91% / 100% (too wide); Mersin 53-58%.
Error-bar factors (dev only, `sigma_factors.json`, r12's method refitted): per architecture from the simulated set
0.97-1.24, then per group petrol 1.50, hybrid 1.0, EV 1.44, plug-in 2.18, EPA 1.11-1.38, ZTBus 1.0, Mersin 1.67, ICCT 3.03.

Phase D targets: beat MOVES-like on hybrids — **met** (13.4% vs 15.3%); on EVs — **not met** (23.2% vs 19.0%); 68% bands
cover 60-75% of VED exam trips — **not met** (59% pooled; petrol 55%, Leafs 41%; v2.0 met it at 61%).

## Limits
- The two exam Leafs stay ~20% low; the one dev Leaf is +4.6%. Old BEVs using more than their twins (RoadSim LIMITS) is
  the likely cause; nothing in dev can fit it.
- A/C: EPA SC03 is still 6% low on petrol cars and 11% on hybrids.
- EPA petrol city: the network reads RoadSim's own dev twins 4.5% low on the FTP and 0.8% low on the highway; the
  remaining generator difference is not found.
- P2 hybrids on VED and plug-in car totals are worse than v1.0.
- The dev score moves by about 0.005 between two sets of five networks; a one-group fix is judged mostly by that noise.
- Buses on Mersin (synthetic speed profiles) and the trucks lose to same-route baselines.
- Hybrid and EV calibration rests on 5 dev hybrids and 1 dev Leaf. VED fuel is a MAF-based estimate; Ann Arbor 2017-18 only.
- RoadSim v2 known gaps (LIMITS.md): older Toyota/Lexus hybrids too frugal on the FTP, TMED P2s too thirsty, plug-in series
  engine-on too low; the network learns around them only where dev data shows them.

## Data and licences
VED (Oh, LeBlanc, Peng, IEEE T-ITS 2020; Apache-2.0). US EPA test-car lists, dynamometer schedules, ALPHA maps,
fueleconomy.gov, USGS 3DEP lidar, IEM ASOS weather, eCFR 40 CFR 1037 (public domain / free use). ZTBus (Widmer, Ritter,
Onder, Scientific Data 2023, ETH Zurich, doi:10.3929/ethz-b-000626723; CC-BY-4.0). Mersin 33M (Abut, Zenodo; CC-BY-4.0). ICCT 2018 tractor-trailer
tests (Rodriguez, Delgado, Muncrief). Code, weights, docs: MIT (ML Ratchet).
