# TripDetective v2.0: model card

**What it does.** Reads one or more 1 Hz trip logs of a light-duty vehicle (speed plus whatever an OBD logger has:
fuel rate, rpm, HV battery current/voltage/SOC, A/C power, with route grade and weather) and infers: architecture
(7 layouts), total mass, engine displacement, CdA, Crr, hybrid control strategy and on/off behaviour, and each trip's
energy ledger (aero, rolling, grade, friction braking, powertrain losses, auxiliaries, kJ/km). Every output has a
1-sigma error bar. Trips of the same car are pooled into one answer. A second network (`TripDetectiveBus`) reads the
mass of a city bus from its power log.

**Trained only on simulation (RoadSim physics v2). Graded on 350 real VED cars and 5,847 real ZTBus legs never used
for choosing anything.** v2.0 was rebuilt on 14 Nov 2026 (Phase D); v1.0 (12 Nov 2026, physics v1) is described at
the end and in `NOTEBOOK.md`.

## Inputs / outputs
- Input per second: v, acceleration, grade, T_amb, headwind (wind x cos(wind_dir - heading)), air density, fuel rate,
  rpm, battery power, SOC, aux power, validity, channel-availability flags, three road-load power channels
  (v(1.03a + g grade), 0.5 rho v_air|v_air| v, g v) and, new in v2.0, a wheel-equivalent source power
  (0.30 x fuel chemical power + 0.90 x battery power). Per trip window: 16 classical numbers (as v1.0) and the
  normal equations of a road-load regression of the source power (15 numbers, see the physics head).
- Windows of at most 1,024 s; longer trips are split and pooled.
- Output: see README. Mass is the total moving mass; curb = total - 136 kg (RoadData's convention).
- Bus: `TripDetectiveBus().read_leg(leg)` with v, total DC power (`power_W`), grade, T_amb, rho_air -> mass with a 1σ range.

## Physics and simulator
RoadSim **physics v2** (`options='roadsim-physics-v2'`, pinned in `training/common.py`): the v1 road loads plus a
tyre model, ALPHA engine maps with a per-map efficiency tilt, P2 converter and start energy, Toyota production rules,
a two-node HVAC cabin, on-road extra loads; all fitted on dev data (shared/vehicle/roadsim/CALIBRATION.md). Known v2
gaps (LIMITS.md): older Toyota/Lexus hybrids ~17 % too frugal on the FTP, TMED P2 ~10 % too thirsty, converter P2s too
frugal in the city, plug-in series engine-on too low. Checks: road-load force within 1.6/0.8/0.5 % of 47 EPA
coast-down targets; ledger closure max 5.6e-4 over 600 stress runs.

The OBD logger emulator now lives upstream (`roadsim/obd.py`, EMU_VERSION 1: this model's v1.0 `obdlog.py` moved there
unchanged); `training/obdlog.py` re-exports it and adds only the VED channel mix of rung p2r9.

## Simulated data in real conditions
- Routes: 1,154 real Ann Arbor fragments (17 unlabelled VED `train_noise` cars, lidar grade, ASOS weather), windows of
  10 light-duty cycles, synthetic stop-and-go; the car sees slightly different conditions than the log reports.
- Prior v4 (released): the RoadSim generator as is plus v1.0's VED-era mass prior (rejection sampling to 85 %
  lognormal(median 1,750 kg, sigma 0.20) + 15 % generator fleet; class mix car 0.60 / SUV 0.30 / pickup / van 0.05).
- Training bank: 2,000 cars x 8 trips (prior v4). Frozen set p2_val_v6: 600 cars x 8 trips (seed 99).
- Buses: 1,500 simulated battery-electric buses (curb 14-24 t log-uniform, 0-110 passengers x 70 kg, CdA 5-8 m2,
  Crr 0.006-0.010, 250-400 kW, accessories 2-8 kW, PTC 6-24 kW) x 6 windows, on ZTBus dev routes (the 'route' half of
  the dev missions; only speed, grade and temperature used), CBD14 and synthetic routes; shown to the network as a
  scaled car (power / 12, mass / 12, CdA / 12).

## Dev calibration (16 dev VED cars; ZTBus dev legs)
- Twin test on physics v2 (`twin.py`): with the plain generator the twins burn 0.95x the real fuel/km (conventional)
  and 1.03x (HEV). v1.0's dev-twin wrappers (priors v2: engine eff_scale 0.88, idle, gearing, HVAC) now over-correct
  (1.07x / 1.11x): physics v2 carries VED-era engine efficiency itself. So the priors were rebuilt as v1 + the v3 mass
  prior (prior v4), as v1.0's notebook proposed.
- Dev Leaf (VED_455, 64 trips): logs 1.5-1.8 kW of heater power below 5 C and 0.15-0.2 kW at 5-15 C; its v2 twin with
  the generator's HVAC drew 2.3-3.0 kW at 5-15 C (energy 1.44x real). Set point 17 C, heat below 3 C, 2 kW, 70 %
  recirculation and series regeneration bring the twin to 1.06x (IQR 0.96-1.15). As a prior (v5) it was rejected
  twice by the ratchet (sim +0.000).
- VED channel mix (dev): conventional and HEV loggers log fuel and rpm and never an HV battery; plug-ins and the Leaf
  log the battery and A/C. As a logger prior (v6) it was rejected twice (sim +0.001 and -0.006).
- Physics-head twin check: on dev twins the pooled road-load coefficient b0 of the real log is 0.83-1.37x the twin's
  (median 1.01); the Leaf 0.83x.
- ZTBus: on 40 dev route-half legs the logged power and the logged traction force both regress to 0.67-0.73 x
  (19 t + 70 kg x passengers), against 0.96 for the RoadSim bus twin.

## Network
1-D CNN (stem k7 + 5 stride-2 blocks, 32-160 channels, BatchNorm, GELU, activations masked outside the trip),
masked mean+max pooling, trip MLP (with the 16 classical numbers) -> 128-d trip embedding; trip embeddings of one car
are averaged (+log count). **Physics-regression head (v2.0):** each window's normal equations of
`P_src = b0 v (1.03 a + g grade) + b1 g v + b2 0.5 rho v_air|v_air| v + b3` (traction seconds, 5 s smoothing) are summed
over the car's trips and solved (ridge 1e-3); the 8 fit features join the car MLP, and the mass output is
`log(b0 / 1800) x skip + head`, so the mass moves with the fit unless the network learns otherwise (skip starts at
1). Heads: architecture (7), strategy (14), 7 regressions (mean, log-sigma; Gaussian NLL); per-trip ledger head.
0.40 M parameters. Trained 2,500 steps of 24 cars x 1-8 trips on the GPU (RTX 3080). numpy == torch to 4.6e-6;
browser (site instrument) == Python to 1.6e-5.

## Ratchet v2.0 (physics v2; frozen sets p2_val_v3 -> v4 -> v5 -> v6; every rung re-scored in order on p2_val_v6; rule as v1.0)

Margin +0.01; dev gate: mean of 4 dev error ratios <= 1.05 and VED-type accuracy drop <= 0.03. Mass response = median
change of the pooled estimate for +800 kg on 64 simulated cars (8 trips; info only).

| rung | change | sim | dev mass/car | dev engine/car | dev type/trip | +800 kg reads | decision |
|---|---|---|---|---|---|---|---|
| p2r0 | CNN + pooling, generator as published, 1,200 cars | 0.378 | 156 kg | 0.39 L | 0.85 | +257 | kept |
| p2r1 | v1.0's dev-twin priors (v2) | 0.363 | 106 | 0.60 | 0.91 | +202 | rejected (dev +14 %) |
| p2r2 | v2 priors, 2,000 cars | 0.377 | 123 | 0.55 | 0.89 | +222 | rejected |
| p2r3 | + road-load physics channels | 0.374 | 103 | 0.56 | 0.85 | +215 | rejected |
| p2r4 | + 16 classical estimates | 0.386 | 125 | 0.67 | 0.91 | +275 | rejected (dev +17 %) |
| p2r5 | classical estimates on the v1 bank | 0.397 | 134 | 0.46 | 0.93 | +331 | kept |
| p2r6 | v3 mass prior + classical (v1.0's recipe) | 0.405 | 122 | 0.51 | 0.89 | +227 | rejected (type -0.043) |
| p2r7 | priors rebuilt: v1 + v3 mass prior (v4) | 0.430 | 126 | 0.43 | 0.92 | +295 | kept |
| p2r8 | BEV cabin heat (v5) | 0.430 | 102 | 0.35 | 0.91 | +226 | rejected (+0.000) |
| p2r9 | VED channel mix (v6) | 0.431 | 121 | 0.32 | 0.93 | +360 | rejected (+0.001) |
| p2r10 | source-power channel (features v4) | 0.432 | 116 | 0.37 | 0.90 | +257 | rejected |
| **p2r11** | **physics-regression head** | **0.450** | **133** | **0.31** | **0.91** | **+524** | **released** |
| p2r12 | pooled multi-trip training | 0.452 | 122 | 0.27 | 0.88 | +519 | rejected |
| p2r13 | GPU: wider CNN, 8,000 steps | 0.491 | 142 | 0.33 | 0.91 | +571 | rejected (dev +6 %) |
| p2r14 | VED channel mix + head | 0.444 | 121 | 0.25 | 0.91 | +503 | rejected |
| p2r15 | BEV heat + head | 0.450 | 128 | 0.32 | 0.90 | +485 | rejected |

Bus ladder (frozen set: 300 buses x 6 windows; dev gate: 757 ZTBus dev legs, level-corrected error <= 5 % worse and
passenger slope not more than 0.1 lower; the first rule on the raw error was replaced in the open after pb2, see
ratchet.yaml): pb1 bus network 0.524, dev error 1,127 kg raw / 805 kg level-corrected, slope 0.42 - released; pb2 +
physics head 0.546, reads 4.8 t light, level-corrected 2,731 kg, slope 1.25 - rejected.

## Exam (run once, 14 Nov 2026, after the release was committed; baselines fitted on dev only)

| | v2.0 | v1.0 (same exam) | classical | dumb |
|---|---|---|---|---|
| VED type, per car (350) | 97.1 % (bal. 97.1 %) | 98.9 % | rule 99.4 % | majority 68.6 % |
| VED type, per trip (10,186) | 91.5 % | 92.9 % | rule 96.7 % | 63.1 % |
| curb mass per car (335), median / mean abs error | **127 / 153 kg**, bias -14 | 116 / 139 kg | physics LSQ 205 / 244 | dev median 227 / 230 |
| curb mass per trip, median abs error | **112 kg** | 114 kg | 264 kg | 227 kg |
| mass by type (per car) ICE / HEV / PHEV / EV | 116 / 181 / 100 / 111 kg | 120 / 93 / 182 / 460 kg | | |
| displacement per car, median abs error | **0.38 L** (bias -0.31) | 0.50 L (bias -0.48) | idle-fuel law 0.64 L | dev median 0.60 L |
| road load vs EPA target, 25 cars, 20/40/60 mph | 24 / 19 / 18 % | 16 / 17 / 18 % | LSQ 66 / 35 / 19 % | |
| mass vs EPA test weight (same 25 cars) | 5.0 % | 7.5 % | | |
| Leafs (2), curb error | -119 / -105 kg | +641 / +278 kg | LSQ +173 / +373 | dev median 0 / 0 |
| ZTBus legs (5,847), bus mass mean abs error | 1,204 kg (752 with dev level) | - | LSQ 3,547 kg | dev median 453 kg |

Coverage (sigmas widened on dev: mass x1.02 per car): mass 1σ 56 % / 2σ 93 % per car, 76 % / 98 % per trip; engine
48 % per car. Pooling (mass mean abs error): 1 trip 165 kg, 4: 138, 8: 138, 32: 150 (v1.0 153 -> 126). Bus: slope
of the estimate on passenger mass 0.70, correlation 0.35, reads +1.0 t; 1σ coverage 68 % (factor 1.62 from dev).

**EPA exam families (semi-real, RoadSim v2).** 150 EPA exam vehicles rebuilt from their test weight and coast-down
targets, 4 trips pooled: mass vs ETW 5.0 % MdAPE (BEV 2.7 %, conventional 6.3 %, HEV 11.2 %, bias -4 %), road load at
60 mph 10.3 %, displacement 27 %, VED type 100 %.

**Identifiability sweep (simulated).** +800 kg moves the pooled estimate by +524 kg (8 trips) / +489 kg (2 trips):
BEV +760, PHEV +679, ICE +448, HEV +384. v1.0 on the same sweep +182 kg. CdA x1.5 moves the CdA estimate by x1.03.

**Energy ledger** (simulated twins of the dev cars, 181 trips, mean abs error vs a median guess, kJ/km): aero 24 vs
54, rolling 35 vs 53, grade 72 vs 75 (no skill), brake 30 vs 56, powertrain 107 vs 636, aux 22 vs 63. Twin test:
mass from the real vs the twin log differ by a median 9.7 %; VED type agrees for 15 of 16 cars.

## Limits (where it fails)
1. Powertrain type got worse (97.1 % per car; v1.0 98.9 %; the rule 99.4 %): 10 HEVs read as conventional, 8 of
   them 1.5 L Prius-c-type cars. Cause not isolated (dev twins still show the engine off more often than real HEVs:
   0.31 vs 0.19 of moving time).
2. Per-car mass is a little worse than v1.0 (127 vs 116 kg median), driven by HEVs (181 vs 93 kg), and pooling more
   than 8 trips no longer helps (138 -> 150 kg): the physics fit carries each car's efficiency bias.
3. CdA and Crr are still weakly identified (CdA x1.5 -> x1.03).
4. Error bars are too narrow (mass 1σ coverage 56 % per car).
5. Buses: the bus network follows passengers only weakly (slope 0.70, r 0.35) and loses to the dev-median guess. The
   ZTBus level itself is unverified (power and traction force both read ~0.7 x 19 t + passengers).
6. Ledger: simulated truth only; no skill on net grade work. Light duty is graded on 2017-18 VED cars only.

## v1.0 (12 Nov 2026, physics v1)
Generator + dev-twin wrappers (priors v2) + VED-era mass prior (v3), feature version 3, no physics head; released
rung r6 (sim 0.402 on val_v3). Exam (same numbers when re-scored with this repository's code on 14 Nov): type 98.9 %
per car, mass 116 kg per car, displacement 0.50 L, road load 16/17/18 %. Its weights and module are in git history
(`models/tripdetective/tripdetective.py` at commit 39f5e53).

## Data and licences
Vehicle Energy Dataset (Oh, LeBlanc, Peng, IEEE T-ITS 2020, doi:10.1109/TITS.2020.3035596), Apache-2.0; ZTBus
(Widmer, Ritter, Onder, Scientific Data 2023, doi:10.1038/s41597-023-02600-6; data doi:10.3929/ethz-b-000626723),
CC BY 4.0; USGS 3DEP lidar (public domain); Iowa Environmental Mesonet ASOS; US EPA test-car lists, drive schedules,
ALPHA maps and fueleconomy.gov (public domain); RoadSim/RoadData (ML Ratchet). Code and weights MIT.
