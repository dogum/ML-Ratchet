# TripDetective v1.0: model card

**What it does.** Reads one or more 1 Hz trip logs of a light-duty vehicle (speed plus whatever an OBD logger has:
fuel rate, rpm, HV battery current/voltage/SOC, A/C power, with route grade and weather) and infers: architecture
(7 layouts), total mass, engine displacement, CdA, Crr, hybrid control strategy and on/off behaviour, and each trip's
energy ledger (aero, rolling, grade, friction braking, powertrain losses, auxiliaries, kJ/km). Every output has a
1-sigma error bar. Trips of the same car are pooled into one answer.

**Trained only on simulation. Graded on 351 real VED cars never used for choosing anything.**

## Inputs / outputs
- Input per second: v, acceleration, grade, T_amb, headwind (wind x cos(wind_dir - heading)), air density, fuel rate,
  rpm, battery power, SOC, aux power, validity, channel-availability flags, three road-load power channels
  (v(1.03a + g grade), 0.5 rho v_air|v_air| v, g v). Per trip window: 16 classical numbers (least-squares fit of
  logged source power to the road-load equation: mass x efficiency, Crr term, CdA term, offset, R2, rows; idle fuel,
  idle rpm, high-load fuel, rpm per speed, engine-off share, fuel/km, battery kWh/km, battery power percentiles).
- Windows of at most 1,024 s; longer trips are split and pooled.
- Output: see README. Mass is the total moving mass; curb = total - 136 kg (RoadData's driver-and-load convention).

## Physics and simulator
RoadSim (ML Ratchet vehicle Phase 1): forward-facing 1 Hz vehicle simulator with road loads (wind along heading, yaw
drag, Crr with temperature, speed, surface and tyre warm-up), EPA ALPHA engine and e-machine maps, AT/CVT/DCT/eCVT,
Thevenin battery, HVAC, 7 architectures, 14 causal strategies. Checks (road/checks): road-load force within 1.6/0.8/0.5 %
of 47 real EPA coast-down targets at 20/40/60 mph; FTP +5.6 %, HWFET -4.3 % (conventional, held-out); ledger closure
max 5e-4 over 600 stress runs; FASTSim cross-check with matched components within ~3 % (conventional).
RoadSim was not edited; all changes are wrappers in `training/priors.py` and `training/obdlog.py`.

## Simulated data in real conditions
- Routes: 1,154 real Ann Arbor route fragments from the 17 unlabelled VED `train_noise` cars (lidar grade, ASOS
  weather), windows of 10 light-duty cycles, synthetic stop-and-go routes. The car experiences slightly different
  conditions than the log says (ASOS T +-1.5 C, wind at car height x0.4-0.9, grade noise 0.3 %).
- OBD emulator (`obdlog.py`): every signal polled on its own clock with cadences measured on the dev cars' raw rows
  (speed 1-1.15 s, MAF/rpm 2-2.4 s, absolute load and fuel-rate PID 5 s, HV current 1 s, voltage 5 s), held and
  quantised, then resampled with RoadData's own resampler; MAF-estimated fuel (misses enrichment, shows motoring air
  during fuel cut, per-car trim bias 4 %), fuel = 0 when rpm < 100, dropouts, random channel sets per car.
- Bank v3 (final): 2,000 cars x 8 trips, frozen validation set val_v3 600 cars x 8 trips (seed 99).

## Dev calibration (16 dev VED cars only)
Twin test (`twin.py`): real dev trips re-driven by each car's RoadSim twin, logged with its own channels.
Found and fixed (priors v2): twin fuel/km 0.82-0.87x real for conventional cars -> VED-era engine efficiency
eff_scale ~ N(0.88, 0.06); idle 0.83x -> idle 650-820 rpm; hybrid engine-off-while-moving 2x too frequent -> strategy
parameters drawn widely (ECMS s 2.4-3.6, rule thresholds lower); Leaf heating 5x too strong at stops -> milder HVAC
priors; older 4-5 speed automatics. After: conventional fuel/km 0.96x, idle fuel 1.00x, hybrid engine-off 1.24x.
Network twin test (`twin_ledger.py`): R0 read twins +230 kg heavy -> cause: the generator fleet (median total mass
2,200 kg) is much heavier than VED-era cars (1,720 kg, dev + train_noise static weights) -> priors v3 mass prior.

## Network
1-D CNN (stem k7 + 5 stride-2 blocks, 32-160 channels, BatchNorm, GELU, activations masked outside the trip),
masked mean+max pooling, trip MLP (with the 16 classical numbers) -> 128-d trip embedding; trip embeddings of one car
are averaged (+log count) -> car MLP -> heads: architecture (7), strategy (14), 7 regressions (mean, log-sigma; Gaussian
NLL); per-trip ledger head (trip + car embedding). 0.40 M parameters. Trained 2,500 steps of 24 cars x 1-8 trips with
channel dropout. numpy == torch to 8e-6; JS == Python to 1e-4.

## Ratchet (frozen set val_v3 after re-scoring every rung; margin +0.01; dev gate: mean of 4 dev error ratios <= 1.05, VED-type accuracy drop <= 0.03)

| rung | change | sim skill | dev mass err/car | dev engine err/car | dev type acc/trip | decision |
|---|---|---|---|---|---|---|
| r0_v1 | CNN + set pooling, generator as published, 1,200 cars | 0.305 | 140 kg (bias +106) | 0.31 L | 0.91 | kept (first) |
| r1_v2 | dev-twin calibrated priors v2, 1,200 cars | 0.362 | 128 kg | 0.52 L | 0.90 | rejected: dev errors +23 % |
| r2_v2full | v2, 2,000 cars | 0.366 | 116 kg | 0.51 L | 0.93 | rejected: +15 % |
| r3_v2phys | + road-load physics channels | 0.363 | 129 kg | 0.53 L | 0.88 | rejected: +23 %, type acc |
| r4_v2cls | + 16 classical per-trip estimates | 0.368 | 136 kg | 0.49 L | 0.90 | rejected: +17 % |
| r5_v1cls | classical estimates on the v1 bank | 0.317 | 139 kg | 0.34 L | 0.91 | kept (+2 %) |
| r6_v3cls | v3 mass prior (VED-era fleet) + classical estimates, 2,000 cars | **0.402** | **110 kg (bias -9)** | 0.44 L | 0.92 | **kept (final)** |

Sim skills of the final rung (val_v3): single trip / whole car: architecture 0.52/0.57 (accuracy 0.66/0.69, VED type
0.95), mass 0.32/0.41 (MdAPE 9.1 %/8.0 %), displacement 0.38/0.49, CdA 0.04/0.06, Crr 0.13/0.15, engine-off share
0.45/0.75, strategy 0.46/0.53 (accuracy 0.60), ledger 0.43.

## Exam (real VED exam cars; at most 40 trips per car, fixed seed; baselines fitted on dev only)

| | Detective | classical baseline | dumb baseline |
|---|---|---|---|
| VED type, per car (351) | 98.9 % (bal. 98.8 %) | rule 99.4 % | majority 68.6 % |
| VED type, per trip (10,186) | 92.9 % (bal. 93.0 %) | rule 96.7 % | 63.1 % |
| curb mass per car (335 with weight), median abs error | **116 kg (7.8 %)**, bias -37 kg, 49 % inside the +-113 kg bin | physics LSQ 205 kg (13.8 %) | dev median 227 kg (16.7 %) |
| curb mass per trip, median abs error | **113 kg (7.4 %)** | 264 kg | 227 kg |
| displacement per car, median abs error | **0.50 L (22.4 %)**, bias -0.48 L | idle-fuel law 0.64 L (27.9 %) | dev median 0.60 L (25 %) |
| road-load force vs EPA target, 25 cars, 20/40/60 mph | 16 / 17 / 18 % (bias +15/+10/+6 %) | physics LSQ 66 / 35 / 19 % | – |
| mass vs EPA test weight (same 25 cars) | 7.5 % | – | – |

By type (mass per car, median abs error): ICE 120 kg, HEV 93 kg, PHEV 182 kg, EV (2 Leafs) +460 kg.
Coverage (sigmas widened on dev: mass x1.0, engine x2.0 per car, x1.4 per trip): mass 1 sigma 73 % / 2 sigma 98 % per
car, 79 % / 99 % per trip; engine 61 % / 98 % per car. Pooling (mass mean abs error): 1 trip 153 kg, 4: 136, 16: 128,
32: 126 kg (rounding floor 57 kg). Finer layout agrees with RoadData's guess for 97 % of high-confidence cars;
plug-ins split series vs power-split poorly (8/14 and 5/8).

**EPA exam families (semi-real).** EPA test cars have no trip logs. 150 EPA exam vehicles (MY 2012+, 50 per
architecture, seed 0) rebuilt from their real test weight and coast-down targets, driven in RoadSim over real routes,
logged by the emulator, 4 trips pooled: mass vs ETW 7.8 % MdAPE, road-load force at 60 mph 9.4 %, displacement
25.5 %, VED type 100 %.

**ZTBus:** not graded (no curb mass; light-duty network). **Energy ledger:** no real truth exists; on simulated twins
of the dev cars (181 trips) mean abs error vs a median guess, kJ/km: aero 30 vs 64, rolling 38 vs 57, grade 74 vs 75
(no skill), brake 32 vs 56, powertrain losses 111 vs 777, aux 28 vs 44. Twin test: mass from real vs twin log differ by
a median 6.7 %; VED type agrees for 14 of 16 cars.

## Limits (where it fails)
1. It reads the kind of car more than the physics of each acceleration: in controlled sweeps (page lab) +800 kg moves
   the estimate by ~200 kg and CdA x1.75 barely registers. Mass, efficiency and drag trade off in 1 Hz MAF-based logs;
   the network leans on fleet regularities. Pooling helps little (153 -> 126 kg over 32 trips).
2. Engines read small (-0.48 L median bias); only narrowly better than a dev-median guess. Every v2-prior rung made
   dev displacement worse; the cause was not isolated (candidates: engine-efficiency spread, gearing priors).
3. The simple rule beats it on architecture (99.4 % vs 98.9 %); in VED the battery channel exists only on plug-ins.
4. Leafs read 460 kg heavy: RoadSim's 2013 Leaf twin uses ~1.5x the real battery energy (heating).
5. Road load +15 % at 20 mph on 25 mostly plug-in cars; CdA/Crr weakly identified even in simulation.
6. Ledger: simulated truth only; no skill on net grade work.
7. Light duty only. RoadSim limits apply (P2 city optimistic, cold too mild, conventional A/C too cheap).

## Data and licences
Vehicle Energy Dataset (Oh, LeBlanc, Peng, IEEE T-ITS 2020, doi:10.1109/TITS.2020.3035596), Apache-2.0; USGS 3DEP
lidar (public domain); Iowa Environmental Mesonet ASOS; US EPA test-car lists, drive schedules, ALPHA maps, and
fueleconomy.gov (public domain); RoadSim/RoadData (ML Ratchet Phase 1). Code and weights MIT.
