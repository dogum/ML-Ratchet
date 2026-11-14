# RoadSim physics v2: calibration log

Every rung of the v2 calibration, kept or rejected, with its dev numbers. **Dev data only**: VED dev cars, the EPA
dev vehicles of `checks/epa_vehicles.json`, the EPA dev pool (`checks/epa_dev_pool.json`, RoadData dev families,
held-out families excluded) and the D3 dev vehicles. No held-out or exam result was computed or looked at. Fitted
numbers live in `roadsim/data/calibration_v2.json` (preset `roadsim-physics-v2`, with the source of each number).

Sign conventions: EPA errors are on fuel economy (+ = the sim burns less than EPA). D3 and VED errors are on fuel or
energy (+ = the sim uses more). D3 conventional errors are over the recognised schedule phases.

## 14 Nov 2026 — calibration agent 1: harness, procedures, conventional and shared physics

### P0 Harness (no physics change)

- `checks/_opts.py`, `--options NAME|JSON` on every check and `run_all.py` (outputs `<name>__<preset>.json`).
  `run_all` no longer re-fits the frozen v1 constants unless `--recalibrate-v1`.
- `roadsim/obd.py`: VED label model (EnergyNet `maf_fuel`, unchanged numbers) and OBD logger emulator
  (TripDetective `obdlog`, EMU_VERSION 1) upstreamed. The models' copies are untouched. Applied only against VED.
- `checks/check_ved_twin.py`: VED dev twins, label matched (k on CdA/Crr, engine eff_scale) to the unadjusted
  fueleconomy.gov city/highway values on the EPA procedures under the run's own options; trips driven warm with logged
  speed, lidar grade and airport weather; fuel through the label model. RoadData's 136 kg fix checked: `mass_kg` is
  the VED curb weight and `payload_kg` 136 is added once by RoadSim.
- `checks/epa_dev_pool.py` + `check_epa_pool.py`: 235 conventional gasoline MY2018+ cars from RoadData EPA dev
  families (max 6 per family, fixed seed), specs built as `epa_vehicles.py` does. Excluded families: every family of
  a check held-out vehicle (FORD|TRANSIT, TESLA|MODEL Y and VOLKSWAGEN|ID.4 are RoadData-dev families with a held-out
  check vehicle) and of the check dev vehicles.
- `check_d3.py` fixes found on the way (data/harness problems, not physics):
  * multi-cycle tests carry unmetered idle gaps (bags off between phases: engine idling at ~700 rpm, fuel record 0).
    The twin idled and was charged for them: CX9 SC03 was +11 % fuel instead of -7 %. Now excluded on both sides.
  * SC03 x3-4 tests switch HVAC per phase (Accord LX "AUTO-AUTO-ECON-OFF", Camry XLE "AUTO 72, AUTO 72, MAX, OFF");
    each phase now runs with its own setting.
  * HWFET files that end with coast-downs: the real car coasts in neutral, the twin followed the trace in gear
    (Camry XLE first HWFET +22 % fuel). Conventional errors are now over the schedule phases only.
  * soaks cool every thermal state; cold starts at < 5 C start with soaked tyres; logged dipstick oil as the start.
- Effect on v1 D3 dev extended conventional (all tests): median -2.7 %, IQR 9.1 (median +0.3 % with the gaps and
  coast-downs counted: errors of opposite sign cancelled).

### P1 EPA procedure corrections (applied to v1 and v2 alike; `checks/epa_procedures.py`)

| change | source | v1 dev effect |
|---|---|---|
| 20 F cold FTP: heat on, A/C off, automatic 72 F, outside air | 40 CFR 1066.710(c)(2), (c)(1)(i)(D) | Sonata conv +7.7 -> +0.5 %; pool cold +7.8 % (n 43) |
| cold FTP starts with soaked tyres (12-36 h soak at -7 C) | 1066.710(e) | included above |
| every soak cools coolant+oil (coupled), transmission, axle, tyres, PTC loop, v1 tyre state | physics | FTP -0.1 % |
| SC03: 505 s UDDS preconditioning, 10-min soak engine off, A/C max (recirculation, auto 72 F) | 1066.835(c)(4)-(6) | Sonata SC03 +9.5 -> +9.1 % |
| SC03 cabin 45 C at the start (D3 dev SC03: CX9 44.2, Accord 45.5, Camry 46.0 C) | measured | (v1's cabin reaches only 32 C in a simulated soak) |

Hybrids now also run the cold FTP in `check_cycles` (dev: RAV4 -7.2, Sonata Hybrid +4.7, Accord Hybrid -6.0 % in v1).
The 24 C FTP keeps warm tyres (v1's cold-start friction was fitted that way; v2's oil friction is re-fitted on it).

### C1 Measured accessory load — kept
`aux_base_W` 350/400 W (class defaults) -> 250 W for light-duty: D3 dev alternator output, hot std tests, HVAC off
(CX9 262, Accord LX 246, Camry XLE 243, Blazer 202 W). D3 ext hot UDDS -1.8 -> -3.4 % fuel, HWFET +7.6 -> +6.4 %
(before the phase fix).

### C2 Converter behaviour — kept (new small options `conv_tc_open_gears`, `conv_tc_slip_rpm`; off = v1)
D3 dev Camry XLE (8AT) and CX9 (6AT) engine minus turbine speed: 1st gear always open (400-740 rpm slip), 2nd
mostly open (CX9 130-166 rpm, Camry 40-310 rpm), gears 3..top-1 run 15-70 rpm controlled slip, top gear 0-15 rpm.
RoadSim locked from 2nd gear above 1050 rpm. Set: open gears 2, slip 30 rpm, light-duty AT only.
D3 ext hot UDDS -3.4 -> -0.5 %, HWFET +6.4 -> +7.0 % (with C1); pool tilt 6.6 -> 6.1.

### C3 Engine efficiency tilt — kept: eta 1.04, low-load 450 W/L, transient fuel 0.5 s
Fitted on the weighted squared median FE error of D3 ext hot UDDS + HWFET phases (weight 2), check-dev conventional
FTP + HWY and pool FTP + HWY (weight 1). Scan (FE %, D3 UDDS / HWFET / US06 | pool FTP / HWY / US06 | check FTP / HWY):

| eta | low-load W/L | transient s | D3 | pool | check | J |
|---|---|---|---|---|---|---|
| 1.00 | 0 | 0 | +0.9 / -3.1 / +7.1 | +1.1 / -2.7 / -0.5 | +2.8 / -3.4 | 6.1 |
| 1.04 | 600 | 0 | +1.9 / -0.0 / +10.4 | +1.0 / -0.1 / +3.1 | +4.0 / -0.2 | 3.0 |
| 1.07 | 300 | 2.0 | +0.3 / -1.4 / +2.3 | +0.3 / -1.0 / -11.3 | +2.2 / -2.7 | 2.1 |
| **1.04** | **450** | **0.5** | +1.2 / -1.0 / +7.6 | +1.3 / -1.0 / -1.4 | +3.5 / -1.4 | 2.7 |

Why a transient term (`eng_transient_fuel`, new, small, light-duty gasoline only): D3 metered fuel binned by load
shows accelerations under-fuelled (sim/meas 0.85-0.98) and steady cruise over-fuelled (1.05-1.09) **at the same
load**; a steady-state map cannot separate them, and the low-load adder contradicts the measured idle fuel (sim idle
already 0.9-1.16x measured). 2.0 s fits D3 best but costs the pool US06 -11 %; 0.5 s balances D3 US06 (+7.6 %) and
pool US06 (-1.4 %). Kept at 0.5 s; the 1 Hz torque noise at cruise also triggers it (see "not closed").
Applied through `eng_family_adj` to the five conventional gasoline maps only (diesel and the hybrid Atkinson map
unchanged; P2 hybrids that use a turbo map get it too).

### C3b Per-map efficiency levels — rejected
Pool per-map FE offsets are large (a25a +15/+4, turbo_honda_15 +11/0, turbo_ford_27 -5/-8, v6 -5/-12, v8 -8/-11 %).
Fitting them (eta x 0.91/0.95/1.07/1.09/1.11) halves the pool IQR (FTP 22 -> 12, HWY 14 -> 7) but makes every metered
D3 car worse (Camry XLE +7.6/+10.1, Accord +6.8/+6.3, CX9 -7.8/-4.7, Blazer -9.8/-12.2 % fuel) and 3 of 7 check-dev
cars (Camry LE -9.2/-10.3, Sonata +10.2, Explorer +10.5 %). The offsets describe the pool's engine mix (performance
turbo engines, old NA engines), not the maps.

### C4 Cold — kept: oil capacity x2, oil_fric_scale 0.33 -> 0.25, tire_dyno_frac 0.6 -> 1.0
- D3 dev -7 C cold-start UDDS, dipstick at 300/600/900/1300 s: the B2 oil node ran 15-20 K hot (Accord 39/57/65/70 C vs
  22/37/43/49 measured). Oil capacity x2: CX9 27/45/55/62 vs 23/40/53/64; Accord 23/38/48/56; Camry 22/37/45/53 vs
  29/44/52/60. 20 F cold-start UDDS +6.5 -> +2.0 % FE, pool cold +7.4 -> +2.0 %.
- oil_fric_scale 0.25 restores the 24 C cold-start penalty (D3 ext UDDS cold start vs hot start: -1.1 vs +1.1 ->
  +0.1 vs +0.9 % FE).
- tire_dyno_frac 1.0 (the whole tyre loss on the roll follows tyre temperature): 20 F UDDS hot start +6.0 -> +4.3 %,
  pool cold +2.0 -> +0.2 %.
- Rejected: cold_enrich 0.012 and warmup_enrich up to 0.025 (< 0.3 % effect: the coolant passes 20 C within a minute),
  axle drag x3 (+1 %), ATF theta 900 (0 %), heater-blower ramp (no D3 support: the Accord holds the blower low ~250 s,
  then full, as B2's step).

### C5 A/C — kept: cop_eta_belt 0.33 -> 0.25
| cop_eta_belt | D3 SC03 AUTO phases | Accord A/C on/off fuel (meas 1.29) | D3 95 F UDDS cold start | pool SC03 (n 47) | Sonata SC03 |
|---|---|---|---|---|---|
| 0.33 | +5.1 | 1.21 | +1.2 | +3.4 | +8.7 |
| 0.28 | +2.5 | 1.25 | -1.3 | +1.1 | +5.9 |
| **0.25** | +0.3 | **1.28** | -3.3 | -0.7 | +3.9 |
| 0.25, evaporator capacity x1.8 | -2.4 | 1.29 | -3.6 | -2.7 | +2.9 (rejected) |

COP at 35 C becomes ~1.1 idling, ~1.4 at speed: an effective number that also carries clutch cycling and reheat.

### C6 On-road (VED dev conventional twins) — kept: road-height wind z0 0.3 m, tyre model, `onroad_aux_W` conv 700 W
Label-matched twins, trip median (car median), %:

| step | trips | cars | <0 / 0-10 / 10-20 / 20-27 / >=27 C |
|---|---|---|---|
| v1 | -5.1 [-11.9, +1.6] | -3.5 | -3.8 / -3.2 / -5.9 / -7.8 / -11.0 |
| v2 before C6 | -9.1 | -8.1 | -9.4 / -7.0 / -10.6 / -9.3 / -11.0 |
| wind 'none' instead of 'log' | -8.3 | -7.3 | windy-vs-calm spread 6.5 pts (log 5.2, z0 0.3 4.7) |
| + tyre model | -8.5 | -7.5 | |
| + gusts | -9.1 | -8.0 | (rejected: no gain, adds a random realisation) |
| **+ onroad 700 W (conv)** | **-2.3 [-8.7, +6.2]** | **-0.4** | -2.5 / -0.5 / -4.9 / -1.5 / -2.2 |

The v2 A/C and heater flattened the temperature dependence that EnergyNet's `rw` (1380 W fuel + 133 W/K above 15 C)
carried; what remains is a constant, time-proportional gap (~600-700 W brake at every speed band). One documented
option, `onroad_aux_W` = {conv: 700 W} on the 12 V supply, road specs only (never a dyno spec), lumps lights and
accessories, real-road rolling, payload and VED fragments that are not warm. Other architectures: 0 until fitted.
Decisions on B3: `substeps` 1 (10 Hz is 10x slower for <= 0.1 % trip energy; option for drivability work), tyre model
on (+8 % run time conventional, +2 % hybrid), gusts off, drafting off (no gap data).

### C7 D3 facts — verified for the conventional dev vehicles (`data/d3_facts.json`, field "verified")
Cross-checked against fueleconomy.gov vehicles.csv and the EPA test-car table. Two errors corrected:
2019 Blazer has auto stop-start (startStop Y; the D3 files show 0 rpm at stops; the twin idled: hot UDDS idle fuel
2.5x measured); 2013 Jetta TDI is turbocharged (cosmetic: diesel map). Unverified: the three European legacy cars
(Mazda 3 i-stop, Smart MHD, Golf TDI BlueMotion) and the 2012 F-150 EcoBoost rating (no EPA test row).
Hybrid, plug-in and BEV entries were not checked (agent 2).

### C8 Heavy duty and buses — decided, not tuned
- Light-duty-only scope for C2 and the transient term (they had made the diesel bus 4.63 -> 3.93 mpg on CBD14).
- v1 -> v2: diesel bus CBD14 4.63 -> 4.30 mpg; hybrid bus advantage +75.5 -> +58.3 % (CBD14), +35.1 -> +31.5 % (HD
  transient); BEV bus 1.66 -> 2.10 kWh/mi (Foothill in service 1.82-2.15); Class 8 BEV 2.51 -> 2.49 kWh/mi.
- Not tuned to the 10-40 % hybrid-bus literature: those are in-service routes; CBD14 has ~7 stops per mile, where a
  hybrid gains most. HD transient (+31.5 %) is inside the range. The CBD14 gap is in the hybrid powertrain (agent 2).
- Class 8 BEV: the duty explains it (65 mph, 19 t payload: wheel energy alone 2.22 kWh/mi, B2); not tuned.
- ZTBus dev legs (new by-month split, 294 dev missions / 1,662 legs; used: 1,643 q_label_ok legs, standstill seconds
  after >= 5 s at rest): median standstill electrical power 4.4 kW at 15-20 C, 8.6-9.2 kW at 0-10 C, 12.3 kW above
  25 C (18 m articulated trolleybus). The v2 40 ft BEV bus at its CBD14 stops: 4.7 kW at 17.5 C (fits), 28.6 kW at
  2.5-12.5 C (PTC at its 24 kW limit: 3x the measured, on a smaller bus), 18 kW at 30 C. Bus heating in v2 is too
  high; fewer air changes did not help (the PTC loop saturates). Not fixed here; the BEV bus 2.10 kWh/mi carries it.

## Dev results, v1 (corrected procedures) -> v2 draft

EPA check dev (FE %, median, n): conventional FTP +2.3 -> +1.1, HWY -3.7 -> -1.8 (n 7; IQR 4.1 -> 6.5, 5.0 -> 6.1),
US06 +0.9 -> -2.4 (n 2), SC03 +9.1 -> +3.4 (Sonata), cold FTP +0.5 -> -6.5 (Sonata). Hybrids (B1/B2 values, not
calibrated here): FTP +9.1 -> +5.9, HWY +2.8 -> -2.0, US06 +4.6 -> -5.2, SC03 +7.2 -> +2.0, cold FTP -6.0 -> -9.8.
BEV UDDS +2.4 -> +2.9, HWY -0.4 -> +0.3.

EPA dev pool, conventional (n 235): FTP +1.2 -> -1.4 (IQR 19.4 -> 19.8), HWY -2.9 -> -1.4 (12.8 -> 13.9), per-car
FTP-HWY tilt +6.6 -> +1.1, US06 +0.1 -> -2.9 (n 57), SC03 +4.4 -> -1.0 (n 47), cold FTP +7.8 -> +1.4 (n 43).

D3 dev conventional, extended era (fuel %): all tests median -2.7 -> -0.3, IQR 9.1 -> 5.6, mean |err| 5.4 -> 3.6
(n 135); hot UDDS -2.6 -> -0.3, HWFET +3.9 -> +1.5, US06 -7.0 -> -6.3, 24 C cold start -1.8 -> +0.4, 20 F cold start
-9.3 -> -1.2, SC03 -10.2 -> -2.3. Legacy era (modal fuel, weighted low): -22.3 -> -22.4.

VED dev twins: conventional trips -5.1 -> -2.3 (car medians -3.5 -> -0.4), idle fuel -16 -> -12 %. Other
architectures are agent 2's (v2 numbers there use B1/B2 values: power-split HEV -22 -> -26 %, engine-on -31 -> -31
points; Leaf -1.6 -> -4.7 %; Energi +36 -> +74 %; Volt -12 -> -1 %).

Ledger stress (600 runs, v2 preset): max closure 5.6e-4 (v1 5.1e-4), none above 0.5 %. Fingerprint (v1): passes.

## Not closed (agent 1)

- US06: D3 dev says the sim burns 6-8 % too little at high load (all four extended cars), the EPA pool says -3 %.
  No high-load enrichment model; the transient term is a compromise.
- The transient term reacts to 1 Hz torque jitter at cruise; a smoothed tip-in signal would be cleaner.
- Per-engine spread (pool IQR ~20 FTP / ~14 HWY) is unchanged: it is engine-to-map mismatch, and per-map levels were
  rejected (C3b). The check-dev conventional IQR rose (FTP 4.1 -> 6.5, n 7). Expect the held-out IQR target (halved)
  to be missed.
- At -7 C the coolant falls to 45-50 C after 10 min with the heater on (D3 62-77 C) and the cabin stays at 14-16 C
  (D3 23-27 C): engine heat to coolant too low or heater-side losses too high. Matters for hybrid heat requests.
- 20 F hot-start tests remain +4 % FE (sim burns too little) after C4; source not found.
- Bus heating 2-3x the ZTBus dev standstill power (C8).
- EPA check-dev Sonata cold FTP went +0.5 -> -6.5 while the pool (n 43) went +7.8 -> +1.4: one car against 43.
