# RoadSim physics v2: calibration log

Every rung of the v2 calibration, kept or rejected, with its dev numbers. **Dev data only**: VED dev cars, the EPA
dev vehicles of `checks/epa_vehicles.json`, the EPA dev pool (`checks/epa_dev_pool.json`, RoadData dev families,
held-out families excluded) and the D3 dev vehicles. No held-out or exam result was computed or looked at. Fitted
numbers live in `roadsim/data/calibration_v2.json` (preset `roadsim-physics-v2`, with the source of each number).

Sign conventions: EPA errors are on fuel economy (+ = the sim burns less than EPA). D3 and VED errors are on fuel or
energy (+ = the sim uses more). D3 conventional errors are over the recognised schedule phases.

## 13 Nov 2026 — calibration agent 1: harness, procedures, conventional and shared physics

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

## 13 Nov 2026 — calibration agent 2: hybrids, plug-ins, BEVs, HVAC, buses; physics v2 frozen

Same rules: dev data only (D3 dev vehicles, VED dev cars, EPA check-dev vehicles, EPA dev pool, ZTBus dev). The
held-out report was computed once, after the commit that froze every value below.

### H0 Harness and twin procedures (applied to v1 and v2 alike; no physics change)

| fix | what was wrong | effect (dev) |
|---|---|---|
| `check_d3.py`: LHV in J/g | `FUELS` is J/kg: the HEV SOC correction was 1000x too small and the PHEV energy (fuel LHV + battery) was 1000x fuel-weighted | PHEV primary is now energy, not fuel; v1 plug-in power-split median +5.8 -> +1.3 % |
| HEV twins hold the test's own start SOC (`soc_target`); a logged SOC outside the twin window (2010 Prius 0.31-0.40 vs 0.4-0.8) uses the mid-point | a twin 0.08 below the window mid charged at 20-40 kW from the first second (Accord Hybrid UDDS: engine at 3,300 rpm while the car itself charged at 3 kW) | |
| plug-in CS tests sustain at the logged SOC (`soc_cs`) | Energi logged 0.16-0.20 vs the twin's CS level 0.20: the twin charged 150-500 Wh per UDDS | |
| D3 spec rule: plug-in pack power = min(traction machines / 0.9, 9 C) | RoadSim's 'energy' default (3 C) gave 23 kW (Energi), 26 (Prime), 55 (Volt); D3 dev 1 Hz maxima are 67 / 69 / 127 kW; the twins started the engine in charge depletion | Energi 68, Prime 79, Volt 123 kW (rule applies to every D3 plug-in) |
| D3 spec rule: a series hybrid's EPA N/V is the engine's; kept only if the motor's 14,000 rpm lands at 140-260 km/h | 2015 Accord Hybrid final drive 2.84 (614 km/h at 14,000 rpm): the motor ran torque-limited at a few hundred rpm; regen capture 0.46 vs 0.64 measured | Accord and 2016 Volt get 9.68; Accord regen 0.64 = measured; v1 Accord median +7.9 -> -8.9 % |
| mild hybrids (P0/P1: Insight, S400h, CR-Z, Civic Hybrid, Malibu eAssist) run as P2 with the new strategy parameter `engine_on_moving` (no electric-only driving) | P1 twins drove electrically: Insight UDDS engine-on 0.27 vs 0.81 | Insight UDDS -35 -> -18 %, S400h -42 -> -3 % (legacy modal fuel) |
| VED: the CR-Z twins (VED_435/440; RoadData labels them hev_powersplit) run as P1 mild hybrids and are **excluded from the hybrid targets**: RoadData matched them with low confidence to a 43/43 mpg label (the CR-Z's own label is 31-35/37-39) and their label match needs eff_scale 1.33 | | reported as hev_parallel_p2 |
| VED plug-in twins: public machines, engine power and the pack rule; road load k = 1, engine level from the CS label | Energi twin: label matching drove k to its 1.8 bound with eff_scale 1.31 (+74 % energy); matched on the electric label, k would be 1.6 (+31 %) | |
| `d3_index.py` reads a time column that advances (2020 Bolt 62009058 had a constant 'Time[s]') | | 1,661 ok tests (was 1,660). The other 34 Bolt and 45 Leaf ePlus failures are DC fast-charging files (no drive trace) |

### H1 P2 converter, starts, K0: kept at the B1 values (looked at, not changed)
The D3 Sonata Hybrid starts with a belt starter-generator (`k0_slip_frac` 0 would suit it, but specs carry no start
device and the EPA check-dev Sonata shows no start-energy excess). The real car makes 29-30 starts per hot UDDS (sim
10-17); many are 4-6 s engine runs at launch (1-2 g each, ~9 % of its UDDS fuel). No option was added for that (one
car). `start_penalty_J` 0 with min-on 4 s gave more starts but D3 Sonata UDDS -9.2 -> -13.9 %: rejected.

### H2 Brake blending on P2: kept, `regen_share` {hev_parallel_p2: 0.75} (new option, off in v1)
D3 dev Sonata Hybrid, battery charge during braking / negative wheel energy (hot std tests): UDDS 0.53 measured vs
0.70 simulated, HWFET 0.50 vs 0.57, US06 0.55 vs 0.69. Power-split (Prime 0.72 vs 0.70), series (Accord 0.64 vs 0.64
after H0) and modern BEVs (Ioniq5 0.70-0.82 vs 0.69-0.81) match: P2 only.

| regen_share P2 | D3 Sonata UDDS / HWFET / US06 (fuel_corr %) | EPA dev Sonata Hybrid FTP / HWY / US06 (FE %) |
|---|---|---|
| 1.0 (v2 draft) | -20.0 / +2.6 / -3.2 | +15.5 / -3.9 / -5.2 |
| **0.75** | **-9.2 / +5.4 / +3.0** | **+4.3 / -5.9 / -9.5** |
| 0.70 | -7.1 / +5.8 / +4.5 | +2.3 / -6.3 / -10.3 |

H2b, rejected: `hybrid_chg_cap_frac` (new option, None). Real hybrids charge more gently than ECMS (Sonata UDDS p10
-11 kW vs -16 sim; Accord -11 vs -24), but capping engine charging at 5 / 10 % of engine power moved the Sonata UDDS
to -14 / -19 % and its HWFET to +12 / +6 %.

### H3 e:HEV direct drive: kept, 22.7 rpm per km/h, window from 65 km/h, `series_dd_prefer` (new option)
Measured on the D3 dev Accord Hybrid: engine rpm / road speed in lock-up 22.7 (mode); lock-up share 0.02 at 40-60,
0.23 at 60-70, 0.79 at 70-80 and 0.80 at 80-100 km/h. With `series_dd_prefer` a running engine inside the window
drives through the clutch (i-MMD engine drive at cruise). D3 Accord HWFET +6.6 -> +1.9 % fuel (engine-on gap -16 -> -11
points); EPA dev Accord FTP -5.1 -> -4.2, HWY -6.9 -> -4.6, US06 -8.5 -> -8.5 % FE.

### H4 Plug-in heat: kept, `phev_engine_heat` (new option) and engine heat below -10 C
- A 'ptc' plug-in heated only electrically, also in charge sustaining, so the engine recharged the battery to feed the
  heater (D3 dev -7 C CS UDDS: Volt fuel +45-50 %, Energi +54-64 %). With `phev_engine_heat` a warm engine (coolant
  >= 50 C) heats the cabin first and the PTC or heat pump does the rest: Volt -0.1 to -11 %, Energi +4 to +16 %.
- The Energis do have electric heat: their D3 CD UDDS at -7 C after a warm start uses 3.0-3.4 kWh with the engine on
  8-11 % (+3.5 kW over the warm test). An engine-only Energi (tried first) ran its engine 64-78 %: rejected.
- `phev_heat_engine_below_C` 0 C put the Volt and Prime -7 C tests at +36 to +62 %. Cold starts only (new option
  `phev_heat_cold_start_only`) fixed the Energi cold starts (-45/-36 -> -3/+10 %) but put the Volt and Prime 8xUDDS
  cold starts at +62 / +36 %: rejected. -10 C (Volt-type engine-assisted heat) kept: Prime -18 C cold start +1 %.

### H5 Toyota production rule: kept, `ps_engine_on_kph` 72 -> 45
VED dev Toyota twins (VED_241 and 258 Prius-type, 375 Highlander-type): the real engine turns 40-60 % of the time at
every temperature, the twin's 15-30 %. Engine-on gap -24 (72 km/h) -> -18 (55) -> -11.6 points (45). EPA dev RAV4
FTP +7.9 -> +5.5, Escape +3.9 -> +1.7 % FE; D3 dev 2010 Prius (legacy) UDDS engine-on +4 -> +9 points.

### H6 On-road extra load for electrified cars: kept, hev 1,250 W, phev and bev 300 W
Fitted as agent 1 did for conventional cars (car median of trip errors -> 0; VED dev label-matched twins; after H5):
hev_powersplit 1,000 / 1,300 W -> -5.5 / +0.7 % (600 W: -13.4 %; 0 W at 72 km/h: -27 %). Applied to every HEV type
(no other VED dev hybrid has a usable label). BEV (2013 Leaf) 0 / 300 / 600 W -> -8.0 / -0.1 / +7.9 %; the same 300 W
on the plug-ins. The hybrid lump is large: it carries what a warm twin on a logged trace misses on real roads (12 V
loads, warm-up running in fragments that are not warm, heat and A/C use), not one physical load.

### H7 Cabin heat loss: kept, `cabin_UA_body_frac` 0.6 -> 0.3; electric A/C `cop_eta_el` 0.36 -> 0.50
- Cause of agent 1's "coolant 45-50 C, cabin 14-16 C at -7 C": the cabin lost too much heat. The heater sat at its
  6 kW cap, drew the coolant down, and the cabin still settled 5-9 K below the D3 measurement (CX9, Accord LX, Camry).
  Heat-flow audit of the CX9 cold-start UDDS: the coolant gains 5.7-7.3 kW in the city, the heater takes 5-6 kW, the
  block loses 2 kW; engine heat to the coolant was not the cause. Conduction share 0.6 / 0.4 / 0.3 / 0.2: cabin error
  -7.3 / -5.4 / -4.4 / -3.6 K; Prius Prime logged heat-pump power +151 / +67 / +43 / +22 %; Leaf ePlus +30-49 / +10 /
  -2 / -15 %. 0.3 kept (36 W/K conduction for a car). CX9 cold-start UDDS at 1,200 s: coolant 55 -> 66 C (measured 74),
  cabin 15.8 -> 19.3 C (21.3). Less fresh air (0.6x) made the cabins colder: rejected. SC03 conventional -1.2 points.
- Logged electric A/C energy on D3 dev hot tests, sim/measured: cop_eta_el 0.36 -> 1.38 (Prime), 1.38 (Leaf ePlus);
  0.45 -> 1.15, 1.23; 0.50 -> 1.06, 1.17; 0.55 -> 0.98, 1.11. 0.50 kept. Cost: EPA dev Accord Hybrid SC03 +3.2 ->
  +12.5 % FE, and the D3 hot tests of the Accord Hybrid and the Volt fall to -17 / -18 %. Their A/C is not logged, so
  the component measurement wins; the hot-weather gap on those two is open.

### H8 Full-load enrichment: rejected (`eng_fullload_enrich`, new option, 0)
10 / 20 % extra fuel ramping in above 85 % of full torque: D3 dev extended US06 -6.3 -> -6.1 / -6.0 % (the twins rarely
reach 85 %). From 60 % torque (0.15): -6.3 -> -4.2 %, but the EPA dev pool US06 already burns 2.9 % too much (opposite
sign). The D3 and EPA high-load evidence disagree: left at 0.

### H9 Bus heating: kept, door exchange `door_cd` 0.6 -> 0.2, `door_min_ms` 0.15 -> 0.03, bus fresh air 2.2 -> 0.8 kg/s
ZTBus dev (18 m trolleybus) adds ~4.5 kW at stops at 0-10 C over its 15-20 C standstill base (4.4 kW). The v2 draft
40 ft BEV bus exchanged 1.4 kg/s through its doors at every stop and blew 0.73 kg/s of 70 % fresh air: 24 kW (the PTC
cap) at stops from 2.5 to 12.5 C, 2.10 kWh/mi on CBD14 (10-32 C). Now 1.83 kWh/mi (10 C 1.96, 18 C 1.47; Foothill in
service 1.82-2.15) and ~9 kW mean heating at 7.5 C. Still open: the PTC loop controller (30 s to target on a large bus
loop) saturates at stops, so the standstill power stays at the cap although the mean fell. Hybrid bus vs diesel on
CBD14 +58 -> +55 % (not tuned: no hybrid-bus trace data; the 10-40 % literature is in service).

### H10 ZTBus twin check (new, `checks/check_ztbus.py`; nothing fitted)
One twin for every ZTBus leg: 18.7 m battery-trolleybus as a RoadSim BEV bus, curb 19,000 kg + 70 kg per passenger
(leg mean), CdA 6.0 m2, Crr 0.007, 320 kW, 200 kWh, class-default bus accessories and HVAC; the leg's own speed,
grade and temperature. Net DC energy vs ZTBus elec_kWh per leg. Dev legs (n 1,643, q_label_ok): v1 -14.9 % median
(IQR -25.5 to -3.9; 1.27 vs 1.56 kWh/km), v2 -3.1 % (IQR -10.2 to +4.7; 1.38 kWh/km); by temperature v2 0-10 C -0.8,
10-20 C -4.9, 20-25 C -1.1, >= 25 C -3.3 %. The spec was set before any leg was simulated and not changed after.

## Physics v2 final (frozen before the held-out run)

`roadsim/data/calibration_v2.json` (preset `roadsim-physics-v2`); PHYSICS_VERSION 'roadsim-physics-v2' and
simulate()'s default. `options='roadsim-physics-v1'` (or None) reproduces v1 exactly (fingerprint, 171 runs).

Dev results, v1 (corrected procedures) -> v2 final:

- EPA check dev (FE %): Sonata Hybrid FTP +27.3 -> +4.3, HWY +2.8 -> -5.9, US06 +4.6 -> -9.5; RAV4 Hybrid FTP +11.6 ->
  +5.5, HWY +2.9 -> -0.7; Escape FTP +6.5 -> +1.7; Accord Hybrid FTP -4.7 -> -4.2, HWY -8.8 -> -4.6, US06 -10.5 ->
  -8.5, SC03 +7.2 -> +12.5; BEVs UDDS -1.9 / +5.9 / +2.4 -> -3.3 / +7.9 / +2.9. Conventional as agent 1 (Sonata SC03
  +9.1 -> +4.1, cold FTP +0.5 -> -5.8).
- EPA dev pool (n 235): FTP +1.2 -> -1.4, HWY -2.9 -> -1.4, US06 +0.1 -> -2.9, SC03 +4.4 -> -0.3, cold FTP +7.8 -> +2.6.
- D3 dev (primary %, median): conventional -4.6 -> -1.5; P2 -19.9 -> -5.4 (UDDS -25.1 -> -8.9); power-split (legacy
  2010 Prius and Fusion, n 8) -23.2 -> -20.7; series -8.9 -> -6.9; plug-in power-split -7.6 -> -8.0 (cold +11.9 ->
  +0.7); plug-in series -1.8 -> -11.2 (cold +25.8 -> +7.2, hot -5.1 -> -18.2); BEV -6.3 -> -6.7 (cold -4.9 -> -0.6).
  Engine-on share, P2 / power-split / series: 0.30 / 0.34 / 0.40 -> 0.29 / 0.44 / 0.43 (measured 0.34 / 0.56 / 0.57).
- VED dev twins (trip median; car median): conventional -5.1 -> -2.4 (-3.5 -> -0.4); Toyota power-split -20.3 -> -0.9
  (-20.5 -> -1.1), engine-on -24.6 -> -10.9 points; CR-Z (excluded) -12.7 -> -6.7; Energi -11.9 -> -12.6; Volt -5.6 ->
  -10.7; Leaf -1.6 -> -0.1. The plug-ins are now -28 to -44 % below 10 C (v1: Volt +6 to +16 %): v2's cabin needs less
  heat and every VED fragment starts with a 21 C cabin. Open.
- Heavy duty: BEV bus CBD14 1.66 -> 1.83 kWh/mi; hybrid bus +75.5 -> +55.1 %; diesel bus 4.63 -> 4.64 mpg; Class 8
  BEV 2.51 -> 2.49 kWh/mi.
- Ledger stress (600 runs, v2): max closure 5.6e-4, none above 0.5 %.

## Not closed (agent 2)
- D3 legacy Ford and Toyota hybrids (2010 Prius, Fusion Hybrid, 2013 Energis in CS) burn 20-28 % more than their
  twins: the modern Atkinson map is better than those engines, and their fuel is the legacy measurement. Not fitted.
- P2: launch engine starts (Sonata), engine-on -11 points; EPA dev Sonata US06 -9.5 % FE.
- e:HEV US06 -8.5 % FE (EPA dev); the Accord Hybrid and Volt hot tests -17 / -18 % after the A/C change.
- Old BEVs (2012-2015 i-MiEV, Leaf SV, Soul EV) use 10-17 % more on D3 than their twins. The BEV loss scale (v1, fitted
  on modern EPA dev BEVs) fits the modern ones (std: Bolt +0.5, Ioniq5 -1.6, Leaf ePlus -1.2 %).
- VED plug-ins in the cold (above); US06 high load (H8); the bus PTC controller (H9).

## Held-out report (computed once, after the freeze commit; nothing changed afterwards)
`checks/run_all.py --options roadsim-physics-v1 --all-splits` and `--options roadsim-physics-v2 --all-splits`;
`checks/scoreboard_roadsim-physics-v1.json` (v1 with the corrected procedures), `scoreboard_roadsim-physics-v2.json`,
`scoreboard_compare_v1_v2.txt`. The table and what looks wrong (TMED P2s now too thirsty, Toyota EPA cycles 5-7 % too
thirsty, plug-in series exam -27 %) are in LIMITS.md; they are reported, not fixed.

## 14 Nov 2026 — calibration agent 3: hybrid fix pass (physics v2.1)

Same rules: dev data only. Two things were frozen before any fitting: a fresh hybrid exam ("exam-2",
`checks/exam2_hybrids.json`, commit b2f10c0) and a v2 fingerprint guard (commit 30a5495). The v2 held-out results of
LIMITS.md were already seen, so they judge nothing here.

### H11 Exam-2, guard and a wider dev set (no physics change)

- **Exam-2** (`checks/exam2_hybrids.py`, sha256 fd180962…): 108 VED exam HEV/PHEV cars (11,157 label trips; no VED
  exam check existed) and 409 EPA test-car hybrid / plug-in records of held-out model families that no check had
  simulated. Excluded: check_cycles vehicles and their carry-over copies (15), D3-matched rows (9), the families of the
  four check_cycles dev hybrids that v2 was fitted on (HYUNDAI|SONATA, TOYOTA|RAV4, FORD|ESCAPE, HONDA|ACCORD: 50
  records), designs RoadSim does not model (two-mode, P4 / through-the-road, P2 with a CVT: 48 records).
- **Guard**: `tests/golden/roadsim_v2_fingerprint.json` (171 runs, options roadsim-physics-v2) and
  `tests/test_roadsim_v2_fingerprint.py`: v2 reproduces it to 1e-9, and every conventional, BEV, bus and Class 8 run
  of v2.1 must match it to 1e-9.
- **Wider dev set**: `checks/hybrid_specs.py` builds specs for EPA test-car hybrids by rule (hardware by model line
  from public spec sheets), `checks/check_hybrids_epa.py --set dev` runs all 55 dev records RoadSim models (every
  hybrid / plug-in of RoadData's dev families: Grand Highlander Hybrid and Hybrid MAX, LC 500h, Altima Hybrid, Infiniti
  Q50/Q70, Audi Q5 Hybrid, Panamera S Hybrid; plug-ins in charge sustaining: BMW 530e, Range Rover PHEV, Panamera
  E-Hybrid, Audi Q5 PHEV, Lincoln Corsair, 2012/2016/2019 Volt; plus the four check_cycles dev hybrids).
  `checks/hybrid_dev.py` runs it with the D3 dev hybrids and plug-ins and the VED dev HEV/PHEV twins in one go.
  Note: the 2012 Volt record is in a dev family but is the EPA row of the D3 exam 2012/2013 Volts.
- Harness fixes (every physics version):
  * `check_d3.py`: a logged charge-sustaining SOC below the twin's window (Prime 0.12-0.15, Volt 0.13-0.16 vs 0.15)
    is on another scale; the twin now starts and sustains at its own CS level. Before, it charged 0.3-0.6 kWh from the
    engine in a UDDS. D3 dev plug-in power-split v2 -8.0 -> -9.8 %.
  * `check_d3.py`: plug-in charge-sustaining tests are scored like HEVs (fuel corrected by net battery energy, SAE
    J1711-style). Counting the battery 1:1 with fuel energy made a twin that drains its pack look frugal. D3 dev
    plug-in series v2 -11.2 -> -5.5 %.
  * `epa_procedures.py`: optional `soc0` (start SOC of a charge-sustaining plug-in test). A local `soc0 = None` in
    `ftp()` first shadowed it: every plug-in FTP started from the default full pack. Found on the cold Volt (raw
    fuel 14 g for a cold FTP, 5 kWh from the pack), fixed before any rung was judged.

Dev scoreboard of v2 with the fixed harness (the reference for every rung below; FE % on EPA, + = sim burns less;
energy % on D3 and VED, + = sim uses more):

| source | v2 |
|---|---|
| EPA dev toyota_ps (n 12) FTP / HWY / US06 / SC03 / cold | +4.0 / -3.0 / +2.0 / +9.1 / -12.7 |
| EPA dev P2 TMED (Sonata Hybrid) | +4.3 / -5.9 / -9.5 / +2.1 / -2.9 |
| EPA dev P2 clutch (n 12: Infiniti, Q5, Hybrid MAX) FTP / HWY | +17.6 / -0.6 |
| EPA dev P2 converter (Panamera S Hybrid) FTP / HWY | +39.6 / +3.8 |
| EPA dev Honda e:HEV (Accord) | -4.2 / -4.6 / -8.5 / +12.5 / -10.6 |
| EPA dev plug-in P2 CS (n 19) FTP / HWY / SC03 / cold | +16.4 / -4.2 / +42.7 / +22.7 |
| EPA dev plug-in power-split CS (Corsair, n 5) FTP / HWY | +7.0 / +1.6 |
| EPA dev plug-in series CS (Volts, n 3) | +11.3 / -6.1 / -0.6 / +17.4 / -16.0 |
| D3 dev P2 / power-split / series / plug-in PS / plug-in series | -5.4 / -20.7 / -6.9 / -9.7 / -5.5 |
| D3 engine-on share sim / measured, P2 / PS / series / plug-in PS / plug-in series | 0.29/0.34, 0.44/0.56, 0.43/0.57, 0.25/0.29, 0.13/0.27 |
| VED dev Toyota car median fuel; engine-on gap | -1.1 %; -10.9 points |

### H12 P2 brake blending re-measured; physical replacements of regen_share — rejected

D3 dev 2021 Sonata Hybrid, hot std UDDS, 10 Hz: the energy the tyres put into the dyno while braking (F_tractive x v
< 0) is 0.757 kWh per test; the pack takes in 0.654 kWh meanwhile (0.86). The twin at regen_share 0.75 charges 0.42 of
its 0.69 kWh (0.61), at 1.0 0.57 (0.83). So the measured capture equals the twin's at full regen; H2's 0.53 must have
used another denominator. The Sonata burns more for other reasons: engine on 24 % of the UDDS vs the twin's 12 %, at
0.85 g/s when on (twin 1.31 g/s), charging gently (median -5.4 kW, twin -12 kW), launching on the engine above 5 kW
below 15 km/h (97-100 % of such seconds), and 19 % of its fuel burnt while braking (engine still running). Measured
capture by speed: 0.55 below 10 km/h, 0.69-0.79 at 10-40 km/h; charge power p99 18 kW, max 30 kW.

| rung | EPA dev P2 FTP median | Sonata FTP / HWY | D3 P2 | verdict |
|---|---|---|---|---|
| v2 (share 0.75) | +17.6 | +4.3 / -5.9 | -5.4 | |
| a: share 1.0 (physical capture) | +29.4 | +15.5 / -3.9 | -13.1 | rejected: every dev P2 too good in the city |
| b: a + regen fade 5-15 km/h (measured) | +28.9 | +15.7 / -4.3 | -12.6 | rejected (fade is real but small) |
| c: b + launch engine (> 5 kW below 15 km/h) + engine on above 60 km/h | +20.8 | +17.1 / -13.7 | -9.6 | rejected: engine-on 0.62 vs 0.34, HWY -14 |

H15, rejected: engine on above 0.3 / 0.5 x P2 motor power (EV only at light load): P2 FTP median +14.1 / +15.5,
Sonata FTP +8.2 / +4.8, HWY unchanged or worse, Panamera HWY -19.6 at 0.3. The city surplus of P2 twins is not a lack
of engine-on time at high power.

### H13 Engine run-on into braking: kept, `hybrid_runon_s` 2 s (new option, hybrids only)

The twin's engine stopped at the first braking second; on the dyno a running engine keeps running into the
deceleration. D3 dev, hot std UDDS/HWFET/US06, braking events entered with the engine on: median run-on 3 s (Sonata
Hybrid, 681 events), 2 s (Accord Hybrid), 2 s (2010 Prius), 1 s (2010 Fusion Hybrid), 1 s (Prius Prime), 2 s (2016 Volt);
0.25-0.8 g/s meanwhile; 5-19 % of the UDDS fuel is burnt while braking (twin: 0). With the option, a running engine
keeps running for 2 s (median of the six medians, measured, not fitted) at the strategy's lowest engine-on point,
charging the pack on top of regen; then it stops as before.

| dev metric | v2 | run-on 2 s | run-on 4 s |
|---|---|---|---|
| VED Toyota car-median fuel; engine-on gap | -1.1 %; -10.9 pts | -0.5 %; -4.9 pts | +0.8 %; -1.6 pts |
| D3 engine-on P2 / PS / series / plug-in PS / plug-in series (measured 0.34 / 0.56 / 0.57 / 0.29 / 0.27) | 0.29 / 0.44 / 0.43 / 0.25 / 0.13 | 0.31 / 0.49 / 0.46 / 0.29 / 0.15 | 0.32 / 0.50 / 0.46 / 0.31 / 0.17 |
| D3 P2 / PS / series / plug-in PS / plug-in series | -5.4 / -20.7 / -6.9 / -9.7 / -5.5 | -5.3 / -20.3 / -6.6 / -9.0 / -5.9 | -5.0 / -19.9 / -6.6 / -9.2 / -6.5 |
| EPA dev Toyota FTP / HWY / US06 / SC03 / cold | +4.0 / -3.0 / +2.0 / +9.1 / -12.7 | +2.9 / -3.4 / +1.3 / +9.5 / -13.7 | +2.4 / -3.7 / +1.1 / +9.4 / -13.8 |
| EPA dev Sonata Hybrid FTP / HWY / US06 / SC03 / cold | +4.3 / -5.9 / -9.5 / +2.1 / -2.9 | +3.0 / -7.5 / -8.7 / -2.7 / -5.3 | +1.9 / -7.3 / -9.2 / +0.7 / -7.0 |
| EPA dev P2 FTP median (n 14) | +17.6 | +15.2 | +14.0 |
| EPA dev Accord Hybrid FTP / HWY / US06 / cold | -4.2 / -4.6 / -8.5 / -10.6 | -4.6 / -4.5 / -9.4 / -11.7 | -5.0 / -5.0 / -9.4 / -12.1 |
| EPA dev plug-in series (Volts) FTP / HWY / US06 / cold | +11.3 / -6.1 / -0.6 / -16.0 | +13.3 / -5.3 / +2.5 / -13.0 | +13.3 / -3.1 / +2.5 / -13.1 |
| EPA dev plug-in power-split (Corsair) FTP / HWY | +7.0 / +1.6 | +6.1 / +0.9 | +5.8 / +0.9 |

Kept at the measured 2 s: it brings engine-on closer for every type and halves the VED Toyota engine-on gap; fuel and
energy move 0.4-2.4 points, both ways. (4 s, the measured p75, is not used: not fitted to the outcome.)

### H14 Gentle charging with run-on: rejected (`hybrid_chg_cap_frac` 0.10, the measured p10 charge power / engine P_max)
Measured charge power while the engine runs (p10): Sonata -11.5, Accord -10.7, Fusion -4.6, Prime -9.2 kW (twin -17 to
-25). Cap 0.10 x P_max on top of run-on: D3 Accord HWFET +1.6 -> +5.5 % (guard: worse by 3.9 points), EPA dev Accord FTP /
HWY / US06 -4.6 / -4.5 / -9.4 -> -6.6 / -6.8 / -11.1, VED Toyota -0.5 -> -1.7 %; D3 P2 -5.3 -> -4.2, series -6.6 -> -4.6.

### H16 Toyota engine-on rule: kept at 45 km/h (with run-on)
| variant | VED Toyota fuel; engine-on gap | EPA dev Toyota FTP / HWY | D3 PS (legacy) |
|---|---|---|---|
| 45 km/h (v2) | -0.5 %; -4.9 pts | +2.9 / -3.4 | -20.3 |
| 55 km/h | +0.3 %; -11.8 pts | +4.5 / -3.4 | -21.3 |
| 72 km/h (v1) | -3.7 %; -18.5 pts | +5.1 / -2.8 | -21.8 |
| 45 km/h above 5 kW only | -1.2 %; -11.1 pts | +3.8 / -3.2 | -20.7 |
The roads (VED) and the EPA dev Toyotas (Grand Highlander Hybrid, LC 500h, Altima Hybrid, RAV4) agree: 45 km/h at any
positive demand. The held-out EPA Toyota deficit of v2 (-5 to -7 %) does not show on the dev cycles (FTP +2.9 %). The
on-road lump (1,250 W) was not re-fitted: with run-on the VED car median is -0.5 %.

### H17 Plug-in charge-sustaining rules: rejected
- Plug-in power-split, speed rule 72 km/h / off (`phev_ps_engine_on_kph`): D3 CS HWFET engine-on 0.85 -> 0.72 / 0.39
  (measured 0.46-0.57: closer), but D3 plug-in PS -9.0 -> -10.0 / -9.9 % and EPA Corsair HWY +0.9 -> +1.6 / -3.7 %.
- Volt-type extended-range rule, engine on above 60 / 50 km/h in CS (`phev_series_cs_on_kph`; the D3 2016 Volt runs
  its engine ~100 % above 60 km/h in CS, the twin 15 % overall vs 27 %): D3 plug-in series -5.9 -> -8.1 / -8.0 %, EPA
  Volts HWY -5.3 -> +10.9 / +9.2, US06 +2.5 -> +8.7 / +9.2 %.
- No dev plug-in series of the gen-1 Volt type other than the 2012 Volt EPA record (which is the D3 exam Volts' row);
  nothing was fitted to it.

### H18-H19 Hybrid cold: rejected
- Power-split heat rule off at 60 C, or 55 / 40 C: no dev number moves (the rule never binds).
- Engine started for heat below 40 C instead of 55 C (new thermal parameter `hybrid_heat_engine_C`): Accord cold -11.7
  -> -11.1, D3 series -6.6 -> -10.5 %: rejected. Warm-up hold 25 C instead of 40 C (`ps_warmup_C`): < 0.7 points.
- Open: EPA dev hybrid cold FTP Toyota -13.7, Accord -11.7, Volts -13.0 % (sim burns more); plug-in P2 +22 %.

### H20 P2 equivalence factor: rejected (`ecms_s0`, new option)
s0 3.2 for P2 (v2 2.6): P2 FTP median +15.2 -> +15.4, Sonata HWY -7.5 -> -5.5. With regen share 1.0 and s0 3.2 / 4.0:
P2 FTP +26.1 / +27.0, D3 P2 -14.2 / -14.4 %.

### Hybrid A/C (d): analysed, not changed
EPA dev SC03 minus FTP error (A/C share): Toyota +6.6, Accord +17.4, plug-in P2 +23.6, Volts +4.5, Sonata -5.7, P2
clutch +3.2 points; D3 dev CS SC03 (fuel_corr): Energis -27 / -37, Volt -37, Prime -28 % against -14 to -20 % on their
std CS UDDS. The one logged electric compressor (Prime, also the Leaf ePlus) matches cop_eta_el 0.50, so no COP
change; the extra fuel is not in the compressor power but in how the cars run their engines and packs when hot
(chilled packs on the Volt and BMW are one candidate). Open.

## Physics v2.1 frozen (before exam-2 and every second-look set)

`roadsim/data/calibration_v2_1.json` (preset `roadsim-physics-v2.1` = v2 + `hybrid_runon_s` 2 s). Guard: v1 and v2
fingerprints reproduce exactly; every non-hybrid run of v2.1 equals v2 (tests/test_roadsim_v2_fingerprint.py); the
v2-improved dev hybrid metrics are not worse: EPA dev Accord HWY -4.6 -> -4.5, D3 Accord HWFET (std) +1.7 -> +1.6, VED
Toyota fuel -1.1 -> -0.5 % and engine-on -10.9 -> -4.9 points, P2 FTP median +17.6 -> +15.2 %. Speed: hybrids 11.4 s for
24 runs in both. **Not the default**: PHYSICS_VERSION stays roadsim-physics-v2, because the fuel and energy gain on dev
is not clear (EPA dev cycle medians move 0.4-2.4 points either way); the gain is in engine operation.

## v2.1 report (computed once, after the freeze commit 1896a7d; nothing changed afterwards)

Exam-2 (the fresh judge; `checks/exam2_report.py` -> `exam2_report.json`), second-look sets (seen before v2.1 was fitted:
`check_cycles --split all`, D3 exam) and `run_all.py --options roadsim-physics-v2.1 --all-splits`;
`scoreboard_roadsim-physics-v2.1.json`, `scoreboard_compare_v2_v2.1.txt` (v2 re-scored with the same harness). Tables
in LIMITS.md. In short:
- **Exam-2 EPA** (409 fresh records, FE %, v2 -> v2.1 FTP): TMED P2 -10.2 -> -9.8, converter P2 +27.9 -> +26.1, clutch
  P2 +17.7 -> +15.4, mild +10.8 -> +8.9, Toyota power-split +18.0 -> +17.2, Ford +9.0 -> +8.9, Honda e:HEV -2.9 -> -3.6,
  plug-in power-split +0.7 -> +0.5, plug-in series +18.5 -> +19.4, plug-in P2 +6.9 -> +7.3. Median absolute error over
  all cycles falls in 6 of 11 groups (by 0.2-1.7 points) and rises in 5 (0.2-1.7).
- **Exam-2 VED** (108 cars, 11,157 trips): Toyota-type HEVs trip median -1.8 -> +0.5 %, car median -1.1 -> +3.1 %,
  engine-on gap -10.1 -> -3.8 points; other HEVs (Ford/Altima labels) +15.0 -> +17.2 %; plug-ins -23 / -20 % unchanged.
- **Second look**: EPA held-out P2 FTP -4.0 -> -3.1, US06 -23.6 -> -17.6, SC03 -17.7 -> -19.8; power-split FTP / HWY
  -5.1 / -7.2 -> -5.6 / -8.9, cold -15.9 -> -17.6; e:HEV -2.0 / -3.8 -> -2.6 / -4.0. D3 exam hybrids -17.8 -> -17.8,
  plug-in power-split -17.8 -> -17.8, plug-in series -22.6 -> -22.6 (v2 re-scored with this harness; first reported
  -16.2 / -27.1). Conventional, BEV, ZTBus, ledger, coast-down, EPA pool, heavy duty: identical to v2 (guard).
- **Generator hybrids** (check_generator): hot UDDS 47.5 -> 47.1 mpg, HWY 45.2 -> 45.1 (EPA MY2022-24 hybrids 59.2 /
  58.2): still ~20 % too thirsty; not addressed.
- **Speed**: hybrids 2-4 % slower than v2 (UDDS single thread 3,085-3,851 x real time).

Verdict: v2.1 improves engine operation (engine-on shares, VED engine-on gap) and moves fuel errors by 0.1-2.5
points both ways. It fixes none of the large hybrid errors, which exam-2 now measures on fresh cars: pre-TNGA Toyota
and Lexus engines (+17 % FTP), TMED P2s (-10 %), converter P2s (+26 %), plug-in series (+19 %), hybrid A/C, and VED
plug-ins (-20 to -23 %). v2 stays the default.

## 14 Nov 2026 — RoadSim v3 pass: hardware on hybrids (physics v3)

Same rules: dev data only. A new clean judge was frozen before any v3 fitting: exam-2, the EPA held-out vehicles, the
D3 exam and the VED exam had all been seen, so they judge nothing here (second looks only).

### H21 Exam-3 frozen (no physics change)

`checks/exam3_hybrids.py` -> `checks/exam3_hybrids.json` (sha256 dcd6ea16…): 64 EPA test-car records no check had
simulated, in held-out families, with each record's test-car fields stored in the file.
- 21 from the MY2000-2007 EPA test-car files (`checks/epa_pre2008.py`; RoadData's table starts at MY2008, so no check
  had ever run on them; downloaded 14 Nov 2026 from epa.gov, git-ignored under data/raw/epa_pre2008/, sha256 of each
  file in the json). Carry-over copies of MY2008+ records are dropped (Prius 2004-2007, RX 400h, Camry, GS 450h and
  Altima 2007 were all copies), which leaves 3 pre-TNGA Toyota records (2001 Prius, 2006 Highlander Hybrid 2WD/4WD).
- 43 MY2008-2026 hybrids and plug-ins that RoadData labels conventional (no hybrid word in the name): FHEV / REX /
  e-tron / TFSIe tokens, hybrid-only model lines (Venza, Crown, Sequoia, Karma) or FTP >= 0.9 x HWY.
- Groups: mild 18, Toyota power-split 14 (3 pre-TNGA, 11 TNGA-era Venza / Crown / Crown Signia / Mazda CX-50),
  converter P2 9 (Explorer and F-150 FHEV, Sequoia), Ford power-split 8 (Maverick, Milan), Honda e:HEV 7, plug-in P2 5,
  plug-in series 2, plug-in power-split 1. 29 of 64 are in families without any exam-2 record.
- Hardware rules for records the old table did not cover (gen-1 Prius and Civic Hybrid, Venza, Crown, CX-50, Prelude,
  Crosstrek plug-in, Sienna, 2026 Maverick, A7 TFSIe) were added as `hybrid_specs.SUPP_HW` / `lookup_ext()` before any
  of these records was simulated; `lookup()` is unchanged.
- Dev additions (`dev_extra`, 9 records of dev families found the same way: Sienna 2021+, Audi Q5 TFSIe / e-tron).
- **Too small for some groups.** No TMED record and 3 pre-TNGA Toyota records are clean; the TMED and older-Toyota
  verdicts stay second looks (exam-2). Plug-ins: 8 records.
