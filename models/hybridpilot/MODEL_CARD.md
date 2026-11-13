# HybridPilot v1.0 — model card

**What it does.** A learned, causal energy-management policy for hybrid vehicles. Every second of traction it decides
engine on/off and the engine power (series, power-split) or engine torque share (P2 parallel), and with it the battery
charge/discharge, from the current state, the past of the trip and (optionally) a navigation preview and distance to go.
It holds charge in HEVs and depletes-then-sustains or blends in plug-ins. Part of ML Ratchet ("Trained on physics.
Graded on reality.").

## How it decides
1. 66 causal features: vehicle parameters (architecture, mass, engine and motor W/kg, battery kWh, CdA/mass, Crr,
   SOC window), state (SOC, engine coolant temperature, on/off timers, battery power limits, aux power, cabin-heat and
   catalyst flags), current demand (speed, acceleration, grade, wheel power), history EMAs (speed 30/300 s, positive
   power 60/300 s, stop share 120 s, distance), optional preview bins (1-5, 6-10, 11-20, 21-30, 31-60 s of positive and
   negative power, speed, stops; masked when absent), optional nav (remaining km, battery energy per km to go).
2. MLP 66 → 96 → 96 → 2 (tanh), 15,938 parameters → log(s + 0.05) and logit P(engine on).
3. s ← s + Kp (SOC_ref − SOC), Kp = 6 (plug-ins: only below their CS level, SOC_lo + 0.05).
4. Among the candidate controls (RoadSim's grid of 20 engine powers or torque fractions plus engine off), choose
   argmin P_fuel + s · P_battery,chemical (+ engine start fuel if starting), min on 4 s / off 2 s, forced on for
   catalyst light-off and cabin heat. The P(on) output is not used by the kept policy (see ratchet R1-R3).

## Physics / simulator
RoadSim (Phase 1, `roadsim-physics-v1`, unmodified): forward-facing 1 Hz vehicle model, EPA ALPHA engine and motor
maps, Thevenin battery, eCVT / P2 / series powertrains, HVAC, ledger closure < 0.5 % (checked on every run).
Its checks: power-split HEV cycles −2.3 % vs EPA, P2 FTP +12 % (optimistic), cold FTP +10-16 % (too mild).
No physics patch was made in this project.

## Teacher and training
- DP (Bellman recursion over engine on/off × 121 SOC points on RoadSim's candidate controls) per trip. Labels at any
  visited state: DP's on/off choice and the costate s* = −∂J/∂SOC / (V_oc Q). HEV teacher values end-of-trip
  charge linearly at the trip's typical s* (2-pass) so labels near the end are causal-meaningful; PHEV teacher depletes
  to SOC_lo + 0.05.
- Data: 400 generated hybrids (18 class × architecture combos: car/SUV/van/pickup power-split, P2, series, PHEV
  power-split and series; series and P2 buses; P2 Class 6 and Class 8 trucks; series-PHEV Class 6) × routes:
  835 VED `train_noise` trips (12 unlabelled cars; real Ann Arbor speed, lidar grade, ASOS weather), EPA/HD cycles with
  random weather, synthetic urban/suburban/highway/hilly/bus-stop routes. −20 to 36 °C, warm fragments and cold soaks,
  random start SOC, random preview length (0-60 s) and nav flag.
- Round 0: 800 trips driven by DP with exploration bursts (428k labelled decisions). DAgger round 1: 600 trips driven
  by the round-0 net (teacher 30 %). Final net: trained on both (750k decisions), Adam, 25 epochs, Huber on log s,
  BCE on on/off. numpy == torch to 5e-6.
- No real labelled example was ever a training sample.

## Grading (matched SOC)
For each strategy on each trip, DP is re-solved to end at that strategy's own final SOC (plus DP points at fixed
targets); F_DP(E) is interpolated at the strategy's net battery energy E.
**% of DP = 1 − (F_strategy − F_DP(E_strategy)) / F_DP,CS**, F_DP,CS = DP fuel when sustaining the start SOC.
Charge holding: HEV ends no farther from the window middle than it started + 15 % of the window and never leaves the
window by > 0.02; PHEV never drops > 0.03 below its CS level.
The Phase 1 fixed-factor correction (1/(0.33·0.90)) over-credits charge: on exam HEV trips it rates ECMS at 105.2 %
of DP while matched SOC gives 99.6 %; A-ECMS 101.1 % vs 99.0 %.

## Ratchet (frozen set: 106 trips, 36 held-out generated vehicles × held-out `train_noise` routes / synthetic / cycles;
dev gate: 16 VED dev cars' routes (twins of the 7 dev hybrids + 18 dev generated vehicles + ZTBus dev legs), 76 trips)
Frozen skill = 1 − Σ_strata gap(policy) / Σ_strata gap(rule strategies), gap = 1 − min(%DP, 1). Keep if skill
> incumbent + 0.01 and dev %DP ≥ incumbent − 0.2 pt and dev charge-holding ≥ incumbent − 3 pt.

| rung | change | frozen skill | frozen %DP | frozen holds | dev %DP | dev holds | decision |
|---|---|---|---|---|---|---|---|
| R0 | RoadSim A-ECMS (incumbent) | 0.564 | 98.88 | 98 % | 98.99 | 100 % | incumbent |
| R1 | behaviour cloning, s and on/off from net | 0.592 | 98.70 | 69 % | 99.64 | 74 % | rejected (dev charge holding) |
| R2 | DAgger round 1 + SOC feedback Kp 6 | 0.596 | 98.88 | 94 % | 99.20 | 99 % | kept |
| R3 | on/off by the cost with learned s (head unused) | 0.702 | 99.31 | 94 % | 99.26 | 99 % | **kept (final)** |
| R4 | R3 + 30 s preview | 0.709 | 99.34 | 94 % | 99.26 | 99 % | rejected (+0.007 < 0.01) |
| R5 | policy-search fine-tune of head knobs on 60 training trips | 0.671 | 99.19 | 96 % | 99.13 | 99 % | rejected |
| R6 | DAgger round 2 (+600 trips by R3 policy) | 0.674 | 99.28 | 94 % | 99.22 | 99 % | rejected |
| R7 | R6 + 30 s preview | 0.677 | 99.23 | 92 % | 99.21 | 99 % | rejected |

## Final exam (never used for any choice)
430 trips: 216 real VED exam routes (2 per exam HEV/PHEV car, fixed random draw) driven by RoadSim twins of those cars,
214 by 54 held-out generated hybrids (3 per combo) on exam routes, ZTBus exam legs (buses) and HD cycles (Class 8).
Cell = mean % of DP (charge-holding rate). HybridPilot: no preview, nav on a random half of trips.

| stratum | n | rule | ECMS (const s) | A-ECMS | MPC 15 s* | HybridPilot | HP − A-ECMS [95 % CI, vehicle bootstrap] | starts/km HP / A-ECMS |
|---|---|---|---|---|---|---|---|---|
| all | 430 | 98.28 (94%) | 99.91 (82%) | 99.35 (99%) | 99.65 (99%) | **99.49 (97%)** | +0.13 [−0.06, +0.29] | 3.8 / 4.1 |
| power-split HEV | 204 | 98.59 (96%) | 99.52 (74%) | 98.90 (99%) | 99.12 (98%) | 99.31 (96%) | +0.41 [+0.29, +0.55] | 4.3 / 4.3 |
| P2 HEV | 39 | 93.83 (87%) | 99.37 (87%) | 99.07 (100%) | 99.66 (100%) | 99.17 (95%) | +0.10 [−0.11, +0.28] | 4.5 / 4.5 |
| series HEV | 24 | 98.14 (71%) | 99.59 (67%) | 99.37 (96%) | 99.55 (96%) | 99.47 (92%) | +0.10 [−0.01, +0.20] | 4.3 / 4.3 |
| power-split PHEV | 57 | 99.92 (98%) | 100.81 (98%) | 99.84 (100%) | 100.15 (100%) | 99.77 (96%) | −0.07 [−0.56, +0.50] | 2.8 / 3.8 |
| series PHEV (EREV) | 46 | 100.23 (98%) | 100.99 (100%) | 100.93 (100%) | 101.31 (100%) | 100.00 (100%) | −0.93 [−2.43, −0.09] | 1.7 / 3.0 |
| bus (series) | 12 | 95.31 (83%) | 100.11 (83%) | 99.12 (100%) | 99.40 (100%) | 99.78 (100%) | +0.66 [+0.45, +0.99] | 4.4 / 4.9 |
| bus (P2) | 12 | 96.44 (100%) | 99.71 (75%) | 98.86 (100%) | 99.34 (100%) | 99.03 (100%) | +0.17 [+0.06, +0.28] | 5.2 / 5.4 |
| truck (P2) | 24 | 96.74 (100%) | 99.91 (71%) | 99.49 (100%) | 99.88 (100%) | 99.54 (100%) | +0.05 [−0.09, +0.18] | 3.5 / 3.7 |
| truck (PHEV series) | 12 | 100.35 (100%) | 100.61 (100%) | 99.97 (100%) | 100.09 (100%) | 100.16 (100%) | +0.19 [−0.27, +0.45] | 2.2 / 3.5 |

\* MPC (15 s exact future) and DP (whole trip) see the future: benchmarks only. Rule = powersplit_rule / parallel_rule /
series_thermostat / phev_cdcs. Constant-s ECMS is the most fuel-efficient at its own energy use but fails to hold charge on
18 % of trips. Values above 100 % mean DP was not tight (warm-engine assumption, no catalyst rule, grid).
Exam skill (rule-gap closed): HybridPilot 0.747, A-ECMS 0.683, MPC 0.782, ECMS 0.843.
Twins of real cars: HP 99.45 % vs A-ECMS 99.32 %; generated vehicles 99.52 % vs 99.39 %.
Error bars: the policy outputs no uncertainty, so there is no coverage calibration; intervals above are
cluster-bootstrap (by vehicle) intervals of the paired difference.

## Real-data sanity check (simulator vs VED logs, RoadSim default A-ECMS on twins, warm fragments)
| | trips | sim ÷ logged fuel (total) | engine on logged / sim | starts/km logged / sim |
|---|---|---|---|---|
| dev power-split HEV | 20 | 0.75 | 59 % / 33 % | 1.9 / 3.6 |
| exam power-split HEV | 252 | 0.80 | 55 % / 35 % | 1.8 / 3.2 |
| exam P2 HEV | 5 | 0.92 | 74 % / 40 % | 1.2 / 3.1 |
| exam power-split PHEV | 33 | 3.6 (CD-CS 2.5) | 14 % / 39 % | 1.1 / 3.7 |
| exam series PHEV | 31 | 3.3 (CD-CS 1.9) | 17 % / 29 % | 0.3 / 3.0 |
Real Toyota hybrids keep the engine turning above ~70 km/h (100 % of dev seconds above 72 km/h) and run it at cold
stops; simulated plug-ins start the engine for cabin heat where real ones heat electrically; MAF-based fuel counts air
through an unfuelled spinning engine. Strategy differences (1-3 %) are far below this ~20 % (HEV) bias: the ranking is a
ranking inside RoadSim's physics. Nothing was tuned on these numbers.

## Limits
- Graded in simulation; the simulator's absolute fuel on real trips is ~20 % low (HEV) and plug-in engine use is too high.
- Small margin over A-ECMS overall (+0.13 pt, CI includes 0); clear gains on power-split HEVs and series buses; worse on
  series plug-ins (−0.9 pt) where DP is not a tight reference.
- Charge holding 97 % vs A-ECMS 99 %.
- Preview did not pass the ratchet; the on/off head is unused.
- DP benchmark assumes a warm engine and no catalyst rule.
- Heavy duty: buses on real Zurich routes, trucks on cycles and city routes only.

## Files
`hybridpilot.py` (features, MLP, decision, RoadSim adapter), `weights/hybridpilot_v1.npz`, `pipeline.py`, `demo.py`,
`examples/` (3 VED exam trips + twin specs), `training/` (all scripts), `requirements.txt`.

## Data and licences
VED (Oh, LeBlanc, Peng, IEEE T-ITS 2020; Apache-2.0) — routes and the example trips redistributed under Apache-2.0.
USGS 3DEP lidar (public domain). IEM ASOS weather. ZTBus (Widmer et al., ETH Zurich, CC-BY-4.0). EPA cycles and
40 CFR 1037 (public domain). RoadSim and RoadData (ML Ratchet Phase 1). Methods: Paganelli et al. 2002 (ECMS), Onori,
Serrao & Rizzoni 2010 (A-ECMS), Sundström & Guzzella 2009 (DP), Ross, Gordon & Bagnell 2011 (DAgger).
