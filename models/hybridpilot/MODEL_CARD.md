# HybridPilot v2.0 — model card

**What it does.** A learned, causal energy-management policy for hybrid vehicles. Every second of traction it decides
engine on/off and the engine power (series, power-split) or engine torque share (P2 parallel), and with it the battery
charge/discharge, from the current state, the past of the trip and (optionally) a navigation preview and distance to go.
It holds charge in HEVs and depletes-then-sustains in plug-ins. Part of ML Ratchet ("Trained on physics. Graded on
reality."). v2 (14 Nov 2026) is a rebuild on RoadSim physics v2; v1 (12 Nov 2026, physics v1) is summarised at the end.

## How it decides
1. 66 causal features: vehicle parameters (architecture, mass, engine and motor W/kg, battery kWh, CdA/mass, Crr,
   SOC window), state (SOC, coolant temperature, on/off timers, battery power limits, aux power, cabin-heat and
   catalyst flags), current demand (speed, acceleration, grade, wheel power), history EMAs (speed 30/300 s, positive
   power 60/300 s, stop share 120 s, distance), optional preview bins (masked when absent), optional nav (remaining km,
   battery energy per km to go).
2. MLP 66 → 96 → 96 → 2 (tanh), 15,938 parameters → log(s + 0.05) and logit P(engine on).
3. SOC feedback as RoadSim's A-ECMS: s ← s + 12 (SOC_ref − SOC) + 0.02 ∫(SOC_ref − SOC) dt (integral clipped at ±30 s;
   plug-ins only below their charge-sustaining level, SOC_lo + 0.05).
4. Among the candidate controls (RoadSim's grid of 20 engine powers or torque fractions plus engine off; series hybrids
   also the same powers through the direct-drive clutch), choose argmin P_fuel + s · P_battery,chemical (+ start fuel if
   starting), with the platform's rules: minimum on 8 s / off 3 s, forced on for catalyst light-off, warm-up, cabin heat,
   the Toyota-type engine-on speed (45 km/h), stay-on under load, and RoadSim's charge-depleting rules for plug-ins.
   The P(on) output is not used.

## Physics / simulator
RoadSim `roadsim-physics-v2` (pinned in every call; v2.1 not used): forward-facing 1 Hz model, EPA ALPHA engine and
motor maps, Thevenin battery, eCVT / P2 (torque converter, start energy, brake blending 0.75) / series (direct drive),
coolant + oil nodes, two-node cabin HVAC, tyre model, on-road extra loads, production hybrid rules. Calibrated on dev
data only by the RoadSim agents (shared/vehicle/roadsim/CALIBRATION.md); known errors in LIMITS.md. No physics patch was
made here. Upstream helpers now used instead of model-folder copies: RoadSim's VED twin builder
(`checks/check_ved_twin.build_twin`) and its OBD label model (`roadsim.obd.label_fuel`).

## Teacher and training
- DP (Bellman recursion over engine on/off × 121 SOC points on RoadSim's candidate controls, stage tables built with the
  v2 powertrain: start energy, brake blending, direct drive). Labels at any visited state: DP's on/off choice and the
  costate s* = −∂J/∂SOC / (V_oc Q). HEV teacher values end-of-trip charge linearly at the trip's typical s* (2 passes);
  PHEV teacher depletes to SOC_lo + 0.05 by the known trip end. When the teacher drives, it obeys the platform's
  forced-on rules.
- Data: 400 generated hybrids (18 class × architecture combos, v2 donor model) × 835 VED `train_noise` routes (12
  unlabelled cars), EPA/HD cycles with random weather, synthetic routes. −20 to 36 °C, warm fragments and cold soaks,
  random start SOC, preview length and nav flag.
- Round 0: 800 trips driven by DP with exploration bursts (436k labelled decisions). DAgger round 1: 600 trips driven by
  the round-0 net (teacher 30 %). Released net: trained on both (763k decisions), Adam, 25 epochs (GPU), Huber on log s,
  BCE on on/off. numpy == torch to 5e-6; the numpy module `hybridpilot.py` reproduces the training policy exactly
  (identical fuel on 23 frozen and dev trips).
- No real labelled example was ever a training sample.

## Grading (matched SOC)
Reference: DP with the physics' minimum on/off times (8 s / 3 s, as every causal strategy) and the real start fuel (no
virtual start penalty). For each strategy on each trip DP is re-solved to end at that strategy's own final SOC;
**% of DP = 1 − (F_strategy − F_DP(E_strategy)) / F_DP,CS**. DP ignores the forced-on production rules, so causal
strategies on Toyota-type power-splits cannot reach 100 %. Charge holding: HEV ends no farther from the window middle
than it started + 15 % of the window and never leaves the window by > 0.02; PHEV never drops > 0.03 below its CS level.

## Ratchet v2 (physics changed, so every decision was re-taken)
Frozen set: 106 trips rebuilt on v2 (36 held-out generated vehicles × held-out `train_noise` routes / synthetic /
cycles). Dev gate: 76 trips (calibrated twins of the 7 VED dev hybrids, 18 dev generated vehicles, full-ZTBus dev legs).
Frozen skill = 1 − Σ_strata gap(policy) / Σ_strata gap(rule strategies), gap = 1 − min(%DP, 1). Keep if skill
> incumbent + 0.01 and dev %DP ≥ incumbent − 0.2 pt and dev charge-holding ≥ incumbent − 3 pt.

| rung | change | frozen skill | frozen %DP | frozen holds | dev %DP | dev holds | decision |
|---|---|---|---|---|---|---|---|
| R0 | RoadSim A-ECMS (incumbent) | 0.255 | 98.40 | 96 % | 98.20 | 97 % | incumbent |
| R1 | behaviour cloning | 0.691 | 99.27 | 63 % | 99.33 | 70 % | rejected (dev holds) |
| R2 | DAgger round 1 + Kp 6 | 0.620 | 99.20 | 78 % | 99.27 | 88 % | rejected (dev holds) |
| R3 | on/off by the cost (v1's release recipe) | 0.719 | 99.51 | 82 % | 99.54 | 87 % | rejected (dev holds) |
| R4 | R3 + 30 s preview | 0.730 | 99.56 | 85 % | 99.34 | 91 % | rejected (dev holds) |
| R5 | policy-search tune of head knobs | 0.734 | 99.55 | 88 % | 99.71 | 88 % | rejected (dev holds) |
| R6 | DAgger round 2 | 0.646 | 99.38 | 85 % | 99.54 | 88 % | rejected (dev holds) |
| R7 | R6 + 30 s preview | 0.677 | 99.48 | 89 % | 99.54 | 91 % | rejected (dev holds) |
| R8 | R3 + A-ECMS's SOC feedback gains (Kp 12, Ki 0.02) | 0.659 | 99.33 | 93 % | 99.19 | 97 % | **kept, released** |
| R9 | re-teach with the twins' mild P1 class in training | 0.657 | 99.32 | 95 % | 99.15 | 97 % | rejected (−0.002) |
| R10 | teacher = DP with 8 s / 3 s timers (causal DP proxy) | 0.624 | 99.27 | 95 % | 99.12 | 95 % | rejected (−0.036) |
| R11 | drivability: 20 kJ start cost in the decision | 0.560 | 99.00 | 91 % | 99.01 | 97 % | rejected (−0.100; starts 3.10 → 2.90 /km) |
| R12 | distance-to-go plug-in plan (nav) | 0.659 | 99.33 | 92 % | 99.18 | 97 % | rejected (−0.001) |
| R13 | 3 × 256 network on the GPU | 0.671 | 99.36 | 95 % | 99.47 | 93 % | rejected (dev holds 93 %) |
| R14 | R8 + 30 s preview | 0.667 | 99.38 | 97 % | 99.10 | 99 % | rejected (+0.007) |
| R15 | policy-search re-tune incl. Ki | 0.682 | 99.40 | 83 % | 99.28 | 88 % | rejected (dev holds) |

What the ladder says: on v2 the network's own s drifts the charge (dev holding 87–91 % with Kp 6, against 99 % on v1);
A-ECMS's stronger SOC feedback fixes holding and costs 0.06 of frozen skill against R3. None of the four new ideas from
V2_PLAN (twins' hardware classes, timed-DP teacher, start cost, distance-to-go plan) beat the margin.

## Final exam (once, after the release choice was committed)
430 trips: 216 real VED exam routes (2 per exam HEV/PHEV car, fixed random draw) driven by RoadSim's calibrated twins of
those cars, 214 by 54 held-out generated hybrids (3 per combo) on exam routes, full-ZTBus exam legs (buses) and HD
cycles (Class 8). Cell = mean % of DP (charge-holding rate). HybridPilot: no preview, nav on a random half of trips.
v1 = the v1 release (its network, Kp 6, cost-based on/off) re-scored on the same exam under v2 physics.

| stratum | n | rule | ECMS | A-ECMS | MPC 15 s* | HybridPilot v2 | v1 release | v2 − A-ECMS [95 % CI] | starts/km v2 / A-ECMS |
|---|---|---|---|---|---|---|---|---|---|
| all | 430 | 97.39 (93%) | 98.53 (74%) | 98.32 (96%) | 98.80 (94%) | **98.84 (95%)** | 99.00 (89%) | +0.53 [+0.20, +0.82] | 4.0 / 4.5 |
| power-split HEV | 148 | 97.92 (97%) | 97.75 (69%) | 97.52 (95%) | 97.60 (93%) | 98.41 (93%) | 98.42 (80%) | +0.89 [+0.75, +1.04] | 4.7 / 4.8 |
| mild P1 (CR-Z type) | 56 | 94.02 (84%) | 98.72 (43%) | 97.51 (96%) | 98.47 (96%) | 98.10 (95%) | 98.52 (91%) | +0.60 [+0.29, +0.83] | 6.2 / 6.6 |
| P2 HEV | 39 | 94.87 (90%) | 99.77 (51%) | 98.56 (85%) | 99.46 (72%) | 99.88 (95%) | 100.18 (77%) | +1.32 [+0.37, +2.85] | 3.5 / 3.3 |
| series HEV | 24 | 97.41 (75%) | 98.97 (54%) | 98.75 (92%) | 99.33 (88%) | 99.78 (96%) | 99.78 (96%) | +1.04 [+0.68, +1.39] | 5.2 / 5.1 |
| power-split PHEV | 57 | 98.82 (96%) | 96.96 (100%) | 98.94 (100%) | 99.30 (100%) | 98.65 (91%) | 98.68 (93%) | −0.29 [−2.31, +1.53] | 2.7 / 4.8 |
| series PHEV (EREV) | 46 | 100.20 (100%) | 100.69 (100%) | 100.40 (100%) | 101.50 (100%) | 100.04 (98%) | 100.21 (100%) | −0.35 [−1.09, +0.12] | 1.4 / 2.2 |
| bus (series) | 12 | 96.52 (92%) | 98.60 (100%) | 96.08 (100%) | 97.04 (100%) | 96.78 (100%) | 97.95 (100%) | +0.70 [+0.21, +1.29] | 5.0 / 4.8 |
| bus (P2) | 12 | 96.68 (92%) | 99.36 (100%) | 98.70 (100%) | 99.09 (100%) | 98.23 (100%) | 99.06 (100%) | −0.48 [−0.63, −0.40] | 4.2 / 4.0 |
| truck (P2) | 24 | 96.64 (100%) | 98.82 (88%) | 99.21 (100%) | 99.35 (100%) | 99.32 (100%) | 99.24 (100%) | +0.11 [−0.11, +0.31] | 2.9 / 3.0 |
| truck (PHEV series) | 12 | 100.37 (100%) | 100.04 (100%) | 99.39 (100%) | 99.71 (100%) | 100.41 (100%) | 100.33 (100%) | +1.02 [+0.41, +1.59] | 2.0 / 3.5 |

\* MPC (15 s exact future) and DP (whole trip) see the future: benchmarks only. Rule = powersplit_rule / parallel_rule /
series_thermostat / phev_cdcs. Values above 100 % mean DP was not tight (warm-engine assumption, grid).
Exam skill (rule gap closed): HybridPilot v2 0.529, v1 release 0.615, A-ECMS 0.321, ECMS 0.502, MPC 0.427.
Twins of real cars: v2 98.60 % vs A-ECMS 98.05 % (+0.55 [−0.01, +1.06]); generated vehicles 99.09 % vs 98.58 % (+0.50
[+0.18, +0.78]). v2 − v1 release: −0.16 [−0.26, −0.06] points of % of DP, with charge held on 95 % vs 89 % of trips.
Error bars: the policy outputs no uncertainty, so there is no coverage calibration; intervals are cluster-bootstrap (by
vehicle) intervals of the paired difference.

## Simulator vs real logs (report only; nothing tuned on exam)
Each VED hybrid trip driven by its calibrated twin (warm fragment, logged start SOC), simulated fuel passed through
RoadSim's OBD label model and compared with VED's logged fuel on valid seconds (plug-ins: fuel + battery energy).
Trip-median error, % (car median):

| | dev (12 trips per car) | exam (3 trips per car) | engine-on gap, exam (points) | starts/km sim / real, exam |
|---|---|---|---|---|
| power-split HEV, HybridPilot | −4.6 (−5.5), 36 trips | **+2.2 (+3.5)**, 168 trips | −7.6 | 3.65 / 1.94 |
| power-split HEV, A-ECMS | −1.0 (−1.2) | +6.7 (+7.0) | −7.0 | 2.51 / 1.94 |
| mild P1, HybridPilot | +2.9 (+0.3), 24 | **+0.3 (+2.4)**, 84 | −3.8 | 3.78 / 1.43 |
| P2, HybridPilot | — | +1.6 (−4.1), 5 | −42.6 | 2.46 / 0.72 |
| plug-in power-split, HybridPilot | −2.9, 12 | **−22.4**, 33 | | |
| plug-in series, HybridPilot | −7.8, 11 | **−30.9**, 31 | | |

The V2_PLAN target (simulator–real gap under 8 %) is met for the exam hybrids (+0.3 to +2.2 % trip median) and on dev for
every architecture, and not met for exam plug-ins (−22 % and −31 %, RoadSim's known plug-in gap: LIMITS.md item 9).
Simulated hybrids start their engine about twice as often as the real cars.

## Limits
- Graded in simulation; plug-ins on real roads read 22–31 % low, P2 engine-on share is far below the real cars'.
- Charge holding 95 % vs A-ECMS 96 %; losses on plug-ins (inside wide intervals) and P2 buses (−0.48 pt).
- The v1 release burns 0.16 pt less on the v2 exam but fails to hold charge on 11 % of trips.
- DP reference ignores the production forced-on rules; it assumes a warm engine.
- Heavy duty: buses on real Zurich routes, trucks on cycles and city routes only.

## v1 (12 Nov 2026, physics v1) in one paragraph
Same network and teacher on RoadSim physics v1 with v1 twins; released rung R3 (Kp 6, on/off by cost). Exam (430 trips,
v1 physics): 99.49 % of DP vs A-ECMS 99.35 % (+0.13 [−0.06, +0.29]), charge held on 97 %; simulated fuel 0.80× logged on
exam power-splits. Full v1 card: git history of this file (commit before 14 Nov 2026) and training/ratchet.json.

## Files
`hybridpilot.py` (features, MLP, decision, RoadSim adapter), `weights/hybridpilot_v2.npz` (v1: `hybridpilot_v1.npz`),
`pipeline.py` (pins roadsim-physics-v2), `demo.py`, `examples/` (3 VED exam trips + calibrated twin specs), `training/`
(all scripts; `v2_*.sh` ran the v2 rebuild; raw logs `ratchet_v2.json`, `ratchet.json` for v1), `requirements.txt`.

## Data and licences
VED (Oh, LeBlanc, Peng, IEEE T-ITS 2020; Apache-2.0) — routes and the example trips redistributed under Apache-2.0.
USGS 3DEP lidar (public domain). IEM ASOS weather. ZTBus (Widmer, Ritter, Onder, Scientific Data 2023; ETH Zurich,
CC BY 4.0). EPA cycles, test-car lists, fueleconomy.gov and 40 CFR 1037 (public domain). RoadSim and RoadData (ML
Ratchet). Methods: Paganelli et al. 2002 (ECMS), Onori, Serrao & Rizzoni 2010 (A-ECMS), Sundström & Guzzella 2009
(DP), Ross, Gordon & Bagnell 2011 (DAgger).
