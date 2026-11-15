# HybridPilot v2.1 — model card

**What it does.** A learned, causal energy-management policy for hybrid vehicles. Every second of traction it decides
engine on/off and the engine power (series, power-split) or engine torque share (P2 parallel), and with it the battery
charge/discharge, from the current state, the past of the trip and (optionally) a navigation preview and distance to go.
It holds charge in HEVs and depletes-then-sustains in plug-ins. Part of ML Ratchet, but unlike most of its models it
is graded in simulation: real routes, simulated cars (see Grading). v2.1 (15 Nov 2026) runs on RoadSim physics v3.1
with a new charge-sustaining rule and a fixed equivalence factor for buses and mild hybrids; it keeps v2.0's network.
v2.0 (14 Nov 2026, physics v2) and v1 (12 Nov 2026, physics v1) are summarised at the end.

## How it decides
1. 66 causal features: vehicle parameters (architecture, mass, engine and motor W/kg, battery kWh, CdA/mass, Crr,
   SOC window), state (SOC, coolant temperature, on/off timers, battery power limits, aux power, cabin-heat and
   catalyst flags), current demand (speed, acceleration, grade, wheel power), history EMAs (speed 30/300 s, positive
   power 60/300 s, stop share 120 s, distance), optional preview bins (masked when absent), optional nav (remaining km,
   battery energy per km to go).
2. MLP 66 → 96 → 96 → 2 (tanh), 15,938 parameters → log(s + 0.05) and logit P(engine on).
3. Buses and mild P1 hybrids (v2.1): s is replaced by a fixed value, series bus 2.8, P2 bus 2.6, mild 2.8 (fitted on 30
   training-distribution simulations; see Ratchet v2.1). Every other class uses the network's s.
4. SOC feedback as RoadSim's A-ECMS, with a dead band on the proportional term (v2.1):
   s ← s + 12 dz(SOC_ref − SOC) + 0.02 ∫(SOC_ref − SOC) dt, dz(e) = sign(e) max(|e| − 0.03, 0) (integral clipped at ±30 s;
   plug-ins only below their charge-sustaining level, SOC_lo + 0.05). v2.0 used dz(e) = e.
5. Among the candidate controls (RoadSim's grid of 20 engine powers or torque fractions plus engine off; series hybrids
   also the same powers through the direct-drive clutch), choose argmin P_fuel + s · P_battery,chemical (+ start fuel if
   starting), with the platform's rules: minimum on 8 s / off 3 s, forced on for catalyst light-off, warm-up, cabin heat,
   the Toyota-type engine-on speed (45 km/h), stay-on under load, and RoadSim's charge-depleting rules for plug-ins.
   The P(on) output is not used.

## Physics / simulator
RoadSim `roadsim-physics-v3.1` (pinned in every call; v2.0 used `roadsim-physics-v2`). v3.1 = v2 + a running engine
keeps running 2 s into braking (v2.1), the P2 converter held open in 1st and 2nd gear and the generator's hybrid hardware
(v3), an engine-generation scale for hybrids on conventional maps by model year (v3.1; VED twins carry no model year, so
it does not act on them). Chosen on dev only: with v2.0 driving the 84 dev twin trips, v2 → v3.1 moved power-split fuel
from −4.6 to −2.4 % of the logs (trip median), the engine-on gap from −9.3 to −4.4 points and engine starts from 3.67 to
2.66 per km (real 2.32); mild fuel +2.9 → +3.4 %, engine-on gap −8.6 → −2.1 points; plug-ins unchanged. The rest as v2:
forward-facing 1 Hz model, EPA ALPHA engine and
motor maps, Thevenin battery, eCVT / P2 (torque converter, start energy, brake blending 0.75) / series (direct drive),
coolant + oil nodes, two-node cabin HVAC, tyre model, on-road extra loads, production hybrid rules. Calibrated on dev
data only (shared/vehicle/roadsim/CALIBRATION.md); known errors in LIMITS.md. No physics patch was made here.
The calibrated twins are rebuilt per physics (each car's road load and engine level re-matched to its fueleconomy.gov
values under v3.1).
Upstream helpers now used instead of model-folder copies: RoadSim's VED twin builder (`checks/check_ved_twin.build_twin`) and its OBD label model (`roadsim.obd.label_fuel`).

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
  (v2.1 rule: identical fuel and final SOC on 24 frozen and dev trips, `training/v21_numpy_check.py`).
- These banks were simulated on physics v2; v2.1 keeps that network. The same recipe on v3.1 banks (R18) was rejected.
- No real labelled example was ever a training sample.

## Grading (matched SOC)
Reference: DP with the physics' minimum on/off times (8 s / 3 s, as every causal strategy) and the real start fuel (no
virtual start penalty). For each strategy on each trip DP is re-solved to end at that strategy's own final SOC;
**% of DP = 1 − (F_strategy − F_DP(E_strategy)) / F_DP,CS**. DP ignores the forced-on production rules, so causal
strategies on Toyota-type power-splits cannot reach 100 %. Charge holding: HEV ends no farther from the window middle
than it started + 15 % of the window and never leaves the window by > 0.02; PHEV never drops > 0.03 below its CS level.

## Ratchet v2.1 (physics v3.1, 15 Nov 2026)
Physics changed (v2 → v3.1, decided on dev: see Physics), so the frozen set (106 trips, same draws and seeds; generated
hybrids carry the v3 generator hardware), the dev gate (76 trips) and the tune set were rebuilt on v3.1, and every v2
rung was re-scored in order with its v2 network and settings (no retraining). Same rule and margin as v2. Then three new
rungs. Their settings were chosen on a new 190-trip probe set from the training distribution (training fleet incl. mild
P1, training routes, new seed, 10 trips per class), never on dev or exam.

| rung | change | frozen skill | frozen %DP | frozen holds | dev %DP | dev holds | decision |
|---|---|---|---|---|---|---|---|
| R0 | RoadSim A-ECMS (incumbent) | 0.144 | 97.50 | 95 % | 98.14 | 97 % | incumbent |
| R1–R7 | v2's R1–R7 re-scored | 0.644–0.738 | 99.03–99.33 | 61–84 % | 98.94–99.55 | 67–91 % | rejected (dev holds, gate 94 %) |
| R8 | v2's R8 re-scored (the v2.0 release) | 0.662 | 99.06 | 90 % | 98.94 | 96 % | kept |
| R9–R15 | v2's R9–R15 re-scored | 0.553–0.663 | 98.67–99.12 | 82–93 % | 98.83–99.19 | 92–99 % | rejected (none beats R8 by 0.01) |
| R16 | dead band ±0.03 on the proportional SOC feedback | 0.689 | 99.19 | 87 % | 99.16 | 95 % | kept (+0.027) |
| R17 | R16 + fixed s for buses (series 2.8, P2 2.6) and mild P1 (2.8) | 0.704 | 99.22 | 87 % | 99.28 | 95 % | **kept, released as v2.1** (+0.015) |
| R18 | re-teach on v3.1 banks (800 teacher + 600 DAgger trips), R17's rule | 0.698 | 99.22 | 88 % | 99.13 | 93 % | rejected (−0.006) |

Every rung's numbers: `ratchet.yaml` and `training/ratchet_v2_1.json`.

Why these rungs. Dev showed the strong A-ECMS feedback costing fuel where constant-s ECMS does well: the dev series
bus burned 95.3 % of DP against 98.3 % for ECMS, the dev mild twins 97.8 % against 99.7 %. The probe set reproduced it
(buses: ECMS 100.0 % vs the v2.0 rule 98.8 / 98.6 %) and showed that A-ECMS with s0 2.6 also beats the network on
buses (99.4 / 99.1 %): the network's s is weak on buses, not only the feedback. Probe skill / charge holding of the
decision-rule variants (v2.0 network): v2.0 rule 0.529 / 93 %; dead band 0.03 0.595 / 92 %; 0.05 0.598 / 91 %;
0.03 without Ki 0.612 / 88 %; Kp 12 without Ki 0.557 / 91 %; SOC error low-passed 120 s 0.455 / 88 %; ECMS cubic window
term with Kp 6 0.575 / 92 %; window-normalised gains 0.532 / 92 %. The DP teacher's median costate on training buses
is 2.21 (P2) / 2.30 (series) (`training/prior_p31r0.json`); as a fixed s it gave 99.41 / 99.55 % on probe buses. A grid
on the 20 probe buses picked P2 2.6, series 2.8 (99.84 / 100.33 %); on the 10 probe mild trips 2.8 was the highest
value that kept every trip in its window (94.56 %, holds 100 %; 3.3 gave 95.43 % but held 70 %). Band 0.03 + these
values: probe skill 0.648 / 92 %. Tried on the probe and not taken to the ratchet: a hard band, band 0.02, a stronger
integral (Ki 0.04). Raw log: `training/explore_v2_1.json`.

What got worse: charge holding. Dev 94.7 % vs 96.1 % for R8 (72 vs 73 of 76 trips), frozen 87 % vs 90 %. The failing
trips end 0.02–0.03 beyond their allowance: the band leaves end-of-trip drift less corrected.

## Ratchet v2.0 (physics v2, 14 Nov 2026)
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

## Final exam v2.1 (once, after the release choice was committed: d9fd0fd)
430 trips: 216 real VED exam routes (2 per exam HEV/PHEV car, fixed random draw) driven by RoadSim's calibrated twins of
those cars, 214 by 54 held-out generated hybrids (3 per combo) on exam routes, full-ZTBus exam legs (buses) and HD
cycles (Class 8). Same draws as v2.0's exam; **everything in this table is on physics v3.1**, so the twins, generated
hardware and every baseline were re-run. Cell = mean % of DP (charge-holding rate). HybridPilot: no preview, nav on a
random half of trips. v2.0 = its network and settings (no dead band, no fixed s) re-scored on v3.1; v1 = the v1 release
(its network, Kp 6, cost-based on/off) re-scored on v3.1.

| stratum | n | rule | ECMS | A-ECMS | MPC 15 s* | HybridPilot v2.1 | v2.0 | v1 release | v2.1 − A-ECMS [95 % CI] | starts/km v2.1 / A-ECMS |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 430 | 97.00 (92%) | 98.31 (75%) | 98.20 (97%) | 98.67 (95%) | **98.57 (93%)** | 98.43 (96%) | 98.54 (91%) | +0.37 [−0.18, +0.84] | 3.0 / 3.2 |
| power-split HEV | 148 | 97.44 (93%) | 97.44 (72%) | 97.05 (97%) | 97.23 (94%) | **97.64 (93%)** | 97.60 (95%) | 97.70 (86%) | +0.59 [+0.42, +0.78] | 3.5 / 3.4 |
| mild P1 (CR-Z type) | 56 | 93.45 (89%) | 98.57 (34%) | 97.28 (96%) | 97.65 (93%) | **98.36 (79%)** | 97.93 (91%) | 98.38 (86%) | +1.08 [+0.79, +1.51] | 3.9 / 4.1 |
| P2 HEV | 39 | 94.26 (87%) | 99.65 (59%) | 98.26 (90%) | 99.31 (85%) | **99.68 (92%)** | 99.54 (97%) | 99.70 (85%) | +1.42 [+0.31, +3.38] | 2.8 / 2.7 |
| series HEV | 24 | 97.13 (71%) | 98.71 (58%) | 98.68 (96%) | 99.36 (96%) | **99.56 (96%)** | 99.45 (96%) | 99.62 (96%) | +0.88 [+0.57, +1.18] | 4.6 / 4.3 |
| power-split PHEV | 57 | 98.64 (96%) | 96.66 (100%) | 99.66 (100%) | 99.81 (100%) | **98.58 (93%)** | 98.44 (95%) | 98.10 (96%) | −1.08 [−4.66, +2.05] | 2.0 / 3.3 |
| series PHEV (EREV) | 46 | 100.22 (96%) | 100.36 (100%) | 100.80 (100%) | 101.93 (100%) | **100.08 (96%)** | 100.07 (96%) | 99.83 (100%) | −0.72 [−1.78, −0.01] | 0.7 / 1.4 |
| bus (series) | 12 | 96.43 (92%) | 98.30 (92%) | 95.49 (100%) | 96.51 (100%) | **96.48 (100%)** | 96.20 (100%) | 97.57 (100%) | +0.99 [+0.79, +1.30] | 4.5 / 4.0 |
| bus (P2) | 12 | 95.71 (100%) | 99.48 (100%) | 98.53 (100%) | 99.41 (100%) | **98.83 (100%)** | 98.08 (100%) | 98.91 (100%) | +0.30 [+0.14, +0.48] | 4.0 / 3.9 |
| truck (P2) | 24 | 96.03 (100%) | 99.01 (88%) | 99.06 (100%) | 99.44 (100%) | **99.07 (100%)** | 99.09 (100%) | 99.26 (100%) | +0.01 [−0.39, +0.41] | 2.6 / 2.6 |
| truck (PHEV series) | 12 | 100.36 (100%) | 100.28 (100%) | 99.35 (100%) | 99.77 (100%) | **100.41 (100%)** | 100.42 (100%) | 100.09 (100%) | +1.06 [+0.26, +1.58] | 1.9 / 2.5 |

\* MPC (15 s exact future) and DP (whole trip) see the future: benchmarks only. Rule = powersplit_rule / parallel_rule /
series_thermostat / phev_cdcs. Values above 100 % mean DP was not tight (warm-engine assumption, grid).
Exam skill (rule gap closed): v2.1 0.558, v2.0 0.516, v1 release 0.587, A-ECMS 0.317, ECMS 0.517, MPC 0.428.
v2.1 − v2.0 (same exam): +0.14 [+0.06, +0.22] points (mild +0.44 [+0.24, +0.68], P2 buses +0.75 [+0.65, +0.94], series
buses +0.28 [−0.59, +1.58]), with charge held on 93 % vs 96 % of trips. v2.1 − v1: +0.03 [−0.12, +0.22] (v1 holds 91 %).
v2.1 − constant-s ECMS: +0.26 [+0.01, +0.52] overall; series buses −1.82 [−4.51, −0.04], P2 buses −0.64 [−1.17, −0.36],
mild −0.21 [−0.53, +0.10]. Twins of real cars: v2.1 98.29 % vs A-ECMS 97.99 % (+0.30 [−0.57, +1.03]); generated
vehicles 98.85 % vs 98.42 % (+0.44 [−0.17, +0.99]).
Error bars: the policy outputs no uncertainty, so there is no coverage calibration; intervals are cluster-bootstrap (by
vehicle) intervals of the paired difference. Raw report: `training/exam_report_v2_1.json`.

What changed against v2.0's exam (physics v2: 98.84 % vs A-ECMS 98.32 %, MPC 98.80 %): on physics v3.1 every causal
strategy moves; v2.0 itself drops to 98.43 %, below MPC (98.67 %). v2.1 recovers 0.14 of it and still beats every causal
baseline overall, but not MPC, and its interval against A-ECMS now includes zero. Fixed bus s closed part of the bus gap
to constant-s ECMS (pooled 97.14 → 97.66 % against 98.89 %), not all of it. Charge holding got worse: 93 % (v2.0 on the
same exam 96 %), mild twins 79 % (91 %): the fixed mild s of 2.8 lets the CR-Z twins drift, which the 10 probe trips
and the dev gate did not show (dev mild holding 75 % for both v2.0 and v2.1).

## Simulator vs real logs (report only; nothing tuned on exam)
Each VED hybrid trip driven by its calibrated twin (warm fragment, logged start SOC), simulated fuel passed through
RoadSim's OBD label model and compared with VED's logged fuel on valid seconds (plug-ins: fuel + battery energy).
Trip-median error, % (car median), HybridPilot v2.1 on physics v3.1 (v2.0 on physics v2 in brackets):

| | dev (12 trips per car) | exam (3 trips per car) | engine-on gap, exam (points) | starts/km sim / real, exam |
|---|---|---|---|---|
| power-split HEV | −7.6 (−6.2), 36 trips [−4.6] | **+2.2 (+2.0)**, 168 trips [+2.2] | −2.0 [−7.6] | 2.42 / 1.94 [3.65] |
| power-split HEV, A-ECMS | +0.4 (−0.4) | +8.6 (+8.9) | −2.3 | 1.93 / 1.94 |
| mild P1 | +3.1 (+0.4), 24 [+2.9] | **+2.3 (+3.6)**, 84 [+0.3] | +4.7 [−3.8] | 2.66 / 1.43 [3.78] |
| P2 | — | +3.4 (−0.7), 5 [+1.6] | −35.9 [−42.6] | 1.43 / 0.72 [2.46] |
| plug-in power-split | −2.9, 12 | **−20.1**, 33 [−22.4] | | |
| plug-in series | −7.8, 11 | **−30.9**, 31 [−30.9] | | |

The simulator–real target (under 8 %) is met for the exam hybrids (+2.2 to +3.4 % trip median) and on dev for every
architecture, and not met for exam plug-ins (−20 % and −31 %, RoadSim's known plug-in gap). Engine starts are much closer
to the real cars' than on physics v2 (power-split 2.4 vs 1.9 per km real, was 3.7).

## Limits
- Graded in simulation; plug-ins on real roads read 20–31 % low, P2 engine-on share is far below the real cars'.
- Charge holding 93 % vs A-ECMS 97 % (v2.0 on the same exam 96 %); mild twins 79 %.
- For buses and mild hybrids the network's s is not used: a fixed s with SOC feedback, i.e. an adaptive ECMS with a dead
  band. Constant-s ECMS still beats it on buses (98.89 vs 97.66 % pooled) and scores 0.21 higher on mild hybrids
  (holding 34 %). Losses on plug-ins against A-ECMS (wide intervals). MPC with 15 s of preview scores 0.10 higher overall.
- The released network was trained on physics-v2 banks; a re-teach on v3.1 banks (R18) did not pass the ratchet.
- DP reference ignores the production forced-on rules; it assumes a warm engine.
- Heavy duty: buses on real Zurich routes, trucks on cycles and city routes only.

## v2.0 (14 Nov 2026, physics v2) in one paragraph
Same network on RoadSim physics v2, released rung R8 (Kp 12, Ki 0.02, no dead band, network s everywhere). Exam (430
trips, v2 physics): 98.84 % of DP vs A-ECMS 98.32 % (+0.53 [+0.20, +0.82]), rule 97.39, ECMS 98.53 (holds 74 %), MPC
98.80, v1 re-scored 99.00 (holds 89 %); charge held on 95 % (A-ECMS 96 %); power-split +0.89, P2 +1.32, series +1.04,
mild +0.60 against A-ECMS; plug-ins −0.29 / −0.35, P2 buses −0.48; constant-s ECMS 98.98 vs 97.50 % on buses and 98.72
vs 98.10 % on mild hybrids. Full report: `training/exam_report_v2.json`.

## v1 (12 Nov 2026, physics v1) in one paragraph
Same network and teacher on RoadSim physics v1 with v1 twins; released rung R3 (Kp 6, on/off by cost). Exam (430 trips,
v1 physics): 99.49 % of DP vs A-ECMS 99.35 % (+0.13 [−0.06, +0.29]), charge held on 97 %; simulated fuel 0.80× logged on
exam power-splits. Full v1 card: git history of this file (commit before 14 Nov 2026) and training/ratchet.json.

## Files
`hybridpilot.py` (features, MLP, decision with the v2.1 defaults; `V20` restores v2.0's rule, RoadSim adapter),
`weights/hybridpilot_v2.npz` (the network of v2.0 and v2.1; v1: `hybridpilot_v1.npz`), `pipeline.py` (pins
roadsim-physics-v3.1), `demo.py`, `examples/` (3 VED exam trips + calibrated twin specs on v3.1), `training/` (all
scripts; `v21_*.sh` / `v21_*.py` ran the v2.1 pass with `HP_PHYS=roadsim-physics-v3.1`, `v2_*.sh` the v2 rebuild; raw
logs `ratchet_v2_1.json`, `explore_v2_1.json`, `exam_report_v2_1.json`, `simcheck_*_v2_1.json`, and the v2 / v1 logs),
`requirements.txt`.

## Data and licences
VED (Oh, LeBlanc, Peng, IEEE T-ITS 2020; Apache-2.0) — routes and the example trips redistributed under Apache-2.0.
USGS 3DEP lidar (public domain). IEM ASOS weather. ZTBus (Widmer, Ritter, Onder, Scientific Data 2023; ETH Zurich,
CC BY 4.0). EPA cycles, test-car lists, fueleconomy.gov and 40 CFR 1037 (public domain). RoadSim and RoadData (ML
Ratchet). Methods: Paganelli et al. 2002 (ECMS), Onori, Serrao & Rizzoni 2010 (A-ECMS), Sundström & Guzzella 2009
(DP), Ross, Gordon & Bagnell 2011 (DAgger).
