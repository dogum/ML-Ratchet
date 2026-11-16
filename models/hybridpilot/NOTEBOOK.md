> **Lab notebook.** HybridPilot was built in a claude.ai cloud session (12–13 Nov 2026) on the shared RoadSim + RoadData
> platform, now in `shared/vehicle/`. The first entry is that session's status document as written; its paths
> (`/home/claude/…`, `/mnt/user-data/…`, `src/`) refer to the cloud sandbox. `MODEL_CARD.md` holds every number.

## 12–13 Nov 2026 — build notes (hybridpilot-status.md, as written)

**What it is.** A learned, causal hybrid energy-management policy on RoadSim. Each second a 66→96→96→2 MLP (15,938 params) reads vehicle parameters, state, trip history and optional preview/nav and outputs an equivalence factor s (fuel J per battery J) and P(engine on). With SOC feedback (Kp 6), the controller picks the candidate control minimising P_fuel + s·P_batt,chem (+ start fuel). It was trained by imitating DP's costate and choices, with DAgger.

- Page: https://claude.ai/artifact/3MU8pXNtGc65HTDnQ8Cgup (lab: 6 vehicle/route combos × rule/ECMS/A-ECMS/HybridPilot/DP, temperature/mass/preview sweeps; the network running live in JS, JS==Python to 1e-5)
- Zip: /mnt/user-data/outputs/hybridpilot-v1.0.zip (numpy module + RoadSim adapter, weights, demo on 3 VED exam trips in 8 s, MODEL_CARD, training/)
- Folder: /home/claude/road/controller (STATUS.md, ratchet.json, data/exam_report.json)

### Recipe as built
- Physics: RoadSim roadsim-physics-v1, unmodified (ledger closure < 0.5 % on every run). No physics patch.
- Training data: 400 generated hybrids in 18 class×architecture combos (power-split, P2, series HEV; power-split and series PHEV; buses, Class 6/8 trucks) × 835 VED train_noise routes (real speed, lidar grade, ASOS weather), EPA/HD cycles, synthetic routes. −20 to 36 °C, warm and cold starts.
- Teacher: DP per trip. Labels: on/off choice and costate s* = −∂J/∂SOC/(Voc·Q). 2,000 trips over 3 banks; final net uses 1,400 trips (DP + exploration, plus DAgger round 1).
- Grading: DP re-solved to each strategy's own end SOC. %DP = 1 − (F − F_DP(E))/F_DP,CS. The fixed Phase-1 factor over-credits charge: ECMS rates 105.2 % with it vs 99.6 % matched.
- Frozen set: 106 trips (36 held-out vehicles × held-out train_noise routes, synthetic routes, cycles). Dev gate: 76 trips on VED dev cars' routes (twins + dev generated vehicles + ZTBus dev legs).

### Rungs (frozen skill = share of the rule strategies' gap to DP that is closed; keep if > +0.01 and the dev gate passes)
| rung | frozen skill | frozen %DP | decision |
|---|---|---|---|
| R0 A-ECMS (incumbent) | 0.564 | 98.88 | incumbent |
| R1 behaviour cloning | 0.592 | 98.70 | rejected: dev charge-holding 74 % |
| R2 DAgger + SOC feedback | 0.596 | 98.88 | kept |
| R3 on/off decided by cost with the learned s | 0.702 | 99.31 | **kept, final** |
| R4 + 30 s preview | 0.709 | 99.34 | rejected (+0.007) |
| R5 policy-search tune | 0.671 | 99.19 | rejected |
| R6 DAgger round 2 | 0.674 | 99.28 | rejected |
| R7 R6 + preview | 0.677 | 99.23 | rejected |

### Exam (430 trips: 216 VED exam routes driven by twins of the exam cars, 214 by 54 held-out generated hybrids incl. buses on ZTBus exam legs and Class 8 on HD cycles)
Mean %DP (charge-holding rate): HybridPilot **99.49 (97 %)**, A-ECMS 99.35 (99 %), rule 98.28 (94 %), MPC-15 s 99.65 (99 %, sees the future), const-s ECMS 99.91 (82 %; fails to hold charge). HP − A-ECMS: +0.13 pt [−0.06, +0.29] overall; power-split HEV +0.41 [+0.29, +0.55]; series bus +0.66; series PHEV −0.93 [−2.43, −0.09]. Starts/km 3.8 vs 4.1 (A-ECMS).

### Real-data sanity check (report only, nothing tuned)
Twins with A-ECMS vs VED logs: exam power-split HEVs burn 0.80× the logged fuel (total). The engine is on 35 % of the time in the sim vs 55 % logged, with 3.2 vs 1.8 starts/km. Real Toyotas keep the engine spinning above about 70 km/h. Simulated PHEVs burn 2–3.6× the logged fuel because their engines start for cabin heat. The strategy differences (1–3 %) are smaller than these simulator biases, so the ranking is a ranking inside RoadSim.

### Limits
Graded only in simulation. Small overall margin. The on/off head is unused, and preview did not help. DP assumes a warm engine and is not tight on cold starts or for PHEVs. Heavy duty data is thin.

### Next ideas
1. Dev calibration rung on the twins: keep the engine spinning above ~70 km/h, electric cabin heat for PHEVs, a start penalty. Rebuild the frozen set and re-score all rungs.
2. A nav-aware PHEV teacher and a dedicated blended-mode test.
3. True RL fine-tuning of the s head on matched-SOC fuel.

### Sandbox gotchas
- `echo $! > file` right after `setsid nohup … &` sometimes failed (cwd race); find PIDs with ps.
- VED HEVs log no SOC or battery current; only PHEVs do.
- Under 3-agent contention, matched-SOC scoring costs about 6 ms per trip-second.

## 13 Nov 2026 — integrated into ML Ratchet

- Moved from `hybridpilot-v1.0.zip` into `models/hybridpilot/`. Sandbox paths now go through `training/_paths.py`
  (work folder `models/hybridpilot/work/`, the vehicle platform at `shared/vehicle/` or `$ML_RATCHET_VEHICLE`).
- The demo reproduces the release's output exactly from the repository (checked against the zip's own run).
- Caveat carried from the handoff: built before RoadData's 13 Nov fixes (EPA city/highway labels were swapped; VED
  mass double-counted the 136 kg driver). Re-score after the fixes are upstreamed into RoadSim.

## 13 Nov 2026 — original lab and live instrument

- The original claude.ai page is now `lab/index.html`, unchanged.
- The site's model page has a live instrument (`web/export_instrument.py`, `site/src/instruments/hybridpilot/`). Three real VED routes driven by RoadSim twins; whole-trip runs precomputed per strategy, the network and decision rule live at recorded moments. Parity with Python: 96 cases, worst 3.9e-15.

## 14 Nov 2026 — v2: rebuilt on RoadSim physics v2 (Phase D)

**Why.** RoadSim physics v2 (13 Nov) changed the hybrid physics (P2 converter and start energy, 8 s / 3 s minimum on/off,
Toyota-type engine-on speed, plug-in CD rules, on-road loads, calibrated twins). Physics changed, so the sets were rebuilt
and every ratchet decision re-taken from R0. Two other agents rebuilt EnergyNet and TripDetective at the same time.

**What moved.**
- All 13 RoadSim calls pinned to `options='roadsim-physics-v2'` (training, `pipeline.py`, `web/export_instrument.py`);
  `hp_env.PHYS` is the one place. v2.1 not used.
- VED cars are driven by RoadSim's calibrated twins: `hp_env.build_twins` calls the upstream builder
  `shared/vehicle/checks/check_ved_twin.build_twin` (115 twins, cached in work/; replaces `twin_spec_v1`, kept for
  reference). CR-Z-type cars become mild P1 twins (`engine_on_moving`), a new stratum. The simulator–real check now uses
  `roadsim.obd.label_fuel` (upstream) instead of raw fuel.
- Generated hybrids use the v2 donor model; WARM fragments also set the v2 oil and axle nodes.
- Bus routes: full ZTBus (`source='ZTBus_full'`), by-month split, 24 dev / 12 exam legs drawn by seed.
- DP stage tables are built with the v2 powertrain (`hp_policy.make_pt`: options, brake blending, start energy).
  Reference for grading: DP with the 8 s / 3 s timers and real start fuel (`dp_backward_timed`, a copy of RoadSim's
  `dp_solve_timed` recursion that shares one stage table across terminal SOCs).
- HybridPilot obeys the platform like every RoadSim strategy: 8 s / 3 s timers (v1 code forced 4 s / 2 s), forced-on
  rules, plug-in CD rules (as `phev_cdcs`), direct-drive candidates for series. The teacher, when it drives, obeys the
  forced-on rules too.
- Evaluation, banks, tuning and the twin check run on 5 processes (`HP_WORKERS`); training on the GPU (CUDA, < 0.5 GB).

**Ladder (frozen skill; dev % of DP / dev charge holding).** R0 A-ECMS 0.255 (98.20 / 97 %). R1–R7, v1's rungs, all
scored 0.62–0.73 on the frozen set but held charge on only 70–91 % of dev trips (gate 94 %): on v2 the learned s drifts
the charge (power-split twins deplete, mild CR-Z twins charge up). R8 = R3 with A-ECMS's own feedback gains (Kp 12,
Ki 0.02): 0.659 (99.19 / 97 %), **kept and released**. Then rejected: R9 mild P1 class in training (0.657), R10
timed-DP teacher (0.624), R11 20 kJ start cost (0.560), R12 distance-to-go plug-in plan (0.659), R13 3 × 256 net on GPU
(0.671, dev holds 93 %), R14 preview 30 s (0.667, +0.007), R15 re-tune incl. Ki (0.682, dev holds 88 %).
Release choice committed (c70a8a3) before the exam ran.

**Exam (430 trips, once).** % of DP (holds): v2 98.84 (95 %), A-ECMS 98.32 (96 %), rule 97.39, ECMS 98.53 (74 %),
MPC 98.80, v1 release re-scored 99.00 (89 %). v2 − A-ECMS +0.53 [+0.20, +0.82]: power-split +0.89, P2 +1.32, series
+1.04, mild +0.60, series bus +0.70; losses: power-split plug-ins −0.29 [−2.31, +1.53], series plug-ins −0.35
[−1.09, +0.12], P2 buses −0.48. Starts/km 4.0 vs 4.5.

**Simulator vs real (twins driven by HybridPilot, OBD label model, trip median).** Exam: power-split HEV +2.2 %, mild
+0.3 %, P2 +1.6 % (5 trips), plug-in power-split −22 %, plug-in series −31 %. Dev: all within 8 % (−7.8 to +2.9).
Target met for hybrids, not for plug-ins (RoadSim's plug-in gap). Simulated starts are about twice the real rate.

**What got worse.** Frozen skill of the released rung (0.659) is below R3–R5's (0.72–0.73): charge holding costs fuel.
v2 burns 0.16 points more than the v1 network on the v2 exam, which however fails to hold charge on 11 % of trips.

**Gotchas.**
- RoadSim v2's `pt.evaluate` reads per-run state (thermal nodes, transient fuel), so a candidate grid re-evaluated after
  the run differs; the instrument exporter now computes each moment's grid inside a second, identical run.
- Series hybrids' candidate count varies (direct-drive window); the exporter pads it.
- The dev gate's charge-holding rule, not the frozen margin, decided every v1 rung on v2.

## 14 Nov 2026 — card corrections after a site audit (no model change)

- **Bus exam row: mixed -> loss.** The row listed only A-ECMS and the rule. Constant-s ECMS, in the same exam table,
  scores 98.98 % of DP pooled over the 24 bus trips (series 98.60, P2 99.36) against HybridPilot's 97.50 % (96.78 /
  98.23), holding charge on 100 % of them. ECMS is now listed and the verdict is a loss. The wins line "series buses
  (+0.70)" now says it is against A-ECMS.
- **Mild row: win -> mixed.** Constant-s ECMS scores 98.72 % against 98.10 % on the 56 mild-hybrid trips but holds
  charge on only 43 % of them (HybridPilot 95 %). ECMS is now listed, with that note.
- **ratchet.yaml:** the v2 note said R1-R7 held charge on 87-91 % of dev trips; R1 held 70 %. Now 70-91 %. Ladder names
  "v1.0 — ..." / "v2.0 — ...", as EnergyNet and TripDetective.
- **MODEL_CARD:** it quoted the collection's motto "Graded on reality", but HybridPilot is graded in simulation; it
  now says so. RoadSim's calibration is credited to the dev-only procedure in CALIBRATION.md.
- ZTBus link: https://doi.org/10.3929/ethz-b-000626723.

## 15 Nov 2026 — v2.1: physics v3.1 and a new charge-sustaining rule (release choice, before the exam)

**Physics rung (dev only).** RoadSim's hybrid presets beyond v2 were tried for this hybrid-only model. On VED twins
v3.1 = v2.1 + v3 hardware rules (VED twins carry no model year, so no era scaling): run-on 2 s into braking, P2 converter
open in 1st/2nd, generator hybrid hardware v3. Dev sim-vs-real check, HybridPilot v2.0 driving the dev twins
(`simcheck.py dev`, 84 trips, OBD label model), v2 -> v3.1:
power-split HEV trip median -4.6 -> -2.4 %, car median -5.5 -> -3.3 %, engine-on gap -9.3 -> -4.4 points, starts/km
3.67 -> 2.66 (real 2.32); mild P1 +2.9 -> +3.4 %, car median +0.3 -> +0.6 %, engine-on -8.6 -> -2.1 points, starts
3.57 -> 2.43 (real 1.01); plug-ins unchanged (-2.9 / -7.8 %). Fuel closer or within 0.5 point, engine operation much
closer: **kept, every HybridPilot run now pins `roadsim-physics-v3.1`** (`hp_env.PHYS`, `HP_PHYS` to re-score on
another preset; caches and logs carry the physics in their names).

**Re-take (physics changed).** New v3.1 frozen set (same draws and seeds; generated hybrids get the v3 generator
hardware), dev, tune and a new training-distribution probe set (190 trips, 10 per class incl. mild; `sets.build_probe`).
R0-R15 re-scored in order with their v2 networks and settings (`v21_ladder0.sh`). Same decisions as on v2: only R8
(the v2.0 release) is kept. R8 on v3.1: frozen skill 0.662, frozen 99.06 % of DP (holds 90 %), dev 98.94 % (holds 96 %),
dev skill 0.656. A-ECMS (R0): 0.144, dev 98.14 % (97 %).

**Exploration on the probe set (training distribution; never dev or exam).** Probe skill / probe charge holding, v2.0
network: R8 rule 0.529 / 93 %; Kp 12 without Ki 0.557 / 91 %; SOC error low-passed 120 s 0.455 / 88 %; ECMS cubic
window term + Kp 6 0.575 / 92 %; window-normalised Kp 3 0.532 / 92 %; dead band 0.03 0.595 / 92 %, 0.05 0.598 / 91 %,
0.03 without Ki 0.612 / 88 %, 0.03 + cubic 0.587 / 92 %, normalised with band 0.599 / 91 %, hard band 0.03 (+priors)
0.591 / 93 %, band 0.02 (+priors) 0.635 / 92 %, band 0.03 + Ki 0.04 (+priors) 0.602 / 93 %. The probe shows the exam's
bus failure: probe buses, constant-s ECMS 100.0 % of DP vs the R8 rule 98.8 (P2) / 98.6 (series), and A-ECMS (s0 2.6)
also beats the network there (99.4 / 99.1): the network's s, not only the feedback, is weak on buses. The DP teacher's
median costate on training buses is 2.21 (P2) / 2.30 (series) (`v21_prior.py`, bank of 800 teacher trips; timed teacher
the same); used as a prior it gives 99.41 / 99.55. A grid over a constant s0 on the 20 probe buses: 2.4 99.71 / 99.73,
2.6 99.84 / 100.05, 2.8 99.74 / 100.33, 3.0 99.80 / 100.20 -> P2 bus 2.6, series bus 2.8. Mild (10 probe trips):
2.2 93.70 % (holds 100 %), 2.5 94.22 (100), 2.8 94.56 (100), 3.0 94.86 (90), 3.3 95.43 (70) -> 2.8, the highest value
that kept every probe trip within its window. Full probe with band 0.03 + these priors: 0.648 / 92 %.

**Rungs on v3.1 (frozen skill; dev % of DP / dev holds).** R16 dead band 0.03 on the proportional SOC feedback:
0.689 (+0.027; dev 99.16 / 95 %) kept. R17 + class priors (buses, mild): 0.704 (+0.015; dev 99.28 / 95 %) kept. R18
re-teach on v3.1 banks (v2.0 recipe, 800 teacher + 600 DAgger trips, `v21_retrain.sh`) with R17's rule: 0.698
(-0.006; dev 99.13 / 93 %) rejected.

**Release choice: v2.1 = R17** (v2.0 network unchanged, `weights/hybridpilot_v2.npz`; new decision rule; physics
v3.1). Against v2.0 on the same physics: frozen skill 0.704 vs 0.662, dev 99.28 vs 98.94 % of DP, dev skill 0.802 vs
0.656 (+0.146, margin 0.01), dev charge holding 94.7 % vs 96.1 % (72 vs 73 of 76 trips; gate: no more than 3 points
worse). What got worse: charge holding, dev -1.4 points and frozen 90 -> 87 % (two P2 and one series trip end 0.02-0.03
beyond the allowance; the band leaves end-of-trip drift less corrected). Numpy module == training policy: identical
fuel and final SOC on 24 dev and frozen trips (`v21_numpy_check.py`). The exam has not been run for v2.1.

## 15 Nov 2026 — v2.1 exam (once, after the release choice d9fd0fd)

Exam rebuilt on physics v3.1 (same draws; twins of the 108 exam cars re-calibrated under v3.1; 430 trips scored, as
v2.0). Every baseline, v2.0 (its network and settings) and the v1 release were re-run on the same physics
(`v21_exam.sh`). % of DP (charge holding): **v2.1 98.57 (93 %)**, v2.0 98.43 (96 %), v1 98.54 (91 %), A-ECMS 98.20
(97 %), constant-s ECMS 98.31 (75 %), MPC 15 s 98.67 (95 %), rule 97.00 (92 %).
v2.1 − A-ECMS +0.37 [−0.18, +0.84]; − ECMS +0.26 [+0.01, +0.52]; − v2.0 +0.14 [+0.06, +0.22]; − v1 +0.03 [−0.12, +0.22].
By stratum against A-ECMS: power-split +0.59, mild +1.08, P2 +1.42, series +0.88, series bus +0.99, P2 bus +0.30,
truck P2 +0.01, plug-in power-split −1.08 [−4.66, +2.05], plug-in series −0.72 [−1.78, −0.01]. Buses against ECMS:
series −1.82, P2 −0.64 (pooled 97.66 vs 98.89; v2.0 on v3.1 97.14). Mild against ECMS −0.21 [−0.53, +0.10].
Starts/km 3.0 vs A-ECMS 3.2.

**What got worse.** (1) The "all trips" row is no longer a win: on v3.1 MPC (sees 15 s of the future) scores 98.67, above
v2.1's 98.57 and v2.0's 98.43; the interval against A-ECMS now includes zero. The physics change, not the new rule, took
most of the margin (v2.0 falls from 98.84 on v2 to 98.43 on v3.1). (2) Charge holding 93 % vs v2.0's 96 % on the same
exam (A-ECMS 97 %). (3) Mild twins hold charge on 79 % of trips (v2.0 on v3.1: 91 %): the fixed mild s of 2.8 lets them
drift; the 10 probe trips (100 %) and the dev gate (75 % for both) did not show it. Rows: 3 wins (power-split, P2 +
series, starts), 3 mixed (all trips, mild, against v2.0), 1 tie (against v1), 3 losses (plug-ins, buses, charge holding).

**Simulator vs real (v2.1 on v3.1, exam trip median).** Power-split +2.2 % (168 trips), mild +2.3 %, P2 +3.4 % (5 trips),
plug-ins −20.1 / −30.9 %. Power-split starts 2.42 per km vs 1.94 real (v2.0 on v2: 3.65); engine-on gap −2.0 points
(−7.6). Dev: power-split −7.6 % (v2.0 on v3.1 −2.4 %); still inside the 8 % target, cause not checked.

**Next.** A charge-holding term at the window edge or toward the trip end, judged on dev holding; the mild s chosen with
holding in the objective on more than 10 probe trips; a bus-specific teacher instead of a fixed s.

## 15 Nov 2026 — round 2: charge holding, plug-ins and buses (release choice v2.2, before the exam)

**Round 1's probes.** The 15 Nov v2.1 entry lists about 15 decision-rule variants scored on the 190-trip probe set. They
were probes, not rungs, and stay that way; none is re-run here except as a named baseline.

**Diagnosis on dev and simulation first.** v2.1 holds charge on 87 % of frozen trips (A-ECMS 95 %) and 95 % of dev trips
(A-ECMS 97 %). Where it fails:
- Plug-ins (frozen): series PHEVs hold 7 of 12, power-split 16 of 18 (A-ECMS all). After depletion they sustain 0.02-0.03
  below their CS level (soc_lo + 0.05) and dip to 0.13-0.16 against a 0.17 limit. The dead band makes it worse: below
  CS the feedback only starts 0.03 lower. Probe plug-ins show the same (87 / 85 %). Dev plug-ins hold 22 of 22.
- Mild P1 (dev, 12 trips of two CR-Z-type twins): v2.1 holds 9, v2.0 9, A-ECMS 10. Two kinds of failure.
  (1) Charge-up on short trips (VED_435_1200, 2.6 km: 0.55 -> 0.68; VED_435_1245: 0.57 -> 0.66). The 0.58 kWh pack
  takes 5-7 kW of load-point charging at s 2.8 (about 1 % SOC per second); the offset is band + (s0 - s_eq) / Kp =
  0.03 + about 1.0 / 12, close to the 0.10-0.13 seen. (2) A 400 s stop with the engine off (VED_435_734): 1.6 kW of
  auxiliaries drain the pack from 0.64 to 0.11-0.16 for every strategy except constant-s ECMS. RoadSim never runs the
  engine at a stop for P2/P1 architectures (`sim._plan`, stopped branch: series and power-split only), so no controller
  can fix this; it needs a RoadSim change (noted for the platform).
- P2 HEVs (dev pickup VED_258_1374, frozen hilly_7 and nycc): the band leaves them 0.12-0.14 below the window middle at
  the end; v2.0 (no band) held all three.
- Buses (frozen): constant-s ECMS beats v2.1 on P2 buses, 100.12 vs 97.40 % of DP (nycc 102.7 vs 93.3; a synthetic bus
  route 98.4 vs 92.9). The Kp 12 feedback pulls s to about 1.8 when the bus starts above the middle and spends charge
  at well below its value (DP's bus costate is 2.2-2.3).

**Rule v2.2 (committed f8c6981 / ff69488 before R19 was scored).** A second path for rungs aimed at charge holding:
frozen + dev holding (182 trips) up by 2 points or more, neither set's holding down, frozen skill no more than 0.01
lower, dev % of DP no more than 0.2 points lower. Path A (frozen skill +0.01 and the dev gate) is unchanged.

**Probes (training distribution only; r2_explore.py; log training/explore_r2.json).** Selection rule for every probe:
the highest mean % of DP among variants holding at least as many trips as A-ECMS on that probe. % of DP / holds:
- New mild probe, 80 trips (40 training mild vehicles, training routes, seed 8600; sets.build_probe_mild): v2.1 96.45 /
  96 %, v2.0 rule 96.18 / 95 %, no band with s0 2.2 96.08 / 96 %, 2.4 96.17 / 98 %, 2.6 96.25 / 98 %, **2.8 96.35 /
  98 %**, band with s0 2.4 96.19 / 96 %, 2.6 96.31 / 98 %, no band Kp 24 with s0 2.8 96.07 / 98 %, 2.6 96.04 / 98 %;
  A-ECMS 95.45 / 98 %, ECMS 95.76 / 51 %. The training mild cars barely drift (3 of 80 fail): the probe does not
  reproduce the dev twins' charge-up.
- Plug-ins, 60 probe trips: v2.1 100.00 / 88 %; below-CS band off 100.12 / 90 %; band off + Kp 24 100.08 / 95 %;
  + Kp 48 100.03 / 95 %; Kp 24 + Ki 0.05 100.06 / 95 %; band off + HEV costate floor at CS 100.20 / 97 %; **band off +
  Kp 24 + costate floor 100.14 / 100 %**; A-ECMS 98.89 / 100 %, ECMS 99.35 / 100 %.
- HEVs, 100 probe trips (P2, power-split, series, P2 trucks): v2.1 98.90 / 92 %; Ki 0.05 98.71 / 94 %; no band 98.74 /
  93 %; band 0.02 98.87 / 92 %; Kp 18 98.84 / 94 %; A-ECMS 98.22 / 90 %. By the selection rule nothing beats v2.1 here;
  Kp 18 was still scored as R21 for its +2 trips of holding.
- Buses, 20 probe trips (bus feedback, on top of R19): Kp 12 (v2.1) 100.09 / 100 %; Kp 6 100.11 / 100 %; Kp 3 100.15 /
  100 %; Kp 6 + band 0.05 100.16 / 100 %; no feedback 100.40 / 75 %; **Kp 3 without the integral 100.26 / 100 %**;
  ECMS 100.01 / 95 %, A-ECMS 99.27 / 100 %.

**Rungs (frozen skill; frozen % of DP / holds; dev % of DP / holds).** Incumbent R17 (v2.1): 0.704; 99.22 / 87 %;
99.28 / 95 %.
- R19 plug-ins at CS: s no lower than the HEV costate (DP teacher medians, power-split 2.42, series 2.70; prior_p31r0),
  below CS Kp 24 without the band: 0.712; 99.27 / 93 %; 99.30 / 95 %. **Kept (path B)**: pooled holding 90.1 -> 94.0 %;
  frozen series plug-ins 7 -> 12 of 12, power-split 16 -> 18 of 18.
- R20 R19 + mild without the band: 0.712; 99.27 / 93 %; 99.23 / 96 %. **Rejected**: frozen has no mild trips; dev mild
  holds 9 -> 10 of 12 (+0.5 points pooled, needs +2) and dev mild % of DP falls 99.21 -> 98.76.
- R21 R19 + HEV Kp 18: 0.691; 99.17 / 96 %; 99.15 / 96 %. **Rejected**: frozen skill -0.022 (limit -0.01). Holding
  would reach A-ECMS's level on frozen (96 %), but at a fuel cost.
- R22 R19 + buses Kp 3 without the integral: 0.774; 99.38 / 91 %; 99.30 / 95 %. **Kept (path A)**: +0.062; frozen P2
  buses 97.40 -> 99.39 % of DP, series 100.08 -> 100.13. Frozen holding falls 93.4 -> 90.6 %: four frozen bus trips
  now end 0.68-0.70 (start 0.54-0.60), as constant-s ECMS does; all stay inside the window.
All rungs are deterministic rule changes on the same network (no training, no seeds), so there is no seed spread.

**Release choice: v2.2 = R22** (v2.0 network unchanged, `weights/hybridpilot_v2.npz`; physics v3.1). Against v2.1:
frozen skill 0.774 vs 0.704, frozen 99.38 vs 99.22 % of DP, frozen holding 90.6 vs 86.8 %, dev 99.30 vs 99.28 % of
DP, dev holding 94.7 % for both. Numpy module (`hybridpilot.py`, DEFAULTS = v2.2; `V21` and `V20` give the older
rules) == training policy: identical fuel and final SOC on 44 dev and frozen trips incl. 22 plug-in and 8 bus trips
(`r2_numpy_check.py`).

**What did not work / what is left.** The main target, mild charge holding, has no fix: the only dev evidence is 12
trips, one failure is a RoadSim limit, the training mild cars do not drift, and the frozen set has no mild stratum, so
no mild rung can pass either path. A frozen mild stratum (a new frozen set) and RoadSim engine-at-stop charging for
P2/P1 cars are what it needs. The exam has not been run for v2.2.

## 15 Nov 2026 — v2.2 exam (once, after the release choice 48e9985)

Same exam set, physics and baseline runs as v2.1 (`r2_exam.sh`); v2.1 re-scored alongside. % of DP (charge holding):
**v2.2 98.66 (399 of 430, 93 %)**, v2.1 re-scored 98.55 (398), v2.0 98.43 (411), v1 98.54 (393), A-ECMS 98.20 (418),
constant-s ECMS 98.31 (321), MPC 15 s 98.67 (410), rule 97.00.
v2.2 − A-ECMS +0.46 [−0.08, +0.95]; − ECMS +0.35 [+0.12, +0.59]; − v2.1 +0.11 [+0.01, +0.24]; − v2.0 +0.23 [+0.10,
+0.41]; − v1 +0.12 [−0.03, +0.31].
What moved (everything else is identical to v2.1, trip by trip): series buses 96.48 -> 99.36 % (ECMS 98.30; +1.06
[+0.86, +1.23] over ECMS) but holding 12 -> 9 of 12; P2 buses 98.83 -> 99.66 % (ECMS 99.48; +0.18 [−0.02, +0.42]),
12 of 12; power-split plug-ins 98.45 -> 98.53 %, holding 53 -> 55 of 57; series plug-ins 100.08 -> 100.07 %, holding
44 -> 46 of 46; plug-in trucks 100.41 -> 100.35 %.
Rows: buses flip from loss to win (99.51 vs ECMS 98.89 pooled). All trips stays mixed (MPC 98.67, 0.01 higher). Mild
stays mixed (unchanged, holds 79 %). Plug-ins stay a loss (99.22 vs A-ECMS 100.17). Charge holding stays a loss (93 vs
97 %). New row against v2.1 re-scored: a win (+0.11 [+0.01, +0.24], same holding). Against v1: a tie.

**What got worse.** Series buses now hold charge on 9 of 12 exam trips (v2.1 12; ECMS 11), as the frozen set showed
(R22: frozen holding 93.4 -> 90.6 %). Plug-in trucks −0.06.

**Reproducibility gap found.** The v2.1 re-score differs from v2.1's own exam run on one trip (generated SUV plug-in on
VED_381_1188: fuel 31.8 g, end SOC 0.455, 102.9 % of DP in the first run; 39.7 g, 0.835, 95.7 % in a fresh process and
in this run). The result depends on which trips the same worker process simulated before, so some vehicle state
survives between RoadSim runs. v2.1's published 98.57 % becomes 98.55 % on a clean re-run. Not fixed here (platform).

**Simulator vs real (v2.2, exam).** Unchanged from v2.1: power-split +2.2 %, mild +2.3 %, P2 +3.4 %, plug-ins −20.1 /
−30.9 % (`training/simcheck_exam_v2_2.json`).

**Live instrument.** Re-exported with the v2.2 rule. The exporter now passes the window and prior key to
`HybridPilotCore.decide` (it passed neither, so its Python reference ignored per-class rules); parity 96 cases, worst
5.0e-15. The TS port (`site/src/instruments/hybridpilot/net.ts`) applies `cls_fb` and `phev_cs_s`.

## 15 Nov 2026 — scoring correction: duplicate exam trip ids (no new version)

**The earlier diagnosis was wrong.** The v2.2 exam entry above says some vehicle state survives between RoadSim runs in
one process. It does not: a separate check ran 432 exam cases forwards and backwards in one process with 0 differences
(`tests/test_roadsim_isolation.py`). The real cause is in this model's scoring. `sets.build_exam` draws 4 exam routes
per generated vehicle with replacement, so the exam set holds 432 cases but only 430 case_ids:
`exam_car_hev_parallel_p2_0|VED_347_2946` (cases 253 and 255, start SOC 0.556 / 0.614) and
`exam_suv_phev_powersplit_1|VED_381_1188` (cases 328 and 330, start SOC 0.90 / 0.49). `evaluate.score_baselines` and
`score_policy` filled dicts keyed by case_id from `imap_unordered`, so the last finisher won. Two trips were never
scored, and a policy's run could be graded against the other draw's DP and baselines. The "102.9 vs 95.7 %" swing on
VED_381_1188 was this. Every HybridPilot exam score on physics v2 and v3.1 was affected; frozen, dev, tune and probe sets
have no duplicates.

**Fix (chooses nothing; the exam set is not rebuilt or reordered).** `sets.assign_keys` gives each case a key in
memory: its case_id, or `case_id#<position>` for a repeated id. `evaluate` keys every result by it and raises on
duplicate keys. The two ambiguous entries were dropped from both baseline caches, the four cases' baselines (DP, MPC,
A-ECMS, ECMS, rule, CD-CS) were run, and every exam policy was re-scored on all 432 trips (`dupfix_exam.sh`). The old
caches are kept in WORK/data/pre_dupfix/.

**Before -> after, physics v3.1 (% of DP).** v2.2 all trips 98.66 -> 98.69 (MPC 98.665: +0.02 [−0.58, +0.54]); v2.2 −
A-ECMS +0.46 -> +0.49 [−0.07, +0.98], − ECMS +0.35 -> +0.38 [+0.13, +0.64]; v2.1 98.55 (published 98.57) -> 98.58;
v2.0 98.43 -> 98.45; v1 98.54 -> 98.56; v2.2 − v2.1 +0.11 [+0.01, +0.24] unchanged; − v2.0 +0.23 [+0.09, +0.41]; − v1
+0.12 [−0.02, +0.31]. P2 + series 99.63 (63 trips) -> 99.64 (64), A-ECMS 98.42 -> 98.44. Plug-ins 99.22 (103) -> 99.29
(104), A-ECMS 100.17 -> 100.13, CD-CS 99.35; power-split plug-ins − A-ECMS −1.13 -> −0.91 [−4.51, +2.16]. Charge held
399 / 430 -> 401 / 432 (A-ECMS 418 -> 420). Power-split, mild, buses, starts: unchanged.
**Physics v2 (v2.0's exam).** v2.0 98.84 -> 98.86, v1 99.00 -> 99.02, A-ECMS 98.32, MPC 98.80, P2 + series 99.84 ->
99.85, plug-ins 99.27 -> 99.33 (A-ECMS 99.59 -> 99.56); power-split, mild, buses unchanged.

**Verdicts.** All trips: v2.2 is now 0.02 above MPC, well inside the interval, so the row moves from mixed to a tie
(not a win). Every other verdict stands. Simulator-vs-real, golden demo and live instrument are not affected (twins of
real cars only; no duplicate there).
