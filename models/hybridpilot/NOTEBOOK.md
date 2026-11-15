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
