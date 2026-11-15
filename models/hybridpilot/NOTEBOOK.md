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
