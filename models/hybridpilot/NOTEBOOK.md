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
