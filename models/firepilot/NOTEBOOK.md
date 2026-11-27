# FirePilot: lab notebook

Historical record. Add dated entries; do not rewrite earlier ones.

## 2026-11-19 — build notes (firepilot-status.md, as written; the private page link removed)

FirePilot is a learned planner for wildfire containment line. It is given a fire at its first afternoon satellite
detection and a crew budget. It picks one line plan for the next 72 h: where the line goes and in which order two
crews build it. It was trained only on FireSim simulations. It was graded once in FireSim on the 48-fire exam replay
subset, each with its real LANDFIRE fuels and terrain, real HRRR weather and real first FEDS detection. Wildfire
Phase 2, project 4.

- **Page:** an explorer of 6 exam fires drawn by a seed, with every plan's simulated 72 h outcome and the observed
  edge; a lab where the network runs live in the browser (re-plan, change the budget, draw your own plan); the exam,
  fidelity, ratchet, method and limits.
- **Release:** `firepilot-v1.0.zip` (208 KB): the numpy + scipy planner `firepilot.py` and the r1 weights; `demo.py`
  on 3 exam fires (about 7 s; reproduces the exam plans exactly) and `pipeline.py`; the model card, README, every
  training, exam and page script, `ratchet.json` and the result JSONs.

### Honest headline

- **Exam, 48 real fires, simulated outcomes.** Mean share of the no-action 72 h growth stopped:

  | Method | 300 m/h | 1000 m/h |
  |---|---|---|
  | **FirePilot** | **0.474** | **0.644** |
  | Anchor-and-flank, best classical plan | 0.316 | 0.480 |
  | Forecast-world search (MPC, 29 simulations) | 0.400 | 0.591 |
  | Perfect-foresight bound | 0.614 | 0.792 |

- **Against anchor-and-flank:** ahead by +0.16 at both budgets (95 % CI about [0.08, 0.25]); wins on 71 % of fires.
- **Against MPC:** ahead by +0.07 [0.01, 0.14] at 300 m/h and +0.05 [−0.02, 0.13] at 1000 m/h, but wins on only 46 %
  of fires.
- **Against the bound:** FirePilot reaches 77–81 % of it.
- **Fidelity.** In the calibrated simulator, no-action growth from detection matches the observed FEDS growth in size
  (×1.1 at 48–72 h) and poorly in shape: F1 on new area 0.34; Dice on total area 0.40 against 0.19 for persistence at
  72 h.
- **The gain is a gain inside FireSim only.**

### Recipe as built

- **Physics:** FireSim from Phase 1, wrapped without editing it. Suppression lines with per-cell build times, rounded
  up to 15 min. Holding rule: a line fails where the fire arrives with flames above 2.4 m (NWCG hauling chart). Crown
  fire on, spotting off. Crews work 08:00–20:00 local solar time, from 2 h after detection, at 300 and 1000 m of line
  per working hour.
- **Truth vs forecast.** Truth: dev-calibrated ROS factors × a hidden f_ros ~ LogNormal(0, 0.5). Forecast: wind speed
  × LogNormal(0, 0.25) and direction + N(0, 20°); no hidden factor.
- **Physics rung p0 → p1 (dev-measured, kept):** the stored replay calibration. Over 72 h on 40 dev fires: Dice 0.29 →
  0.45, F1-new 0.26 → 0.33, new-area ratio ×3.9 → ×1.5.
- **Env fix U1:** 8-connected lines (see the upstream candidates below).
- **Plan space:** build order (centre-out or ends-in from anchors), centre angle, half-width, and offsets at the head
  and the rear (90 m – 8 km).
- **Training bank:** 297 synthetic scenarios. Landscapes 60 % real LANDFIRE from 28 non-gate dev fires, 40 %
  parametric; weather 75 % real HRRR, 25 % synthetic; the fire grown from a point ignition for 6–30 h up to an
  afternoon detection; 10 plans per scenario, scored in the truth world.
- **Network:** 31 features (13 on the scenario as seen through the forecast, 18 on the plan against the forecast,
  including "shadow" and connectivity proxies); a 31 → 64 → 64 → 2 tanh MLP with 6,338 weights, heads the mean and
  log σ; planning: 205 candidates plus 2 CEM rounds on the net, about 3 s per plan.
- **Checks:** numpy == torch 8.7e-8; JS == Python predictions 8.5e-7 (one timing feature differs by one 0.05 h clock
  step); release demo == exam plans 6/6.
- **Dev gate:** 12 seeded dev fires × 2 budgets, on real landscape and weather, scored in the simulator.
- **Error bars:** the widen-only factor fitted on the gate is 1.0.

### Rungs

Rule: a rung is kept if its frozen-sim val mean saved fraction (val_p1, 160 scenarios) beats the incumbent by more
than 0.01 AND its dev gate score is no more than 5 % below the incumbent.

| Rung | Change | Sim val | Dev gate | Kept |
|---|---|---|---|---|
| r0 | proxy planner: argmax of a connectivity proxy on the forecast; no learning | 0.403 | 0.383 | base |
| **r1** | **value net, 297 scenarios × 10 truth-scored plans** | **0.484** | **0.481** | **yes, released** |
| r2 | full 596-scenario bank | 0.470 | 0.473 | no |
| r3 | ensemble of 4 seeds of r1 | 0.476 | 0.494 | no (val −0.008) |
| r4 | + on-policy round, 195 scenarios planned by r1 (DAgger-like) | 0.483 | 0.483 | no |

- Baselines on the gate: anchor-and-flank 0.263, best indirect 0.111. The perfect-foresight CEM (70 sims) gets 0.588,
  so r1 reaches 82 % of the bound.
- Winner's curse (r4 round): r1 predicted 0.568 on its own picks; the truth was 0.454. Retraining on these rows did
  not improve the picks.

### Exam details

- **Stages.** The exam ran in two stages, split before any exam output existed, to save compute on the shared
  machine. Both finished on all 48 fires, 0 skipped, 0 failed. Stage A: no action, 5 fixed plans, r0 and FirePilot.
  Stage B: MPC and the perfect-foresight bound, in a seeded order.
- **By period:** main 2017–21 (32 fires) 0.446 / 0.631 against anchor and flank 0.285 / 0.422; recent 2022–24
  (16 fires) 0.530 / 0.670 against 0.379 / 0.595 (300 / 1000 m/h).
- **By detected-area tercile, 300 m/h:** small (≤ 24 ha) 0.74, anchor and flank 0.50, bound 0.86; mid (≤ 160 ha)
  0.49, 0.32, 0.65; large 0.20, 0.14, 0.33.
- **Oracle comparison:** the best of the 5 fixed plans per fire gets only 0.401 / 0.562.
- **Coverage:** 90 % coverage 0.98 / 1.00, 50 % coverage 0.69 / 0.77; prediction bias −0.11 / −0.07 (optimistic).
  σ is wide, so the bars are honest but not very informative.

### Upstream candidates (FireSim)

- **U1: time-blocked lines leak diagonally.** In `firesim/kernel.py` the corner rule blocks a diagonal step only when
  both side cells have `fuel_row < 0`; it ignores `t_block <= tt`, so a 4-connected timed suppression line lets fire
  pass between two diagonal line cells. Test `training/leak_test.py` → `results/upstream_U1_leak.json` (uniform grass,
  closed ring built at t = 0, cells burned outside the ring after 48 h): no line 3,972; 4-connected timed line 3,551
  (leaks); the same ring as non-burnable fuel 91: 0; 8-connected timed line 0. Suggested fix: in the corner rule,
  treat a side cell as blocked when `fuel_row < 0 or t_block[c] <= tt`. Phase-1 docs say "lines are 4-connected so
  fire cannot pass diagonally"; that is true for non-burnable barriers, not for suppression lines.
- **U2 (minor):** the stored replay landscape windows are tight. On 45 of 48 exam fires a 15 km crop reaches the
  window edge, so long runs are truncated at the domain edge.

### Limits

1. Simulated outcomes only. The simulated fire is often in the wrong place (F1 0.34 on new area), and the real fires
   were fought.
2. One line, two crews, one decision at detection: no re-planning, no aircraft, no spotting and no values at risk.
3. Large fires: only 0.20 stopped at 300 m/h.
4. Wide σ.
5. Domain-edge truncation.

### Next rungs

1. Re-plan each morning (receding horizon): the value net at each shift with the new detection. The biggest realism
   gain and probably the biggest score gain, since the bound is about 15 points higher.
2. Imitate the perfect-foresight search: the truth-CEM plans on the training bank as targets, or as features for a
   ranking loss. The MSE value net suffers from the winner's curse on its own argmax.
3. Fix U1 upstream, then rebuild the frozen sets (p2) and re-score every rung. Add values-at-risk targets (WUI cells)
   and multi-segment lines.

### Sandbox notes

- Four agents shared 2 cores: load average 4–5, cgroup memory 4–5 GB of 6.3 GB.
- `setsid nohup` forks, so `$!` is not the job's PID; use `pgrep -f` with the exact command.
- Stage-B exam cost: about 7 min CPU per fire. The stage split kept the exam to about 1 h 20 min of wall time.

## 2026-11-19 — rule notes and upstream items from the Wildfire handoff (sections 4-6)

- FirePilot split its exam into two stages for compute. Both finished on all 48 fires, so the split changed nothing.
- All four Phase 2 agents were interrupted once (an interface stall) and resumed from their status notes; each checked
  that no exam had started before resuming.
- Dev data in the gates: the 40 dev fires fed the physics p1 calibration (the dev replays) and supplied FirePilot's 12
  gate fires, so the gate is partly in-sample for the physics. The exam was untouched.
- Handoff section 4 lists U1 first among the upstream candidates for FireSim: fix the corner rule, then rebuild every
  frozen set as physics p2 and re-score; it touches all four Wildfire models. Section 4 also lists the domain edge (U2)
  and a fire-persistence module (simulated fires grow on 91 % of half-days, real ones on 50 %).
- Handoff section 5, FirePilot's next rungs: re-plan each morning; imitate the perfect-foresight search or use a
  ranking loss; fix U1, then physics p2 and re-score; values-at-risk targets, multi-segment lines, aircraft.

## 2026-11-21 — imported into ML Ratchet

From `firepilot-v1.0.zip` (19 Nov 2026). No retraining, no new exam scoring, no number changed.

What changed on import:
- **Moves (contract):** `demo_cases/` → `examples/` (bytes unchanged; its README.txt became `examples/README.md` with the
  data credits); `ratchet.json` → `training/ratchet.json`; `results/*.json` → `training/results/` (exam_summary,
  fidelity_dev_p0, fidelity_dev_p1, page_extra, sigma_widen, upstream_U1_leak, as shipped). The release README's
  content went into `model.yaml`; `ratchet.yaml` normalises `ratchet.json`.
- **Unchanged:** `firepilot.py` and `firepilot_weights.npz` (bytes identical to the zip). The weights file has no JSON
  `meta` entry, which the contract asks for; left as shipped.
- **demo.py** reads `examples/` instead of `demo_cases/`; its output is identical to the zip's demo. It needs only numpy
  and scipy: each example stores what the planner sees (the detected area, the forecast run's arrival time and flame
  length, the line cost and the crew clock), and the exam's truth-world outcomes are stored numbers, printed for
  comparison. They need FireSim's truth world (a hidden spread error), so nothing in the demo runs FireSim.
- **pipeline.py:** the sandbox defaults for FireSim and the replay calibration now point at `shared/fire/firesim` and
  `shared/fire/checks/replay_calibration.json`; it puts its own folder on `sys.path` so `import firepilot` works from
  anywhere. Only `plan_fire` imports FireSim. Tried on a FireSim parametric landscape (120 × 120 cells, synthetic
  weather, a 25-cell detection, 300 m/h): plan, prediction and build times in 11.7 s.
- **Training paths:** every sandbox path goes through `training/_paths.py`: WORK = `models/firepilot/work` or
  `$ML_RATCHET_WORK/firepilot` for everything a script writes (banks, rung weights, results, logs, the release),
  FireSim and FireData from `shared/fire`, the data root `$ML_RATCHET_FIRE`, the calibration
  `shared/fire/checks/replay_calibration.json`; shipped JSONs are read through `_paths.latest()`. `run_banks.sh` and
  `run_eval.sh` run from `training/` and log to WORK. `build_release.py` reads `training/fp/` and writes its zip into
  WORK. `web/build_page.py` reads its template from `web/` and writes into WORK. Nothing was re-run.
  `import fp.env` works against `shared/fire/firesim`; `fp.scenarios` also needs FireData's dependencies (shapely and
  pyarrow, not in the repository's base environment).
- **Lab:** `lab/index.html` is the published page without the page host's wrapper (`web/make_lab.py`). The published
  page was checked to equal `web/template.html` + `web/fp.js` + its DATA line to the byte, apart from the wrapper.
  `build_page.py` cannot be re-run: it needs `results/exam/<fire>_ctx.npz` and the per-budget masks, not shipped.
- **Instrument:** `web/export_instrument.py`, `site/src/instruments/firepilot/` (net.ts, parity.ts, ui.ts),
  `site/public/instruments/firepilot/` (0.66 MB: float32 weights 25 KB, the three example fires 0.57 MB, 90 parity
  cases). The browser computes the 31 features and the network for any plan and budget, and runs a lighter version of
  the planner's search with the kit's seeded random numbers.

Checks against the shipped files (what reproduced):
- `build_release.module()` regenerated from `training/fp/core.py` and `planner.py` is byte-identical to the shipped
  `firepilot.py`.
- The demo plans all 6 fire-budgets exactly as in the exam (seeded by each fire's hidden-draw index).
- Every exam number in the card and status doc matches `training/results/exam_summary.json`: the means, the 95 %
  intervals, the paired intervals (+0.158 [0.084, 0.228] and +0.164 [0.078, 0.258] against anchor and flank; +0.074
  [0.012, 0.139] and +0.053 [−0.020, 0.128] against MPC), the win rates (70.8 %, 45.8 %), the shares of the bound
  (77.1 %, 81.2 %), the hectares, the periods, coverage and bias; 48 fires scored, 48 in stage B, none skipped.
- The fidelity tables match the summaries in `exam_summary.json` (exam) and `fidelity_dev_p0/p1.json` (dev: Dice 0.292
  → 0.446, F1 0.258 → 0.326, new-area ratio 10^0.594 = ×3.9 → 10^0.167 = ×1.5).
- The ratchet table matches `training/ratchet.json`, and every decision follows the rule (r1 gate +25.8 %; r2 −0.014;
  r3 −0.008; r4 −0.001 on val).
- The site instrument's TypeScript port matches `firepilot.py` on 90 cases (3 fires × 3 budgets × 10 plans; 31
  features, the mean and σ, km, the proxy and the line cell count) to 4.4e-16.

Found on import (recorded, not fixed):
- **The page's one-step timing difference, probably float precision.** The status doc's "one timing feature differs by
  one 0.05 h clock step (6.9e-4)" is most likely float32 against float64: in `firepilot.py` the cost raster is float32
  and `cost × cell_len` (a Python float) stays float32 under numpy's promotion rules, so the per-cell cost and its
  cumulative sum are float32, while the page's `fp.js` works in float64; near a clock-step boundary that can move a
  build time by one 0.05 h step. `page_check.json` was not shipped, so this is not confirmed. On the three example
  fires (90 reference cases, stored float32 clocks) the shipped `fp.js` differs from `firepilot.py` by at most 8.2e-8
  (the mean line cost and log km), with no step jumps. The site instrument emulates numpy's float32 arithmetic and
  pairwise sums and matches to 4.4e-16. `web/fp.js` (the lab page) is left as shipped.
- **297 or 298 training scenarios.** The card, status doc and r1 say 297; r3's entry in `ratchet.json` says "same 298
  training scenarios". Not checkable (the banks were not shipped).
- **Perfect-foresight bound cost:** the card says about 44 truth simulations per fire; `exam.py`'s docstring says ~46.
- **Not checkable here:** the tercile table (small / mid / large) and the per-fire rows behind it (`results/exam/` was
  not shipped); "about 3 s median, 53 s max" planning time on the exam; numpy == torch 8.7e-8; the network's selection
  RMSE 0.124 and Spearman 0.82.
- **U1 touches every outcome here.** The leak is in FireSim's kernel, which scored every training target, the ratchet
  and the exam. FirePilot avoids it with 8-connected lines whose cells share the 4-connected length, so a fix (physics
  p2) means rebuilding the frozen sets and re-scoring every rung and baseline. Recorded in `model.yaml` limits and next.

## 2026-11-22 — exam re-run on rebuilt data (reproduction check, nothing chosen)

The 48 replay fires' landscapes and HRRR were rebuilt from LANDFIRE and AWS on Windows. `firepilot_weights.npz` was
copied to `work/models/r1.npz` (build_release.py copied it the other way) and `exam.py A`, `B` (12 shards, numba
0.68) and `summary` ran once into `work/results/`. Stage B finished all 48 fires, as shipped.
- **Planner outcomes reproduce:** all 324 numbers in `exam_summary.json`'s planner tables match within 2e-9. Share of
  growth stopped 0.474 / 0.644 (300 / 1000 m/h); perfect-foresight bound 0.614 / 0.792; r0 0.365 / 0.524.
- **Fidelity Dice on total area does not, because of a later fix.** With the current FireSim the 24 h median is
  0.494 (shipped 0.484) and the 48 h median 0.4355 (0.435); 72 h, F1 on new area and persistence are unchanged.
  Running the same 48 fires with FireSim as imported (bcaf80b) gives the shipped 0.4836 / 0.435 / 0.405 exactly. The
  cause is the 21 Nov PR #7 review fix that burns a mask ignition's interior at t = 0: in the free run, pockets
  inside the first detected perimeter used to burn later, which lowered Dice on the total area at 24-48 h. Planner
  outcomes score growth outside the start mask and are unaffected.

## 2026-11-22 — pipeline.py pinned to FireSim physics p1

FireSim's default physics became p2 today (U1, U9, U10, U11 fixed; `shared/fire/STATUS.md`). FirePilot v1.0 was
trained, gated and graded in p1, so `pipeline.plan_fire` now passes `physics='p1'` with the calibrated ROS factors.
Its forecasts are unchanged. The demo reads stored examples and does not call FireSim. Moving FirePilot to p2 means
a p2 frozen set (val_p2) and a re-score, not this edit.

## 2026-11-22 — r1 scored on a FireSim physics p2 frozen set; p1's lines were helped by two solver bugs

`FIREPILOT_PHYSICS` (p1 default, p2) now sets the planning world. `calibrated_params` passes FireSim's solver
explicitly, and so does the weather generator. Banks go to `work/data/val_<physics>`; results go to
`work/results/eval_val_<physics>_<rung>.jsonl`. Before this, the training code called FireSim with its default,
which became p2 today, while still writing to `*_p1` folders.
- **p1 rebuild (`val_p1`, 160 scenarios):** everything matches the ratchet: r1 0.4841, r0 0.4033, direct attack
  0.107, indirect 1 / 2 / 4 km 0.261 / 0.203 / 0.078, anchor and flank 0.195.
- **p2 (`val_p2`, 159 scenarios; one had no growth):**
  - r1 0.460 (median 0.453), r0 0.403. r1 still beats r0 by +0.057.
  - Indirect 1 / 2 / 4 km 0.269 / 0.215 / 0.079.
  - Direct attack 0.054 and anchor and flank 0.114: both close lines near the fire.
- **Why the near-fire plans fell:** the same scenario and the same 42 line cells saved 0.873 under p1 and 0.007
  under p2 (case v0117). Two p1 solver bugs let a line take credit for cells the fire reached before the line was
  built:
  - U12: the cell itself stayed unburned (8 such cells in v0117).
  - U13: spread in transit from a cell was cancelled once a line was built on it, if the step crossed an hour.
  - Test: under p1, removing from the plan the line cells the free fire reaches before they are built takes
    direct attack's saved share from 0.87 / 0.78 / 0.67 / 0.75 / 0.30 to 0.004 / 0.071 / 0.012 / 0.068 / 0.052
    (v0117, v0145, v0007, v0058, v0017). Under p2 the same removal changes nothing.
- **What this means for v1.0:** every FirePilot number was scored in p1, so plans with lines close to the fire were
  credited for some line built behind the front. That includes the exam and the fixed-plan baselines. r1's
  advantage over r0 holds under p2 on val. The exam and the gate were not re-run under p2. `pipeline.py` stays on p1.
- Only r1's weights were shipped, so r2 to r4 were not re-scored and no decision was re-taken.

## 2026-11-22 — gate and exam re-run under FireSim physics p2 (released r1 weights, nothing chosen)

The dev gate (12 gate fires x 2 budgets) was rebuilt twice, as `gate_p1` and `gate_p2`, from the same seeds. r0 and
r1 were scored on both, and the widen factor was refitted on each. The 48-fire exam then ran once under p2 with the
released r1 weights: stage A, stage B on all 48 fires (16 shards), and the summary. Nothing was retrained or chosen.
The p2 outputs are kept in `training/results/` as `exam_summary_p2.json`, `sigma_widen_p2.json` and
`fidelity_dev_p2.json`. `exam.py` and `fit_widen.py` now follow `FIREPILOT_PHYSICS`. p1 keeps the shipped names
(`results/exam/`, `exam_summary.json`, `sigma_widen.json`); p2 adds a `_p2` suffix.

- **Gate, p1 rebuild:** identical to the ratchet log. r1 0.4815, r0 0.3828, and all five baselines match to the last
  digit. The widen fit is identical too: bias −0.040, 90 % coverage 0.917, factor 1.0.
- **Gate, p2 (24 cases):** r1 0.487 (median 0.526), r0 0.354.
  - Anchor and flank falls from 0.263 to 0.209.
  - Direct attack falls from 0.144 to 0.133.
  - Indirect 1 / 2 / 4 km: 0.104 / 0.118 / 0.033 (p1: 0.107 / 0.111 / 0.025).
  - The widen factor is still 1.0 (90 % coverage 0.958, bias −0.042).
- **Exam, p2 (48 fires, all scored, stage B on all 48), p1 → p2 mean share of the no-action growth stopped:**

  | 300 / 1000 m/h | p1 (shipped) | p2 |
  |---|---|---|
  | FirePilot r1 | 0.474 / 0.644 | 0.401 / 0.629 |
  | MPC (forecast search) | 0.400 / 0.591 | 0.351 / 0.526 |
  | r0 | 0.365 / 0.524 | 0.347 / 0.526 |
  | Anchor and flank | 0.316 / 0.480 | 0.198 / 0.372 |
  | Direct attack | 0.179 / 0.215 | 0.138 / 0.176 |
  | Indirect 1 km | 0.122 / 0.147 | 0.120 / 0.147 |
  | Oracle best fixed plan | 0.401 / 0.562 | 0.316 / 0.480 |
  | Perfect-foresight bound | 0.614 / 0.792 | 0.582 / 0.780 |

  - **300 m/h:**
    - FirePilot minus MPC is +0.050 (paired 95 % −0.013 to 0.115). In p1 it was +0.074 (0.012 to 0.139).
    - FirePilot minus r0 is +0.054 (−0.025 to 0.131). In p1 it was +0.108 (0.065 to 0.153).
    - So at the short budget FirePilot no longer clearly beats MPC or r0. It still beats anchor and flank (0.128 to
      0.278) and direct attack.
    - FirePilot beats MPC on 64.6 % of fires (p1 45.8 %).
  - **1000 m/h:**
    - FirePilot minus MPC is +0.103 (0.030 to 0.179). In p1 it was +0.053 (−0.020 to 0.128).
    - MPC lost more than FirePilot did, so the tie becomes a win.
    - FirePilot beats MPC on 56.2 % of fires, ties on 6.2 %.
  - **Share of the bound:** 0.689 / 0.806 (p1 0.771 / 0.812).
  - **Hectares kept unburned, summed:** FirePilot 15,529 / 51,870, MPC 15,601 / 46,880.
  - **Recent fires (16), 300 / 1000 m/h:** FirePilot 0.487 / 0.651, MPC 0.413 / 0.565.
  - **Main-period fires (32):** FirePilot 0.358 / 0.618, MPC 0.319 / 0.506.
  - **Error bars:** the network predicts 0.589 / 0.718 for its own picks, optimistic by 0.188 / 0.089 (p1 0.113 /
    0.069). The 90 % range still covers 100 % / 100 % of fires; the 50 % range covers 64.6 % / 75.0 %.
- **Where the 300 m/h loss is:** 44 of 48 FirePilot picks at 300 m/h are centre-out (mode 0); their mean fell from
  0.511 to 0.434. Only 23 % of them are the same plan as under p1, because the forecast run the planner reads
  changes with the physics too. This is consistent with the val_p2 finding (short budgets mean lines built close
  to the fire, which U12 and U13 credited), but the U12/U13 cell-removal test was not repeated on exam fires.
- **Fidelity, free run vs the observed perimeters (48 exam fires, median Dice on total area 24 / 48 / 72 h):**
  - p1: 0.484 / 0.435 / 0.405 as shipped (0.494 / 0.436 / 0.405 with today's mask-ignition fix).
  - p2: 0.500 / 0.427 / 0.411. F1 on new area 0.267 / 0.346 / 0.341 (p1 0.271 / 0.343 / 0.341).
  - The dev fidelity (40 dev fires) moves by less than 0.01 from the p1 rebuild. The p1 rebuild's 24 h Dice is
    0.581 against 0.527 shipped, from the same mask-ignition fix.
- **What it means:** r1 still beats r0 on the p2 gate (0.487 against 0.354), so the r1 decision stands under p2. On
  the exam, part of v1.0's 300 m/h lead came from line credited behind the front. At 1000 m/h the result holds and
  gets stronger against MPC. The fixed near-fire plans, anchor and flank and direct attack, were also overstated in
  p1. `pipeline.py` and the released weights are unchanged, and v1.0's headline stays the p1 exam, which is what was
  graded. The model card adds the p2 numbers.

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). The value network only saw afternoon starts (34 working hours). The reader now warns on other clocks. The exam fires all had 34 h; training on varied start times is queued.
