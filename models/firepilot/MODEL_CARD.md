# FirePilot v1.1 — model card

FirePilot is a learned planner for containment line. It is given a wildfire at its first afternoon satellite
detection and a crew budget (metres of line per working hour). It chooses one line plan for the next 72 hours:
where the line goes and in which order the crews build it. The aim is to keep as much as possible of the area the
fire would otherwise add from burning.

It learned only from simulations. It was graded once in the simulator, FireSim, on 48 real exam fires. Each exam
fire had its real LANDFIRE fuels and terrain, its real HRRR weather and its real first FEDS detection.

**A win in FireSim is not a win on the ground.** Every outcome below is simulated. The fidelity section shows how
closely the simulator follows the real fires.

ML Ratchet Wildfire, Phase 2, project 4. Released 19 Nov 2026. Imported into the repository on 21 Nov 2026: paths
below are repository paths (the release zip's `demo_cases/`, `results/` and `ratchet.json` are now `examples/`,
`training/results/` and `training/ratchet.json`); no number changed. Lab page: `lab/index.html`; live instrument on
the model's site page.


## v1.1 (27 Nov 2026): trained on varied crew start times

The audit found that every training scenario started in the afternoon, so the crew-clock input never varied. r5vs is
the same network and recipe on 299 scenarios whose crews start between 06 and 18 h local solar time. The ratchet kept
it (val 0.495 vs 0.484, +0.011 against a +0.01 margin; gate 0.492 vs 0.482). Graded once on the same 48 fires: 0.469 /
0.625 of the growth stopped at 300 / 1000 m/h against v1.0's 0.474 / 0.644 (paired differences −0.005 and −0.019,
both intervals include zero). It still beats MPC at 300 m/h and ties it at 1000 m/h. What improved is calibration: its
prediction for its own pick is optimistic by 0.054 / 0.005 (v1.0 0.113 / 0.069). Start hours 06-18 cover only 34-36
working hours in 72 h; the reader warns outside that. Everything below describes v1.0 unless it says otherwise;
`model.yaml` and `NOTEBOOK.md` (27 Nov 2026) have v1.1's numbers.
## What it does

- **Input:**
  - the burned mask at detection (90 m cells)
  - one FORECAST FireSim run from that mask: arrival time and flame length, with the forecast's wind error
  - a line-cost raster from the fuel map
  - the crew clock (crews work 08:00–20:00 local solar time, starting 2 h after detection)
  - the budget in metres of line per working hour, for two crews
- **Plan space (5 numbers):**
  - build order: centre-out (mode 0), or ends-in from anchors (mode 1)
  - the centre angle φc relative to the forecast head
  - the half-width ω (5–180°)
  - the offset from the fire edge at the plan centre, d_h, and at the far side, d_f (90 m – 8 km)
  - The line is the 8-connected inner boundary of the region {distance ≥ d(Φ)} inside the sector, where
    d(Φ) = d_f + (d_h − d_f)·((1 + cos(Φ − φc))/2)².
- **Holding rule.** A line cell blocks spread from the moment it is built. It fails if the fire arrives there with a
  flame length above 2.4 m (8 ft, the NWCG hauling-chart limit for hand and dozer line).
- **Output:** the chosen plan, its line cells and build times, the km built, and the network's predicted share of
  the no-action growth stopped, with a σ.
- **Planning procedure:**
  - 205 candidate plans: the 5 baselines, a 40-plan grid and 160 random plans
  - 2 rounds of CEM refinement on the network (48 plans each)
  - the plan with the highest predicted score is chosen
  - about 3 s median on one CPU core (max 53 s on a 650k-cell window)

## Physics and simulator

- **FireSim (Phase 1):**
  - Rothermel 1972 surface spread with Albini 1976
  - Scott & Burgan 2005 fuel models
  - Van Wagner 1977 crown fire
  - Nelson dead-fuel moisture
  - minimum-travel-time spread on a 16-neighbour stencil
- FireSim was verified against BehavePlus-style reference cases in Phase 1 (`shared/fire/checks/`).
- **FirePilot's planning world** wraps FireSim without editing it. It uses suppression lines with per-cell build
  times, rounded up to 15 min, and no spotting. Crown fire is on.
- **Two worlds per scenario:**
  - **truth:** the scenario's weather and the dev-calibrated ROS factors (`shared/fire/checks/replay_calibration.json`),
    times a hidden factor f_ros ~ LogNormal(0, 0.5)
  - **forecast:** what the planner sees; wind speed × LogNormal(0, 0.25) and direction + N(0, 20°), held constant;
    the calibrated ROS factors without the hidden factor
- **Physics rung p0 → p1 (kept, measured on 40 dev fires).** We ran the simulator freely for 72 h from first
  detection and compared it with the observed FEDS perimeters:

  | 40 dev fires, 72 h | uncalibrated FireSim (p0) | dev-calibrated ROS (p1) |
  |---|---|---|
  | Dice on total burned area | 0.29 | 0.45 |
  | F1 on newly burned area | 0.26 | 0.33 |
  | Simulated / observed new area (median ratio) | ×3.9 | ×1.5 |

- **Line geometry U1 (env fix, upstream candidate).** FireSim's corner rule blocks a diagonal step only when both
  side cells are permanently non-burnable. Cells blocked by a timed suppression line are not checked, so a
  4-connected line leaks diagonally. Test (`training/leak_test.py`, `training/results/upstream_U1_leak.json`): a closed
  4-connected ring was built at t = 0 around a point ignition in uniform grass.
  - As a timed line, 3,551 of 3,972 outside cells still burned in 48 h.
  - The same ring as non-burnable fuel held, with 0 cells burned outside.
  - FirePilot uses 8-connected lines, which hold (0).
  - The line length is counted from the 4-connected boundary × π/4, so the 8-boundary does not inflate the km.

## Training data (simulated only)

- **Scenarios:**
  - 60 % real LANDFIRE windows from 28 dev fires (the 12 gate fires are excluded) and 40 % FireSim parametric
    landscapes
  - 75 % real HRRR weather from those dev fires and 25 % FireSim synthetic weather
  - 140–360 cells per side
  - the fire is grown from a point ignition for 6–30 h up to an afternoon detection
  - the budget is drawn at random
- **Training bank:** 297 scenarios (seeds 0–149 and 300–449) × 10 plans scored in the truth world:
  - 2 baselines
  - 4 prior draws
  - the 4 plans ranked best by a geometric proxy
  - 2,970 rows in all
- **Noise:** the forecast/truth gap, a hidden ROS factor plus wind-speed and wind-direction errors, is the noise the
  network must live with. Real HRRR weather and real LANDFIRE fuels supply the real background.
- No real fire outcome is ever a training target.

## Network

- **31 features:**
  - 13 describe the scenario as seen through the forecast: detected and forecast area, growth at 12/24/48 h,
    budget, working hours, spread speed, the share of the forecast growth with flames above 2.4 m, median flame
    length, edge contact, mean line cost, and buildable line vs the forecast perimeter
  - 18 describe the plan against the forecast: plan geometry, km buildable, the share of line built before the
    forecast fire arrives, the share overrun, timing, the "shadow" area behind the line, and two connectivity
    proxies of the area saved
- **Architecture:** a 31 → 64 → 64 → 2 tanh MLP with 6,338 weights. The heads are the mean (sigmoid, MSE on the
  saved fraction) and log σ (Gaussian NLL on the detached residual).
- **Training:** 300 epochs, early stopping on a 10 % simulated selection set: RMSE 0.124, Spearman 0.82 (the
  geometric proxy alone gives 0.70).
- **Checks:**
  - numpy == torch: 8.7e-8
  - JS == Python on the 6 fires embedded in the page (66 plans, `web/check_js.js`): predictions within 8.5e-7; features within 1e-7, except one timing feature that differs by one 0.05 h clock step (6.9e-4)

## Ratchet

- **Primary sim metric:** the mean share of the no-action 72 h growth stopped, on the frozen simulated validation
  set val_p1 (160 synthetic scenarios, seeds 900000+).
- **Dev gate:** 12 seeded dev fires × 2 budgets (300 and 1000 m/h). The gate uses the real landscape and weather
  and is scored in the simulator.
- **Keep rule:** sim val > incumbent + 0.01 AND gate no more than 5 % (relative) below the incumbent.

| Rung | Change | Sim val | Dev gate | Decision |
|---|---|---|---|---|
| r0 | proxy planner: argmax of a connectivity proxy on the forecast run; no learning | 0.403 | 0.383 | base |
| **r1** | **value net, 297 scenarios × 10 truth-scored plans; argmax + 2 CEM rounds on the net** | **0.484** | **0.481** | **kept, released** |
| r2 | same net, full 596-scenario bank | 0.470 | 0.473 | rejected (val −0.014) |
| r3 | ensemble of 4 seeds of r1 | 0.476 | 0.494 | rejected (val −0.008) |
| r4 | r1 data + 195 on-policy scenarios (r1's plan + 4 jitters + 2 prior, truth-scored) | 0.483 | 0.483 | rejected (val −0.001) |

Fixed baselines on val: direct attack 0.107, indirect 1 km 0.261, indirect 2 km 0.203, indirect 4 km 0.078,
anchor-and-flank 0.195. On the gate: 0.144, 0.107, 0.111, 0.025 and 0.263. On the gate, the perfect-foresight CEM
search (70 truth simulations) gets 0.588, and r1 gets 82 % of that.

The on-policy round showed the expected winner's curse. r1 predicted 0.568 on its own chosen plans, and the truth
was 0.454. Retraining with those rows removed part of the bias, but it did not pick better plans.

## Final exam (run once)

**Set-up.**
- The fires are the 48-fire exam replay subset (frozen list in `shared/fire/data/proc/baselines/replay_subset.json`):
  32 main-period fires (2017–21) and 16 recent ones (2022–24).
- Each fire has a seeded hidden draw (truth ROS factor, forecast wind errors). Both budgets share it.
- The exam ran in two stages. The split was fixed before any exam output existed, to save compute on a shared
  machine. Both stages finished on all 48 fires.
  - **Stage A:** the no-action run, the 5 fixed plans, r0 and FirePilot.
  - **Stage B:** the forecast-world CEM search (MPC: 2 iterations × 6 plans per mode plus seeds, 29 forecast
    simulations) and the perfect-foresight bound (CEM in the truth world, 3 × 6 per mode, seeded with every other
    plan, about 44 truth simulations).
- 95 % intervals are bootstrap intervals over fires.

**Mean share of the no-action 72 h growth stopped (simulated), 48 fires:**

| Plan | 300 m/h (3.6 km per shift) | 1000 m/h (12 km per shift) |
|---|---|---|
| No action | 0 | 0 |
| Direct attack at the head | 0.179 | 0.215 |
| Indirect 1 km | 0.122 | 0.147 |
| Indirect 2 km | 0.056 | 0.082 |
| Indirect 4 km | 0.010 | 0.026 |
| Anchor and flank (best classical) | 0.316 [0.226, 0.403] | 0.480 [0.365, 0.586] |
| Best fixed plan per fire (oracle pick of 5) | 0.401 | 0.562 |
| r0 proxy planner | 0.365 | 0.524 |
| MPC: forecast-world search, 29 sims | 0.400 [0.308, 0.492] | 0.591 [0.483, 0.693] |
| **FirePilot** | **0.474 [0.366, 0.572]** | **0.644 [0.541, 0.735]** |
| Perfect-foresight bound, ~44 truth sims | 0.614 [0.513, 0.708] | 0.792 [0.707, 0.868] |

- **FirePilot minus anchor-and-flank:** +0.158 [0.084, 0.228] at 300 m/h and +0.164 [0.078, 0.258] at 1000 m/h.
  FirePilot wins on 71 % of fires at both budgets.
- **FirePilot minus MPC:** +0.074 [0.012, 0.139] at 300 m/h and +0.053 [−0.020, 0.128] at 1000 m/h. FirePilot
  wins on only 46 % of fires. It wins by more when it wins.
- **Share of the bound:** FirePilot reaches 77 % / 81 % of the perfect-foresight bound.
- **Hectares kept unburned, summed over 48 fires:**

  | Budget | FirePilot | MPC | anchor and flank | bound |
  |---|---|---|---|---|
  | 300 m/h | 20,651 ha | 18,467 ha | 8,248 ha | 45,768 ha |
  | 1000 m/h | 65,230 ha | 51,543 ha | 15,454 ha | 95,853 ha |

- **By period, 300 / 1000 m/h:**

  | Period | FirePilot | anchor and flank |
  |---|---|---|
  | main (32 fires) | 0.446 / 0.631 | 0.285 / 0.422 |
  | recent (16 fires) | 0.530 / 0.670 | 0.379 / 0.595 |

- **By detected-area tercile, 300 m/h:**

  | Tercile | FirePilot | anchor and flank | bound |
  |---|---|---|---|
  | small, ≤ 24 ha | 0.74 | 0.50 | 0.86 |
  | mid, ≤ 160 ha | 0.49 | 0.32 | 0.65 |
  | large | 0.20 | 0.14 | 0.33 |

- **Error bars.** The widen-only factor was fitted on the dev gate and came out at 1.0 (the raw gate coverage90 was
  0.92).
  - Exam coverage at 90 %: 0.98 / 1.00. Coverage at 50 %: 0.69 / 0.77.
  - The network is optimistic about its own pick: truth minus prediction is −0.11 / −0.07.
  - The σ is wide, often ±0.8 or more on a 0–1 score. The bars are honest but not very informative.

## Fidelity: simulator vs the real fires

The test is a free run of the truth physics (no hidden error, no action) from the first detection, compared with
the observed FEDS PM perimeters. Medians over 48 exam fires:

| Lead | Dice on total area | Persistence Dice | F1 on new area | Sim / obs new area |
|---|---|---|---|---|
| 24 h | 0.48 | 0.47 | 0.27 | ×1.55 |
| 48 h | 0.43 | 0.25 | 0.34 | ×1.11 |
| 72 h | 0.40 | 0.19 | 0.34 | ×1.14 |

The simulator gets the size of the growth about right at 48–72 h. It gets the shape only partly right: F1 on the
new area is about 0.34. The real fires were also fought, so "no action" is not their real counterfactual.

## Limits

1. **Simulated outcomes only.** The planner's gain is a gain inside FireSim, with the holding rule above. It has not
   been tested against real line-building records. With F1 about 0.34 on new area, the simulated fire is often in
   the wrong place.
2. **The plan space is simple:** one line, two crews, chosen once at detection. There is no re-planning, no
   spotting, no aircraft, no values at risk and no line loss other than the flame-length rule. Large fires
   (detected area above about 160 ha) get only 0.20 at 300 m/h.
3. **Error bars are wide** (90 % coverage 0.98–1.00), and the network is optimistic about its own choice (bias
   −0.07 to −0.11).
4. **The domain is the stored LANDFIRE window.** On 45 of 48 exam fires the 15 km crop box reaches the window edge.
   Growth past it is cut off for every method alike.
5. **Training scenarios are synthetic fires** grown from point ignitions on dev landscapes. Real detections have
   more ragged shapes.

## Files

- `firepilot.py`: numpy + scipy planner, generated from the training code
- `firepilot_weights.npz`: r1
- `demo.py`, `examples/`: 3 exam fires drawn by a seed, one per size tercile
- `pipeline.py`: plan your own fire; needs FireSim (`shared/fire/firesim`) for the forecast run
- `training/ratchet.json`, `training/results/*.json`: exam summary, fidelity, sigma widening, U1 leak test
- `training/`: every script (banks, training, rungs, exam, fidelity and release builders) and `fp/`
  (environment, scenarios, planner); `training/_paths.py` maps the sandbox paths to the repository
- `web/`: the page builder, its template and `fp.js` (the JavaScript planner); `web/export_instrument.py` writes the
  site instrument
- `lab/index.html`: the published page, unwrapped (see `lab/README.md`)

## Data credits and licences

- FEDS-MTBS fire progression: Chen et al. 2022 method; Zenodo 20187963; CC BY 4.0
- MTBS perimeters: USGS/USFS; public domain
- LANDFIRE fuels, canopy and terrain: public domain
- HRRR analyses: NOAA, via the University of Utah hrrrzarr archive on AWS Open Data
- gridMET: Abatzoglou 2013
- FireSim and FirePilot code: ML Ratchet project, open source
