# Wildfire handoff: what exists, what's missing, how to finish it

For Greg's code agent, 19 Nov 2026. This is the "make it complete" list for the wildfire work in ML Ratchet
(github.com/dogum/ML-Ratchet). It is the fifth big-system project, after RoadSim, GridSim, BuildingSim and WaterSim.

Read this first. Then read:
- `FIRE_BRIEF.md` (the contract) and `PHASE2_BRIEF.md`
- `firesim/README.md`
- `data/DATA.md`, especially section 2, "Units and conventions (gotchas first)"

## 1. What exists

| Piece | Where | State |
|---|---|---|
| FireSim simulator (`firesim`) | `firesim-firedata-v1.0.zip` → `firesim/` | v1, verified (section 3) |
| FireData loader (`firedata`): frozen splits, exam guard, metrics, baselines, the 40 dev-fire packages, fetch scripts | same zip → `firedata/`, `data/` | v1 |
| Physics checks and their JSONs | same zip → `checks/` | pass |
| FireCast (next-day spread probability) | `firecast-v1.0.zip` | v1.0 |
| FireSize (escape probability and final size from an ignition record) | `firesize-v1.0.zip` | v1.0 |
| FireDetective (what is driving a fire) | `firedetective-v1.0.zip` | v1.0 |
| FirePilot (containment-line planner, graded in simulation) | `firepilot-v1.0.zip` | v1.0 |
| Status docs | claude.ai project "Physics Fun": `wildfire-status.md`, `wildfire-handoff.md`, `firecast-status.md`, `firesize-status.md`, `firedetective-status.md`, `firepilot-status.md`; also in `wildfire-status-docs.zip` | current |

- **Every model zip has:**
  - numpy-only inference and weights
  - a demo on real exam data (all four were run from unzipped copies on 19 Nov)
  - `MODEL_CARD.md` and the training scripts
- **Not in the Phase 1 zip.** Rebuild these with `data/scripts/`, all of which are resumable:
  - FEDS-MTBS progression (171 MB)
  - WildfireSpreadTS (607 fires, 269 MB processed)
  - FPA-FOD (1.59 M ignitions)
  - Globe-LFMC
  - gridMET at fire and ignition points
  - the exam fires' landscapes and HRRR (48 replay fires)
  - the per-fire exam baseline rows (`spread_exam.jsonl`, 89 MB)
  - the pyretechnics reference environment (`firesim/.refenv`)
  - all simulated banks
- **In the zip:**
  - splits, fire tables, the exam guard footprints and baseline summaries
  - LANDFIRE landscapes plus hourly HRRR for the 40 dev fires, so the real generators and the dev replays work offline
- **Licences:**
  - FEDS-MTBS: CC BY 4.0
  - GOFER, WildfireSpreadTS: see DATA.md
  - Globe-LFMC: CC0
  - LANDFIRE, MTBS, FPA-FOD, HRRR, gridMET: US public data
  - Prior Labs licence: only for Prior Wars, not here

## 2. The vision vs what was built

| Area | Status | Gap |
|---|---|---|
| Surface fire (Rothermel/Albini, 53 fuel models, WAF, slope) | Done; matches published tables | — |
| Crown fire (Van Wagner, Rothermel 1991, Cruz 2005) | Done | Crown spotting uses the surface plume formula |
| Spread solver (minimum travel time, 16/32-cell stencils, level-set-like) | Done | `huygens` not implemented; elongated ellipses lose up to 6 % of area at the default stencil |
| Spotting (Albini + stochastic embers) | Present | **Uncalibrated**; the replays run without it; FireDetective can't read spotting even in simulation (AUC 0.47) |
| Fuel moisture (Nelson-type dead fuels, GSI live) | Done | 2.4 points drier than gridMET NFDRS. **No real dead-fuel stick data**: NFMD moved into FEMS and has no public download. Live moisture is constant unless supplied |
| Fire–atmosphere coupling, plume winds, terrain wind downscaling | **Not started** | Wind-event nights are missed by an order of magnitude (Riverside 2020: 496 km² overnight) |
| Suppression | Partial: timed lines, ROS rasters, initial attack | No validated suppression data (ICS-209); FireSize uses a dev-fitted effective containment race with parameters at their bounds |
| Fire persistence (burnout, stalls, night inactivity) | **Missing** | Simulated fires grow on 91 % of half-days against 50 % for real ones. This one gap drives the over-prediction in FireCast, FireDetective and the replays |
| Satellite emulator | VIIRS calibrated on dev counts and FRP | MODIS and GOES uncalibrated; no real cloud masks; false alarms not measurable |
| Fire danger indices | FWI (matches xclim) | NFDRS ERC/BI not implemented (gridMET's ERC/BI used instead) |
| Real data | Strong for CONUS 2012–2024 | Hourly weather only from Aug 2016 (HRRR); hourly precipitation missing before mid-2018; LANDFIRE 2016 fuels used up to 2022 (stale, and scarred by 2012–16 fires); no 2024 disturbance layer |

## 3. Verification baseline (do not regress)

From `checks/*.json`.

- **Published Rothermel/BehavePlus tables** (GTR-371, GTR-143): 436 of 436 within half the last published digit.
- **vs pyretechnics** (flat and slope-only, 6,720 cases): ROS median 4.9e-5, maximum 5.2e-5. Crown ROS agrees to 1e-7 and fire type 100 %. Wind on slope differs by a median 1.3 %, by design.
- **Ellipse tests** (55 cases, 30–90 m, 7 wind directions):

  | Solver | Mean area ratio | Minimum area ratio | Grid-orientation bias |
  |---|---|---|---|
  | Default | 0.981 | 0.940 | ≤ 3.4 % |
  | `level_set` | 0.992 | 0.973 | — |

- **Two-fuel strip and fuel break:** two-fuel strip 2.9e-9; the fuel break leaks 0 cells.
- **Crown hand calculations:** exact.
- **Moisture:** Simard isotherm RMSE 0.41 points.
- **FWI:** standard test row within 0.005; matches xclim.
- **Speed:** 30 × 30 km, 60 m, 24 h in 0.47 s; batch 0.09 s per ignition at 90 m.
- **Dev replays** (40 fires, 470 growth days; F1 on newly burned cells):
  - uncalibrated 0.153
  - dev-calibrated 0.194 in sample, 0.184 leave-one-out
  - area over-predicted 4–12×
- **VIIRS emulator:** dev-calibrated count ratio +0.01 log10, Spearman 0.80.

## 4. Upstream candidates (physics found by Phase 2, currently in each model's own folder)

1. **Bug U1: timed suppression lines leak diagonally** (FirePilot `training/leak_test.py`).
   - Cause: in `firesim/kernel.py` the corner rule blocks a diagonal step only when both side cells are non-burnable; it ignores cells blocked by a line (`t_block <= tt`).
   - Evidence: a closed 4-connected ring in grass let 3,551 of 3,972 outside cells burn.
   - Fix: treat a side cell as blocked if `fuel_row < 0 or t_block[c] <= tt`. Then rebuild every frozen set as physics p2 and re-score.
2. **Fire persistence / burnout knob.**
   - FireDetective v2/v3 added night fine-fuel recovery, anchor-and-flank containment and a wind-sensitive WAF. The WAF change raised sim-vs-real per-step growth ρ from 0.36 to 0.44.
   - FireCast found a fire-age spread factor (early days too slow, days 5+ too fast; twin median log10 ratio +0.16).
   - Unify these into one documented persistence module.
3. **Agriculture fuel.** LANDFIRE 93 should burn, e.g. as GR1 by season. It was FireSize's biggest single gain. Also move ignitions that land on non-burnable cells to the nearest burnable cell (≤ 180 m).
4. **A sub-cell analytic ellipse** for fires under 0.81 ha (FireSize).
5. **A containment race module**: Fried & Fried style, with two attack modes and reporting floors. It is in FireSize's `supp.py` and `calib_v1rep2.json`.
6. **A FEDS-like perimeter emulator**: from simulated VIIRS pixels to FEDS-style perimeters, calibrated on dev to a median area ratio of +0.012. FireCast and FireDetective each built one; merge them into `firesim.observe`.
7. **Memory-bounded `replay.replay_day` ensembles.** Today each member copies the prepared fields, so 12 members on crops over 500k cells need more than 1.2 GB.
8. **Domain edge (U2).** On 45 of 48 exam replay fires the 15 km crop reaches the stored landscape edge. Store larger windows for replay fires.

## 5. Per-model state and next rungs

### FireCast v1.0: beats every baseline, in every period, region and size bin
**Exam: 3,030 fires, 25,105 growth fire-days**

| Method | AP | F1 |
|---|---|---|
| FireCast | 0.323 | 0.239 |
| Best baseline (yesterday's growth from the active fire line) | 0.252 | 0.184 |

- **Replay subset:** FireCast AP 0.355 vs the dev-calibrated physics replay's 0.165; the replay over-predicts about 3.5×.
- **Win rate:** better AP than the best baseline on 71 % of fire-days.
- **Limits:**
  - Probabilities are over-confident above 0.35 (a 0.9 forecast burned 61 % of the time).
  - Persistence still wins total-area Dice (0.964 vs 0.922).
  - Simulated AP 0.59 vs real dev 0.32.
  - LANDFIRE adds only 0.002–0.004 AP.
- **Next:**
  1. Monotone calibration on dev (needs a rule change from widen-only).
  2. Finish twin p1 with less memory, then a p2 fire-age factor.
  3. Suppression and wind-night priors, with a twin test by wind tercile.
  4. Lower thresholds.
  5. A 12 h product.

### FireSize v1.0: best at ranking large fires, loses overall to a dev-fitted GBM
**Exam: 655,869 ignitions**

| Method | Escape AUC | Large-fire AUC |
|---|---|---|
| FireSize | 0.727 | **0.815** |
| GBM on dev real labels | **0.777** | 0.803 |
| Climatology | 0.688 | |
| ERC/BI/wind | 0.634 | |

- **By region:** beats the GBM in CA, NW and RM; far worse in SE (0.566 vs 0.704) and SP.
- **Calibration fails:** mean P(escape) is 0.055 against a real rate of 0.027.
- **Size:** biased +0.22 log10; MTBS-size fires under-sized by 3.3 log10.
- **Twin test:** the network matches the simulator at AUC about 0.91, but the simulator matches reality at only 0.67. The ceiling is the physics and suppression model, not the network.
- **Next:**
  1. Keep T = 1 for P(escape) and widen only the size tail.
  2. SE/SP physics: agricultural and debris-burn fuel states, a WUI suppression term.
  3. Containment beyond 96 h with multi-day weather.
  4. Model FPA-FOD's rounded reported sizes.
  5. Recover the escaped dev twins the guard skipped.
  6. Widen the suppression-parameter bounds.

### FireDetective v1.0: two partial wins, one failure
**Exam: 944 fires, 2017–23**

| Target | FireDetective | Best classical | Notes |
|---|---|---|---|
| Burn severity, forest fires (Spearman ρ) | **0.58** | FWI 0.54, ERC 0.53 | Across all fuels the Van Wagner crown-fire potential wins: 0.55 vs 0.48 |
| Wind on fire days, read without wind inputs (ρ) | **0.30** | Rothermel back-calculation 0.13 | Level wrong; recent fires 0.14 vs 0.21 |
| Live fuel moisture (ρ) | 0.17 | Climatology **0.34** | **Fails** |

- **Error bars:** severity bands too narrow (80 % band covers 69 %).
- **Rule note:** the exam *input* build ran before the ratchet closed. Nothing was scored or opened; this is disclosed in the card and in `ratchet.json`.
- **Next:**
  1. The persistence knob (upstream item 2).
  2. Simulate a dNBR-like severity label.
  3. Conformal severity bands; drop or rework live moisture.
  4. Validate against GOFER hourly perimeters.

### FirePilot v1.0: wins in simulation; real-world value untested
**Exam: 48 real exam landscapes and weather, simulated, mean share of the no-action 72 h growth stopped**

| Method | 300 m/h | 1000 m/h |
|---|---|---|
| FirePilot | **0.474** | **0.644** |
| Anchor and flank (best classical) | 0.316 | 0.480 |
| MPC forecast search | 0.400 | 0.591 |
| Perfect-foresight bound | 0.614 | 0.792 |

- **Win rates:** wins on 71 % of fires against anchor and flank, but on only 46 % against MPC.
- **Fidelity:** a free run against observed perimeters gives Dice 0.40 (persistence 0.19) and F1 0.34; the size of the growth is about right, its shape only partly.
- **Winner's curse:** the value net predicted 0.568 on its own picks; the truth was 0.454.
- **Next:**
  1. Re-plan each morning (receding horizon).
  2. Imitate the perfect-foresight search, or use a ranking loss.
  3. Fix U1, then physics p2 and re-score.
  4. Values-at-risk targets, multi-segment lines, aircraft.

## 6. Rule notes, reported honestly

- **FireDetective** built its exam inputs (including truth values) before the ratchet closed. They were never opened or scored before close.
- **FireCast:**
  - Read one exam fire's landscape array shape outside the exam script, to size a memory guard.
  - Its first physics-replay baseline attempt was killed after 2 fires for memory and re-run from scratch, deterministically. The killed attempt is kept.
  - The exam process peaked at about 1.35 GB and was allowed to finish rather than abort a once-only run.
- **FireSize:**
  - Read FPA-FOD records straight from parquet, filtered by its split, because firedata has no FOD accessor.
  - Several dev-calibrated suppression parameters sit at their bounds.
- **FirePilot** split its exam into two stages for compute. Both finished on all 48 fires, so the split changed nothing.
- **Interruption:** all four Phase 2 agents were interrupted once (an interface stall) and resumed from their STATUS notes. Each checked that no exam had started before resuming.
- **Dev data in the gates:** dev twins and dev replays fed physics calibration *and* the gates in several models, so the gates are partly in-sample. The exams were untouched.

## 7. Suggested order for the code agent

1. **Fix U1 and add a persistence module.** Then physics p2, rebuild the frozen sets and re-score every rung. This one physics change touches all four models.
2. **Merge the FEDS emulator into `firesim.observe`,** and add agriculture fuel and ignition relocation.
3. **FireCast calibration rule change** (monotone, on dev), logged as a method decision.
4. **Data:**
   - FEMS dead-fuel moisture (help_wanted)
   - ICS-209 suppression records
   - the per-fire MTBS severity mosaics (an e-mail queue)
   - larger replay windows
5. **Wire into the repo:**
   - `shared/fire/` with `firesim` and `firedata`
   - `mlratchet data fire`
   - the four models
   - instruments on the site
