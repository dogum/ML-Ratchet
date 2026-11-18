# ThermoPilot status (BuildingSim phase 2)

A learned, causal thermostat / heat-pump controller for homes and small commercial buildings. Every hour it picks a heating and a cooling setpoint (36-pair menu: off, band edge, pre-heat / pre-cool up to 3 K) to minimise bill + λ × kelvin-hours outside the occupant's comfort band. Trained only in BuildingSim by imitating a perfect-foresight DP teacher; graded on simulated twins of real exam units under real weather and real tariffs.

- Page: `lab/index.html`, the original page (week explorer over BuildingSim runs, live JS network lab, exam, WPuQ reality check, ratchet).
- Release: `thermopilot-v1.0.zip` (numpy `thermopilot_core.py`, weights, `demo.py` replaying 3 exam episodes in ~1 s, `pipeline.py`, MODEL_CARD, training code).
- Code: the sandbox's `bldg/thermopilot/` (tp/, scripts/, STATUS.md, ratchet.json); in this repository `training/`.

## Recipe as built
- **Physics:** BuildingSim 5R1C + equipment curves at 1 h through the step API. The teacher and the classical MPC use BuildingSim's own one-step zone maps. Numba DP == numpy DP exactly. JS port == Python on 2,232 / 2,232 demo hours. numpy == torch to 0.
- **Teacher:** a DP over the mass temperature (0.25 K grid, 24 h horizon, hourly replanning) with the true weather, the true house and the true state. On demand tariffs it does a two-pass monthly peak target using exact re-simulation. "Perfect foresight" = best of that DP and the rules.
- **Training data (simulation only):**
  - Buildings: generated homes and small commercial buildings, plus RECS *prior* twins (nearest ResStock match). The "true" house is a perturbed copy of the controller's nominal one.
  - Weather: 2016 BDG2 site weather and 17 TMYx cities.
  - Tariffs: 18 URDB tariffs, synthetic TOU/demand tariffs and ERCOT 2018 RTP.
  - Forecast error grows with lead time.
- **Student:** MLP 298→384→384→36 (277k params). Loss = soft-target CE + expected regret. BC, then 3 DAgger rounds (533 episode-months, 391k hours), plus a 6 h comfort shield.
- **Dev gate:** 30 episodes (11 RECS dev respondents + 4 BDG2 dev buildings × Jan/Jul 2017). Frozen sim val: 60 episodes.

## Rungs (skill: 0 = fixed thermostat, 1 = perfect foresight; MPC with forecast = 0.83 val / 0.83 dev)
| rung | change | val | dev | decision |
|---|---|---|---|---|
| R1 | behaviour cloning, 173 episodes | 0.08 | −3.02 | rejected (dev comfort) |
| R2 | + 4 h shield | 0.38 | −0.81 | rejected |
| R3 | + 6 h shield | 0.42 | −0.19 | rejected |
| R4 | + DAgger 1 | 0.52 | 0.04 | kept |
| R5 | + DAgger 2 | 0.50 | −0.07 | rejected |
| R6 | 384×384 | 0.55 | 0.20 | kept |
| R7 | + DAgger 3 | 0.56 | 0.36 | **kept = v1.0** |

## Exam (run once, R7)
Bill saving vs fixed thermostat; K h = kelvin-hours outside the band per episode; 90 % bootstrap over units.

| set | ThermoPilot | setback (own schedule) | MPC with forecast | perfect foresight | TP skill |
|---|---|---|---|---|---|
| RECS exam twins, 96 home-months, 12 tariffs | 2.2 % [1.6, 2.8], 1.4 K h | 2.2 %, 2.5 K h | 3.5 %, 0.9 K h | 3.6 % | 0.33 |
| BDG2 small commercial, 32 | 2.8 % [1.4, 4.2], 0.05 K h | 1.2 % | 3.1 % | 7.2 % | 0.45 |
| HUE BC homes, 21 | 4.3 %, 0.4 K h | 5.9 %, 3.1 K h | 5.7 % | 5.8 % | 0.54 |
| WPuQ heat-pump homes, 38 | 3.1 % | 5.6 % | 7.2 % | 7.5 % | 0.35 |

- ThermoPilot always beats fixed. Its comfort is better than setback.
- On homes it is roughly equal to setback or worse.
- It is close to MPC only on commercial demand tariffs. The classical MPC beats it everywhere.
- Standard setback (−4/+4 K) saves 8–11 % but leaves the band for ~700–870 K h per month.
- Setback raised the bill on 2 of 12 RECS tariffs here (phase 1's 1R1C house: 6 of 11).

**WPuQ reality check (report only):**
- Generic twins overstate the heating slope 2.7× (median). The twins also put heat-pump load at night, while the real homes run flat (hourly-profile correlation r = 0.21).
- On generic vs envelope-scaled twins, the controller order holds with Kendall tau 0.70. MPC > ThermoPilot > fixed holds.
- ThermoPilot vs setback flips in 18 of 37 homes, so that comparison cannot be trusted from the simulator.

## Limits
- The exam is simulated: there are no real thermostat logs (ecobee gated).
- Time-shifting savings are the least certain part.
- The model uses an hourly ideal thermostat: no short-cycling, no humidity control.
- Twins are survey characteristics plus a ResStock match. HUE and WPuQ use stand-in US tariffs.
- Heat pumps are under-represented in training.
- The student trails the MPC by 0.1–0.5 skill.

## Next ideas
1. More DAgger rounds, plus a heat-pump-heavy bank (val still rose with every round).
2. Make the student a residual on a cheap MPC, or distil the classical MPC's value function.
3. Calibrate residential twins on RECS dev annual energy, then build a new frozen val.
4. Get real thermostat logs (ecobee DYD application) for a real exam.

## Sandbox gotchas
- The shared 2-core machine ran at load 8–10 for the first 2 h, so episodes took 4–5× longer.
- Recovery was resumable banks and baseline files, skipping work already done.
- An oracle that only amortised demand charges was worse than a fixed thermostat on commercial demand tariffs. The fix was a two-pass peak-target oracle.
- Ratio-of-sums skill was dominated by large commercial bills; it is now relative per episode.

## 17 Nov 2026: imported into ML Ratchet

What changed in the import (code and weights as released; only paths and file places moved):

- `thermopilot_core.py`, `thermopilot_v1.npz`, `demo.py`, `pipeline.py` and `MODEL_CARD.md` sit at the model root as
  in the zip. `demo_episodes.npz` moved to `examples/bdg2/` because it is BDG2-derived (2017 site weather of Cockatoo,
  Hog and Gator; the third episode is a twin of the BDG2 building Gator_public_Geraldine); `demo.py` reads it from
  there. `examples/bdg2/LICENSE-DATA.md` and `lab/LICENSE-DATA.md` carry the CC BY-SA 4.0 notice.
- `training/_paths.py` is new: `SRC` (training/), `WORK` (`models/thermopilot/work/` or `$ML_RATCHET_WORK/thermopilot`),
  `BLDG` (`shared/building/` or `$ML_RATCHET_BUILDING`, holding `buildingsim` and `buildingdata`) and `PROC`
  (`BLDG/data/proc`). `tp/__init__.py` and `tp/env.py` import it in place of the sandbox's hard-coded package roots;
  the HUE and WPuQ parquet reads in `tp/env.py`, `tp/cases.py` and `scripts/wpuq_check.py` use `PROC`.
- In the sandbox the scripts' `ROOT` was the `thermopilot/` folder (tp/, scripts/, data/, results/, ratchet.json).
  In the repository `tp/` and `scripts/` are in `training/`, and the scripts' `ROOT` is now `WORK`: `data/cases`,
  `data/raw/nasa_yvr`, `results/<set>/`, `ratchet.json` and `exam_*.json` from a re-run land there. The shipped
  `training/ratchet.json`, `training/results/exam_r7.json` and `training/results/wpuq/` are the v1.0 logs, kept as
  shipped. `build_page.py` reads those logs from `training/` (as shipped it pointed one folder too high for the
  release layout) and writes the page to `WORK`. `scripts/build_release.sh` assembles `WORK/release/` and zips it to
  `WORK/thermopilot-v1.0.zip` instead of the sandbox's output folder.
- `lab/index.html` is the original page, copied byte for byte (5.4 MB). Its "Student" paragraph still describes the
  R1-era 298→256→256→36 network (152k parameters); the released network is 298→384→384→36 (276,516 parameters).
- `web/export_instrument.py` (new) writes the site's live instrument from the lab page's data: the three RECS exam
  twins in five 2017 climates, two weeks and five tariffs, with BuildingSim's hourly house maps and the full
  BuildingSim runs as the truth. The browser runs the network closed-loop on the linear house; on the 150 lab weeks
  ThermoPilot's bill there is a median 0.953× the full-simulator bill (IQR 0.896–1.003). Parity with Python: 20
  cases, worst relative difference 8.5e-15.

Re-run here: `python demo.py` (numpy only, about 1 s) replays the three exam episodes and chooses the recorded
setpoints in 744 / 744 hours each; its bills, comfort and J for every controller are the closed-loop BuildingSim
results stored in the file. Recorded as `tests/golden/thermopilot.txt`. Nothing in `training/` was re-run: it needs
BuildingSim, BuildingData and their processed data (`shared/building/`), which arrive with the platform zip
(buildingsim-buildingdata-v1.0), plus numba (DP) and torch (training).

Rule notes from the build (no exam result was used to choose anything): the RECS exam uses 4 of the 6 respondents
drawn per tariff slot, a cut made for compute and decided before the exam was run. During the build the agent once
ran `pkill -f` with its own script name to stop a job; the pattern could have matched other processes, but it matched
only its own.

Waits on the platform zip and numba: `pipeline.py`, every training script, rebuilding the lab page and the demo
file, and re-running the WPuQ check. `shared/building/` arrived in the worktree during the import; with it,
`pipeline.py` finds BuildingSim and BuildingData through `_paths.py` and stops at `import numba` (`tp/dp_fast.py`),
which is in neither of the repository's virtual environments. Not installed here.

## 2026-11-17 — Integrity pass: the plan, written before anything is re-run

Nothing in `training/` has been run in this repository yet. This entry is the plan for an integrity pass on v1.0. No
new rung is trained here. The exam sets were all taken once by v1.0, so every exam number below is a **seen** re-score;
nothing will be chosen from it.

**Setup.** numba (the DP teacher and the MPC need it) goes into `.venv-train` from PyPI. The episode code also reads
parquet, and `.venv-train` has no pyarrow, so episodes run with `.venv`'s Python (same Python 3.14 and numpy 2.5.4)
with `.venv-train`'s site-packages appended after its own, for numba only. Training uses `.venv-train` (torch).
At most 3 worker processes. Everything writes to `work/` through `training/_paths.py`.

1. **Re-run the exam with the released weights.** Freeze the case lists again from the same seeds
   (`evaluate.py <set> freeze`); run `baselines` (fixed, setback, standard setback, pre-cool, phase-1 pre-cool, MPC,
   perfect foresight) and `policy thermopilot_v1.npz r7`; then `exam_report.py r7`. Compare with the shipped
   `training/results/exam_r7.json` set by set (bill sums, savings, skill, K h) and say how close. HUE needs NASA POWER
   irradiance for Vancouver (`work/data/raw/nasa_yvr/`), which is not in the repository or the platform data; it is
   not downloaded in this pass, so HUE keeps its v1.0 numbers without a new interval. First time one episode of each
   controller, then decide what is affordable; anything over budget is cut by set and said.
2. **Dev re-score.** The same on the 30 dev episodes; ThermoPilot should reproduce `ratchet.json`'s R7 dev skill
   0.360 and the MPC's 0.831.
3. **Fairness of the baselines (code read, then dev).** Check that the MPC uses the nominal twin (the card ThermoPilot
   gets), the same noisy forecast and an observer, not the true house or true state. Check whether the setback and
   pre-cool rules were tuned on anything. If a rule has free settings that were never tuned, tune them on the 30 dev
   episodes only (pre-cool: lead 1-4 h × depth 1-3 K, chosen by dev skill) and add the tuned rule as a baseline on the
   exam sets (seen). Setback is the occupant's own schedule and has no setting to tune.
4. **Intervals.** For every exam row (bill saving, skill on cost + comfort, K h per episode, on each set): 95 %
   bootstrap interval on ThermoPilot minus the best baseline, 2,000 resamples, seed 20261117, resampling units (homes /
   buildings; the episodes of one unit move together). Best baseline: the strongest baseline on that row, not perfect
   foresight (a bound, not a controller). For bill rows, a baseline counts only if it passes the ratchet's comfort
   test (K h per episode ≤ fixed + 1); standard setback and the phase-1 pre-cool rule are listed but fail it. Verdict:
   win if the interval is on the better side of zero, loss if on the worse side, tie if it spans zero. State n. Also
   ThermoPilot minus setback (occupant's schedule) on bill with its interval. The old 90 % intervals are replaced.
5. **Dev twin test of what the network adds** (prepares the residual rung). On the 30 dev episodes, run the classical
   MPC with forecast closed-loop (the "cheap MPC" anchor). Every hour, on the same state, also ask the teacher (true
   house, perfect weather, true state, the demand peak target of `bank.py`) and ThermoPilot. Correlate ThermoPilot's
   setpoints minus the MPC's with the teacher's minus the MPC's (heating and cooling, setpoints clipped to 4 K outside
   the occupied band). Also report the teacher's regret ($ per episode, from its own action values) for the MPC's
   action and for ThermoPilot's, how often the teacher disagrees with the MPC, and where (hour of day, month, tariff
   type, heating kind).
6. **Readiness.** Here (Windows, 16 cores shared, ≤ 3 processes): one bank episode (DAgger mode with the v1.0
   student), one evaluate episode, a 1-epoch train into `work/`. Time them; extrapolate a full DAgger round and a
   retrain. List blockers.
7. **The residual-on-cheap-MPC rung**, written as a plan in a later entry (not run in this pass).
8. Update model.yaml, MODEL_CARD.md and the ladder note; results go in a new dated entry that says nothing was changed
   after a result was seen.

## 2026-11-17 — Integrity pass: results (all exam numbers seen)

Everything below followed the plan in the entry above; nothing was changed after a result was seen. Logs:
`training/results/integrity_2026-11-17/` (intervals, the re-run exam report, the dev pre-cool grid, the twin test,
the MPC wind check). New scripts: `training/scripts/intervals.py`, `precool_tuned.py`, `twin_test.py`,
`mpc_nowind.py`.

**Setup as run.** numba 0.68.1 (+ llvmlite 0.50.0) went into `.venv-train` from PyPI; numpy stayed 2.5.4. Episodes
ran with `.venv`'s Python with `.venv-train`'s site-packages appended (numba only). Two things were missing:
(a) the ERCOT real-time prices (`shared/grid/data/proc/prices/ercot_rtm_hourly.parquet`) are not in this worktree;
the first pass failed on the 10 `rtp_ercot` episodes, so every run used `ML_RATCHET_GRID` pointing at the main
checkout's GridData copy (read only) and those episodes were re-run; (b) the HUE homes need NASA POWER irradiance for
Vancouver (`work/data/raw/nasa_yvr/`), which is not here and was not downloaded, so HUE was not re-run.

**Cost.** Baselines (7 controllers) on 30 dev + 96 RECS + 32 BDG2 + 38 WPuQ episodes: about 14 min wall on 3
processes (RECS 15 s, BDG2 26 s, WPuQ 8 s per episode). ThermoPilot on all of them: 2.5 min. Val (60 episodes): 3.5
min on 3 processes.

**What reproduced and what did not.**
- WPuQ (38 homes): every controller reproduces `exam_r7.json` exactly (bill sums to the cent).
- BDG2 (32 building-months): fixed, both setbacks, both pre-cool rules and ThermoPilot reproduce exactly; perfect
  foresight to +0.02 % of the bill. The MPC does not: 2.69 % saving here against 3.05 % shipped (bill +0.37 %). On the
  demo episode Gator_public_Geraldine_7 the MPC differs by 0.005 %. The MPC is a DP with an argmin over 36 actions every
  hour; on demand tariffs one different hour can move the monthly peak. Not resolved further.
- RECS (96 home-months) and dev: do not reproduce. The draws, tariffs, comfort bands, forecasts and base loads are
  identical (checked on the two RECS demo episodes), but the twin houses differ: RECS_109160 UA 124.5 → 133.1 W/K;
  RECS_116893 UA 287 → 434 W/K and thermal mass light → medium (C 18.4 → 27.6 MJ/K). WPuQ twins, which use the same
  ResStock prior and `resstock_to_spec`, reproduce, so the difference is in the RECS respondent → twin step (the
  processed RECS table or `recs_twin` at the time of the sandbox run). The sandbox's frozen case files were not
  shipped, so it cannot be settled here. RECS now: fixed bill −0.5 %; saving ThermoPilot 2.18 → 2.56 %, MPC 3.50 →
  4.10 %, setback 2.24 → 3.01 %; skill ThermoPilot 0.33 → 0.24, MPC 0.84 → 0.84.
- Ratchet re-score with the released weights: val skill 0.556 (logged 0.562; MPC 0.859, logged 0.83), dev skill 0.482
  (logged 0.360; MPC 0.815, logged 0.831). Dev has 11 RECS twins, hence the larger shift.

**Fairness of the baselines.**
- The MPC with forecast uses the nominal twin (`Episode.model`, the house the building card is made from), the same
  noisy forecast and a dead-beat observer from the measured air temperature. It never sees the true house or the true
  mass temperature. Nothing in ThermoPilot's pipeline is calibrated on dev (dev is only a gate), so the nominal twin
  is the fair house for both.
- One leak: the MPC's one-step zone maps take the infiltration wind term from the weather file, i.e. the true future
  wind; ThermoPilot sees no wind. With the wind replaced by its monthly mean, the MPC's dev skill drops 0.815 → 0.787.
  Small next to its lead over ThermoPilot (0.48), so the exam MPC is kept as run.
- The MPC also reads the indoor humidity ratio (for the cooling coil); ThermoPilot does not. Not measured.
- Setback is the occupant's own schedule: nothing to tune. Standard setback (−4 / +4 K) and the phase-1 pre-cool rule
  have fixed settings and leave the band (835 and 69 K h per RECS episode).
- The in-band pre-cool rule (lead 3 h, depth 2 K) was never tuned. Tuned on dev (lead 1-4 h × depth 1-3 K, by dev
  skill): 4 h, 3 K, dev skill 0.18 (v1.0 rule −0.21). It is the grid's edge; longer leads were not tried. Added as
  "pre-cool rule tuned on dev" on every exam set (seen): RECS 3.07 % / skill 0.35, BDG2 −0.03 % / 0.16, WPuQ the same
  as setback.

**Exam rows with 95 % intervals** (ThermoPilot minus the best baseline, bootstrap over units, 2,000 resamples, seed
20261117). Bill rows count only baselines with K h ≤ fixed + 1 (the ratchet's comfort test), which leaves out setback
on RECS, standard setback and the phase-1 rule. The best baseline is the MPC on every bill and skill row.

| set (n units) | row | ThermoPilot | best baseline | difference [95 %] | verdict (v1.0) |
|---|---|---|---|---|---|
| RECS (48 homes, 96 ep) | bill saving | 2.56 % | MPC 4.10 % | −1.55 [−2.28, −0.93] pts | loss (loss) |
| RECS | skill | 0.24 | MPC 0.84 | −0.60 [−0.85, −0.39] | loss (loss) |
| RECS | K h / episode | 1.28 | fixed 0.36 | +0.91 [+0.20, +2.00] | loss (loss) |
| BDG2 (16 buildings, 32 ep) | bill saving | 2.77 % | MPC 2.69 % | +0.08 [−0.85, +1.09] pts | **tie** (loss) |
| BDG2 | skill | 0.451 | MPC 0.452 | −0.00 [−0.11, +0.10] | tie (new row) |
| BDG2 | K h / episode | 0.05 | fixed 0.00 | +0.05 [0.00, +0.09] | tie (new row) |
| WPuQ (38 homes) | bill saving | 3.13 % | MPC 7.16 % | −4.03 [−4.66, −3.47] pts | loss (loss) |
| WPuQ | skill | 0.35 | MPC 0.95 | −0.60 [−0.68, −0.53] | loss (new row) |
| WPuQ | K h / episode | 0.00 | fixed 0.00 | 0.00 [0.00, 0.00] | tie (new row) |
| HUE (17 homes, 21 ep) | bill, skill, K h | 4.29 %, 0.54, 0.36 | MPC 5.75 %, 0.95; fixed 0.14 | not re-run, no interval | loss (point) |

ThermoPilot minus setback (occupant's schedule) on bill: RECS −0.46 [−1.73, +0.56] pts (tie), BDG2 +1.58 [+0.30,
+2.81] (better), WPuQ −2.45 [−3.16, −1.54] (worse). The BDG2 bill row moves from loss to tie partly because the MPC's
BDG2 saving did not reproduce (3.05 → 2.69 %); with 16 buildings the interval would span zero either way
(ThermoPilot's own 95 % interval is [1.19, 4.53] %).

**Dev twin test of what the network adds** (`twin_test.py`, 30 dev episodes, 22,320 hours). The cheap MPC drives the
house; setpoints are compared at the band edge, so only pre-heating / pre-cooling inside the band counts.
- The teacher (perfect foresight) picks different setpoints from the cheap MPC in only **4.1 %** of hours, and on
  average in no direction (teacher − MPC: −0.007 K heating, −0.011 K cooling).
- ThermoPilot moves away from the MPC almost everywhere: mean +0.88 K heating and −0.52 K cooling. It pre-conditions
  far more than the teacher wants.
- Correlation of ThermoPilot − MPC with teacher − MPC: **0.10** (heating) and **0.14** (cooling) over all hours;
  **0.37 and 0.53** in the hours where the teacher differs; the same sign in 54 % of those.
- In the teacher's own action values, the MPC's choices cost 2.5 % of the MPC's J per episode, ThermoPilot's 6.7 %.
  In dollars it is nearly all commercial demand tariffs (MPC $50 of $52, ThermoPilot $124 of $127 per episode).
- Where the teacher departs from the MPC: demand tariffs (6.6 % of hours), July (5.7 %; January 2.5 %), away hours
  (6.3 %), heat pumps (5.8 %), mornings 06-11 (5.8 %). ThermoPilot follows it best on summer cooling (r 0.66 where the
  teacher differs; 0.76 asleep) and worst on commercial cooling (0.28) and electric-furnace heating (−0.11).
- Reading: the network learned roughly where the teacher wants something different, but adds large corrections in
  hours where nothing is needed. The headroom over the cheap MPC is small and sparse, so a residual has to default to
  "no change".

**Readiness (Windows, this worktree, 1 process unless stated).**
- One bank episode (DAgger, the v1.0 student, beta 0.1): 3.6 s; 11 more: 2.3-12.0 s, 54 s in all (two seeds gave no
  case, as in training). One evaluate episode (7 controllers): 8 s on val, 26 s on a BDG2 demand building.
- A 1-epoch train on that bank (8,065 hours) into `work/ckpt/`: 3 s; numpy == torch to 1.2e-6. One epoch at v1.0
  scale (391k × 298, random data, 1 thread): 10 s, so 24 epochs take about 4 min.
- A full DAgger round (about 100 new episodes: 8 min on 1 process, 3 on 3; retrain 4 min; val + dev policy 1.5 min):
  about 10 min. Rebuilding all 533 episode-months: about 45 min on 1 process, 15 on 3.
- Blockers: (1) numba is in `.venv-train`, pyarrow only in `.venv` (a path wrapper was used; add pyarrow to
  `.venv-train` or numba to `.venv`); (2) the ERCOT prices are not in the worktree's `shared/grid/data` (`mlratchet
  data grid`, or `ML_RATCHET_GRID`); (3) HUE needs the NASA POWER YVR download; (4) the v1.0 banks and the R1 / R4 / R6
  students were not shipped, so v1.0's training set cannot be rebuilt exactly; a new rung builds new banks; (5) RECS
  twins differ from the sandbox's, so a new rung must be compared with R7 re-scored here (val 0.556, dev 0.482), not
  with the logged numbers; (6) another session edited BuildingSim (opt-in phase-2 wrappers) in this worktree while
  these runs went; with no flags it keeps v1.0 physics, and BDG2 / WPuQ reproduced, so no effect was seen.

## 2026-11-17 — Plan for the next rung: a residual on the cheap MPC (written before it runs; not run)

Why: the twin test says the cheap MPC already matches the teacher in 96 % of hours, and the v1.0 network adds
corrections where none are needed. The MPC is the best baseline on every exam row. So the next student should start
from the MPC and learn only when to depart from it.

1. **Anchor.** The released controller must stay numpy-only and work from the building card, so it cannot call
   BuildingSim's 5R1C maps. Step 0: a card MPC in numpy (the DP of `tp/mpc.py`'s numpy loop over a 1R1C model from the
   card's UA and C, internal gains from the expected other load, a solar gain fitted online from the measured
   temperature), scored on val and dev next to the classical MPC. Reference rows, not candidates: card MPC alone,
   classical MPC alone, R7. If the card MPC loses more than 0.1 dev skill to the classical MPC, the anchor is the
   first rung, not the residual.
2. **Residual network.** Inputs: the 298 v1.0 features plus the anchor's 36 action values (divided by the 24 h price
   scale S, as the targets are). Output: 36 logits added to −Q_anchor / τ. Last layer initialised to zero, so the
   untrained rung is exactly the anchor. Same loss (soft targets from the teacher's values + expected regret), same
   6 h shield.
3. **Data.** New banks (the v1.0 banks were not kept): round 0 = states the anchor visits, labelled by the teacher, with
   the anchor's values saved every hour; then DAgger rounds with the residual student executed at beta 0.3, 0.2, 0.1;
   about 530 episode-months from the v1.0 seed ranges. From the timings above: about 15 min per 500 episodes on 3
   processes, times about 2 for the anchor's plan every hour.
4. **Rule.** The same ladder rule (val skill > incumbent + 0.01, dev gate) with R7 re-scored here (val 0.556, dev
   0.482) as the incumbent, and one more condition: the rung must also beat the anchor alone on val by 0.01, or the
   anchor is the better controller and that is what gets reported. Three seeds if the margin is under twice the seed
   spread. Before the decision, repeat the twin test on the rung: its corrections should correlate with the teacher's
   above v1.0's 0.10 / 0.14 and stay near zero where the teacher agrees with the anchor.
5. **Exam.** All four sets are seen. Any release is re-scored on them as seen, with the same interval rule, next to the
   card MPC and the classical MPC. HUE only if the NASA POWER file is fetched.

## 2026-11-17 — The RECS twin drift: cause found (not the platform edit)

Checked on request, because another session edited BuildingSim while the re-score ran (now commit 38870b1).
- With BuildingSim and BuildingData exported from dfafdcf (before the edit) and from the current tree, on the same
  data, the 60 RECS dev and exam twins are identical (same ResStock dwelling and the same spec hash for all 60). So
  the edit did not cause it, and the RECS and dev rows stand as re-scored.
- The cause is in `tp/twins.py` `recs_twin`: `top = np.argsort(score)[:5]` picks among the 5 best-scoring ResStock
  dwellings, and the score ties a lot. For RECS_116893, 68 dwellings tie at 0.075; for RECS_109160, 7 tie at 0.473.
  numpy's default sort is not stable, and the order of ties depends on the numpy version and the CPU's SIMD sort
  (here numpy 2.5.4, AVX2). So a different dwelling comes out on another machine.
- Evidence: the logged card of RECS_109160 (UA 124.55 W/K) is ResStock dwelling 461147 exactly. It ties with
  today's pick 408171 (UA 133.08) and ranks 6th here. For RECS_116893, tied dwelling 181802 gives UA 286.8 W/K against
  the logged 287.0, with light mass as logged. Today's pick, 546506, gives 434 W/K with medium mass.
- The sandbox's tie order cannot be recovered here, so the v1.0 RECS twins cannot be rebuilt exactly. A fix (stable
  sort with the dwelling id as tie-break) would make twins reproducible from now on, but it would change them again.
  It is left for the next rung, not done in this pass. The same line was ported into BuildingSim 1.0.1
  (`stock.recs_to_spec`, `shared/building/buildingsim/buildingsim/stock.py` line 617) and has the same problem.

## 2026-11-17 — Residual rung (R8): the plan as fixed, written before anything is built or run

This fixes the "residual rung plan" above. Physics stays BuildingSim 1.0 (no physics version was adopted this
round; `shared/building/NOTEBOOK.md`). No fresh exam exists for ThermoPilot (no open thermostat logs), so every exam
number below will be a **seen** re-score and nothing is chosen from it. Runs use `.venv-train` (torch, numba, pyarrow)
with `ML_RATCHET_GRID` pointing at the main checkout's GridData copy (read only), at most 3 worker processes,
everything written to `work/` through `training/_paths.py`. HUE is out of scope: the NASA POWER file is not fetched,
and HUE keeps its v1.0 point values without an interval.

1. **Reproducible twins first.** `tp/twins.py` `recs_twin` ranks tied ResStock matches by `bldg_id`
   (`np.lexsort((bldg_id, score))`, as `stock.recs_to_spec(stable=True)` does since d7d9c91). This changes RECS
   twins against v1.0 again (training RECS-prior twins, val, dev and the RECS exam). The old val and dev case files
   and results move to `work/archive_unstable_twins/`; val (60) and dev (30) are frozen again from the same seeds;
   the 7 baselines are re-run on both. **R7 (the released weights) is re-scored there first; that is the incumbent.**
2. **The anchor: a card MPC in numpy** (`thermopilot_core.CardMPC`). The same receding-horizon DP as `tp/mpc.py`
   (0.25 K mass grid, 24 h, hourly replanning, the same 36 actions, the same demand-charge amortisation and running
   peaks, λ × K h), but its house is a 12-number linear map on the building card, not BuildingSim:
   T_air = A + At·T_mass + Aq·q, T_mass' = C + Ct·T_mass + Cq·q, with At, Aq, Ct, Cq constant and A, C linear in the
   forecast outdoor temperature, the forecast irradiance and the expected non-HVAC load. The 12 numbers are fitted by
   least squares to the nominal twin's hourly 5R1C maps over the episode (the same nominal house the card already
   comes from; a thermostat could identify them from its own logs). No wind, no indoor humidity (the classical MPC
   reads both). Plant from the card's tables (+ heat-pump / AC speed, outdoor-air conductance, as card fields).
   Observer: dead-beat from the measured air temperature through the card map. Check first, on dev: card MPC alone
   vs the classical MPC. If the card MPC loses more than 0.1 dev skill, the anchor is fixed before any residual is
   trained (one revision, logged here before it runs). The shield setting of the rung (none or v1.0's 6 h shield)
   is chosen once on dev with the anchor alone, before training.
3. **The residual (R8).** Inputs: the 298 v1.0 features + the anchor's 36 action values (minus their minimum,
   divided by the 24 h price scale S, as the teacher's). Output: 36 logits added to −Q_anchor / (S·τ), τ = 0.1, the
   same τ as the soft targets. Last layer initialised to zero, so the untrained rung is exactly the anchor. MLP
   334→384→384→36, same loss (soft-target CE from the teacher's values + expected regret), 24 epochs.
4. **Banks (new; seeds 20000+, not used before).** Round 0: 230 episode-months where the anchor acts and the teacher
   labels every hour, with the anchor's values saved. Rounds 1-3: about 100 episode-months each, the residual
   student acting with the teacher's action executed with probability β = 0.3, 0.2, 0.1 (each round's student is
   trained on all banks so far, seed 0). About 530 episode-months, as v1.0.
5. **Rule.** The ladder's rule: val skill > incumbent + 0.01 and the dev gate (dev skill > 0, ≥ incumbent dev − 0.03,
   dev K h ≤ fixed + 1), plus: **val skill > anchor alone + 0.01**. Reference rows, not candidates: anchor alone,
   classical MPC, R7. The final rung is trained with 3 seeds on the same banks; if the margin over the binding
   comparator is under twice the seed spread, the decision is on the 3-seed mean; if kept, the seed with the median
   val skill is released. Before the decision, the twin test is repeated on the rung (its correction vs the
   teacher's, both relative to the anchor).
6. **If released as v1.1:** weights (MIT); `thermopilot_core.py` runs the anchor + residual; demo + golden;
   model.yaml / ladder / card / notebook. Seen exam re-scored for v1.1 and v1.0 with the same baselines (+ the card
   MPC) and 95 % intervals as `intervals.py` (2,000 resamples over units, seed 20261117): RECS exam frozen again with
   the stable twins (baselines re-run), BDG2 and WPuQ as frozen. Instrument: the card MPC runs in the browser on the
   same linear house, parity checked. **If not released:** v1.0 stays, and the ladder logs the rungs.

## 2026-11-17 — Residual rung (R8): results. Rejected; v1.0 stays

Everything followed the plan above; nothing was changed after a result was seen. Logs: `training/ratchet_r8.json`
(every row), `training/results/residual_2026-11-17/` (`scores.json`: val / dev of every row, bank sizes, training
runs; `twin_report_r8.json`). Code: `tp/twins.py` (stable ties), `tp/card_mpc.py` (card MPC + residual pilot, numpy;
it imports v1.0's features and menu from `thermopilot_core.py`, which is unchanged), `tp/features.py`
(`card_map`, `full_card`), `tp/policy.py` (adapters), `scripts/bank_r8.py`, and residual options in `train.py`,
`evaluate.py` (`anchor`, `merge`, index ranges), `ratchet.py` (anchor rule, reference rows), `twin_test.py`
(`TWIN_ANCHOR=card`) and `intervals.py` (card MPC as a baseline; not used, no release). Setup as run: `.venv-train`'s
Python with `.venv`'s site-packages appended after its own (BuildingSim needs pvlib, which `.venv-train` lacks), and
`ML_RATCHET_GRID` at the main checkout's GridData. About 75 min of runs on ≤ 3 processes.

**Stable twins.** Val and dev were frozen again from the same seeds. Same case ids; 19 of val's 28 RECS-twin
episodes and 22 of dev's 24 RECS episodes (11 of 12 homes) got a different house. The RECS exam list was also frozen
again (39 of 48 homes change house) and its v1.0-era files moved to `work/archive_unstable_twins/`; nothing was run on
it. The exam rows in model.yaml stay as re-scored this morning (old tie order).

**Rows on the rebuilt sets** (skill: 0 = fixed thermostat, 1 = perfect foresight; K h per dev episode):

| row | val skill | dev skill | dev K h | role |
|---|---|---|---|---|
| R7 (v1.0 weights) | 0.561 | 0.471 | 0.09 | incumbent (old sets: 0.556 / 0.482) |
| classical MPC with forecast | 0.862 | 0.765 | 0.22 | reference |
| card MPC alone, no shield | 0.573 | 0.572 | 0.91 | reference |
| **anchor**: card MPC + 6 h shield | **0.700** | **0.720** | 0.00 | reference (the rule's second bar) |
| R8 seed 0 | 0.670 | 0.718 | 0.06 | rejected |
| R8 seed 1 | 0.644 | 0.673 | 0.12 | rejected |
| R8 seed 2 | 0.686 | 0.711 | 0.00 | rejected |
| R8 mean (sd) | 0.667 (0.021) | 0.701 (0.024) | | |

**The anchor.** The card house map fits the nominal twin's hourly maps to 0.14-0.19 K (air) and 0.08-0.18 K (mass)
RMS on three dev episodes checked; a numpy plan takes 3-4 s per episode-month. Without the shield the card MPC
loses 0.19 dev skill to the classical MPC, mostly comfort (0.91 K h per dev episode: it does not anticipate the band
tightening after setback). With v1.0's 6 h shield it loses 0.045 (0.720 vs 0.765), inside the plan's 0.1, so the
anchor was not revised. The shield (on) was chosen on dev by the anchor alone before any training, as planned.

**Banks.** 507 episode-months, 371,663 hours (round 0: 212 episodes with the card MPC acting; DAgger rounds 1-3: 100,
96, 99; 29 seeds gave no case or an unsupported boiler). The card MPC picks the teacher's action in 95.0-95.9 % of
hours, and an action with zero teacher regret in 96.0-96.7 %.

**Decision.** R8 beats R7 on every seed (val +0.08 to +0.13, dev +0.20 to +0.25, dev gate passed), but it loses to
the anchor alone on every seed (3-seed mean 0.667 vs 0.700; the rule needs > 0.710). Rejected. v1.0 stays released.
The anchor alone would pass this ladder's rule against R7 (val +0.14, dev +0.25), but it was fixed as a reference
row, not a candidate, so it is not promoted in this round.

**What failed: the residual does not default to "no change".** Dev twin test, seed 0, the anchor driving the house
(`twin_report_r8.json`, 22,320 hours):
- The teacher departs from the card MPC in 5.6 % of hours (mean −0.008 K heating, −0.013 K cooling). R8 departs in
  96.7 % (mean +0.81 K heating, −0.54 K cooling), the same over-correction v1.0 showed against the classical MPC.
- Correlation of R8 − anchor with teacher − anchor: 0.03 (heating) and 0.13 (cooling); 0.13 and 0.46 in the hours
  where the teacher departs; same sign in 53 % of those.
- In the teacher's own action values R8's choices still cost less than the anchor's: $57 vs $69 per episode (nearly
  all on commercial demand tariffs). The bank hold-out says the same (hard regret 0.0028-0.011 vs the anchor's
  0.010-0.018). But closed loop, on val, R8 loses: against the anchor it is worse in 17-21 episodes and better in
  7-12; most of the loss is in commercial demand-tariff episodes (−0.011 to −0.040 skill there).
- Why the zero-initialised residual moves almost everywhere: the anchor's prior logits −Q/(S·τ) are flat. In the
  round-0 bank a median 12 of the 36 actions tie exactly at the anchor's minimum and 18 are within 0.5 logit
  (heating setpoints in July, cooling in January), so a tiny network output decides among them. The teacher is indifferent among ties in its own model, but the card house is
  not the true house, and the moves are not free there. The bank hold-out regret was lowest after epoch 0-1 and rose
  with training (seed 0: 0.0028 → 0.0098), so the checkpoints kept are early ones.

**Not done** (no release): the seen exam was not re-scored for any new controller, the demo, golden, weights and
instrument are unchanged, and HUE stays at v1.0's point values (NASA POWER not fetched).

**Next ideas (not run).** (1) Promote the anchor (card MPC + shield) as its own rung: it needs a plan entry and the
same rule, and it is numpy-only already (`tp/card_mpc.py`). (2) A residual that cannot move on ties: add the
correction only where the anchor's top two values differ by more than a set fraction of S, or penalise departures
from the anchor in the loss in proportion to the anchor's own value gap. (3) Train the residual on the teacher's
regret measured in the card house's closed loop, not the one-step values.
