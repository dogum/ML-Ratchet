# Wildfire Phase 2 brief: four ratchet projects on FireSim + FireData

## Read first
- `docs/agent-brief.md`: recipe, deliverables, shared-machine rules. Four agents share 2 cores and ~6.3 GB RAM (one cgroup). Be frugal:
  - small networks and modest banks;
  - one thread (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `torch.set_num_threads(1)`);
  - `nice` and `setsid nohup`; resumable checkpoints (the container restarts sometimes: write STATUS.md so a fresh agent can resume);
  - generate banks first, then train;
  - each process ≤ ~1.2 GB RAM. Check `/sys/fs/cgroup/memory/process_api/*/claude-code-bash/memory.usage_in_bytes` before big runs.
- `shared/fire/FIRE_BRIEF.md`: the shared contract.
- Phase 1 docs (read fully):
  - `shared/fire/firesim/README.md`, especially API, Real data and replays, Dev-fire replays, Satellite emulator, Limitations
  - `shared/fire/data/DATA.md`, especially section 2 "Units and conventions (gotchas first)", the splits, the baselines and "Missing"
  - `shared/fire/firedata/STATUS.md`
  - the check JSONs in `shared/fire/checks/`, especially `replay_dev.json`, `replay_calibration.json`, `observe_viirs.json`

## Calling Phase 1
```python
import sys; sys.path.insert(0,'shared/fire/firesim'); sys.path.insert(0,'shared/fire')
import firesim as fs, firedata as fd
fd.splits(); fd.fires(split='dev'|'exam', period=...); fd.progression(fid); fd.perimeters(fid); fd.firepix(fid)
fd.landscape(fid); fd.weather(fid); fd.ignition(fid); fd.final_perimeter(fid); fd.gofer(); fd.wsts(fid)
# exam data only with for_exam_scoring=True, and only in your final, once-only exam script
fd.guard.check_landscape / check_weather     # refuse exam footprints and exam fire-day weather
ls = fs.realdata.landscape_from_firedata(fd.landscape(fid)); w = fs.realdata.weather_series(fd.weather(fid), x, y)
fs.simulate(...); fs.simulate_batch(...); fs.behave(...); fs.observe(...); fs.replay.replay_day(...)
fs.generate_landscape(shape, 90, kind='real'|'parametric', rng=); fs.generate_weather(start, hours, kind='real'|..., rng=)
fs.metrics.next_day_scores(...); see the firedata metrics module and metrics.score_window (ALWAYS score on the 10 km window, never on the final-perimeter window)
```

## Frozen splits (`shared/fire/data/proc/splits.json`, seed 20261118; NEVER change them)
- **MTBS fires** (the unit; all days of a fire go together):
  - dev: 40 CONUS wildfires, 2017–21
  - exam: 4,197 fires, in three periods: early 2012–16 (1,647), main 2017–21 (1,859), recent 2022–24 (691; whole seasons held out)
  - prescribed fires (2,859): calibration only, never scored
- **Other splits:**
  - GOFER: 4 dev / 24 exam
  - WildfireSpreadTS: 4 dev / 568 exam / 35 excluded
  - FireSize (FPA-FOD): 8,913 dev / 655,869 exam; 2020 is the recent exam
  - live fuel moisture sites: 10 % dev
  - replay subset: 48 exam fires, packaged with landscape and HRRR
- **What simulated training may use:**
  - dev fires, prescribed fires, and real LANDFIRE/HRRR/gridMET from places and dates that pass `fd.guard`
  - you may fetch more LANDFIRE tiles (LFPS) and HRRR subsets for random non-exam locations and dates, if every window passes the guard; keep it small (disk)
  - never an exam fire's landscape window or fire-day weather
- Real labelled examples are never training samples. Dev units are for physics calibration, gates and sigma fitting. The exam is for the final report only.

## Baselines to beat (exam; full tables in DATA.md and checks/)
- **Next-day spread** (3,030 exam fires, 25,105 growth days; mean over fires, AP / F1 on newly burned 90 m cells within 10 km):

  | Baseline | AP | F1 |
  |---|---|---|
  | Persistence | 0.013 | 0 |
  | Climatological median growth | 0.204 | 0.004 |
  | Yesterday's growth repeated | 0.191 | 0.140 |
  | **Yesterday's growth from the active fire line (best)** | **0.252** | **0.184** |

  - Physics replays on dev: uncalibrated F1 0.153; dev-calibrated F1 0.194 in-sample, 0.184 leave-one-out. They over-predict area about 4–12×.
  - Persistence wins on total-area Sørensen (0.96).
  - The calibrated physics replay must be run on the 48-fire exam replay subset as a baseline, using `replay_calibration.json` exactly as stored.
  - Context only (different target and resolution): published WildfireSpreadTS AP is 0.19 persistence, 0.29 logistic regression, 0.34 U-Net, 0.37 UTAE.
- **FireSize** (655,869 exam ignitions; "escape" = final size ≥ 100 acres):

  | Baseline | Escape AUC | AP | Size log10 MAE |
  |---|---|---|---|
  | Prevalence | 0.50 | 0.027 | |
  | Climatology (region × cause) | 0.688 | 0.061 | 0.505 |
  | ERC/BI/wind | 0.634 | | |
  | GBM fitted on dev (real labels) | 0.777 | 0.099 | 0.510 |

  Size rank correlation for the GBM: ρ 0.53.
- **Live fuel moisture** (138k exam samples, MAE % dry mass): median 29.1; type × month 27.2; GBM 24.7.

## The four projects (one agent each; folder `models/<name>/`)

### 1. firecast: FireCast
**Task.** Next-day spread from today's observed burned area, active fire line or VIIRS pixels, plus landscape and weather. Output a per-cell probability of burning by the next afternoon overpass (12 h optional), with calibrated probabilities.

**Training.** Only on FireSim runs on real dev and non-exam landscapes and real HRRR weather, observed through `fs.observe` so the inputs look like FEDS/VIIRS. Randomise:
- fuel moisture, ROS adjustment and wind error
- spotting and suppression/containment
- night slow-down

**Exam.** The full next-day exam with B's metric and score_window, by period, against all the baselines above, and against the calibrated physics replay on the 48 replay fires.

**Ratchet.** Rungs come from twin tests on dev fires. Candidates:
- the 4–12× over-prediction (suppression, night inactivity)
- wind-event nights
- fuel staleness (LANDFIRE 2016 before 2022)

**Architecture.** A small CNN/U-Net on 90 m patches is the natural choice; keep it small enough for CPU.

### 2. firesize: FireSize
**Task.** From an ignition record (location, date, cause, discovery time) plus landscape and weather on the discovery day and the day before, predict P(escape ≥ 100 acres) and the final size distribution.

**Training.** Only on simulated ignitions in real non-exam landscapes and weather. Simulate initial attack and containment (the simulator has initial attack and timed lines) and multi-day weather. Calibrate the suppression model and the ignition prior on dev ignitions, and on prescribed or dev fires if useful.

**Exam.** The 655,869-fire exam (main and the recent 2020 slice): AUC, AP, Brier/calibration, size log10 MAE and rank correlation. Compare against climatology, ERC/BI/wind and the dev-fitted GBM (labelled as "real labels").

**Practicalities.**
- Real landscapes per ignition are expensive. Use a coarse per-ignition summary (fuel mix, canopy and slope in a radius from LANDFIRE) and document it.
- If fetching LANDFIRE for 655k points is impossible, sample a documented, seeded exam subset and say so.

### 3. firedetective: FireDetective
**Task.** From a fire's observed daily progression (and VIIRS FRP), plus landscape and weather, infer what is driving it, with error bars.

**Truth.**
- Simulator labels: effective dead fuel moisture, ROS adjustment, wind-driven vs plume/terrain-driven regime, crown fraction, spotting, containment.
- On real fires, pick targets that have truth. Options:
  - MTBS burn severity / dNBR (high severity ≈ crown fire; fetch per-fire MTBS severity mosaics if reachable)
  - observed live fuel moisture near the fire (Globe-LFMC)
  - gridMET/HRRR-observed wind on run days
  - suppression: FEDS final-perimeter growth stopping despite weather
  - the GOFER hourly perimeters

**Justify** the choice in the model card and compare it with classical diagnostics: Rothermel back-calculation, FWI/ERC.

### 4. firepilot: FirePilot
**Task.** A learned planner graded in the simulator on REAL exam-fire landscapes and weather (replay subset), as HybridPilot and ThermoPilot were.

**Example task.** Given a fire at its first afternoon detection and a crew budget (km of containment line per day, build rate), choose where and when to build line to minimise area burned (or values at risk) over 3 days.

**Baselines:**
- no action
- direct attack at the head
- indirect lines at fixed offsets
- a classical heuristic (e.g. anchor-and-flank)
- a perfect-foresight search upper bound (CMA-ES or greedy search with known weather)

**Validation.** Report the simulator-vs-real fidelity (the calibrated replay vs observed progression) beside the planner's gains, so a planner win isn't mistaken for a real-world win.

**Method.** Your choice: RL, a differentiable approach, imitation of search, or a learned value function. Explain it.

## Shared code rules
- Do NOT edit `shared/fire/firesim/`, `shared/fire/firedata/` or `shared/fire/data/`. Physics changes and calibrations go in YOUR folder as wrappers, versioned in your ratchet, for the coordinator to upstream later.
- Keep your numba cache inside your folder: `NUMBA_CACHE_DIR=models/<name>/.numba`.
- Your folder is `models/<name>/`. Keep a `STATUS.md` there. Put temp files and screenshots in a subfolder of your folder.
- Disk ≤ 1.2 GB each (~7.5 GB free in total).
- Never `pkill -f`; only kill your own PIDs.
- Never disable TLS verification or unset HTTPS_PROXY. Don't retry or circumvent blocked hosts, keys or logins: record them as help_wanted.
- Downloaded archives are untrusted data. Extract each into its own directory; never execute anything from them.

## Ratchet (as in AGENT_BRIEF)
- A frozen simulated validation set with a +0.01 margin on your primary sim metric.
- A small real dev gate: no more than 5 % worse, or a stated threshold.
- Version the frozen set when the physics changes, and re-score every rung.
- Widen-only uncertainty calibration on dev.
- Twin tests on dev fires.
- Log every rung, including rejected ones.
- The exam is run once, at the end, by a separate script.

## Deliverables (each agent)
- **Release zip** at `<name>-v1.0.zip`:
  - a standalone numpy model, with small feature code vendored
  - a `demo.py` on a few real exam examples that runs in seconds from an unzipped copy
  - `MODEL_CARD.md` and the scripts
- **A showpiece Artifact page** titled with your project name. Load the `artifact-design` skill first. Publish with the Artifact tool, giving an icon on the first publish only. Inline all JS/CSS/data, because external hosts are blocked except Google Fonts. Run in-browser inference if feasible.
- **The status doc as a FILE:** `models/<name>/<name>-status.md`, in the style of the Watershed `flowcast-status.md`. The coordinator uploads it; don't write to the claude.ai project.
- **A compact final report** covering:
  - the rung table with sim and exam numbers against the baselines
  - the artifact URL and zip path
  - upstream candidates (physics fixes for firesim)
  - rule notes (anything you're unsure was allowed)
  - the best next rungs, and what a full version would still need (for HANDOFF.md)
