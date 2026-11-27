> **Lab notebook.** The build notes as recorded during the original build in a cloud project (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that build machine: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# SunDoctor — status (30 Oct 2026, v2.0)

- What it is: a neural network trained only on solar-physics simulations. It diagnoses real PV systems day by day, with error bars:
  - dead strings, with failure and repair days
  - shading, clipping, outages, soiling, snow, and stuck or stowed trackers
  - the kWh lost to each cause
- Second project in the physics-ML collection, after ChirpNet. Collection name is still open; the owner leans toward "ML Ratchet" (also liked First Light, Ratchet, Ab Initio).
- v2 adds two things the owner picked:
  1. Power-only "home" mode, for systems with no irradiance sensor.
  2. A second real farm: PVDAQ 9068 in Colorado, with single-axis trackers and snow.
- Page (same link, updated to v2): https://claude.ai/artifact/VqpkFf7pWwWhodohiaqE97. Sections:
  - California cases, with a sensor / power-only toggle
  - a "No sensor" section
  - a Colorado console with a tracker-angle answer-key carpet
  - a lab with a sensor toggle and a snowstorm slider
  - both ratchet ladders, the exams, and limits
- Release: `sundoctor-v2.zip` (2.5 MB) contains:
  - `sundoctor.py` (numpy + pandas): fixed or tracker mount, with sensor or power-only
  - weights (591,455)
  - two real example CSVs and `demo.py`
  - `MODEL_CARD.md`, `README.md`
  - `training/`: all scripts and both ratchet logs
- The cloud workspace (`/home/claude/sun`) is ephemeral; the zip and the page are the durable copies.
- v1 notes (rungs 1–5, s3 released) are in the v1 zip. v1 results on California:
  - 93.9% of inverter-days counted right
  - 54 of 56 string failures caught

## v2 build
- **Simulator additions** (row shading matches pvlib `shaded_fraction1d` to 1e-15):
  - Single-axis trackers, with and without backtracking. True-tracking rows shade each other (Anderson & Jensen 2024).
  - Snow that falls on cold overcast days and slides off.
  - Stuck and stowed tracker rows.
  - CdTe thin film.
  - Sensor misalignment, morning dew dropouts, and 40% of fortnights with no sensor.
  - Rung 8 only: weather fronts, i.e. day-to-day temperature swings with cold fronts on overcast days.
- **Network:** 12 input channels (adds a sensor flag and a clear-sky expectation) and 2 new outputs (snow, stuck). v1 weights were transplanted in.
- **Ratchet:** a new frozen v2 exam of 2,000 fortnights (seed 778), half with no sensor. The objective is the mean of the with-sensor and no-sensor scores.
  - Start: v1 seen through v2 inputs, 0.341.
  - Rung 6 (v2): 1.579. Kept.
  - Rung 7 (polish): 1.615. Kept.
  - Rung 8 (weather fronts): 1.628. Kept and **released**.

## Real exams (rung 8)
- **California, PVDAQ 2107** (24 sibling inverters as the answer key):

  | Metric | With sensor | Power only | v1, power only |
  |---|---|---|---|
  | Dead strings counted right, all days | 95.9% | 94.2% | — |
  | ...when strings really are dead | 63% | 52% | 3% |
  | String failures caught within a day | 55/56 | 47/56 | 1/56 |
  | Inverters offline caught | 44/49 | 39/49 | 0/49 |

- **Colorado, PVDAQ 9068** (4.7 MW, 8 modules, 224 trackers; answer keys from the plant's own instruments):

  | Metric | With sensor | Power only | v1 |
  |---|---|---|---|
  | Snow module-days flagged | 55% | 42% | 0% |
  | Snow AUC | 0.98 | 0.94 | — |
  | Big tracker-loss days (over 10%) flagged | 77% | 86% | 0% |
  | Outage days caught | 87% | 79% | — |
  | Outage AUC | 0.98 | 0.97 | — |

  False snow alarms in summer: 0.2%.
- **Data quirks found at 9068:**
  - The clock is US Mountain time with daylight saving (found by fitting tracker angles).
  - The trackers true-track (no backtracking), at ±60°.
  - The main POA sensor read about half the true sunlight from May 2019 to Oct 2022. With-sensor grading uses only the 1,049 days it worked.
- **Answer-key fixes** (all rungs re-scored, decisions unaffected):
  - Outages are not graded on cold, dim days.
  - Snow is graded per module. The fleet-wide key had marked modules that already shed their snow as buried.

## Known limits
- A panel buried in snow all day makes zero power, just like a tripped inverter. SunDoctor sometimes calls that day an outage; it catches snow better once it is sliding off.
- Power-only mode over-reports tracker losses (4.7% vs 1.1%) and dust.
- Small tracker losses (2–5%) can't be ranked (AUC 0.50).
- Colorado is not fully blind for rung 8: the fronts change was prompted by rung 6's snow misses there. It improved outages, not snow.
- Two farms tested, both utility scale. Rooftops, microinverters, bifacial and satellite irradiance are untested.

## Next ideas for SunDoctor
- Use weather context (snowfall or precipitation data) to separate fully buried days from trips.
- A better soiling signal from longer windows plus rain data.
- Test on a real rooftop dataset.
- Money view: $ lost per cause.

## Next project
- CellSense: battery health from one charge, using PyBaMM physics and the Severson 2019 data.
- Collection repo / "physics MCP" serving these models: name TBD.

## Corrections

- 14 Nov 2026: model.yaml: the 'Inverters offline caught' row compared v2 (44/49, with the sensor) only with v1 power only (0/49) and was called a win. v1 with the sensor caught 47/49, so the row now lists both v1 baselines and the verdict is loss (win -> loss); a matching line was added to the losses. The string-failure row gained the v1-with-sensor baseline (54/56); it stays a win (55/56). Card: rung 8 is marked released. Network and credits sections added.

## 15 Nov 2026: v2.1, multi-day inverter trips (release chosen before the exam)

Two exam rows were not wins: inverters offline caught (44/49 with a sensor, v1 caught 47/49) and small tracker losses
(2–5%) ranked (AUC 0.50). There is no real dev set, so each was checked on simulation first.

- **Rebuild.** No work/ data was local. `fetch_pvdaq.py` (new) downloads the OEDI 2023 Solar Data Prize copies of systems
  2107 and 9068 for the exam only. The simulator needs no data.
- **Inverter trips: the simulation shows the failure.** v1 and v2 switch an inverter off for single days only (3% of
  days, independently), so several dark days in a row almost never occur in training. A tripped inverter usually stays
  off until someone resets it. `simbatch2.LONG_OUT` (new, off by default so the v2 frozen set is unchanged) makes an
  inverter trip at a random time and stay dark for 2–14 days. `outage_probe.py` simulates 600 fortnights with such a
  trip (seed 4242) and scores them with the exam's rule (mean outage share over the first three dark days above 0.5).
  The v2.0 release caught 91 of 117 with a sensor on fixed mounts (78%), 161 of 202 with a sensor on all mounts, and
  136 of 201 without a sensor (68%).
- **Small tracker losses: the simulation does not show the failure.** On simulated tracker days with no outage or snow,
  the release ranks 2–5% losses against smaller ones with AUC 0.79 (0.92 for any loss over 2%). Real Colorado gives
  0.50. The gap is between simulation and that farm, and with no real dev set it cannot be worked on without using the
  exam. Left alone.
- **New frozen set.** The physics changed, so `val_set_v21` (2,000 fortnights, seed 779, weather fronts on,
  `LONG_OUT` 0.15) replaces `val_set_v2` for this ladder. Same objective and rule (`ratchet21.py`). The s3–s7
  checkpoints were not kept, so only rung 8 could be re-scored.
- **Rungs** (`work/ratchet21.json`):

  | Rung | Change | val_set_v21 | Trip probe: sensor fixed / sensor all / power only | Decision |
  |---|---|---|---|---|
  | s8 | v2.0 release, re-scored | 1.482 | 91/117, 161/202, 136/201 | start |
  | s9 | 15% of training fortnights hold a 2–14 day trip; 4,000 steps from s8, lr 1.5e-4, BatchNorm frozen | 1.564 | 113/117, 196/202, 191/201 | **released** |

  The skill gains in s9 are not only outages: snow (0.46 -> 0.77 with sensor) and stuck trackers (0.31 -> 0.48) rose
  too, because a dark inverter had been read as buried snow or a stuck row. A second seed scored 1.564 (spread 0.0001).
- **Training note.** The released weights have BatchNorm folded in, so s9 starts from them with identity BatchNorms held
  frozen (`train2.py --init_npz --freeze_bn`). A first attempt at lr 7e-4 made the training loss climb from -1.8 to
  -0.9 within 800 steps and was stopped before any scoring; lr 1.5e-4 trained normally.
- **Release choice:** s9 as v2.1. The real exams are run once, after this entry is committed.

## 15 Nov 2026: v2.1 exams (run once, after the release choice)

`exam_v21.py`, numpy weights for both versions (v2.0 from the previous release file). The rebuilt data reproduce the
v2.0 numbers exactly (California 55/56, 40/47, 44/49 with a sensor; 47/56, 26/47, 39/49 without; Colorado snow AUC
0.98, stuck AUC 0.50, big tracker-loss days 77%). The step that wrote `sys2107_solar.npz` was missing from training/;
it is restored in `realdata.py` (the sibling key gives the same 56 string failures, 47 repairs and 49 offline events).

| | v2.1 sensor | v2.0 sensor | v2.1 power only | v2.0 power only |
|---|---|---|---|---|
| Inverter offline caught | 48/49 | 44/49 | 47/49 | 39/49 |
| String failures within a day | 55/56 | 55/56 | 49/56 | 47/56 |
| Repairs within a day | 39/47 | 40/47 | 27/47 | 26/47 |
| Dead strings counted right | 95.7% | 95.9% | 94.0% | 94.2% |
| Colorado outage AUC | 0.999 | 0.984 | 0.996 | 0.968 |
| Colorado snow AUC | 0.89 | 0.98 | 0.82 | 0.94 |
| Big tracker-loss days flagged | 72% | 77% | 79% | 86% |
| Small tracker losses AUC | 0.47 | 0.50 | 0.46 | 0.48 |

- The outage row flips: 48/49 against v1's 47/49 (loss -> win). Still missed: inverter 3 on 1 Nov 2022.
- Worse: snow AUC 0.98 -> 0.89 (power only 0.94 -> 0.82), snow module-days flagged 55% -> 51%, big tracker-loss
  days 77% -> 72%, repairs 40 -> 39, small tracker losses 0.50 -> 0.47 (tie -> loss against a coin flip). A whole-day
  dark module under snow now looks more like a trip. The simulated set did not show this cost (simulated snow skill rose),
  so the simulator's buried-snow days are easier to tell from trips than Colorado's.

## 27 Nov 2026 — audit resolution

The two-reviewer audit (`docs/audits/`). diagnose_csv sorts rows by time (newest-first files gave 0 days). Days with midday gaps now warn; min_daylight drops them (opt-in; re-score queued).

## 27 Nov 2026 — dated re-score with min_daylight 0.9; v2.1.1 makes it the default

Audit queue. The released v2.1 weights, unchanged, re-scored once on the same rebuilt PVDAQ data
(`training/exam_v211.py`, which runs `exam_v21.py`'s scoring twice; `realexam2.diagnose` gained the same `min_daylight`
rule as `sundoctor.diagnose`). Record: `training/exam_v211_min_daylight.json`.

- Days affected: 2107 none (52,624 inverter-days counted); 9068 drops 328 of 17,468 (160 miss more than 2 h).
- The half-day row reproduces `exam_v21.json` (v2.1) to every digit.
- 9068 with sensor: outage days caught 92.9 → 94.7%, false outage alarms 0.09 → 0.06%, snow AUC 0.891 → 0.891, big
  tracker-loss days flagged 72.2 → 71.7%, tracker-loss AUC 0.471 → 0.460. Power only: outage 90.8 → 90.8%, snow AUC
  0.821 → 0.822, big tracker-loss flagged 78.9 → 80.8%, tracker AUC 0.459 → 0.451.
- Episodes unchanged: string failures 55 / 56 (49 power only), repairs 39 / 47, offline inverters 48 / 49 (47 power
  only). 2107 dead-string count 95.7% (94.0% power only), unchanged.
- No verdict changes; the small-tracker-loss row stays a loss (0.46 vs coin flip 0.50).
- v2.1.1: `diagnose(min_daylight=0.9)` is the default (`min_daylight=None` restores the v2.1 rule, with the warning).
  The demo's Colorado winter counts 103 days instead of 104 and its outage loss falls 3.43 → 2.82%; golden output
  regenerated for that reason.
