> **Lab notebook.** The build notes as recorded during the original work in a cloud sandbox (Oct–Nov 2026), kept
> as written. Paths like `/home/claude/...` and `/mnt/user-data/outputs` refer to that sandbox: in this repository
> the code is in `training/` and everything it writes goes to `work/` (see `training/_paths.py`). Page links to
> claude.ai are the original private copies; the lab now lives at `lab/index.html`.

# SunDoctor — status (30 Oct 2026, v2.0)

- What it is: a neural network trained only on solar-physics simulations. It diagnoses real PV systems day by day, with error bars:
  - dead strings, with failure and repair days
  - shading, clipping, outages, soiling, snow, and stuck or stowed trackers
  - the kWh lost to each cause
- Second project in Greg's physics-ML collection, after ChirpNet. Collection name is still open; Greg leans toward "ML Ratchet" (also liked First Light, Ratchet, Ab Initio).
- v2 adds two things Greg picked:
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
- CellSense: battery health from one charge, using PyBaMM physics and the Severson 2019 data. Greg has a Tesla Model Y, so there may be a personal-data angle.
- Collection repo / "physics MCP" serving these models: name TBD.
