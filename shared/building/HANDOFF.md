# BuildingSim handoff: what exists, what's missing, how to finish it

For the owner's code agent, 17 Nov 2026. This is the "make it complete" list for the buildings work in ML Ratchet
(github.com/dogum/ML-Ratchet), the third big-system project after RoadSim (vehicles) and GridSim (power grid).
Read this first. Then read `BUILDING_BRIEF.md` (the contract), `PHASE2_BRIEF.md`, `buildingsim/README.md`,
`buildingsim/LIMITS.md` and `data/DATA.md`.

## 1. What exists

| Piece | Where | State |
|---|---|---|
| BuildingSim simulator (`buildingsim`) | `buildingsim-buildingdata-v1.0.zip` → `buildingsim/` | v1, verified (section 3) |
| BuildingData loader (`buildingdata`), frozen splits, BDG2, RECS/CBECS, fault runs, homes, tariffs, baselines | same zip → `buildingdata/`, `data/` | v1 |
| Physics checks + JSONs | same zip → `checks/` (`run_all.sh`, about 30 min) | pass |
| LoadCast (hourly building energy per meter) | `loadcast-v1.0.zip`, `models/loadcast/lab/index.html` | v1.0 |
| BuildingDetective (infers the building from meters) | `buildingdetective-v1.0.zip`, `models/buildingdetective/lab/index.html` | v1.0 |
| FaultFinder (AHU/RTU fault detection and diagnosis) | `faultfinder-v1.0.zip`, `models/faultfinder/lab/index.html` | v1.0 |
| ThermoPilot (learned thermostat / heat-pump control) | `thermopilot-v1.0.zip`, `models/thermopilot/lab/index.html` | v1.0 |
| Status docs | the claude.ai project: `buildingsim-status.md`, `loadcast-status.md`, `buildingdetective-status.md`, `faultfinder-status.md`, `thermopilot-status.md`; also exported in `building-status-docs.zip` | current |

- **Every model zip has:**
  - numpy-only inference
  - weights
  - a demo on real exam data (all four were run from unzipped copies on 17 Nov)
  - `MODEL_CARD.md`
  - `training/`, including the wrappers listed in section 4
- **Not in the zip, rebuild with `data/scripts/`:**
  - ResStock/ComStock characteristics, hourly samples and weather (`data/proc/stock`, 1.2 GB)
  - the raw downloads (1.4 GB)
  - BuildingSim's ResStock/ComStock weather cache (`refdata/stock_weather`, 55 MB)
  - the reference venv (`buildingsim/.refenv`: CoolProp, pythermalcomfort)
  - simulated training banks

## 2. The vision vs what was built

| Area | Status | Gap |
|---|---|---|
| Zone heat balance (5R1C + multinode walls, solar, infiltration, ventilation, moisture, comfort) | Done | **Single zone only**: no perimeter/core, no simultaneous heating and cooling, no inter-zone air |
| HVAC (13 systems: heat pumps with defrost, DX, furnaces, boilers, chillers + towers, VAV AHU with reheat, PTAC/RTU, district, DHW) | Done | Part-load and short-cycling are first-order; no actuator dynamics; no return fan, exhaust damper or duct model; no chiller-off or CHW sensor faults |
| Faults (20 types, BMS-style trend log) | Done | Logs actual positions, not commands (FaultFinder wrapper fixes this); damper is linear (real installed characteristic is not) |
| Controllers (callable + step API with exact rollback), tariffs, PV, battery | Done | Ideal thermostat within a step; no humidity control |
| Generator (ResStock / ComStock / CBECS priors) | Done | Site-energy intensity off by use type: schools 1.8×, healthcare 1.9×, labs 2.0×, food 1.5×, single-family 1.4×, lodging 0.66× |
| Campus / district behaviour | **Weak** | Default twins read campus chilled water, steam and hot water about 3× low. Real meters are flatter: summer district heat, night chilled water, base loads |
| Real data | Strong for non-residential hourly; annual for homes | BDG2 (1,636 buildings), RECS/CBECS, 237 real/lab fault runs, 66 homes (HUE, WPuQ), 19 tariffs. No real thermostat logs |
| EnergyPlus reference | **Blocked** | No PyPI wheel with an engine; GitHub blocked. BESTEST ranges and ResStock/ComStock twins stand in |

## 3. Verification baseline (do not regress)

From `checks/*.json`. Re-run `checks/run_all.sh` after any change.

- **Analytic checks:**
  - steady conduction to 1.8e-9
  - RC step within 0.0035 K
  - degree-day ratio 0.999996
  - ISO 13786 wall: amplitude −1.6 %, lag −0.01 h
  - every ledger closes (≤ 1e-8)
- **BESTEST:** 29 of 34 values in the published ranges, with all 20 annual heating/cooling values in. The ranges were transcribed from the literature. Get the official ASHRAE 140 tables.
- **Psychrometrics and comfort:** humidity ratio within 0.55 % of CoolProp; PMV within 0.0025 of pythermalcomfort.
- **ResStock twins (126 homes):** site energy median 1.10× (IQR 0.98–1.26), daily r 0.92.
- **ComStock twins (45):** site energy 0.86× (aggregate 0.97), daily r 0.89.
- **Real BDG2 dev twins from metadata alone:**

  | Meter | Median annual ratio | Notes |
  |---|---|---|
  | Electricity | 0.98 | IQR 0.50–1.42, daily r 0.34 |
  | Chilled water | 0.28 | |
  | Steam | 0.29 | |
  | Hot water | 0.15 | |

- **Speed:**

  | Case | Time per building-year |
  |---|---|
  | Single zone, 15 min | 0.7 s |
  | Single zone, 1 h | 0.2 s |
  | Multinode | 2 s |

- **Model exam numbers** are in each model's status doc and model card. Record them before changing anything.

## 4. Prioritised work list

Every physics change must:
1. bump the BuildingSim version;
2. rebuild each model's frozen simulated set;
3. re-score every rung in order;
4. never touch `data/proc/splits.json` (seed 20261116).

### P0: make it part of ML Ratchet
1. **Paths.** Replace the hard-coded sandbox paths (`/home/claude/bldg`, `/mnt/user-data`) with one configurable root. Done when a grep for the sandbox home folder is empty and all demos run from a fresh clone.
2. **Package layout.**
   - Shared modules: `mlratchet/shared/buildingsim` and `mlratchet/shared/buildingdata`.
   - Models: `models/loadcast`, `models/buildingdetective`, `models/faultfinder`, `models/thermopilot`, each with `model.yaml` and `ratchet.yaml`.
   - Done when `mlratchet validate` passes and the site builds them.
3. **Licences** (DATA_LICENSES.md). **Important:**
   - BDG2 is **CC BY-SA 4.0**. Anything derived from it (example series in zips, pages, processed tables) must be shared alike and credited. Keep BDG2-derived files in clearly marked data folders, separate from the MIT code.
   - **Public domain / US government:** RECS, CBECS.
   - **Open, with citation:**
     - NREL ResStock/ComStock (OEDI): CC-BY 4.0
     - ASHRAE RP-1312, FLEXLAB, ORNL and Drexel fault sets: per their figshare/OpenEI licences (the subset used is CC0)
     - HUE: check (Harvard Dataverse, CC-BY)
     - WPuQ: CC-BY 4.0
     - URDB tariffs: OpenEI, open
     - TMYx: climate.onebuilding.org, check terms
     - NASA POWER and Iowa Mesonet: credit
4. **Upstream the Phase 2 wrappers into BuildingSim**, each behind a flag, then re-run the checks:
   - **LoadCast:**
     - country holiday calendars (US/CA/UK/IE)
     - BDG2-style spec maker: the meters a building has decide its plant
     - campus defaults from dev: 24/7 operation in 22 of 52 buildings, outdoor air ×2.5, VAV minimum flow
     - a per-meter district scale and base load
     - bounded-memory mode for long bank runs: one process was killed for running out of memory
   - **BuildingDetective:**
     - W1: intensity per use type from RECS/CBECS prior + dev
     - W2: campus district factors (heat 0.18, CHW 0.33)
     - W3: flat base loads. Real summer district heat is 0.72 of the mean vs 0.06 simulated; night/day CHW 0.69 vs 0.31; electricity 0.77 vs 0.66.
     - label cleaning: three training buildings had base_share −8e12
   - **FaultFinder:**
     - log commands rather than actual actuator positions
     - installed damper characteristic: a real 100 % command mixes 81–90 % outdoor air
     - larger static sensor offsets (1–1.5 K)
     - operator start/stop jitter
   - **ThermoPilot:**
     - numba DP kernel
     - two-pass demand-peak oracle
     - RECS respondent twins with comfort bands from the respondents' own thermostat settings
     - a DST-safe tariff price builder: `bd.tariff_prices` reindexing fails on the November DST hour, so fix that in BuildingData too
     - ERCOT RTP pairing

### P1: simulator fidelity (each has a measured gap)

| Gap | Measured | Likely fix | Calibrate on | Done when |
|---|---|---|---|---|
| Campus district magnitude and shape | thermal meters ~3× low; real meters much flatter | District plant model: distribution losses, summer reheat, plant serving several buildings, seasonal shutdowns; process loads | BDG2 dev 2016 | Generic dev twins within ±30 % annual, and daily r ≥ 0.7 on thermal meters |
| Electricity weather sensitivity | real PRISM r² 0.07 vs twins 0.24 | Unconditioned space, process and plug base loads, campus 24/7 | BDG2 dev | Real and twin r² distributions match |
| Heat-pump signature | Detective heat pump vs resistance AUC 0.40 on RECS (0.94 sim); WPuQ twins overstate the heating slope 2.7× and peak at night while real heat pumps run flat (r 0.21) | Backup strips, defrost, controls, emitter/buffer behaviour, weather compensation curves | RECS dev, WPuQ/HUE dev | Detective heat pump vs resistance AUC > 0.7 on RECS dev; WPuQ profile r > 0.6 |
| Generator intensity by use type | schools 1.8×, healthcare 1.9×, labs 2.0×, lodging 0.66× | Recentre the priors on CBECS/RECS prior respondents | Prior split | Within ±15 % per use type |
| Multi-zone | single zone | Perimeter/core with inter-zone air; simultaneous heating and cooling; VAV reheat per zone | ComStock twins, BDG2 dev | ComStock heating within ±25 % (now 0.75×) |
| Air-side detail for faults | no return fan, exhaust damper, ducts, chiller-off, CHW sensor | Add these and their faults (13 more exam fault types diagnosable) | Fault dev runs | FaultFinder diagnosis beats majority |
| Residential heating in mild weather | 1.3× ResStock | Shading, natural ventilation, internal gains | ResStock twins (sim-vs-sim) | Within ±10 % |
| Official BESTEST | ranges transcribed from literature | Get the ASHRAE 140 tables and run all cases | — | All cases in range |

### P1: data still missing
- **Real thermostat logs:** ecobee Donate Your Data (application). This would give ThermoPilot and BuildingDetective a real exam for time constants and setpoint behaviour.
- **LBNL full fault datasets:** behind a name/email form at faultdetection.lbl.gov. They add a second real RTU exam and more field data. The owner can fill in the form.
- **UK heat-pump field trials:** RHPP, Electrification of Heat. Both need UK Data Service registration.
- **Large open smart-meter sets** (IDEAL homes, Low Carbon London): open but large, so they were skipped. They would add a residential LoadCast exam.
- **EnergyPlus binary:** for a true reference engine. Install outside the sandbox (GitHub blocked here).

### P2: model next steps (from each model's report)
- **LoadCast:**
  1. A median-aimed log-magnitude head plus a shape net. No-history forecasts lose to a median guess and run about 2× low on thermal meters.
  2. A heating-meter physics rung (it loses to TOWT on steam, hot water and gas).
  3. Train the short-history mode on 4–12 week histories from every season; fall back when history is out of season. Two weeks of winter history gives CV 87 % on summer chilled water.
- **BuildingDetective:**
  1. Physics v3: weather-insensitive electricity.
  2. Heat-pump realism.
  3. Domain-adversarial or twin-consistency training on unlabelled 2016 BDG2 meters, plus a dev-fitted population prior for use classes.
  4. The weakest real outputs: primary use is 11 % against 40 % for the majority class; year built has no skill; hourly input reads floor area no better than an annual total.
- **FaultFinder:**
  1. Compare each day with the unit's own recent days: most real false alarms are unit habits. Today it flags 33 % of healthy days at 0.5.
  2. The new air-side faults (P1).
  3. A 2-parameter calibration or a larger dev set.
  4. Diagnosis is no better than the majority class. RTU loses to the one-line rule.
- **ThermoPilot:**
  1. More DAgger rounds plus heat-pump-weighted banks (val skill rose every round).
  2. A residual on a cheap MPC, or a distilled MPC value function. It trails the classical MPC everywhere by 0.1–0.5 skill.
  3. Calibrate residential twins on RECS dev annual energy.
  4. A real exam once ecobee access exists.

### P3: product
- **MCP tools:**
  - `buildingsim.simulate`
  - `loadcast.forecast`
  - `buildingdetective.read`
  - `faultfinder.diagnose`
  - `thermopilot.schedule`
- **A "building audit" page:** upload a year of meter data, then:
  1. Detective reads the building.
  2. LoadCast forecasts next year.
  3. ThermoPilot estimates savings on the user's tariff.
- **The four Artifact pages are private on claude.ai.** Share them or self-host. (17 Nov 2026: self-hosted as each model's `lab/index.html` on the site.)

## 5. Caveats to keep in the docs
- **BDG2 data quirks:**
  - BDG2 timestamps were mixed (local DST vs standard) and are fixed to UTC.
  - Gator electricity is daily totals spread hourly, so it is excluded from hourly labels.
  - Eagle's thermal meters were ÷1000.
  - Four sites were located by temperature matching.
  - Peacock's weather doesn't match its stated location.
- **NASA POWER** runs up to 2.5 °C cold at BDG2 sites; use station temperature for loads.
- **ThermoPilot is graded in simulation** (real weather and tariffs, twin buildings); there are no real thermostat logs. Its WPuQ check shows the simulator cannot settle ThermoPilot vs setback (it flips in 18 of 37 homes).
- **FaultFinder's fixed-false-alarm detection rates** set thresholds from the exam's own healthy days; they are report-only.
- **Rule notes from Phase 2 agents** (no exam result was used to choose anything):
  - **LoadCast** changed its frozen-set objective twice before R3, re-scoring all rungs each time. Its noise bank includes dev buildings' 2017 residuals, which makes dev-gate numbers slightly optimistic.
  - **BuildingDetective** manually withdrew a kept rung (3×512 network) for exceeding the parameter cap. Its real dev gate drifted 0.331 → 0.306 across kept rungs, each step within tolerance.
  - **ThermoPilot** used 4 of 6 RECS respondents per tariff slot for compute, decided before the exam. It once ran `pkill -f` on its own script name; it matched only its own process.

## 6. Definition of "complete" for v2
1. All P0 done, including the CC BY-SA handling for BDG2-derived files.
2. Campus/district, heat-pump and generator-intensity gaps inside their "done when".
3. Multi-zone.
4. Official BESTEST.
5. At least one of these real datasets added as dev+exam by unit: ecobee DYD, LBNL field RTU, a UK heat-pump trial.
6. All four models re-trained on the new BuildingSim version and re-scored, each with a published win/loss table.
