# Buildings project (BuildingSim) — status (17 Nov 2026)

The owner's third big-system project for ML Ratchet, built like RoadSim and GridSim.
- **Phase 1:** two parallel agents built a shared simulator (BuildingSim) and a shared real-data package (BuildingData) against a written contract.
- **Phase 2:** four parallel agents built the models (they were interrupted by a container restart and resumed from their STATUS.md notes).
- **Handoff:** `buildingsim-handoff.md` (also `HANDOFF.md` in the zip).

## Deliverables
- **`buildingsim-buildingdata-v1.0.zip`** (230 MB):
  - `buildingsim` package, checks and briefs
  - `buildingdata` loader, frozen splits
  - BDG2 (cleaned), RECS/CBECS, fault runs, HUE/WPuQ homes, tariffs, TMY, baselines
  - Not included (rebuildable): ResStock/ComStock samples and the raw downloads.
  - BDG2-derived files are CC BY-SA 4.0.
- **Models:**

  | Model | Page | Zip | Status doc |
  |---|---|---|---|
  | LoadCast v1.0 | `models/loadcast/lab/index.html` | `loadcast-v1.0.zip` | `models/loadcast/NOTEBOOK.md` |
  | BuildingDetective v1.0 | `models/buildingdetective/lab/index.html` | `buildingdetective-v1.0.zip` | `models/buildingdetective/NOTEBOOK.md` |
  | FaultFinder v1.0 | `models/faultfinder/lab/index.html` | `faultfinder-v1.0.zip` | `models/faultfinder/NOTEBOOK.md` |
  | ThermoPilot v1.0 | `models/thermopilot/lab/index.html` | `thermopilot-v1.0.zip` | `models/thermopilot/NOTEBOOK.md` |

- All the demos, including BuildingSim's, were run from unzipped copies on 17 Nov. Pages are private until shared.
- `building-status-docs.zip` holds all six docs as files.

## Phase 1: BuildingSim
- **Physics:**
  - Zones: ISO 13790 5R1C plus a multinode wall option, Perez solar, LBL infiltration, ventilation and economizer, foundations, moisture, PMV/PPD.
  - Equipment: 13 HVAC systems, using EnergyPlus/DOE-2 curves, defrost, a chiller with a Braun tower, VAV with reheat, district energy and DHW.
  - Faults and control: 20 fault types with a BMS trend log, a controller hook with a step API and exact rollback, tariffs, PV and battery.
- **Checks:**
  - Analytic cases to 1e-9; every ledger closes.
  - BESTEST: 29 of 34 values in range, including all 20 annual heating/cooling values.
  - CoolProp and pythermalcomfort references match.
- **Twins:**

  | Comparison | Result |
  |---|---|
  | ResStock twins (simulator vs simulator) | site 1.10×, daily r 0.92 |
  | ComStock twins | 0.86× site (aggregate 0.97), daily r 0.89 |
  | Real BDG2 dev twins from metadata, electricity | 0.98× (wide scatter, daily r 0.34) |
  | Real BDG2 dev twins, campus chilled water / steam / hot water | 0.15–0.29× (about 3× low) |

- **Speed:** 0.7 s per building-year at 15 min (0.2 s at 1 h).
- **EnergyPlus:** blocked here (no wheel with an engine; GitHub blocked).

## Phase 1: BuildingData
- **BDG2:** 1,636 buildings and 3,053 meters, hourly 2016–17, quality-flagged and fixed to UTC. Several site fixes: Eagle ÷1000; Gator daily-spread electricity excluded; four sites located by temperature matching.
- **ResStock 2025 R1 / ComStock 2025 R3:** whole-stock characteristics plus 3,097 hourly profiles, labelled MODELLED.
- **Surveys:** RECS 2020 (18,496 homes) and CBECS 2018 (6,436 buildings).
- **Faults:** 237 real or lab AHU/RTU fault runs (Iowa RP-1312, FLEXLAB, ORNL, Nesbitt Hall), 27-type taxonomy.
- **Homes:** HUE (28) and WPuQ (38 heat-pump homes).
- **Tariffs:** 19 URDB tariffs.
- **Splits** (seed 20261116):

  | Data | Split |
  |---|---|
  | BDG2 | 52 dev / 1,584 exam buildings; 2016 history → 2017 forecast |
  | RECS | 9,247 prior / 1,848 dev / 7,401 exam |
  | CBECS | 3,217 prior / 661 dev / 2,558 exam |
  | Faults | 20 dev / 217 exam runs |

- **Baselines (exam):**
  - Electricity hourly CV: change-point 19.1 %, TOWT 19.6 %.
  - Fault rules: AUC 0.69 (AHU), 0.72 (RTU).
  - Setback raised the bill on 6 of 11 TOU tariffs (simple model).
- **Gated:** LBNL full fault datasets (form), ecobee DYD, UK heat-pump trials, Pecan Street.

## Phase 2 results (exam never used for choosing)

### LoadCast
BDG2 exam, 2017, median CV(RMSE). Warm mode uses 2016 history; cold mode uses no history.

| Meter | Warm | TOWT | Change-point | Cold | Median guess |
|---|---|---|---|---|---|
| Electricity | **18.8 %** | 19.6 % | 19.1 % | 59.3 % | 53.8 % |
| Chilled water | **42.9 %** | 45.7 % | 42.4 % | 104 % | 101.6 % |
| Steam | 45.5 % | **39.8 %** | 40.5 % | — | — |
| Hot water | 94.5 % | 83.1 % | — | — | — |
| Gas | 110 % | 95.7 % | — | — | — |

- Warm mode ties the classical models on electricity and chilled water, and loses on steam, hot water and gas.
- Cold mode loses to the median guess and underpredicts thermal meters by about half.
- 90 % bands cover 88–92 %.

### BuildingDetective
- **Wins:**
  - RECS heating fuel 87.3 % vs a rule's 85.3 %.
  - CBECS heating type 74.7 % vs 68.6 %.
  - CBECS chiller AUC 0.77.
  - BDG2 chilled-water/heat meter presence from electricity alone: AUC 0.75 / 0.70.
- **Ties:** floor area (BDG2 48–50 % vs 51.5 %).
- **Fails:**
  - Primary use: 11 % vs 40 % majority.
  - Year built: no skill.
  - Heat pump vs resistance on RECS: AUC 0.40.
- **Simulator-vs-simulator only:** UA within 0.10 log10 (PRISM 0.85). On ResStock the UA estimate is biased (+0.25).

### FaultFinder
- **AHU detection:** AUC 0.78 vs rules 0.69.
- **Real occupied Nesbitt Hall:** 0.78 vs 0.54.
- **Weaknesses:**
  - 33 % false alarms at P = 0.5; rules catch more at strict alarm rates.
  - Diagnosis is no better than guessing the most common fault (20 % top-1).
  - RTU: 0.65 vs rule 0.72.

### ThermoPilot
Simulated exam: real weather and tariffs on twins of real units.

| Set | ThermoPilot | Setback | MPC with forecast | Perfect foresight |
|---|---|---|---|---|
| RECS twins | 2.2 % | 2.2 % (more discomfort) | 3.5 % | 3.6 % |
| BDG2 small commercial | 2.8 % | — | 3.1 % | — |

- Always beats the fixed thermostat; trails the classical MPC everywhere.
- The WPuQ reality check shows the simulator cannot settle ThermoPilot vs setback (the order flips in 18 of 37 homes).

## Rule notes
- **LoadCast:** changed its frozen-set objective twice before R3, re-scoring all rungs each time. Its noise bank includes dev 2017 residuals.
- **BuildingDetective:** manually withdrew one kept rung for exceeding the parameter cap.
- **ThermoPilot:** ran `pkill -f` once on its own script, which matched only its own process. It sub-sampled 4 of 6 RECS respondents per tariff slot (decided before the exam).
- No exam result was used to choose anything.

## Next
See `buildingsim-handoff.md`:
- **P0:** paths, ML Ratchet packaging, BDG2 CC BY-SA handling, and upstreaming each model's wrappers into BuildingSim.
- **P1 physics:** campus/district behaviour, weather-insensitive electricity, the heat-pump signature, generator intensity by use type, multi-zone, air-side fault detail, official BESTEST.
- **Data:** ecobee DYD, LBNL field RTU, UK heat-pump trials.
