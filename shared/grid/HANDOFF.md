# GridSim handoff: what exists, what's missing, how to finish it

Written 14 Nov 2026 for the repository integration pass. This is the "make it complete" list for the power-grid work in ML Ratchet
(github.com/dogum/ML-Ratchet). Read it first. Then read:
- `GRID_BRIEF.md` (the contract) and `PHASE2_BRIEF.md`
- `gridsim/README.md` and `gridsim/LIMITS.md`
- `data/DATA.md`

The vehicle project's `HANDOFF.md` follows the same pattern.

## 1. What exists

| Piece | Where | State |
|---|---|---|
| GridSim simulator (`gridsim`) | `gridsim-griddata-v1.0.zip` → `gridsim/` | v1, verified (section 3) |
| GridData loader (`griddata`), frozen splits, fleets, events, prices, weather, networks, baselines | same zip → `griddata/`, `data/` | v1 |
| Physics checks + JSONs | same zip → `checks/` (`run_all.sh`) | pass |
| GridPulse: MW lost, inertia and nadir from a frequency trace | `gridpulse-v1.0.zip`, page now `models/gridpulse/lab/` | v1.0 |
| BatteryPilot: storage arbitrage | `batterypilot-v1.0.zip`, page now `models/batterypilot/lab/` | v1.0 |
| PlantDetective: unit character from its output | `plantdetective-v1.0.zip`, page now `models/plantdetective/lab/` | v1.0 |
| DemandCast: demand, wind and solar from weather | `demandcast-v1.0.zip`, page now `models/demandcast/lab/` | v1.0 |
| Status docs | the cloud project (now each model's first `NOTEBOOK.md` entry; GridSim's in `STATUS.md`): `gridsim-status.md`, `gridpulse-status.md`, `batterypilot-status.md`, `plantdetective-status.md`, `demandcast-status.md` | current |

- Every model zip has numpy-only inference, weights, a demo on real exam data (all four were run from unzipped copies on 14 Nov), `MODEL_CARD.md`, and `training/` with the wrappers listed in section 4.
- **Included in the zip:** fleets, networks, splits, baselines, GB/ERCOT event traces, prices, weather, and EIA-930 hourly for the 8 focus BAs.
- **Not included, rebuild with `data/scripts/` (sources in `data/DATA.md`):**
  - the full EIA-930 tables for all 71 BAs, plus interchange/subregion (≈0.5 GB)
  - CAMPD hourly unit data (138 MB processed, 2018–2025 Texas)
  - raw downloads (0.9 GB)
  - the reference-library venv (`gridsim/.refenv`, 1.2 GB: pandapower, PYPOWER, PyPSA, ANDES; recreate it with pip)
  - simulated training banks

## 2. The original vision vs what was built

The vision was a deep, wide simulation of a big system with lots of real data and many ML cases on top.

| Area | Status | Gap |
|---|---|---|
| AC/DC power flow, PTDF, DC-OPF with congestion | Done, exact vs PYPOWER | AC-OPF and voltage/reactive limits in dispatch not built |
| Unit commitment / economic dispatch (MILP, LP, merit) | Done | Coal-vs-gas split follows gas price (−43 % to +28 % coal by year); scarcity pricing weak; 4–11 % of days need a fallback re-solve |
| Frequency dynamics (swing, governors, FFR, synthetic inertia, UFLS, multi-area) | Done, checked vs ANDES and GB 9 Aug 2019 | Load damping fitted in-sample; unfitted GB events ~30 % too deep; generic governors; no voltage or angle stability |
| Wind / solar from weather | Done | Annual wind +6–8 %; reanalysis weather (50 km) limits complex terrain |
| Weather-driven demand | Done (regression) plus DemandCast's building-stock physics (wrapper) | Upstream the building model |
| Storage | Done (dispatch) plus BatteryPilot's wear model | Upstream rainflow wear |
| Markets | Partial | Energy only; no ancillary services, no ORDC in core (BatteryPilot has one), DA/RT only in a wrapper |
| Regions | ERCOT and GB deep; CISO, PJM, MISO, SWPP, ISNE, NYIS, BPAT data only | Other regions not dispatched or calibrated |
| Real data | Strong | EIA-930, CAMPD hourly (Texas), 470 GB and 329 ERCOT events, complete ERCOT prices, weather, test grids |
| Not built | — | Transmission expansion / capacity planning, hydro cascades, DER/distribution feeders, gas–electric coupling, outages from weather (beyond BatteryPilot's cold curve), DispatchNet model |

## 3. Verification baseline (do not regress)

From `checks/*.json`; re-run `checks/run_all.sh` after any change.

- **Power flow:** AC/DC power flow matches PYPOWER to 1e-11° and 3e-10 MW on IEEE 9–300 and ACTIVSg200/2000. DC-OPF cost to 3e-13, LMPs to $0.013.
- **Frequency:** RoCoF exact. Nadir within 0.03 mHz analytically. Kundur two-area vs ANDES: 15 mHz RMS. GB 9 Aug 2019: first nadir 49.11 Hz at 24.5 s (reported 49.1 / 24.5 s), LFDD 75.6 s (reported 75.9 s); 44 mHz RMS vs the NESO 1 s trace before LFDD.
- **ERCOT dispatch vs EIA-930:**
  - 2019 calibrated: coal +6 %, gas −2 %, CO2 −1 %.
  - 2022 held out: coal +28 %, gas −11 %, CO2 +1.8 %.
  - Prices: median $46.6 vs $48.0 (2022), rank correlation 0.87.
- **Speed:**
  - ERCOT year: MILP 13–21 min, LP ~3 min, merit ~2 s.
  - Frequency event: 0.3 s (batched ~0.1 s, simple states 0.5 ms).
  - 2,000-bus AC power flow: 36 ms.
- **Model exam numbers:** in each model's status doc and model card. Record them before changing anything.

## Status, 14 Nov 2026 (repository integration)

- **P0.1 paths: done.** No `/home/claude` or `/mnt/user-data` in any script. One root, `$ML_RATCHET_GRID` (default
  `shared/grid`), holds `data/`, `gridsim/refdata/` and `checks/out/`; code is found next to each file.
- **P0.2 package layout: done**, as `shared/grid/gridsim` and `shared/grid/griddata` (importable with
  `pip install -e ".[grid]"`), and `models/gridpulse`, `models/batterypilot`, `models/plantdetective`,
  `models/demandcast`, each with `model.yaml` and `ratchet.yaml`.
- **P0.3 data rebuild: done** as `mlratchet data grid` (`data/build.py`): `fetch` (the hand-fetched sources), then the
  scripts in order, then `verify` against `data/MANIFEST.json` (sha256 of all 1,064 shipped files) and the split hash.
  `--from-zip` unpacks the v1.0 release instead. The NESO data come through the CKAN datastore API; CAMPD through the
  no-key bulk route.
- **P0.4 licences: done** (`DATA_LICENSES.md`, checked on the providers' pages; CAISO's terms page did not load).
- **Checks reproduce** (`checks/REPRODUCED.md`): `checks/run_all.py` re-ran every check on this repository's machine
  and compared every number with the v1 JSONs. Exact: analytic frequency, GB 9 Aug 2019, dev events, ANDES, generator,
  renewables, demand, the dev-2019 calibration. ERCOT 2019 and 2022 by fuel, CO2 and prices to the published digits
  (MILP noise in the 4th-6th digit). One real difference: the ACTIVSg2000 24 h MILP prices in hours 14-16.
- **Models re-scored from their shipped weights** on this data: GridPulse, PlantDetective (395 units, on CAMPD rebuilt
  from EPA), DemandCast (actual weather) exactly; BatteryPilot ERCOT exactly, CAISO 0.641 vs 0.642 (price decimals).
- **P0.5 outage-rate leak: fixed (physics 1.1, 14 Nov 2026).** Forced and planned outage shares for coal, CC, CT and
  gas steam now come from sources published before 2023 (`fleet.OUTAGE_DEFAULTS`; `GRIDSIM_PHYSICS=1.0` reproduces the
  release). The dev calibration (coal adder −6.62 → −5.73 $/MWh) and ERCOT damping (2.5 → 2.75 %/%) were re-fitted on
  dev; the checks moved as in `gridsim/LIMITS.md` D (held-out 2022 coal +27.7 → +34.4 %, hours above $200 99 → 54
  vs 238 real): the leaked values flattered the checks. Models: BatteryPilot used them directly (rebuilt, see its
  NOTEBOOK); PlantDetective through 69 GridSim units and GridSim prices behind 37 of 110 shards (measured, no
  decision changes); GridPulse and DemandCast never read them. Other exam-year citations (generator unit sizes and
  heat-rate medians, an IBR-share bound, a 2023 wind report) move their priors by at most 0.5 % (heat rates) and
  3 % (sizes) and are left as cited.
- **P0.7 GB_20191203 label: fixed** (1,788 → 854.7 MW, Elexon-cited correction table; six exam events with the same
  FUELINST pattern are listed in `data/DATA.md` section 5 for a decision).
- **P0.6 upstream fixes:** done — per-state loss lists (`_as_events`), EIA-930 solar timestamp flags
  (`gd.eia930_solar_qc()`), per-technology aggregation in `generate_states` (flag), soft history constraints in
  `dispatch` (flag). Not done: the new-physics items (Dynamic Containment two-segment droop, year-aware priors, ORDC
  in core, cold-outage curve, GT ambient derate, DA price generator, self-commitment layer, building-stock demand
  model) and two speed tools (single-LP perfect foresight, tabulated farm curves).
- **Not done:** everything in P1-P3.

## 4. Prioritised work list

Every physics change must:
- bump the GridSim version;
- rebuild each model's frozen simulated set;
- re-score every rung in order;
- never touch `data/proc/splits.json` (sha256 895b4e9a…). 2026 is a reserve year: keep it untouched for a future exam.

### P0: make it part of ML Ratchet
1. **Paths.** Replace every hard-coded `/home/claude/grid` (and `/mnt/user-data`) with one configurable root. Done when `grep -r /home/claude` is empty and all demos run from a fresh clone.
2. **Package layout.**
   - Add `mlratchet/shared/gridsim` and `mlratchet/shared/griddata`.
   - Add `models/gridpulse`, `models/batterypilot`, `models/plantdetective`, `models/demandcast`, each with `model.yaml` + `ratchet.yaml`.
   - Done when `mlratchet validate` passes and the site builds them.
3. **Data rebuild.** One command runs `data/scripts/` in order and ends with the splits hash check. It must include the NESO API route (the direct NESO files redirect to a host the cloud proxy refused) and the no-key CAMPD bulk route `api.epa.gov/easey/bulk-files/emissions/hourly/state/emissions-hourly-<year>-tx.csv`.
4. **Licences** (DATA_LICENSES.md):
   - Public domain (US government): EIA-930, EIA-860/923, EPA CAMPD.
   - NESO data portal: NESO Open Data Licence. Check the terms.
   - Elexon BMRS: check the licence.
   - ERCOT public reports and prices: public; cite.
   - CAISO OASIS: public; cite.
   - NASA POWER: free, with credit.
   - Iowa Mesonet: free, with credit.
   - MATPOWER cases including ACTIVSg: BSD / CC-BY per case.
   - RTS-GMLC: BSD-3.
   - Open-Meteo: CC-BY 4.0.
5. **Possible exam leak to remove.** GridSim's outage-rate defaults cite Texas RE's 2024/2025 reliability-assessment aggregates, published during the exam years. Replace them with pre-2023 NERC GADS values, re-run the checks, and re-score. BatteryPilot's model card notes this; GridPulse and PlantDetective did not use outage rates.
6. **Upstream fixes found by the Phase 2 builds.** Add each behind a flag, then re-run the checks.
   - **GridPulse:**
     - `_as_events` mis-handles a list of per-state loss lists (pass an object array).
     - Aggregate units per technology inside `generate_states` (~40× faster).
     - NESO Dynamic Containment two-segment droop model.
     - Effective ERCOT damping ~1.0 %/% for generated states (Phase 1 used 2.5, fitted on real-fleet states).
     - Year-aware priors.
   - **BatteryPilot:**
     - ORDC scarcity adder (μ −4,000 MW, σ 5,000 MW, fitted on dev 2019/2021).
     - Cold-weather outage curve (55 % at about −3 °C, fitted on Uri).
     - Gas-turbine ambient derate.
     - Day-ahead price generator built from EIA-930 forecast errors.
     - Single-LP perfect-foresight baseline (6× faster).
   - **PlantDetective:**
     - `dispatch(outages='random')` goes infeasible when an outage starts while a unit is on. Its wrapper turns outages into near-zero derates. Fix this in core with soft history constraints.
     - Self-commitment DP and CAMPD measurement emulator as a GridSim "unit behaviour" layer.
     - Re-check CC/CT heat-rate priors against the Texas fleet.
   - **DemandCast:**
     - Building-stock demand model (`src/buildings.py`) to replace GridSim's demand regression.
     - Fast tabulated wind farm curve and fleet grouping (`src/renew.py`).
     - In GridData, flag the EIA-930 solar timestamp shifts (CISO led by 1 h in 2019/21, matched in 2023, lagged in 2024–25; PJM changed in 2025) and negative night solar.
7. **Data label issue.** GB dev event GB_20191203T200030 is labelled 1,788 MW, but its frequency drop implies ~850 MW (GridPulse used the 854 MW metered output on dev). Fix the label in GridData with a note; it is dev, so the splits are unchanged.

### P1: simulator fidelity (each has a measured gap)

| Gap | Measured | Likely fix | Calibrate on | Done when |
|---|---|---|---|---|
| Coal vs gas commitment | coal −43 % to +28 % by year | Coal contracts/must-run and self-commit behaviour (PlantDetective's DP), coal delivered prices from EIA-923 receipts, real heat-rate curves from CAMPD (dev plants) | Dev years 2019, 2021 | Coal and gas annual within ±10 % on held-out training-input years |
| Scarcity pricing | spikes too rare, wrong hours | ORDC in core, unforecast outages, 15-min ramp scarcity | Dev years | Hours above $1,000 within ±25 % and timing correlation up |
| Inertia at low net load | 2022 min 76 vs 115 GW·s | Commitment of synchronous units at low net load (must-run, reserves, voltage support) | ERCOT dev inertia reports | Min inertia within ±15 % |
| Frequency damping / response | unfitted GB events 30 % too deep | Year-aware response (DC, DR, batteries), real governor populations, measured deadbands | GB and ERCOT dev events | Unfitted dev twins inside the real distribution |
| Daily fallback re-solves | 4–11 % of days | Soft history constraints across days | — | < 0.5 % fallbacks |
| Network in ERCOT runs | copper plate | Use ACTIVSg2000 or a reduced real-topology model; West Texas congestion | Dev prices by hub | Hub spreads within ±25 % |
| Other regions | data only | Dispatch PJM, MISO, CISO with their fleets | Dev years | Per-fuel error like ERCOT |

### P1: data still missing
- **Sub-second frequency** (PMU or FNET/GridEye; the FNET site timed out). GridPulse's inertia comes mostly from demand because 1 s data blur the first-second slope.
- **Day-ahead forecast-weather archive.** The Open-Meteo Previous-Runs API returned HTTP 429; only 2024 population points were fetched. Needed for a full DemandCast forecast-mode exam and for training on real forecast errors.
- **CAISO prices before 2023-08.** OASIS serves only ~3 years; an archive is needed for a longer CAISO exam.
- **17 missing ERCOT monthly operations reports.** These hold more ERCOT events.
- **CAMPD for other states.** Currently Texas only; adding more would widen PlantDetective's exam beyond ERCOT units.
- **GB loss sizes.** Noisy outside 9 Aug 2019; better unit-trip data would sharpen the MW-lost exam.

### P2: model next steps (from each model's report)
- **GridPulse:**
  1. Year-aware priors to fix the 2023–25 GB inertia overestimate (~30 %).
  2. A better 30–60 s model: secondary response as droop plus delayed reserves. Its 10 s and 60 s forecasts lose to persistence.
  3. Inertia from the ambient-noise spectrum.
  4. Coverage on real MW lost is too narrow (48 % inside 1σ).
- **BatteryPilot:**
  1. Make simulated RT spikes less predictable from DA until the dev-twin capture matches real (91–94 % twin vs 80 % real).
  2. A 1 h specialist or 15-min execution head: it loses to the DA optimiser on 1 h batteries and in CAISO.
  3. West Texas congestion basis; real CAISO import/hydro curve; ancillary services.
- **PlantDetective:**
  1. Re-centre CC/CT heat-rate priors (CC 21 % and CT 15 % too high). Make trust in EIA-923 learnable per technology.
  2. Unit-specific prices and coal costs.
  3. Learn run statistics as corrections to counted statistics.
  4. Note: its exam fed 2023–25 ERCOT prices as optional context (no choices were made from them). Decide whether to keep that or restrict to pre-exam context.
- **DemandCast:**
  1. Simulate random 0/±1 h reporting lags so solar stops failing on timestamp shifts. Solar loses to persistence; it fails in CISO and BPAT.
  2. Real forecast-error noise in training, plus a 2023–25 forecast-weather exam.
  3. Better-calibrated bands: 64 % inside the 68 % band.
  4. Higher-resolution weather for wind in complex terrain.
- **DispatchNet (not built):** a learned dispatcher vs the MILP. There is no real "optimal dispatch" truth. Grade it on EIA-930 generation by fuel and real prices once the coal/gas gap is fixed, and on speed vs MILP.

### P3: product
- **MCP tools**, each returning error bars and the model card's limits:
  - `gridsim.dispatch`
  - `gridsim.frequency_event`
  - `gridpulse.read`
  - `batterypilot.run`
  - `plantdetective.read`
  - `demandcast.forecast`
- **A "control room" page** on the site: one ERCOT day showing demand forecast, dispatch, a trip event and a battery's response together.
- Share or self-host the four original pages: done, they are each model's `lab/` (14 Nov 2026).

## 5. Caveats to keep in the docs
- **ERCOT events have no public 1 s traces.** ERCOT tests use the operations reports' summary numbers. GB has traces, but loss sizes are noisy.
- **CAMPD gross load:** it excludes the steam turbine for many combined-cycle units, and is 5–8 % above EIA net generation.
- **EIA-930:** demand = generation − interchange fails for CISO, MISO, BPAT and PJM, so use the demand column. Fuel categories changed in 2024-07. CISO's own forecast is misaligned by 1 h.
- **Fleet defaults by technology:** ramp, start cost, inertia and governor values are technology defaults, not unit data.
- **DemandCast's headline:** its demand result (beats grid operators in 6 of 8 BAs) uses actual weather and is an upper bound. With real forecast weather (2024 only) it wins 4 of 8.
- **Rule note:** one Phase 2 build used `pkill -f` once on its own job (the pattern matched only its own process), against the shared-machine rule. No other build's work was affected.

## 6. Definition of "complete" for v2
1. All P0 done, including the outage-rate leak fix.
2. P1 coal/gas, scarcity and inertia gaps inside their "done when".
3. Sub-second frequency data or a documented substitute.
4. A full forecast-weather archive.
5. All four models re-trained on the new GridSim version and re-scored, each with a published win/loss table.
6. DispatchNet built.
7. At least one more region (PJM or CISO) dispatched and calibrated like ERCOT.
