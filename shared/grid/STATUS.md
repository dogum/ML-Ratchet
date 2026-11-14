> Kept as written on 14 Nov 2026 in the cloud project where the grid work was built, with names of people and build roles made neutral. The `.zip` names refer to that project's releases; in this repository the platform is `shared/grid/` and the models are `models/gridpulse`, `models/batterypilot`, `models/plantdetective` and `models/demandcast`.

# Power-grid project (GridSim) — status (14 Nov 2026)

The second big-system project for ML Ratchet, built like the vehicle project (RoadSim):
- **Phase 1 (2 parallel builds):** a shared simulator (GridSim) and a shared real-data package (GridData) against a written contract.
- **Phase 2 (4 parallel builds):** models on top.
- **Handoff:** `gridsim-handoff.md` (also `HANDOFF.md` in the zip).

## Deliverables
- **`gridsim-griddata-v1.0.zip`** (245 MB):
  - `gridsim` package, checks, briefs
  - `griddata` loader, frozen splits, fleets for 8 BAs, test grids, GB/ERCOT events, ERCOT/CAISO prices, weather, EIA-930 for the 8 focus BAs, baselines
  - not included, rebuildable: CAMPD hourly data, all-BA EIA-930, raw downloads
- **GridPulse v1.0:** page `models/gridpulse/lab/` · `gridpulse-v1.0.zip` · status in `models/gridpulse/NOTEBOOK.md`
- **BatteryPilot v1.0:** page `models/batterypilot/lab/` · `batterypilot-v1.0.zip` · status in `models/batterypilot/NOTEBOOK.md`
- **PlantDetective v1.0:** page `models/plantdetective/lab/` · `plantdetective-v1.0.zip` · status in `models/plantdetective/NOTEBOOK.md`
- **DemandCast v1.0:** page `models/demandcast/lab/` · `demandcast-v1.0.zip` · status in `models/demandcast/NOTEBOOK.md`
- All demos (including GridSim's) were run from unzipped copies on 14 Nov. The pages are now each model's `lab/`.

## Phase 1: GridSim
- **Power flow and dispatch:**
  - AC/DC power flow, PTDF and DC-OPF are exact vs PYPOWER on IEEE 9–300 and ACTIVSg200/2000.
  - Unit commitment and dispatch run in MILP, LP or merit mode, with ramping, min up/down, starts, reserves, storage, hydro budgets and network.
- **Frequency dynamics:** multi-area swing equation, governors per technology, FFR, synthetic inertia, RoCoF relays and UFLS.
  - Analytic cases exact; ANDES two-area within 15 mHz RMS.
  - GB 9 Aug 2019: nadir 49.11 Hz at 24.5 s (reported 49.1), LFDD at 75.6 s (75.9 s); load damping fitted on this event.
- **ERCOT dispatch vs EIA-930:**

  | Year | Coal | Gas | CO2 |
  |---|---|---|---|
  | 2019 (calibrated) | +6 % | −2 % | −1 % |
  | 2022 (held out) | +28 % | −11 % | +1.8 % |

  The coal/gas split follows gas price, unlike reality. Prices: median $46.6 vs $48.0 (2022), rank correlation 0.87; scarcity spikes too rare.
- **Renewables and demand:** wind +6–8 % annual, hourly correlation 0.90; solar correlation 0.97; demand regression MAPE 3.8–4.3 %.
- **Speed:**

  | Task | Time |
  |---|---|
  | ERCOT year, MILP | 13–21 min |
  | ERCOT year, LP | ~3 min |
  | ERCOT year, merit | ~2 s |
  | One frequency event | 0.3 s |
- **Possible exam leak:** outage-rate defaults cite Texas RE 2024/25 published aggregates. To be replaced with pre-2023 NERC GADS values.

## Phase 1: GridData
- **EIA-930:** hourly for all 71 BAs, 2018–2026, with quality flags.
- **Fleets:** 8 focus BAs (ERCO: 2,393 generators) in the contract spec, with a source for every value.
- **EPA CAMPD:** hourly, every Texas unit 2018–2025 (452 units, 28 M unit-hours). The no-key bulk route works.
- **Frequency:**
  - GB: 470 events, each with an 11-minute 1 s trace and NESO's inertia estimate; 4 full months at 1 s; per-minute summaries 2018–2026.
  - ERCOT: 329 events from monthly operations reports; no public 1 s data.
- **Prices:** ERCOT DA/RT 2018–2026 (complete); CAISO 2023-08 → 2026 (OASIS limit).
- **Weather:** NASA POWER + ASOS for every BA.
- **Test grids:** 11, IEEE through ACTIVSg2000.
- **Splits (frozen, sha256 895b4e9a…):**
  - training-input years 2018, 2020, 2022 · dev 2019, 2021 · exam 2023–25 · reserve 2026
  - plants: dev 10 plants (40 units), exam 407 units
  - events: GB dev 4 / exam 412; ERCOT dev 5 / exam 301
- **Baselines (exam):**
  - **Demand:** each BA's own day-ahead forecast scores 2.0–4.3 % MAPE (CISO's is misaligned, 6.9 %).
  - **Heat rate:** EIA-923 annual average, 13.4 % hourly.
  - **Battery:** perfect-foresight vs spread rule, e.g. ERCOT Houston 2023: $152.6 vs $94.5/kW-yr.
  - **Frequency:** RoCoF formula 27–31 % on MW lost; ERCOT MW/Hz constant 20.1 %.

## Phase 2 results (exam never used for choosing)
- **GridPulse:**
  - GB inertia: 16.4 % vs constant 20.4 %, but a dev-fitted demand rule gets 15.1 % (the inertia skill comes from demand, not the trace).
  - GB MW lost: 25.9 % vs RoCoF formula 27.1 % (medium labels).
  - Nadir forecast at 5 s: small win; 10 s and 60 s forecasts lose to persistence.
  - ERCOT MW lost from report metrics: 15.7 % vs 20.1 %.
  - Overestimates 2023–25 GB inertia by about 30 %.
- **BatteryPilot** (share of perfect hindsight, net of rainflow wear):

  | Strategy | ERCOT 2023–25 | CAISO 2024–25 |
  |---|---|---|
  | BatteryPilot | 0.667 | 0.642 |
  | Classical DA optimiser | 0.685 | 0.681 |
  | Spread rule | 0.561 | 0.589 |

  - Ties the classical optimiser: wins with 4 h batteries, loses with 1 h and in CAISO.
  - Sim-to-real gap: dev twins capture 91–94 % vs 80 % on real data.
- **PlantDetective:**
  - **Wins:**
    - minimum stable load 5.7 points vs 8.0
    - ramps 14 % vs 39 %
    - heat rate for coal 4.4 % vs EIA-923 10.3 %, and for gas steam 4.7 % vs 22 %
  - **Loses:**
    - combined-cycle heat rate 21 % (biased high) vs default 7 %
    - combustion-turbine heat rate 16 % vs EIA-923 6.5 %
  - **Ties:** with a year or more of history, start statistics are no better than plain counting.
- **DemandCast:**
  - Demand MAPE beats the grid operators' own day-ahead forecasts in 6 of 8 BAs with actual weather (pooled 2.38 vs 3.41 %; an upper bound). With real forecast weather (2024 only) it wins 4 of 8 (2.92 vs 3.20 %).
  - Wind: 19.2 % vs 43.5 % persistence.
  - Solar loses to persistence (29.9 vs 19.1 %) because EIA-930 solar timestamps shifted in CISO and PJM.
  - Error bars are too narrow.

## Rule notes
- PlantDetective used `pkill -f` once on its own job (the pattern matched only its own process). Its exam also fed 2023–25 ERCOT prices as optional context; no choices were made from them.
- The DemandCast build overwrote some generic-named files in the shared scratchpad (screenshots only).

## Next
See `gridsim-handoff.md`:
1. **P0:** fix paths, package into ML Ratchet, fix the outage-rate leak, upstream each model's fixes.
2. **P1:** coal/gas commitment, scarcity pricing, low-net-load inertia, frequency response by era.
3. **Data still missing:** sub-second frequency, a day-ahead forecast-weather archive, older CAISO prices, CAMPD for more states.
4. **Not built yet:** DispatchNet.
