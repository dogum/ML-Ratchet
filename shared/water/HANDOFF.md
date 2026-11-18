# Watershed handoff: what exists, what's missing, how to finish it

For Greg's code agent, 18 Nov 2026. This is the "make it complete" list for the watershed work in ML Ratchet
(github.com/dogum/ML-Ratchet). It is the fourth big-system project, after RoadSim (vehicles), GridSim (power grid) and
BuildingSim (buildings).

Read this first, then:
- `WATER_BRIEF.md` (the contract) and `PHASE2_BRIEF.md`
- `watersim/README.md`, especially "Using it in Phase 2" and "Known limitations"
- `waterdata/DATA.md`, especially section 8, "Gotchas"

## 1. What exists

| Piece | Where | State |
|---|---|---|
| WaterSim simulator (`watersim`) | `watersim-waterdata-v1.0.zip` → `watersim/` | v1, verified (section 3) |
| WaterData loader (`waterdata`), frozen splits, attributes and specs, baselines, fetch scripts | same zip → `waterdata/`, `data/` | v1 |
| Physics checks and their JSONs | same zip → `checks/` (`run_all.sh`) | pass |
| FlowCast (ungauged daily streamflow, plus a gauged nowcast mode) | `flowcast-v1.0.zip`, https://claude.ai/artifact/7iJtRbBuCUfAhmm6eo7SvH | v1.0 |
| SnowSense (SNOTEL SWE from weather alone) | `snowsense-v1.0.zip`, https://claude.ai/artifact/QQdrSbSKeN7SH2eqLFLbEW | v1.0 |
| BasinDetective (decade-to-decade change, regulation, hidden stores) | `basindetective-v1.0.zip`, https://claude.ai/artifact/SbwK1wQbrkWUz56dnaQAUk | v1.0 |
| ReservoirPilot (causal learned dam release policy) | `reservoirpilot-v1.0.zip`, https://claude.ai/artifact/JNZJvZe5RT4C1A4XvHDvGt | v1.0 |
| Status docs | claude.ai project "Physics Fun": `watershed-status.md`, `flowcast-status.md`, `snowsense-status.md`, `basindetective-status.md`, `reservoirpilot-status.md`; also in `watershed-status-docs.zip` | current |

- **Every model zip has:**
  - numpy-only inference
  - weights
  - a demo on real exam data (all four were run from unzipped copies on 18 Nov)
  - `MODEL_CARD.md`
  - the training scripts, including the physics wrappers listed in section 4

- **Not in the Phase 1 zip.** It is 10 MB and leaves out the real time series. Rebuild them with `data/scripts/` (all resumable), in this order:
  - `fetch_caravan.py` streams Caravan v1.6 once (24.8 GB tar.gz, never stored). Output: `proc/caravan`, 1.7 GB.
  - NWIS daily values and peaks: `proc/nwis`.
  - NWM v3.0 retrospective at CAMELS gauges: `proc/nwm`. Expensive: about 385k range requests.
  - SNOTEL: `proc/snotel`.
  - ResOpsUS, GRanD and CDEC: `proc/reservoirs`.
  - Baseline simulations: `proc/baselines/sims`.

  The R/airGR/TUWmodel reference environment (`watersim/.refenv`, 1.3 GB) and the simulated training banks are also not included.
- **Licenses:**
  - Caravan: CC-BY 4.0. Cite Kratzert et al. 2023 and each national dataset.
  - ResOpsUS: CC-BY. Cite Steyaert et al. 2022.
  - USGS, NOAA and NRCS data: public domain.
  - DATA.md lists every citation.

## 2. The vision vs what was built

| Area | Status | Gap |
|---|---|---|
| Runoff physics (HBV, GR4J, SAC-SMA, VIC-lite) | Done | Soil and groundwater are lumped per basin; no glacier melt; no explicit wetland or lake storage; no karst |
| Snow (degree-day with elevation bands; daily energy balance) | Done | EB is daily (no diurnal melt-refreeze); EB defaults are off at SNOTEL points (see section 4); no wind redistribution or canopy interception |
| Evapotranspiration (FAO-56 PM, PT, Hargreaves, Oudin) | Done | ERA5-Land SRAD is *net* shortwave, so incoming is reconstructed approximately |
| Routing (UH, Muskingum-Cunge, kinematic, networks) | Done | Channel geometry from power laws; no floodplain storage; no backwater |
| Hourly / flash floods | Partial | HBV, SAC-SMA and VIC-lite run hourly with infiltration excess, but there is **no hourly real dataset** wired up (only NWM hourly 2015–2023). GR4J and reservoirs are daily only |
| Reservoirs and irrigation | Done (single reservoir) | No cascades, environmental flows, evaporation in ReservoirPilot, or real operating rules (flood-control curves, water rights) |
| Groundwater pumping, urban drainage, water quality, sediment, temperature | **Not started** | |
| Real forcing | **ERA5-Land only** | NCAR GDEX (original CAMELS Daymet/Maurer/NLDAS) and the Daymet API timed out from the sandbox. ERA5-Land undercatches mountain and Pacific Northwest precipitation (50 basins have runoff ratio > 1) and drifts between decades (about 6 %, 1980s vs 2000s). This caps **every** method and makes literature comparisons invalid |
| Real data coverage | Strong in the US | CAMELS-US 671; GB 671, CL 505, BR 870, AUS 561 (time series and splits frozen, but **no Phase 2 model used non-US regions**); SNOTEL 880; ResOpsUS 678 + CDEC 18; NWIS to Nov 2026; NWM 650 gauges |
| Forcing after 2023 | Missing | Caravan v1.6 forcing ends 2023-12-30, so the recent exam stops at WY2023 |
| Basin polygons | Not kept | SNOTEL-to-basin matching is by HUC8 or within 50 km, not inside the polygon |

## 3. Verification baseline (do not regress)

From `checks/*.json`; re-run `checks/run_all.sh` after any change.

- **Reference models:**
  - GR4J vs airGR 1.7.9 (compiled from CRAN in an isolated R environment): max |ΔQ| 1.0e-7 mm; NSE 0.79882 vs published 0.7988.
  - HBV vs TUWmodel 1.1-1: snow, soil and ET identical to ≤ 1e-12 mm; runoff NSE 0.9990 without routing and 0.9998 with it.
  - The published HBV-light example could not be obtained, because HBV-light is a Windows GUI.
- **Water balance:** 121 of 121 runs close (worst 1.4e-15 relative). Restarting from `final_state` reproduces one continuous run exactly.
- **Analytic checks:**

  | Check | Error / score |
  |---|---|
  | Linear-reservoir recession | 4e-15 |
  | UH S-curve | 6e-13 |
  | Nash cascade | 2e-10 |
  | Muskingum X = 0 vs ODE | 0.17 % |
  | Muskingum-Cunge vs diffusion wave | NSE 0.9999998 |
  | Kinematic wave | NSE 0.9998 |
  | Degree-day and EB terms | exact against hand calculations |
  | FAO-56 Example 18 | 3.88 vs 3.9 mm/day |

- **Reservoirs:** the storage-elevation power law fits the 15 dev dams with R² 0.999.
- **Weather generator:** wet-day frequency, persistence, annual variability and dry spells all match at a ratio of about 1.00.
- **Generator vs real dev signatures:** mean KS 0.14–0.18; 2–4 % of real basins fall outside the simulated cloud.
- **Snow vs SNOTEL** (33 dev stations, WY2011–25):

  | Model | Calibrated NSE | Uncalibrated NSE |
  |---|---|---|
  | Degree-day | 0.89 | 0.70 |
  | Energy balance | 0.93 | 0.82 |

- **Calibrated dev benchmarks** (52 basins; validation 1980–89 and 2008–14): median validation NSE 0.47–0.58. Snow basins reach 0.8–0.9, humid rain basins about 0.4, arid basins ≤ 0.
- **Speed:**
  - Numba core: 0.08–0.27 ms per basin-year.
  - Batch: 1,000 parameter sets × 10 years in 1.2–3.9 s.
  - `simulate()` per basin-decade: 11–65 ms.

## 4. Upstream candidates (physics found by Phase 2, currently only in each model's own folder)

Each was found by a twin test or a dev calibration. Fold them into `watersim` and re-run the checks.

1. **ERA5-Land water-yield multiplier** (FlowCast `src/calib_water.py`, `simbank.p_multiplier`):
   - FlowCast fit: `log m = −0.088 + 0.058·elev_km + 0.153·frac_snow + 0.397·log(P/3)`, σ 0.146.
   - BasinDetective fit (simpler): `log m ~ N(−0.076 + 0.30·frac_snow, 0.12)`.

   Reconcile the two into one documented prior.
2. **Gauge-noise model:**
   - BasinDetective `calib_gauge.round_sig`: ρ 0.999, cv 0.02, rounding to 3 significant figures in cfs (real records have more flat days than twins).
   - FlowCast found the default forcing error too small: a calibrated twin scores 0.88 against its clean run, while reality allows about 0.56.
3. **Forcing drift between decades:** about 6 % between 1980–89 and 1999–2008, plus about 10 % year-to-year precipitation error (BasinDetective v3/v4). Measure it per region and era.
4. **Snow at points (SnowSense):**
   - EB defaults: TURB 0.17 (default 1.0), ALB_DECAY 2.4 d (default 12), GHEAT 0.32. With the defaults, peaks are about 20 % low.
   - Degree-day TT is about +1.0 °C at points.
   - Add a perennial-snow guard: drop or flag windows with SWE > 5 mm on 1 Aug.
5. **Prior shift for flashiness:** the full dev shift (not half) matches flashiness 1.00×, but FlowCast networks trained on it lose NSE (rung r2). Keep both priors available.
6. **Reservoir task generator** (ReservoirPilot `gen.py`):
   - demand and Qmax derived from the operator
   - back-calculated inflow noise (negative inflow on about 2 % of days)
   - upstream regulation and data spikes
   - a non-stationary decade factor, exp(N(0, 0.3))
7. **Bug:** `realdata.random_forcing` window arithmetic. `span*365` days is shorter than `DateOffset(years=span)`, so a window can end a few days inside an exam period. The guard catches it with a `PermissionError`, so no exam data leaked, but fix the arithmetic.
8. **Doc note:** NSE* loss weighted by target std biases simulation-trained *ungauged* models low. The std depends on hidden parameters. Scale by an input-derived quantity instead, e.g. the Fu runoff ratio (FlowCast r0b).

## 5. Per-model state and next rungs

### FlowCast v1.0: lost to regionalisation overall

**Ungauged exam (WY1990–99, 619 basins):**

| Method | Median NSE |
|---|---|
| FlowCast | 0.341 |
| U-dev nearest-neighbour regionalisation | 0.372 |
| U-k12 | 0.385 |
| GR4J (gauged) | 0.473 |
| NWM | 0.550 |

- **Recent exam:** 0.378 vs U-dev 0.433.
- **Where it wins:** ties or wins in arid, regulated and large basins.
- **Where it loses:** snow (0.40 vs 0.56) and humid basins (0.49 vs 0.62).
- **Flood peaks:** −46 % bias.
- **Bands:** too wide (90 % band holds 97 % of days).
- **Gauged nowcast mode:** a clear win over persistence.

  | Lead | NSE | Persistence |
  |---|---|---|
  | 1 day | 0.569 | 0.475 |
  | 3 days | 0.474 | −0.19 |
  | 7 days | 0.426 | −0.47 |

**Next rungs:**
1. A snow rung driven by twins on the snowy dev basins: TT −0.66 °C, SFCF +0.11, and an elevation-dependent ERA5 temperature bias.
2. Per-basin water yield with uncertainty, or ERA5 SWE and soil-moisture states as inputs.
3. A peak head or high quantile, and more capacity so that flashier physics doesn't cost NSE.
4. Out-of-domain exams on GB, CL, BR and AUS.
5. Hourly flows.
6. Narrower bands.

### SnowSense v1.0: beat every ungauged baseline

**Exam (621 stations):**

| Method | Median NSE |
|---|---|
| SnowSense | 0.888 |
| Uncalibrated energy balance | 0.853 |
| Uncalibrated degree-day | 0.691 |
| Climatology | 0.701 |
| Per-station calibrated degree-day (ceiling) | 0.925 |

- **Other errors:** peak SWE 17.5 % abs (bias −9 %), melt-out 8.3 d, peak date 3 d.
- **Weakest:** stations below 1,500 m (0.79) and the Southwest (0.78).
- **Bands:** conservative (68 % band holds 78 %).

**Next rungs:**
1. A snowfall-undercatch prior that depends on wind or exposure, to fix the peak bias.
2. Rain-on-snow and thin low-elevation packs: sub-daily melt-refreeze, or rain energy in the degree-day routine.
3. A gridded ERA5-band mode, so SnowSense can feed FlowCast (the optional SnowSense → FlowCast test was not done).
4. An initial-snow input for stations that keep snow through summer.
5. Checks across the SNOTEL temperature-sensor change.
6. More Alaska stations.

### BasinDetective v1.0: tied GR4J, regulation failed

**Exam, change between decades (Spearman ρ):**

| Signature | BasinDetective | GR4J |
|---|---|---|
| Mean flow | 0.76 | 0.79 |
| Runoff ratio | 0.45 | 0.50 |
| BFI | 0.20 | 0.23 |
| Q5 | 0.44 | 0.42 |
| Q95 | 0.71 | 0.72 |
| Half-flow date | 0.63 | 0.57 |

- **What it has that GR4J doesn't:** no per-basin fitting, and error bars that hold (they are a bit wide).
- **Validation-decade levels:** clearly better than B's detective table (BFI ρ 0.98 vs 0.79; Q5 0.975 vs 0.78).
- **Regulation detection fails on real basins:** AUC 0.49 against storage > 10 % of annual flow (simulated AUC 0.91).
- **Hidden properties on twins:** ρ 0.70–0.79.

**Next rungs:**
1. Realistic reservoirs in the training bank: fitted to the ResOpsUS dev dams, with hydropower peaking and twins of the regulated dev basins.
2. Per-region and per-era forcing drift, measured on both non-exam windows. This should also fix the band width.
3. A small sequence encoder next to the features, plus an ungauged attribute-only mode.

A full version needs real truth for storage and groundwater; GRACE or well data are candidates.

### ReservoirPilot v1.0: big win, with a caveat

**Exam (244 ResOpsUS dams), median cost J:**

| Policy | Median J |
|---|---|
| ReservoirPilot | 0.0154 |
| Sim-tuned classical rule (hedging + flood pool) | 0.0285 |
| SOP | 0.108 |
| Rule curve | 0.143 |
| Historical operations | 0.183 |
| Perfect-foresight DP | 0.0063 |

- **CDEC recent exam:** 0.019 vs SOP 0.095.
- **Win rates:** beats SOP on 84 % of dams and historical operations on 95 %.
- **Caveat:** J charges nothing for storage, so the policy holds reservoirs lower (median 38 % of capacity vs SOP's 59 %). It ends at least 20 % of capacity lower on 34 % of dams. It loses to SOP on 15 % of dams, mostly large supply reservoirs whose scored decade was dry.
- **Method:** the network is trained by back-propagating J through a differentiable reservoir simulator. Behaviour cloning of DP lost.

**Next rungs:**
1. A storage-aware objective, or a storage-floor input, to show the price of holding reservoirs low.
2. MPC, or an inflow-forecast input; FlowCast's nowcast is a natural partner.
3. Adaptation to a drier or wetter decade (an inflow EMA).
4. Cascades, evaporation, minimum flows and an hourly flood mode.

## 6. Rule notes, reported honestly

- **Dev data used to fit physics and also to gate rungs (partly in-sample):**
  - FlowCast's 4 water-yield coefficients.
  - BasinDetective's decadal-drift mean.
  - SnowSense's parameter prior (dev SWE up to WY2010; gate on WY2011–25).

  The exam was untouched in every case.
- **Free-window forcing from exam basins** was used in training banks, as the brief allows. No exam Q, SWE or reservoir data was used for training.
- **Climate statics at exam time.** FlowCast computes them from the forcing of the window being predicted, which is exam-period forcing but contains no Q.
- **ReservoirPilot:**
  - Used aux-dam statistics (6 complete dams, plus inflow ratios on 29 aux dams) to set the generator's ranges.
  - Its climatology features use each exam dam's calibration-half inflow, as B's rule curve does.
  - Two void training runs (the optimiser step was missing) were removed and logged.
- **SnowSense:**
  - R6 was initialised from the rejected R5 weights.
  - The demo was run on 3 exam stations before the final exam, for code testing only.
  - One process peaked at about 1.34 GB, above the 1.2 GB guidance.
- **Ratchet noise:** sim-metric differences of about 0.01 are noise-level for FlowCast. A 300-basin subset and the full 800 disagreed on rung r5. Use larger frozen sets.

## 7. Suggested order for the code agent

1. **Daymet/NLDAS forcing.** Re-run every exam with it to separate method skill from ERA5-Land bias. The single biggest gap.
2. **Upstream the physics in section 4** into `watersim` v1.1. Version the frozen sets and re-score every rung on them.
3. **FlowCast:** a snow rung, then out-of-domain exams on non-US regions.
4. **SnowSense → FlowCast coupling.**
5. **ReservoirPilot:** a storage-aware objective, then MPC with the FlowCast nowcast.
6. **Hourly flash-flood track:** NWIS instantaneous values plus hourly forcing.
7. **Wire into the repo:**
   - `systems/watershed/` with `watersim`, `waterdata` and the four models.
   - Artifact pages to GitHub Pages.
   - CI running `checks/run_all.sh` with a short subset.
