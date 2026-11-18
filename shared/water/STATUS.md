# Watershed project (WaterSim) status, 18 Nov 2026

This is Greg's fourth big-system project for ML Ratchet, built like RoadSim, GridSim and BuildingSim. It covers rivers, snow, floods and reservoirs.
- **Phase 1:** two parallel agents built a shared simulator (WaterSim) and a shared real-data package (WaterData) against a written contract.
- **Phase 2:** four parallel agents built the models.
- **Handoff:** `watershed-handoff.md`, also in the zip as `HANDOFF.md`.

## Deliverables
- **`watersim-waterdata-v1.0.zip` (10 MB)** contains:
  - the `watersim` package, its checks and the two briefs;
  - the `waterdata` loader and the frozen splits;
  - attributes and specs for 3,278 basins;
  - baseline scores;
  - fetch scripts.

  The real time series are not included. They are rebuildable with `data/scripts/`: about 2.9 GB from Caravan, NWIS, NWM, SNOTEL, ResOpsUS and CDEC.
- **Models**

  | Model | Page | Zip | Status doc |
  |---|---|---|---|
  | FlowCast v1.0 | https://claude.ai/artifact/7iJtRbBuCUfAhmm6eo7SvH | `flowcast-v1.0.zip` | `claude/flowcast-status.md` |
  | SnowSense v1.0 | https://claude.ai/artifact/QQdrSbSKeN7SH2eqLFLbEW | `snowsense-v1.0.zip` | `claude/snowsense-status.md` |
  | BasinDetective v1.0 | https://claude.ai/artifact/SbwK1wQbrkWUz56dnaQAUk | `basindetective-v1.0.zip` | `claude/basindetective-status.md` |
  | ReservoirPilot v1.0 | https://claude.ai/artifact/JNZJvZe5RT4C1A4XvHDvGt | `reservoirpilot-v1.0.zip` | `claude/reservoirpilot-status.md` |

- All demos, and a WaterSim smoke test, were run from unzipped copies on 18 Nov. The pages are private until shared.
- `watershed-status-docs.zip` holds all six docs as files.

## Phase 1: WaterSim
- **Physics**
  - Snow: degree-day with elevation bands, refreeze and liquid water, or a daily energy balance.
  - Runoff models:
    - HBV (matches TUWmodel);
    - GR4J (a port of airGR; matches to 1e-7 mm);
    - SAC-SMA;
    - VIC-lite.
  - PET: FAO-56 Penman-Monteith, Priestley-Taylor, Hargreaves or Oudin.
  - Routing:
    - unit hydrographs;
    - Muskingum-Cunge;
    - kinematic wave;
    - reach networks.
  - Reservoirs:
    - storage-area-elevation curves;
    - spill and evaporation;
    - hydropower;
    - operating policies, or any function you supply;
    - a `step()` API for reinforcement learning.
  - Irrigation and hourly infiltration excess.
- **Checks**
  - 121 of 121 water balances close to within 1e-15.
  - Analytic routing tests pass, as does FAO-56 Example 18.
  - Snow against SNOTEL (dev stations) gives NSE 0.89 calibrated and 0.93 for the energy balance.
- **Speed:** the numba core needs 0.1–0.3 ms per basin-year; batches of 1,000 parameter sets run in a few seconds.
- **Generator:** four latent factors drive the parameters of all four models. Its priors are corrected on dev basins. It also has a stochastic weather generator and reuses real forcing from non-exam windows only.
- **Calibrated dev benchmark:** median validation NSE 0.47–0.58 with ERA5-Land forcing.

## Phase 1: WaterData
- **Caravan v1.6** (CC-BY):
  - CAMELS-US: 671 basins.
  - Also GB 671, CL 505, BR 870 and AUS 561 basins.
  - ERA5-Land forcing ends 2023-12-30.
- **USGS NWIS:** daily flows to Nov 2026, plus 48,769 flood peaks.
- **NWM v3.0 retrospective** at 650 gauges.
- **SNOTEL:** 880 stations.
- **Reservoirs:** ResOpsUS (678 dams, plus GRanD) and CDEC (18 dams).
- **Splits** (seed 20261118):
  - CAMELS-US basins: 52 dev, 619 exam. Dev basins are stratified by HUC2, snow, aridity and regulation.
  - SNOTEL stations: 39 dev, 772 exam.
  - Reservoir dams: 15 dev, 250 exam.
  - Exam windows: WY1990–99 is the main exam and WY2016–23 the recent one.
  - Ungauged protocols: U-dev and U-k12.
- **Exam baselines** (median NSE, main window):

  | Baseline | NSE |
  |---|---|
  | Climatology | 0.065 |
  | Persistence | 0.475 |
  | GR4J calibrated per basin | 0.473 |
  | Nearest-neighbour regionalisation | 0.372 |
  | NWM | 0.550 |

  - SNOTEL: calibrated degree-day 0.925.
  - Reservoir cost J: SOP 0.108, perfect-foresight DP 0.006.

## Phase 2 results (exam, run once)

| Model | Result | Against |
|---|---|---|
| FlowCast (ungauged) | NSE 0.341 | regionalisation 0.372 (loses), NWM 0.550 |
| FlowCast nowcast (gauged), 1/3/7-day lead | 0.569 / 0.474 / 0.426 | persistence 0.475 / −0.19 / −0.47 |
| SnowSense | NSE 0.888 | uncalibrated energy balance 0.853, degree-day 0.691, calibrated ceiling 0.925 |
| BasinDetective, decadal change (ρ) | ties GR4J, e.g. mean flow 0.76 vs 0.79; Q5 0.44 vs 0.42 | levels: BFI 0.98 vs GR4J 0.79 |
| BasinDetective, regulation | AUC 0.49 (fails) | simulation AUC 0.91 |
| ReservoirPilot | J 0.0154 | SOP 0.108, historical 0.183, DP 0.0063 |

ReservoirPilot's win comes with a caveat: J does not charge for holding storage low.

## Lessons (cross-system)
- **The biggest real-world gap is the forcing, not the physics.** ERA5-Land undercatches rain and snow in the mountains and drifts between decades. Twin tests in all three streamflow-facing models found it, and it caps every method, the classical ones included.
- **Where simulation training wins:**
  - well-posed physics read from local measurements (SnowSense);
  - control problems with an exact cost (ReservoirPilot).
- **Where it ties or loses:**
  - ungauged streamflow, where regionalisation borrows real behaviour from similar gauged basins;
  - anything human (regulation detection).
- **NSE\* loss with target-std weighting** biases sim-trained ungauged models low. Scale by a quantity derived from the inputs instead.

## Next steps
See `watershed-handoff.md`, section 7. In short:
1. Daymet/NLDAS forcing.
2. Upstream the twin-test physics into WaterSim v1.1.
3. A FlowCast snow rung and out-of-domain regions.
4. SnowSense → FlowCast coupling.
5. A storage-aware objective and MPC for ReservoirPilot.
6. An hourly flash-flood track.
