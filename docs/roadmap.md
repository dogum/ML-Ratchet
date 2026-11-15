# Roadmap

Where the collection goes next, beyond each model's own next rungs (in its `model.yaml` under `next`).

## Exams waiting on data (help wanted)

- ~~**ChirpNet O3 exam.**~~ Done 12 Nov 2026 once GWOSC was reachable: 7 of 7 in-range events found, every catalogue
  median inside the 90% posteriors; GW190521 (outside the training range) missed; one O3 background window above
  threshold. See `models/chirpnet/training/logs/o3_exam.json`.
- **CellSense, other chemistries.** Partly done 12 Nov 2026: a second, separate network for NMC811 (LG M50T), from
  Pozzato et al. 2022 (OSF, CC BY 4.0), reading slow C/20 discharges cut short. Health only: those cells lost 3-9% and
  never reached end of life. See `models/cellsense/MODEL_CARD.md` (v2). Still waiting: Sandia NCA / NMC / LFP cycling
  (Preger et al. 2020; BatteryArchive asks for an email) for fast-rate curves, other temperatures and lifetime, and
  Zhu et al. 2022 (Zenodo) for a second NMC/NCA cell.

## The collection as a tool

- **pip-installable models.** `pip install ml-ratchet[quakenet]`, with weights fetched from GitHub Releases on first
  use instead of shipping in the wheel.
- **A physics MCP server.** Each model becomes one tool that returns its readings *with* error bars and the card's
  limits: `chirpnet.scan`, `sundoctor.diagnose`, `cellsense.health`, `transitnet.read`, `quakenet.read`,
  `bearingnet.diagnose`, `swellcast.read`, `beatnet.classify` (with its not-a-medical-device notice),
  `starquake.read`, for the vehicle models `energynet.predict`, `tripdetective.read_car`, `hybridpilot.compare`, and
  for the grid platform `gridsim.dispatch`, `gridsim.frequency_event`, `gridpulse.read`, `batterypilot.run`,
  `plantdetective.read`, `demandcast.forecast`. `mlratchet.load()` already imports any model in isolation, which is most of the work.
- **One design language for the labs.** The interactive pages were each built with their own visual identity.
  The website now wraps them in one system; porting each lab natively into the site (shared JS network runtime,
  shared chart components) is the next step.
- **A shared JS runtime** for in-browser networks: numpy-equivalent conv/GELU/BatchNorm in TypeScript, with parity
  tests against the Python modules.

## The method

- **A reusable twin-test harness** in `mlratchet`, since it found the biggest simulator gaps in three projects.
- **Per-stratum σ widening** (StarQuake showed one pooled factor isn't enough across observing lengths).
- **Saliency on real vs twin** inputs, to show where a network looks when it misreads real data.

## The vehicle platform

What is left to make it complete is in [shared/vehicle/HANDOFF.md](../shared/vehicle/HANDOFF.md); its status as of
13 Nov 2026 is in [shared/vehicle/STATUS.md](../shared/vehicle/STATUS.md). In short: move the models' dev-calibrated
wrappers into RoadSim (a new physics version), rebuild the frozen sets and re-score every rung; a `mlratchet data
vehicle` command that rebuilds RoadData from the raw downloads; real heavy-duty traces for at least two truck and
three bus types.

## The grid platform

GridSim, GridData and four models (GridPulse, BatteryPilot, PlantDetective, DemandCast) came in on 14 Nov 2026; see
[grid-platform.md](grid-platform.md). What is left is in [shared/grid/HANDOFF.md](../shared/grid/HANDOFF.md), in order:

1. **The possible exam leak.** GridSim's default outage rates cite Texas RE 2024/2025 aggregates, published during
   the exam years. Replace them with pre-2023 NERC GADS values, re-run the checks, re-score.
2. **Upstream each model's fixes behind a flag** (GridPulse's per-technology aggregation and two-segment Dynamic
   Containment, BatteryPilot's ORDC adder and cold-outage curve, PlantDetective's self-commitment layer and soft history
   constraints, DemandCast's building-stock demand model and tabulated farm curves), and fix GridData's 3 Dec 2019 dev
   label (1,788 MW labelled, about 850 MW implied).
3. **GridSim v2:** coal/gas commitment, scarcity pricing and low-net-load inertia inside their "done when"; era-aware
   frequency response; one more region (PJM or CAISO) dispatched and calibrated like ERCOT. Then every grid model's
   frozen set rebuilt and every rung re-scored, each with a published win/loss table.
4. **Data:** a day-ahead forecast-weather archive (Open-Meteo previous runs; rate-limited from the cloud sandbox,
   reachable from the repository's own machine), sub-second frequency (PMU or FNET/GridEye), CAISO prices before
   2023-08, the 17 missing ERCOT monthly reports, CAMPD for more states than Texas.
5. **DispatchNet,** a learned dispatcher graded on EIA-930 generation by fuel, real prices and speed, once the coal/gas
   gap is closed.
6. **A control-room page:** one ERCOT day with the demand forecast, the dispatch, a trip and a battery's response
   together.

## New models

The power-grid project is in (above). The next big-system candidates are BuildingSim (NREL ResStock/ComStock, Building
Data Genome 2; DemandCast's building-stock model is a head start), Watershed (USGS NWIS, SNOTEL) and WakeNet (Kelmarsh
and Penmanshiel SCADA). See [backlog.md](backlog.md) for them and the other candidates with physics, data and an exam
lined up.
