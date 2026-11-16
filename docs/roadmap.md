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
  The website now wraps them in one system, and every model runs live on its page; porting each original lab natively
  into the site (shared chart components) is the next step.
- ~~**A shared JS runtime** for in-browser networks.~~ Done 12 Nov 2026: `site/src/lib/nn` (numpy-equivalent layers in
  TypeScript) runs all 16 instruments, each with parity tests against its Python module (`npm test` in `site/`).

## The method

- **A reusable twin-test harness** in `mlratchet`, since it found the biggest simulator gaps in three projects.
- **Per-stratum σ widening** (StarQuake showed one pooled factor isn't enough across observing lengths).
- **Saliency on real vs twin** inputs, to show where a network looks when it misreads real data.
- **Gate each dev year separately.** BatteryPilot v1.3 passed a dev gate pooled over 2019 and 2021, carried by Winter
  Storm Uri alone, and lost ground on the exam (15 Nov 2026).
- **Commit every split as a file.** The 15 Nov 2026 pass could not rebuild TransitNet's v1.0 split or QuakeNet's dev
  events from today's catalogues (one v1.0 TransitNet exam planet now lands in dev), and SwellCast's dev storms had to
  be recovered from a results file. Each needs a frozen, committed list before its next rung can be judged.
- **Unique case keys in every scorer.** HybridPilot's exam held two trips under one id and graded them against each
  other's baselines (found and corrected 15 Nov 2026); scorers now refuse duplicate keys.

## The vehicle platform

RoadSim physics v2 (13 Nov 2026) took in the models' dev-calibrated wrappers, `mlratchet data vehicle` rebuilds
RoadData from the raw downloads, Argonne D3 and the full ZTBus set are in, and the three models were rebuilt on v2 on
14 Nov 2026; see [vehicle-platform.md](vehicle-platform.md). Status as of 14 Nov 2026:
[shared/vehicle/V2_PLAN.md](../shared/vehicle/V2_PLAN.md) (decisions log) and
[shared/vehicle/roadsim/LIMITS.md](../shared/vehicle/roadsim/LIMITS.md). What is open:

1. **Real heavy-duty traces.** NREL Fleet DNA (truck and bus traces with fuel) is still blocked from scripts; heavy
   duty needs at least two truck and three bus types graded on real traces.
2. **The hybrid gaps after v3.1.** TMED P2s 8-10 % too thirsty, converter P2s +15 to +28 % too frugal in the city,
   older Toyota/Lexus hybrids +14 to +44 % on the FTP, plug-ins on VED roads 20-23 % low, hybrid A/C. No clean judge
   is left for a further hybrid pass: the next one needs new data first. RoadSim also never runs a mild or P2 engine to
   charge at a stop, which drains packs on long stops; HybridPilot's mild charge holding (79 % of exam trips) cannot be
   worked on until that is modelled and its frozen set has a mild stratum.
3. **Heavy-duty traces into the models:** EnergyNet's trucks and Mersin buses still run on cycles and synthetic speed
   profiles.

## The grid platform

GridSim, GridData and four models (GridPulse, BatteryPilot, PlantDetective, DemandCast) came in on 14 Nov 2026; see
[grid-platform.md](grid-platform.md). The first follow-up round (merged 15 Nov 2026) fixed the outage-rate exam leak
as GridSim physics 1.1, rebuilt BatteryPilot as v1.1, re-scored PlantDetective, corrected the GB 3 Dec 2019 dev label
(1,788 to 854.7 MW), upstreamed four fixes behind flags and fetched the Open-Meteo forecast-weather archive. The second
round (15 Nov 2026) added GridSim physics 1.2, opt-in real-time scarcity the day-ahead market cannot see, and
BatteryPilot v1.3 on it, which came out worse than v1.1 on the exam. Status and the full list:
[shared/grid/HANDOFF.md](../shared/grid/HANDOFF.md). What is open, in order:

1. **Six GB exam events** carry the same interconnector-ramp pattern as the corrected dev label. They are listed in
   `shared/grid/data/DATA.md` section 5; changing exam labels needs a documented decision, not a quiet fix.
2. **The new-physics upstream items:** Dynamic Containment two-segment droop, year-aware priors, ORDC scarcity
   pricing in core, the cold-outage curve, gas-turbine ambient derate, a day-ahead price generator, a self-commitment
   layer and DemandCast's building-stock demand model.
3. **GridSim v2:** coal/gas commitment (held-out 2022 coal is +34 % on physics 1.1), scarcity pricing in the dispatch
   itself (54 hours above $200 against 238 real; physics 1.2 covers only the real-time layer, needs a CAISO fit and
   year-to-year tightness, and under-rates the day-ahead optimiser on dev twins), low-net-load inertia and era-aware frequency response; one more region dispatched and
   calibrated like ERCOT. Then every grid model's frozen set rebuilt and every rung re-scored, each with a published
   win/loss table.
4. **Data:** sub-second frequency (FNET/GridEye has no public download), CAISO prices before 2023-08, the 17 missing
   ERCOT monthly reports, CAMPD for more states than Texas.
5. **DispatchNet,** a learned dispatcher graded on EIA-930 generation by fuel, real prices and speed, once the coal/gas
   gap is closed.
6. **A control-room page:** one ERCOT day with the demand forecast, the dispatch, a trip and a battery's response
   together.

## New models

The power-grid project is in (above). The next big-system candidates are BuildingSim (NREL ResStock/ComStock, Building
Data Genome 2; DemandCast's building-stock model is a head start), Watershed (USGS NWIS, SNOTEL) and WakeNet (Kelmarsh
and Penmanshiel SCADA). See [backlog.md](backlog.md) for them and the other candidates with physics, data and an exam
lined up.
