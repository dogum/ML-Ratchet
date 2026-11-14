# Vehicle platform v2: plan

Goal: take RoadSim and the three vehicle models from a first baseline to the strongest part of ML Ratchet. Every
known gap gets a measured target ("done when"), is calibrated on **dev only**, and is checked on held-out data. Every
physics change bumps `PHYSICS_VERSION`, rebuilds the frozen sets and re-takes every ratchet decision. The splits
(`data/proc/splits.json`, sha256 d917df78…) never change. Started 13 Nov 2026; the work list it draws on is
[HANDOFF.md](HANDOFF.md) and [roadsim/LIMITS.md](roadsim/LIMITS.md).

## What changed since v1

- **Compute.** v1 was built on 2 shared sandbox cores. This machine has 16 cores and a CUDA GPU (RTX 3080, 10 GB):
  banks that took 2 h take minutes, and the networks can be larger and trained longer.
- **Data now reachable** (checked 13 Nov 2026 from this machine): VED (GitHub), USGS 3DEP lidar, IEM weather, EPA,
  fueleconomy.gov, and the **ETH Research Collection (full ZTBus, 1,409 missions — v1 had 10)**.
- **Still blocked from scripts:** Argonne D3 (anl.gov, HTTP 403) and NREL Fleet DNA / data.nrel.gov (timeout).
  Not routed around; Greg can download D3 in a browser.

## Phase A — foundation

1. `mlratchet data vehicle`: one resumable command that downloads the raw sources and runs `data/scripts/` in order,
   then checks the splits hash and that `baselines.json` matches the shipped one (HANDOFF P0.3).
2. A training extra (`pip install -e ".[train]"`: torch with CUDA). Inference stays numpy-only.
3. Freeze the v1 scoreboard: every check, every model's frozen-set skill and exam numbers, in `checks/v1_baseline.json`.

## Phase B — RoadSim physics v2 (each fix calibrated on dev, judged on held-out)

| Gap (v1) | Fix | Done when |
|---|---|---|
| Wrappers living in model folders | Upstream: MAF fuel-cut label model, on-road loads, PHEV hardware, OBD logger emulator, VED-era mass prior; matched-SOC grading the default | One copy, behind flags, models import it |
| Toyota HEV: engine on 35% vs 55% real; fuel 0.80× logged | Production-style calibration: engine on above ~72 km/h, warm-up and catalyst holds, start penalty | Twin fuel ±8%, engine-on share ±10 points (VED dev) |
| Leaf twin 1.5× real energy; PHEV twins 2–3.6× fuel | PTC heater curve and cabin model from VED's logged heater power; electric cabin heat and CD-mode rules for PHEVs | BEV twin ±10% per trip; PHEV total ±15% |
| P2 city FTP +8 to +31% | Torque converter in P2 automatics, engine-start and clutch-engagement energy, minimum on-time | Held-out P2 FTP ±8% |
| e:HEV HWFET −9% | Direct-drive clutch at cruise | ±6% |
| Cold FTP +10–16%, SC03 A/C +9–15% | Enrichment, driveline oil and tyre warm-up vs ambient; belt A/C compressor + condenser fan + hot-soak pull-down | Held-out cold FTP and SC03 ±6% |
| Conventional tilt FTP +5.6% / HWFET −4.3%; per-engine ±5–10% | Low-load losses, converter slip; per-family `eff_scale` fitted from EPA dev FTP/HWY | Both medians ±3%; IQR halved |
| Generated hybrids 20% too thirsty | Hybrid bodies drawn from efficient donors | Generator HEV FE matches EPA |
| Bus +76% vs literature 10–40%; BEV Class 8 2.5 kWh/mi | Bus accessories and door-open HVAC; light-load motor losses on large machines | Inside published ranges; full ZTBus |
| 1 Hz, no tyre, perfect driver, steady wind | **10 Hz sub-step mode** for shifts, starts and tip-ins; **tyre model** (slip, load transfer, friction ellipse, EU Crr classes) setting regen limits; **driver model** (lag, overshoot, eco ↔ aggressive); gusts and road-height wind | Ledger still closes; 1 Hz results unchanged within 0.5% where the new physics is off |
| Battery: no aging | Capacity fade and resistance growth (CellSense's model), SOC-window effects | Documented, with a check |

Kept from v1 and re-run after every change: ledger closure (600 stress runs), EPA held-out checks, FASTSim cross-check.

## Phase C — more real data

- Full ZTBus (1,409 missions, passengers per second): EnergyNet's bus exam, TripDetective's bus mass from passengers.
- Argonne D3 (10 Hz dyno tests of modern HEVs, PHEVs, BEVs with fuel, current, engine speed, temperature): **needs
  Greg's browser download.** Best calibration data for P2, e:HEV, cold and A/C. Dev and a second exam, split by vehicle.
- A model-year input for EnergyNet from fueleconomy.gov, so modern cars stop inheriting the old-fleet prior.

## Phase D — rebuild and re-score the three models on physics v2 (GPU)

- **EnergyNet:** fix the hybrid (+12%) and Leaf (−20%) biases through the twins, per-architecture σ, cold-start and
  model-era inputs, buses on full ZTBus. Target: beat MOVES-like on hybrids and EVs; 68% bands cover 60–75% on VED.
- **TripDetective:** make mass and drag identifiable (a physics-regression head that separates mass from efficiency,
  pooled multi-trip training, sharper channels), rebuild priors on v1 + the v3 mass prior, fix BEV heat. Target: mass
  responds to +800 kg by ≥ 500 kg; Leafs within 150 kg; beat the rule on powertrain type.
- **HybridPilot:** re-teach on the calibrated twins, a distance-to-go PHEV teacher, a drivability term, a causal DP
  proxy with minimum on/off times. Target: a clear win over A-ECMS overall, and the simulator–real gap under 8%.

Every rung logged, kept or not; new win/loss tables; instruments re-exported.

## Phase E — the garage

One page where a visitor picks a vehicle and a real route and sees all three models on the same trip: EnergyNet's
forecast, the RoadSim run, TripDetective reading the log back, HybridPilot driving it.

**Done 14 Nov 2026.** Page `site/src/pages/garage.astro` (in the site nav; a button on the three model pages), code in
`site/src/garage/`, data in `site/public/garage/` (6.5 MB) written by `shared/vehicle/garage/export_garage.py`.
- Grid, RoadSim physics v2 pinned, 156 runs: six vehicles (RoadSim's calibrated twins of VED exam cars 470 and 10,
  HybridPilot's twins of VED exam cars 428 and 492, a generated P2 hybrid, the ZTBus trolleybus twin) x five real exam
  routes (the four EnergyNet / HybridPilot example trips of 10 minutes or more, one ZTBus exam leg drawn with seed
  2026; the bus drives only that one) x air temperature -10 / 20 / 35 C (EnergyNet's own perturb) x load +0 / +400 kg
  (bus +60 riders). A third load level would have taken it past 8 MB: HybridPilot's inputs at every decision second
  dominate the size.
- Per run: ledger, traces, the VED-style OBD log TripDetective reads, and for the three hybrids HybridPilot, A-ECMS,
  the rule and DP graded as HybridPilot's exam grades. The networks are the instruments' own ports, run live.
- Parity (`site/tests/garage.test.ts`): 156 EnergyNet cases worst 9.5e-8, 83 TripDetective cases 6.4e-5, 30
  HybridPilot cases 6.4e-15 (tolerance 2e-3). HybridPilot's s from the stored float16 inputs is within 2.6e-3 of the
  run's own.
- What it shows, against RoadSim (not reality), median [range] over each vehicle's runs: EnergyNet vs RoadSim energy
  sedan -9.7 % [-24, +4] (low with +400 kg, about -11 %, and at 35 C), Leaf -1.5 % [-12, +16], bus -6.8 % [-16, +2], power-split +3.9 %
  [-14, +18], P2 +15.5 % [-5, +57], plug-in started near charge sustaining +33 % [-29, +57]; inside its 68 % band on 119
  of 156 runs. HybridPilot minus A-ECMS, points of DP: power-split +0.82 [-0.19, +5.91], P2 +4.51 [-12.5, +13.2],
  plug-in +4.64 [+0.84, +8.86]. TripDetective's single-trip mass: median |error| 112-268 kg by vehicle.

## Decisions log

- **13 Nov 2026 — ZTBus split for the full set (owner's decision).** v1's rule left 2 of 1,409 missions as dev. The
  by-month split (fixed 13 Nov, before any full trace was processed) is now in force: 294 dev missions, 1,115 exam; the
  v1 trace-exam sample missions stay exam. EnergyNet v1's numbers on ZTBus mission totals are re-scored on the 1,115
  exam missions before comparing with v2. Details: data/DATA.md, proc/hd/ztbus_full_splits.json.
- **13 Nov 2026 — physics v2 frozen and made the default (calibration agent 2).** `simulate()` without options runs
  `roadsim-physics-v2` (PHYSICS_VERSION); `options='roadsim-physics-v1'` or `None` reproduces v1 exactly (fingerprint
  test). Every RoadSim call under models/ is pinned to v1, so the three released models, their demos and
  tests/golden are unchanged until Phase D rebuilds them. VED CR-Z twins are P1 mild hybrids and are excluded from the
  hybrid targets (low-confidence label). Decisions: roadsim/CALIBRATION.md, rungs H0-H9.
- **14 Nov 2026 — physics v2.1 frozen, not made the default (calibration agent 3).** A hybrid fix pass on a wider dev
  set (55 EPA dev-family hybrid / plug-in records, D3 dev hybrids and plug-ins at 10 Hz, VED dev twins), with a fresh
  exam frozen first (`checks/exam2_hybrids.json`: 108 VED exam HEV/PHEV cars, 409 unseen EPA hybrid records) and a v2
  fingerprint guard. Kept: engine run-on into braking (2 s, measured on six D3 dev hybrids). Rejected: every physical
  replacement of the P2 brake-blending share, P2 engine rules, Toyota speed-rule changes, plug-in CS rules, cold and
  heat thresholds (CALIBRATION.md H12-H20). `options='roadsim-physics-v2.1'` runs it; PHYSICS_VERSION stays
  roadsim-physics-v2 because the dev fuel/energy gain is not clear (the gain is in engine operation: VED Toyota
  engine-on gap -10.9 -> -4.9 points). models/ stay pinned to v1.
