# The grid platform: GridSim + GridData

The collection's second **shared platform**: one verified power-system simulator and one real-data package, under four
models — [GridPulse](../../models/gridpulse/), [BatteryPilot](../../models/batterypilot/),
[PlantDetective](../../models/plantdetective/) and [DemandCast](../../models/demandcast/). New power-grid models start
here instead of from scratch. It was built like the [vehicle platform](../vehicle/): Phase 1 built the simulator and
the data package in parallel against one contract ([GRID_BRIEF.md](GRID_BRIEF.md)); Phase 2 built the four models on
top ([PHASE2_BRIEF.md](PHASE2_BRIEF.md)).

```
shared/grid/
├── gridsim/        GridSim, the power-system simulator (Python package `gridsim`, numpy/scipy/pandas)
│   ├── gridsim/      the package: network, fleet, dispatch, frequency, renewables, demand, generator
│   ├── README.md     physics, API, verification numbers      LIMITS.md   what it gets wrong, measured
│   ├── tools/        build_heat_rates.py
│   ├── refdata/      MATPOWER cases, EIA-923 fuel receipts and heat rates, Henry Hub (git-ignored; restored by the data command)
│   └── demo.py       one ERCOT day dispatched + a 1,300 MW trip at the evening peak
├── griddata/       GridData's loader (Python package `griddata`, pandas/numpy/pyarrow): EIA-930, fleets, CAMPD,
│                   frequency events and traces, prices, weather, test networks, baselines
├── data/           DATA.md (sources, cleaning, splits, baselines), build.py (the data command), scripts/,
│   │               MANIFEST.json (sha256 of every shipped v1.0 data file), sources/ (URL lists)
│   └── proc/splits.json   the frozen dev / exam split — content sha256 895b4e9a…; never change it
├── checks/         every physics check, its JSON (v1 numbers) and run_all.py (re-run + compare)
├── HANDOFF.md      what exists, what's missing, and the prioritised path to v2 — start here to extend it
├── STATUS.md       the platform's status at release (14 Nov 2026)
├── GRID_BRIEF.md   the shared contract: time series, regions, generator spec, simulator API, verification
└── PHASE2_BRIEF.md how the four models were set up
```

## Get the data

The data are not in git (about 290 MB processed; 1.7 GB with raw downloads and CAMPD). One resumable command fetches
or unpacks them into `shared/grid/data/` and `shared/grid/gridsim/refdata/`:

```bash
pip install -e ".[grid-data]"
mlratchet data grid --from-zip gridsim-griddata-v1.0.zip   # unpack the v1.0 release and verify all 1,064 files (exact)
mlratchet data grid                                        # or: download every source and rebuild (hours; resumable)
mlratchet data grid --only verify                          # sha256 of every file vs data/MANIFEST.json
mlratchet data grid --list                                 # the stages
```

Set `ML_RATCHET_GRID` to keep the data somewhere else: it is the one root that holds `data/`, `gridsim/refdata/` and
`checks/out/` (default: this folder). The rebuild never writes `proc/splits.json`: `make_splits.py` must reproduce its
content hash, and the frozen file is restored byte for byte either way. A rebuild from live hosts is not byte-identical
to v1.0 (EIA revises EIA-930; NESO and ERCOT re-issue files), so the verify stage lists the files that differ.

## Use it

```bash
pip install -e ".[grid]"            # from the repo root: makes `gridsim` and `griddata` importable anywhere
cd shared/grid/gridsim && python demo.py
python shared/grid/checks/run_all.py --quick     # re-run the checks and compare every number with v1
```

```python
import griddata as gd
from gridsim import dispatch, frequency_event, powerflow, generate_states
sp = gd.splits()                                   # frozen dev / exam splits
ev = gd.events('GB', split='dev')                  # 4 GB dev events (1 s traces via gd.event_trace)
d = gd.eia930('ERCO', split='dev')                 # hourly demand, day-ahead forecast, generation by fuel
```

Without installing, add `shared/grid/gridsim` and `shared/grid` to `sys.path`. Model training scripts find both through
their `training/_paths.py`.

## What it is

- **Power flow and dispatch:** AC/DC power flow, PTDF and DC-OPF, exact against PYPOWER on IEEE 9-300 and ACTIVSg200/
  2000 (angles to 1e-11°, flows to 3e-10 MW, DC-OPF cost to 3e-13). Unit commitment and economic dispatch as MILP, LP
  or merit order, with ramping, minimum up/down, starts, reserves, storage, hydro budgets and a network.
- **Frequency dynamics:** multi-area swing equation, governors per technology, fast frequency response, synthetic
  inertia, RoCoF relays and under-frequency load shedding. Analytic cases exact; the ANDES Kundur two-area case within
  15 mHz RMS. GB 9 Aug 2019: first nadir 49.11 Hz at 24.5 s (reported 49.1 Hz, 24.5 s), demand disconnection at
  75.6 s (75.9 s).
- **ERCOT dispatch vs EIA-930:** 2019 (calibrated) coal +6 %, gas −2 %, CO2 −1 %; 2022 (held out) coal +28 %, gas
  −11 %, CO2 +1.8 %. Prices: median $46.6 vs $48.0/MWh (2022), rank correlation 0.87.
- **Weather to wind, solar and demand:** annual wind +6-8 %, hourly correlation 0.90; solar 0.97; demand regression
  MAPE 3.8-4.3 %.
- **Speed:** an ERCOT year in 13-21 min (MILP), ~3 min (LP), ~2 s (merit); one frequency event 0.3 s; a 2,000-bus AC
  power flow 36 ms.
- **Real data (GridData):** EIA-930 hourly for 71 balancing authorities 2018-2026; generator fleets for 8 focus
  regions (ERCO 2,393 generators) with a source for every value; EPA CAMPD hourly for every Texas unit 2018-2025
  (452 units, 28 M unit-hours; rebuilt, not shipped); 470 GB frequency events with 11-minute 1 s traces and NESO
  inertia; 329 ERCOT events from the monthly operations reports; ERCOT prices 2018-2026, CAISO from 2023-08; weather for
  every region; 11 test networks.
- **Splits (frozen, seed 20261113):** training-input years 2018, 2020, 2022 · dev 2019, 2021 · exam 2023-2025 ·
  reserve 2026 (untouched, a later exam). Plants: 10 dev plants (40 units), 407 exam units. Events: GB dev 4 / exam
  412; ERCOT dev 5 / exam 301.

## Known caveats

- ERCOT publishes no 1 s frequency traces: ERCOT events have summary numbers only. GB loss labels are noisy outside
  9 Aug 2019; dev event GB_20191203T200030 is labelled 1,788 MW but its frequency drop implies about 850 MW.
- The coal/gas split follows the gas price, unlike reality (coal −43 % to +28 % by year); scarcity prices are too rare.
- GridSim's outage-rate defaults cite Texas RE 2024/2025 reliability aggregates, published during the exam years: a
  possible exam leak, listed first in `HANDOFF.md` P0.5 (not yet replaced).
- CAMPD gross load excludes the steam turbine of many combined-cycle units and is 5-8 % above EIA net generation.
- EIA-930: demand = generation − interchange fails for CISO, MISO, BPAT and PJM (use the demand column); fuel
  categories changed in 2024-07; CISO's own day-ahead forecast is misaligned by 1 h.
- Fleet ramps, start costs, inertia and governors are technology defaults, not unit data.
- `gridsim/LIMITS.md` lists every measured simulator gap; `HANDOFF.md` section 4 is the work list to close them.

## Licences

GridSim and the loader: MIT (this repository). Data: see the grid rows of [DATA_LICENSES.md](../../DATA_LICENSES.md)
and `data/DATA.md` section 2.
