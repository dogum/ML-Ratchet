# The vehicle platform: RoadSim + RoadData

The collection's first **shared platform**: one verified physics simulator and one real-data package, under three
models — [EnergyNet](../../models/energynet/), [TripDetective](../../models/tripdetective/) and
[HybridPilot](../../models/hybridpilot/). New vehicle models start here instead of from scratch.

```
shared/vehicle/
├── roadsim/        RoadSim, the whole-vehicle simulator (Python package `roadsim`, numpy/scipy)
│   ├── roadsim/      the package: sim, vehicle, powertrain, strategies (rule … ECMS, A-ECMS, MPC, DP), generator, maps
│   ├── tools/        rebuild the maps, cycles and generator calibration (fetches the EPA ALPHA archive)
│   ├── README.md     physics, verification numbers      LIMITS.md   what it gets wrong, measured
│   └── demo.py       one car per powertrain over a cycle, with its energy ledger
├── realdata/       RoadData's loader (Python package `realdata`, numpy/pandas): trips, vehicles, EPA, cycles, heavy duty
├── data/           the processed tables that ship (24 MB) + scripts/ to rebuild everything else; DATA.md = sources
│   └── proc/splits.json   the frozen dev / exam split — sha256 d917df78…; never change it
├── checks/         every physics check and its JSON (run_all.sh)
├── HANDOFF.md      what exists, what's missing, and the prioritised path to v2 — start here to extend it
├── VEHICLE_BRIEF.md   the shared contract: trip format, vehicle spec, simulator API, energy ledger
└── PHASE2_BRIEF.md    how the three models were set up
```

## Use it

```bash
pip install -e ".[vehicle]"          # from the repo root: makes `roadsim` and `realdata` importable anywhere
cd shared/vehicle/roadsim && python demo.py
```

```python
from roadsim import simulate, load_vehicle, cycle, list_strategies
import realdata as rd
epa = rd.epa_testcars(split="dev")   # one row per EPA test-vehicle configuration
cyc = rd.cycles()                    # 24 dynamometer schedules
```

Without installing, set `ML_RATCHET_VEHICLE` to this folder (or add `shared/vehicle` and `shared/vehicle/roadsim` to
`sys.path`). Model training scripts find it through their `training/_paths.py`.

## What it is

- **Architectures:** conventional (gasoline, diesel), P2 / series / power-split hybrids, power-split and series
  plug-ins, battery-electric — across cars, SUVs, pickups, vans, buses, medium trucks and Class-8 tractors.
- **Physics:** ALPHA engine and motor maps with cold start; AT, DCT, CVT, AMT, eCVT; battery equivalent circuit with
  a thermal model; regen blending; wind along the heading, temperature- and surface-dependent rolling resistance, real
  lidar grade, HVAC. Forward-facing, 1 Hz, with a driver controller tracking the speed trace.
- **Control strategies:** rule, power-follower, thermostat, P2 assist, Prius-style split, ECMS, A-ECMS, MPC, DP,
  plug-in CD-CS and blended, BEV regen variants — and the learned HybridPilot.
- **Verified** (`checks/`; re-run after any physics change):
  - energy ledger closes in all 600 stress runs (max 5e-4)
  - about 60,000× real time (conventional, BEV) and 5,000× (hybrids), single thread
  - EPA on 33 held-out vehicles: conventional FTP +5.6% / HWFET −4.3%, power-split hybrids −2.3%, BEV −0.5% / −1.7%
  - FASTSim 3.1 with matched components: within 3% (conventional), 4% (BEV)
- **Real data:** 33,015 Vehicle Energy Dataset trips from Ann Arbor with USGS lidar grades and airport weather
  (rebuildable, ≈1.2 GB, not shipped); 80,227 EPA tests; fueleconomy.gov labels; 24 drive cycles; heavy duty from
  ZTBus (Zurich trolleybuses), Mersin (diesel and CNG buses) and ICCT truck tests. Dev and exam vehicles were split once
  and frozen.

## Known caveats

- VED fuel is estimated from the MAF sensor and fuel trims for 351 of 384 cars, not metered.
- On 13 Nov 2026 RoadData fixed two bugs (EPA city/highway labels swapped; VED mass double-counted the 136 kg
  driver); the three v1.0 models were built before the fixes — see each model card.
- Heavy-duty grading is thin (2 trolleybuses, 2 Mersin bus types, 6 truck dyno numbers).
- `LIMITS.md` lists every measured simulator gap; `HANDOFF.md` section 4 is the work list to close them.

## Licences

RoadSim and the loader: MIT (this repository). Data: see the vehicle rows of [DATA_LICENSES.md](../../DATA_LICENSES.md)
and `data/DATA.md`.
