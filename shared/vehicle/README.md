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
├── garage/         export_garage.py: the RoadSim grid behind the site's garage page (site/public/garage/)
├── HANDOFF.md      what exists, what's missing (status as of 14 Nov 2026) — start here to extend it
├── V2_PLAN.md      the v2 plan, each phase marked done, and the decisions log
├── VEHICLE_BRIEF.md   historical brief (12-13 Nov 2026): trip format, vehicle spec, simulator API, energy ledger
└── PHASE2_BRIEF.md    historical brief: how the three v1.0 models were set up
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
- **Verified** (`checks/`; re-run after any physics change). Physics v2, the default since 13 Nov 2026, calibrated on
  dev data only (`roadsim/CALIBRATION.md`); held-out numbers from `roadsim/LIMITS.md`:
  - energy ledger closes in all 600 stress runs (max 5.6e-4)
  - about 30,000× real time for conventional cars (29,500-29,900×), 33,400× for BEVs and 4,600-5,900× for hybrids,
    single thread, UDDS
  - EPA on 33 held-out vehicles: conventional FTP +0.4% / HWFET −1.9% (n 15), BEV −0.4% / −1.8% (n 8), P2 hybrids
    FTP −4.0% (n 5), power-split hybrids FTP −5.1% (n 4; v1 +1.0%)
  - FASTSim 3.1 with matched components, conventional UDDS / HWFET / US06: −1.9 / −1.2 / −6.4%
- **Real data:** 33,015 Vehicle Energy Dataset trips from Ann Arbor with USGS lidar grades and airport weather
  (rebuildable, ≈1.2 GB, not shipped); 80,227 EPA tests; fueleconomy.gov labels; 24 drive cycles; Argonne D3, 65
  vehicles and 1,661 usable 10 Hz dynamometer tests, dev and exam by vehicle (`data/D3.md`); heavy duty from the full
  ZTBus set (1,409 Zurich trolleybus missions, by-month split), Mersin (diesel and CNG buses) and ICCT truck tests. Dev
  and exam vehicles were split once and frozen.

## Known caveats

- VED fuel is estimated from the MAF sensor and fuel trims for 351 of 384 cars, not metered.
- On 13 Nov 2026 RoadData fixed two bugs (EPA city/highway labels swapped; VED mass double-counted the 136 kg
  driver). The three v1.0 models were built before the fixes; the v2.0 releases (14 Nov 2026) were rebuilt on the
  fixed tables — see each model card.
- Heavy-duty grading is thin (2 trolleybuses, 2 Mersin bus types, 6 truck dyno numbers).
- `roadsim/LIMITS.md` lists every measured simulator gap; `HANDOFF.md` section 4 is the work list to close them.
- The model contract is [docs/model-contract.md](../../docs/model-contract.md); `VEHICLE_BRIEF.md` and
  `PHASE2_BRIEF.md` are the briefs the platform was first built from, kept as written.

## Licences

RoadSim and the loader: MIT (this repository). Data: see the vehicle rows of [DATA_LICENSES.md](../../DATA_LICENSES.md)
and `data/DATA.md`.
