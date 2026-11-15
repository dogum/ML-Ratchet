# GridSim Phase 2 brief: four ratchet projects on GridSim + GridData

> Kept as written on 13-14 Nov 2026 in the cloud project where the grid work was built, with names of people and build roles made neutral. Paths such as `/home/claude/grid/...` refer to that sandbox; in this repository the platform is `shared/grid/` and the models are `models/gridpulse`, `models/batterypilot`, `models/plantdetective` and `models/demandcast`.

Read first:
- `/home/claude/AGENT_BRIEF.md`: recipe, deliverables, shared-machine rules. 2 cores are now shared by **four** parallel builds, so be frugal: small networks, modest banks, one thread, `nice`.
- `/home/claude/grid/GRID_BRIEF.md`: the shared contract.
- Phase 1 docs:
  - `/home/claude/grid/gridsim/README.md` (section "Using it in Phase 2") and `LIMITS.md`
  - `/home/claude/grid/data/DATA.md`
  - the check JSONs in `/home/claude/grid/checks/`

## Calling Phase 1
```python
import sys; sys.path.insert(0, '/home/claude/grid/gridsim'); sys.path.insert(0, '/home/claude/grid')
import gridsim as gs, griddata as gd
states, losses, labels = gs.generate_states(N, seed=1, systems=('ERCO','GB'))
ev = gs.frequency_event(states, losses, t_end=60, dt=0.1, record=False)
res = gs.dispatch(fleet_or_path, demand, wind_MW=..., solar_MW=..., imports_MW=..., fixed_MW={...}, method='milp'|'lp'|'merit', ...)
gs.generate_fleet(seed=...); gs.fit_demand(wx, load); gs.renewables.fleet_profiles(...)
```
Loaders:
- `gd.splits()`
- `gd.eia930(ba, start, end)`
- `gd.fleet(ba, plant_safe=...)`
- `gd.campd_units()` / `gd.campd_unit()`
- `gd.events('GB'|'ERCO', split=)` / `gd.event_trace()`
- `gd.frequency()` / `gd.gb_inertia()` / `gd.gb_genmix()`
- `gd.prices(...)`
- `gd.weather(...)`
- `gd.network(...)`
- `gd.baselines(...)`

## Frozen splits (`/home/claude/grid/data/proc/splits.json`, sha256 895b4e9a…). NEVER change them.

| Split | Contents |
|---|---|
| Training-input years | 2018, 2020, 2022 (real inputs only: weather, demand shapes, price shapes, quiet noise) |
| Dev | 2019, 2021 |
| Exam | 2023–2025 |
| Reserve | 2026 (do not touch) |
| CAISO prices | dev 2023-08 → 2023-12, exam 2024–2025 |
| Plants | dev = 10 ERCOT plants (40 units); exam = every other Texas CAMPD unit (407) |
| GB events | dev = 9 Aug 2019 + 3 others (4); exam = 412 |
| ERCOT events | dev = 5; exam = 301 |

Only dev may be used for calibration, gates and any fitting. The exam is for the final report only.

## Phase 1 caveats to respect
- **Simulator calibration:** GridSim's frequency load damping was fitted in-sample on dev events (GB 1.5 %/Hz, ERCOT 2.5 %/%). Unfitted GB events come out about 30 % too deep.
- **Dispatch weaknesses:**
  - The coal/gas split follows gas price, unlike reality.
  - Prices lack real scarcity spikes.
  - Minimum inertia in 2022 is 34 % low.
- **Data quirks:**
  - CAMPD gross load excludes the steam turbine for many combined-cycle units.
  - GB loss sizes are noisy, apart from 9 Aug 2019. The GridSim build thinks dev event GB_20191203T200030 (1,788 MW) is about 2× too large; check it yourself on dev before using it.
  - ERCOT events have official summary metrics but no 1 s traces.
- **Possible leak:** GridSim's outage-rate defaults cite Texas RE 2024/2025 published fleet-wide reliability aggregates. If your model uses outage rates, prefer pre-2023 NERC GADS values and say so.

## Shared code rules
- Do NOT edit `/home/claude/grid/gridsim/`, `/home/claude/grid/griddata/` or `/home/claude/grid/data/`.
- Physics changes or calibrations go in YOUR folder as wrappers, versioned in your ratchet, for the coordinator to upstream later.
- Your folder: `/home/claude/grid/<name>/`. Keep a `STATUS.md` there.
- Disk: ≤ 2 GB each.

## Deliverables (each build)
- Release zip `/mnt/user-data/outputs/<name>-v1.0.zip`. Prefer a standalone numpy model; vendor small feature code.
- A showpiece Artifact page titled with your project name (load the artifact-design skill first).
- Project doc `claude/<name>-status.md`.
- A compact final report, including upstream candidates and best next rungs.
