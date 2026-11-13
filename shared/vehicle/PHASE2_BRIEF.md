# Phase 2 brief: three ratchet projects on RoadSim + RoadData

Read first:
- `/home/claude/AGENT_BRIEF.md`: the recipe, deliverables, shared-machine rules (2 cores now shared by THREE agents).
- `/home/claude/road/VEHICLE_BRIEF.md`: the contract.
- Phase 1 docs: `/home/claude/road/roadsim/README.md`, `/home/claude/road/roadsim/LIMITS.md`, `/home/claude/road/data/DATA.md`, and the check JSONs in `/home/claude/road/checks/` and `/home/claude/road/data/proc/checks/`.

## How to call Phase 1
```python
import sys; sys.path.insert(0, '/home/claude/road/roadsim'); sys.path.insert(0, '/home/claude/road')
from roadsim import simulate, load_vehicle, make_trip, list_strategies, sim_to_contract
from roadsim.generator import generate_vehicles          # generate_vehicles(n, seed=, cls=, arch=)
import realdata as rd                                     # load_trips('dev'|'exam'|'train_noise', arch=, source=, label_ok=True, ...)
out = simulate(spec, trip_arrays, strategy=None, init_soc=None, init_temps=None, record=True)   # out['summary'], out['ledger'], sim_to_contract(out)
```

**Simulator speed:**

| Vehicle | × real time |
|---|---|
| Conventional / BEV | ~60,000 |
| HEV | ~5,000 |

- DP takes 0.8 s per UDDS; MPC 1.2 s.
- VED trips are warm fragments: pass `init_temps={'engine': 85, 'driveline': 70, 'tire_warm': 1, 'cabin': 21}` and start from the logged speed. Cold soak is the default.
- `dp` and `mpc` see the future: benchmarks only.

**Real data facts (from RoadData):**
- **Frozen splits:** `/home/claude/road/data/proc/splits.json`. NEVER change them.
- **dev:** 16 VED cars:
  - 8 ICE
  - 5 power-split HEV
  - 1 PHEV series (Volt)
  - 1 PHEV power-split
  - 1 BEV Leaf

  Plus 10% of EPA model families.
- **exam:** all other labelled VED cars (240 ICE, 84 HEV, 2 P2 HEV, 22 PHEV, 2 BEV) and 90% of EPA families. Also ZTBus (electric trolleybus legs and missions), Mersin diesel/CNG bus trips, and the ICCT truck table.
- **train_noise:** 17 unlabelled VED cars giving 1,092 real Ann Arbor routes. Use these as route inputs, together with the EPA cycles and synthetic routes you build.
- **VED limits:**
  - Fuel is estimated from MAF/trims for most cars; it is not metered.
  - Make, model and year are hidden; architecture is inferred from engine size.
  - Mass is rounded to 500 lb; CdA, Crr and gearing are unknown.
  - Trips are warm fragments.
  - PHEV mode per trip is unknown.
  - T_amb is from the weather station (±2 °C); wind is the airport 10 m wind.
  - Grade is good: USGS 1 m lidar, repeatability 0.36 % vs 2.1 % between road cells.
  - AC/heater power is logged only on PHEV/EV.
- **Classical baselines to beat (VED exam, median per-trip |error|, fitted on dev):**

  | Fleet | Label MPG × distance | Speed–power bins (MOVES-like) | Class median |
  |---|---|---|---|
  | ICE | 13.0 % | **10.3 %** | 22.8 % |
  | HEV | 16.2 % | **15.3 %** | 17.2 % |
  | EV | 21.5 % | **19.0 %** | 27.1 % |
  | PHEV | 46.1 % | **41.3 %** | 42.7 % |

  `rd.baselines('summary')` has everything, including the EPA regressions:
  - conventional: FTP 8.8 %, HWY 7.0 %, US06 6.9 %
  - HEV: 21.5 / 16.1 %
  - BEV kWh/mi: 9.8 / 6.5 %

  And heavy duty:
  - ZTBus legs 22 %
  - Mersin 7–9 %
  - ICCT 7 %
- **Simulator known limits:**
  - P2 hybrids are optimistic in the city (+8 to +31 %).
  - Cold weather is too mild (−7 °C FTP +10 to +16 %).
  - A/C on conventional cars is too cheap.
  - The hybrid bus is too good.

  Several of these are natural dev-calibration rungs for you.

## Rules for the shared simulator (important with three agents)
- **Do NOT edit `/home/claude/road/roadsim/` or `/home/claude/road/data/`.** They are frozen for Phase 2.
- If your project needs a physics change (e.g. a cold-weather or P2 fix calibrated on dev), implement it in YOUR folder as a wrapper or patch module and version it in your ratchet. Record it so the coordinator can upstream it later.
- Work only in your own folder under `/home/claude/road/<yourname>/`. Keep a `STATUS.md` there.
- Keep disk under 3 GB each. Use one thread. `nice` the heavy jobs. Never kill processes you didn't start.

## Deliverables (each agent)
Follow the AGENT_BRIEF deliverables:
- Release zip at `/mnt/user-data/outputs/<name>-v1.0.zip`. Include a `requirements` note saying it needs the `roadsim` package only if your inference truly needs the simulator. Prefer a standalone numpy model, and vendor any small feature code.
- A showpiece Artifact page titled with your project name.
- Project doc `claude/<name>-status.md`.
- A compact final report.

Make each page's lab use the simulator's ideas in JS where feasible (e.g. drive a route, change wind, temperature, mass or architecture). If a full JS port is too heavy, show precomputed sweeps.
