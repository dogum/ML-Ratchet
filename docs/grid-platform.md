# The grid platform

The second shared platform. **GridSim** is a verified power-system simulator — network, dispatch, frequency, weather
to wind, solar and demand — and **GridData** is real grid data, split once into training-input, dev, exam and reserve
periods and frozen. Four models sit on it, and new power-grid problems start from it instead of from scratch.

| Model | The question |
|---|---|
| [GridPulse](../models/gridpulse) | A power station just tripped: how many MW were lost, how much inertia does the grid have, and where will the frequency bottom out? |
| [BatteryPilot](../models/batterypilot) | Hour by hour, when should a grid battery charge and discharge to earn the most, net of wear? |
| [PlantDetective](../models/plantdetective) | From a power plant's hourly output alone, what kind of unit is it: heat rate, minimum load, ramps, starts? |
| [DemandCast](../models/demandcast) | How much electricity will a region use tomorrow, and how much will its wind and solar make? |

Code and data: [`shared/grid/`](https://github.com/dogum/ML-Ratchet/tree/main/shared/grid).

## GridSim: the simulator

- **The network:** AC and DC power flow, PTDF and DC optimal power flow with line limits. Exact against PYPOWER on
  the IEEE 9- to 300-bus cases and the Texas A&M synthetic grids ACTIVSg200 and ACTIVSg2000: angles to 1e-11°, flows
  to 3e-10 MW, optimal-power-flow cost to 3e-13. A 2,000-bus AC power flow takes 36 ms.
- **Running a fleet:** unit commitment and economic dispatch as a mixed-integer program, a linear program or a merit
  order, with ramping, minimum up and down times, start costs, reserves, storage, hydro budgets and the network.
  Marginal prices come from the duals. An ERCOT year takes 13-21 min (MILP), about 3 min (LP) or 2 s (merit order).
- **Frequency after a trip:** the swing equation with time-varying inertia, governors per technology (steam reheat,
  combined cycle, gas turbine, hydro), deadbands, load damping, fast frequency response, synthetic inertia, RoCoF
  relays and under-frequency load shedding, in one or several areas. Analytic cases exact; the four-machine Kundur
  case against ANDES within 15 mHz RMS. One event takes 0.3 s.
- **Checked against a real blackout:** GB, 9 August 2019. Simulated first nadir 49.11 Hz at 24.5 s (reported 49.1 Hz
  at 24.5 s); demand disconnection at 75.6 s (reported 75.9 s); 44 mHz RMS against NESO's 1 s trace.
- **Checked against a real year:** the real ERCOT fleet dispatched against EIA-930. 2019 (calibrated): coal +6 %,
  gas −2 %, CO2 −1 %. 2022 (held out): coal +28 %, gas −11 %, CO2 +1.8 %; median price $46.6 vs $48.0/MWh, rank
  correlation 0.87.
- **Weather to power:** wind farms from hub-height wind (annual +6-8 %, hourly correlation 0.90), solar from irradiance
  with pvlib (correlation 0.97), demand from weather (MAPE 3.8-4.3 %).

## GridData: the real data

- EIA-930 hourly demand, day-ahead forecasts, generation by fuel and interchange for all 71 US balancing authorities,
  2018-2026.
- Generator fleets for 8 focus regions (ERCOT, CAISO, PJM, MISO, SPP, ISO-NE, NYISO, BPA) in one generator spec, with a
  source for every value: ERCOT alone has 2,393 generators.
- EPA CAMPD hourly output, heat input and CO2 for every Texas fossil unit, 2018-2025: 452 units, 28 million unit-hours.
- Frequency: 470 GB events with 11-minute 1 s traces and NESO's inertia estimate; 329 ERCOT events from the monthly
  operations reports (summary numbers only: ERCOT publishes no traces).
- Prices: ERCOT day-ahead and real-time 2018-2026; CAISO from August 2023 (OASIS serves about three years).
- Weather for every region (NASA POWER, airport stations) and 11 test networks.
- **Frozen splits** (`splits.json`, content sha256 `895b4e9a…`, seed 20261113): training inputs 2018, 2020, 2022;
  dev 2019, 2021; exam 2023-2025; 2026 kept untouched for a later exam. Plants: 10 dev plants, 407 exam units.
  Events: 4 GB and 5 ERCOT dev events; 412 GB and 301 ERCOT exam events.

The data are not in git. `mlratchet data grid` downloads every source and rebuilds them (resumable), or unpacks the
v1.0 release and checks all 1,064 files against their published hashes.

## What it gets wrong

The simulator's measured gaps are its next rungs, each with a size and a "done when" in
[`HANDOFF.md`](https://github.com/dogum/ML-Ratchet/blob/main/shared/grid/HANDOFF.md):

- Coal versus gas follows the gas price, unlike reality: coal is off by −43 % to +28 % depending on the year.
- Scarcity price spikes are too rare and land in the wrong hours.
- Too little inertia at low net load (2022 minimum 76 vs 115 GW·s).
- Frequency response is fitted on one 2019 event: unfitted GB events come out about 30 % too deep, and the 2023-2025
  grids, with gigawatts of batteries, are stiffer than the simulator.
- Only ERCOT is dispatched and calibrated; the other seven regions have data only.
- A possible exam leak: the default outage rates cite Texas reliability aggregates published during the exam years.
  They are to be replaced with pre-2023 values before any model is re-trained.

The models show where those gaps bite. Physics-trained models win on the physics-heavy part and lose where the real
data has reporting quirks, and they also lose on physics they model too simply:

- DemandCast beats the grid operators' own day-ahead demand forecasts in 6 of 8 regions with the weather that actually
  happened, an upper bound; with archived forecast weather (2024) it wins in only 4 of 8. It loses on solar, because
  EIA-930 solar timestamps moved by an hour between years.
- PlantDetective reads coal and gas-steam heat rates far better than EIA-923 but loses on combined cycles: its 3 dev
  combined-cycle plants were cogeneration or reported only the gas turbine, while most exam units report the whole
  block, so the dev calibration pushed its combined-cycle heat rates too high.
- GridPulse's 10 s nadir and 60 s forecasts lose to simply holding the lowest or the current value: the simulator's
  response after the first seconds (secondary response, market actions) is too simple.
- BatteryPilot loses to the classical day-ahead optimiser in both markets (ERCOT 0.667 vs 0.685 of perfect hindsight,
  CAISO 0.642 vs 0.681); it beats it only with 4-hour batteries.

## Using it

```bash
pip install -e ".[grid]"
mlratchet data grid --from-zip gridsim-griddata-v1.0.zip    # or `mlratchet data grid` to rebuild from the sources
cd shared/grid/gridsim && python demo.py
python shared/grid/checks/run_all.py --quick
```

```python
import griddata as gd
from gridsim import dispatch, frequency_event, powerflow, generate_states
```

Every physics change bumps the GridSim version, rebuilds each grid model's frozen simulated set and re-takes their
ratchet decisions in order. `splits.json` never changes.
