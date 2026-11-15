# GridSim: shared plan (read with /home/claude/AGENT_BRIEF.md)

> Kept as written on 13-14 Nov 2026 in the cloud project where the grid work was built, with names of people and build roles made neutral. Paths such as `/home/claude/grid/...` refer to that sandbox; in this repository the platform is `shared/grid/` and the models are `models/gridpulse`, `models/batterypilot`, `models/plantdetective` and `models/demandcast`.

The next big-system project for ML Ratchet (github.com/dogum/ML-Ratchet, "Trained on physics. Graded on reality."),
built the same way as the vehicle project (RoadSim):

- **Phase 1 (now, two parallel builds):**
  - **A: GridSim.** The shared, verified power-system simulator.
  - **B: GridData.** The real-data package with frozen dev/exam splits and classical baselines.
- **Phase 2 (later):** several ratchet models on top, in parallel. Candidates, chosen after Phase 1:
  - **GridPulse:** MW lost and system inertia from a frequency event.
  - **DemandCast:** demand and wind/solar from weather, with error bars.
  - **PlantDetective:** a unit's heat-rate curve, minimum load, ramp limits and start behaviour from its output history.
  - **DispatchNet:** a learned dispatcher vs the MILP optimum.
  - **BatteryPilot:** storage arbitrage graded on real historical prices, which gives a real exam.
- **Then:** a `HANDOFF.md` listing what is missing.

Root: `/home/claude/grid/`.
- A works in `grid/gridsim/` (Python package `gridsim`) and `grid/checks/`.
- B works in `grid/data/` (`raw/`, `proc/`) and `grid/griddata/` (Python package `griddata`).
- Neither touches the other's folders. Each keeps a `STATUS.md` in its own folder.

## Shared contract (both follow it exactly)

### Time series
- Hourly data are UTC-indexed pandas DataFrames / parquet, with columns in MW (power), MWh (energy), $/MWh (price), °C, m/s, W/m².
- Sub-second frequency data are 1 s or faster npz arrays: `t` [s], `f` [Hz], plus a meta json (`region`, `nominal_hz`, `start_utc`, `source`).

### Regions
- A region is a balancing authority (BA) code as EIA-930 uses it (ERCO, CISO, PJM, MISO, SWPP, NYIS, ISNE, BPAT, …) or `GB` for Great Britain.
- **Primary focus:** ERCOT (ERCO; an islanded interconnection, so simplest to close the balance) and GB (best public 1 s frequency data). Then CAISO, PJM and MISO as breadth.

### Generator spec (dict/JSON; B fills it from EIA-860 / PUDL / CAMPD, A's simulator consumes it)
- `id`, `plant_id`, `region`, `lat`, `lon`
- `tech`: `nuclear` | `coal_steam` | `gas_cc` | `gas_ct` | `gas_steam` | `gas_ice` | `oil` | `hydro_res` | `hydro_ror` | `pumped` | `wind` | `solar_pv` | `battery` | `biomass` | `geothermal` | `other`
- `pmax_MW`, `pmin_MW`, `ramp_up_MW_per_min`, `ramp_down_MW_per_min`
- `min_up_h`, `min_down_h`, `start_cost_$`, `start_time_h`
- `heat_rate_curve`: list of [MW, MMBtu/MWh] points, or a quadratic
- `fuel`, `fuel_cost_$_per_MMBtu`, `co2_kg_per_MMBtu`
- `inertia_H_s` (on its own MVA base), `mva_base`
- `governor`: {`droop`, `deadband_Hz`, `T_s`, `headroom_frac`}
- `ibr` (bool, inverter-based); `ffr`: {`MW`, `trigger_Hz`, `delay_s`}
- For storage: `energy_MWh`, `eta_rt`, `soc_min`, `soc_max`
- For wind/solar: `hub_height_m`, `power_curve` id, `tracking`, `dc_ac`, `tilt`, `azimuth`
- `retired_date`, `online_date`

Unknown fields are None, with a documented default source.

### Simulator API (A builds; B's baselines and Phase 2 call)
```python
from gridsim import dispatch, frequency_event, powerflow, demand_model, renewables
res = dispatch(fleet, demand_MW, wind_cf, solar_cf, prices=None, network=None, reserves=..., horizon='day', method='milp'|'lp'|'merit')
#  -> hourly dispatch per unit, commitment, generation by tech, marginal price, CO2, curtailment, reserve
ev = frequency_event(system_state, loss_MW, t_end=60, dt=0.02, ...)
#  -> t, f (Hz), RoCoF, nadir, settling, governor and FFR responses, UFLS actions
#  system_state: online units with H, MVA, governor; load damping; IBR share
pf = powerflow(network, injections, method='dc'|'ac')
#  -> flows, angles, voltages, losses
```

### Physics to implement (A)
- **Network:** DC and AC (Newton–Raphson) power flow, line limits, losses; DC-OPF / PTDF for dispatch with congestion. Verify against pandapower or PyPSA (PyPI reachable) on the IEEE and Texas A&M ACTIVSg test grids (`electricgrids.engr.tamu.edu`, reachable).
- **Operations:** unit commitment (MILP with `scipy.optimize.milp` / HiGHS, or LP relaxation plus rounding for speed), economic dispatch with ramping, min up/down, start costs, reserves, storage, curtailment, imports. Marginal prices from the duals. Keep it fast enough for years of hourly simulation.
- **Frequency dynamics:**
  - aggregated swing equation with time-varying inertia from online units
  - governor and turbine models per tech (steam reheat, gas turbine, hydro with water hammer)
  - deadbands
  - load damping
  - inverter fast frequency response and synthetic inertia
  - UFLS stages
  - optional multi-area / multi-machine runs, with ANDES (PyPI) as a check
  - output at ≥ 10 Hz
- **Renewables:** wind power from hub-height wind speed (shear, air density, farm power curve, wake and availability losses); solar PV from irradiance (pvlib, used in SunDoctor).
- **Demand:** weather-driven (heating/cooling degree response, humidity, solar gain), calendar, behind-the-meter solar.
- **Emissions** per unit from heat input.

### Verification A must record (`grid/checks/*.json`)
- **Power flow:** matches pandapower/PyPSA on several test grids (angles and flows within tolerance).
- **Frequency events:**
  - Reproduce published real events: GB 9 Aug 2019 (nadir ~48.8 Hz, LFDD) and other NESO-reported trips; ERCOT events in public reports.
  - Check the swing equation's RoCoF = ΔP·f0 / (2·H·S) and nadir against analytical cases.
- **Dispatch:** with the real ERCOT fleet (from B's files when ready, or a copy) and real hourly demand / wind / solar, compare simulated generation by fuel against EIA-930 for at least a year: hourly and annual error by fuel. Also compare simulated prices against real ERCOT/CAISO prices if B provides them.
- **Emissions:** annual CO2 vs EPA CAMPD.
- **Speed:** one simulated year of hourly dispatch for ERCOT in ≤ ~10 min single thread, or say what it takes.

### Real data B must assemble (endpoints checked by the coordinator on 13 Nov)
- **EIA-930 hourly by BA:** demand, day-ahead demand forecast, net generation by fuel, interchange.
  - Six-month bulk CSVs need no API key, e.g. `https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv` (200).
  - The INTERCHANGE and SUBREGION files follow the same pattern. 2018 onward.
  - The api.eia.gov API needs a key the project does not have: don't use it.
- **Plant and generator inventory:** EIA-860 and EIA-923 (eia.gov pages reachable), or PUDL (Catalyst Cooperative), which is cleaner. `https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/` is reachable; the full sqlite is large, so prefer the parquet tables you need, or the Zenodo release (API reachable). Check the size before downloading.
- **EPA CAMPD:** hourly unit-level gross load, heat input, CO2, operating time, for every fossil unit ≥ 25 MW. This is the truth for heat-rate curves, minimum load, ramping and starts. `campd.epa.gov` is reachable; find the documented bulk-file route (the S3 bulk files behind the CAMPD API). If a key is required, say so and stop there.
- **Frequency:**
  - NESO GB 1 s system frequency data (data portal page `https://www.neso.energy/data-portal/system-frequency-data` is reachable; the CKAN API timed out once, so retry gently).
  - The NESO/Ofgem event reports (9 Aug 2019 and others) for MW lost and timing.
  - Any public ERCOT frequency event data or reports.
- **Prices:** ERCOT settlement-point prices (find documented public files) and CAISO OASIS LMPs (`oasis.caiso.com` reachable).
- **Weather:** representative stations for ERCOT and GB (Iowa Mesonet ASOS for US; NASA POWER for irradiance and wind; for GB, find an open source).
- **Test grids:** Texas A&M ACTIVSg200 / 2000 (+ coordinates), MATPOWER/IEEE cases.
- **Splits** (decide once, freeze in `grid/data/proc/splits.json`):
  - Time-based: train/calibration years vs held-out exam years, e.g. dev = 2019 and 2021, exam = 2023–2025.
  - Unit-based for plant-level models: by plant, never by hour.
  - Event-based for frequency: dev = a few named events, exam = the rest.
  - Simulated training may use real *inputs* (weather, demand shapes) from non-exam periods only. Exam-period labels are never touched until the exam.
- **Classical baselines on the exam:**
  - demand: persistence, same-hour-last-week, and the BA's own day-ahead forecast (in EIA-930)
  - inertia: the textbook RoCoF formula
  - heat rate: the EIA-923 annual average
  - prices: perfect-foresight arbitrage bound and a simple threshold rule for BatteryPilot

## Shared-machine rules
As in AGENT_BRIEF.md:
- 2 cores, 7 GB RAM, ~19 GB free disk.
- One thread each; `nice` heavy jobs.
- Never kill others' processes; never `pkill -f`.
- Resumable work.
- ≤ 6 GB disk each this phase (stream big files; keep compact parquet).
- No changes outside one's own folders. Nothing written to the claude.ai project in Phase 1.
