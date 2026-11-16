# GridSim

A power-system simulator built from the physics, for training networks on. It solves the network (DC and AC power
flow, DC optimal power flow with line limits), runs a whole fleet hour by hour (unit commitment and economic dispatch
with ramping, minimum up/down times, start costs, reserves, storage, curtailment and imports, plus marginal prices
and CO2), simulates the system frequency in the seconds after a power plant trips (inertia, governors, fast frequency
response, load damping, under-frequency load shedding), and turns weather into wind, solar and demand. A generator
draws random but realistic fleets and system states for training.

Part of ML Ratchet ("Trained on physics. Graded on reality."), Phase 1 of the power-grid project. Phase 2 trains the
models on it (GridPulse, DemandCast, PlantDetective, DispatchNet, BatteryPilot). Known errors are in
[LIMITS.md](LIMITS.md); every check is a JSON file in `grid/checks/` with its numbers.

## Quick start

```python
from gridsim import dispatch, frequency_event, powerflow, demand_model, renewables, generate_states

# 1. a day of ERCOT dispatch from the real fleet and real hourly inputs (pandas Series, UTC index, MW)
res = dispatch('shared/grid/data/proc/fleet/fleet_ERCO.parquet', demand_MW,
               wind_MW=wind, solar_MW=solar, imports_MW=net_imports, method='milp')
res.gen_by_fuel        # MW by EIA-930 fuel (COL, NG, NUC, WND, SUN, WAT, OIL, OTH), hourly
res.price              # $/MWh (dual of the energy balance), res.co2_t (t/h per cluster), res.frame()

# 2. what happens to frequency if a 1,300 MW unit trips in hour 17 of that dispatch
ev = frequency_event(res.system_state(17, region='ERCO'), loss_MW=1300, t_end=60, dt=0.02)
ev['nadir_Hz'], ev['rocof_Hz_s'], ev['t'], ev['f']

# 3. power flow on a MATPOWER case
pf = powerflow('shared/grid/gridsim/refdata/matpower/case_ACTIVSg2000.m', method='ac')

# 4. 10,000 random system states + losses for training, simulated as one vectorised batch
states, losses, labels = generate_states(10000, seed=0, systems=('ERCO',))
events = frequency_event(states, losses, t_end=30, record=False)
```

`python3 demo.py` dispatches 15 Aug 2019 in ERCOT from real EIA-930 inputs, prints the energy balance and the
simulated-vs-real generation by fuel, trips a 1,300 MW unit at the evening peak and writes `demo_day.png`.

The reference libraries used only for checks (pandapower, PYPOWER, PyPSA, ANDES) live in an isolated venv,
`gridsim/.refenv`, because pandapower pins pandas 2.x and the shared environment uses pandas 3. gridsim itself needs
numpy, scipy, pandas and pvlib; `highspy` (HiGHS's own Python API, MIT) is used when present and gives duals for the
MILP prices, otherwise it falls back to `scipy.optimize.milp` + `linprog` (same HiGHS solver).

## API (the shared contract)

### `dispatch(fleet, demand_MW, wind_cf=None, solar_cf=None, prices=None, network=None, reserves='auto', horizon='day', method='milp', ...)`

| argument | meaning |
|---|---|
| `fleet` | generator specs in the contract (DataFrame, list of dicts, `.parquet`/`.json` path). Units are filtered by `online_date`/`retired_date`; missing fields are filled from `TECH_DEFAULTS` (`prepare_fleet` records what was filled in `src_fill`). |
| `demand_MW` | hourly Series (UTC). |
| `wind_cf`, `solar_cf` | capacity factors (× capacity online that hour), or `wind_MW`, `solar_MW` available MW. |
| `prices` | **fuel** prices: `None`/`'eia923'` (each plant's, else the state's, monthly EIA-923 delivered cost; gas also follows the daily Henry Hub move), `'constant'` (spec values), `{fuel: $/MMBtu or daily Series}`, or a DataFrame (days × unit). |
| `network` | optional DC network: `dict(net=Network, gen_bus={unit id: bus}, load_share=array, wind_share=..., solar_share=...)`. Every hour gets PTDF line limits (added lazily: solve, find overloads, add, re-solve); units are committed individually; `res.lmp` (hours × buses) and `res.line_flow_MW`. `fleet_from_case(ppc)` builds a fleet + network dict from a MATPOWER case (e.g. ACTIVSg2000). |
| `reserves` | `'auto'` = max(largest available unit, 6 % of demand) of 10-min spinning reserve, a number, or a Series. Shortage is priced by a 3-block operating-reserve demand curve ($250 / $1,500 / $5,000). |
| `method` | `'milp'` (integer commitment for clusters of ≤ 50 units), `'lp'` (commitment relaxed, ~2× faster), `'merit'` (hourly merit order, no inter-temporal constraints, a year in ~2 s). |
| `horizon`, `lookahead_h` | rolling optimisation: 24 h kept + 12 h look-ahead, state (commitment, output, start/stop history, storage SOC) carried day to day. |
| `imports_MW` | fixed net imports (+ = into the region). `fixed_MW={tech: Series}` removes that technology from the optimisation and injects the given MW (e.g. observed nuclear). `hydro_MW`: hourly hydro energy available (used as a daily budget). |
| `outages` | `'expected'` (each cluster derated by forced-outage rate + planned outages with a spring/autumn season), `'random'` (Markov forced outages + scheduled planned outages per unit, `seed`), `'none'`. |
| `must_run`, `offer_adders` | technologies always committed (default nuclear); $/MWh added to a technology's offers. The dev-calibrated ERCOT values (coal self-commitment, coal −$6.62/MWh, fitted on 2019 only) are in `gridsim/data/calibration.json`. |
| `pricing` | `'relaxed'` (default): prices are the duals of the LP relaxation (approximate convex-hull pricing, carries commitment costs into prices); `'fixed'`: duals of the LP with the MILP commitment fixed. On 4 dev weeks of ERCOT 2019, relaxed gave price MAE $8.7 vs $12.7/MWh and rank correlation 0.88 vs 0.76. |
| `n_clusters`, `cluster` | thermal units are grouped into clusters of similar units (coal and nuclear per unit; ~24 CC, 10 CT, 10 steam clusters); `cluster=False` gives unit-level commitment (slower). |

Returns a `DispatchResult`: hourly arrays `p`, `u` (units on), `su`/`sd` (starts/stops), `r` (reserve), `seg`
(heat-rate segments) per cluster; `wind`, `solar`, `hydro`, `ch`/`dis`/`soc` (storage), `shed`, `dump`,
`reserve_short`, `price`, `reserve_price`, `fuel_MMBtu`, `co2_t`, `fuel_cost`; views `gen_by_tech`, `gen_by_fuel`,
`curtailment_MW`, `inertia_MWs`, `summary`, `frame()`; `system_state(hour)` builds the frequency model's state from
the committed units (H × MVA, output, headroom including the reserve each cluster carries). `balance_residual_MW`
is checked on every run (≤ 1e-9 MW in all checks).

### `frequency_event(system_state, loss_MW, t_end=60, dt=0.02, h=0.01, record=True)`

`system_state` from `make_system_state(units, load_MW, region, **kw)` or `DispatchResult.system_state(hour)`, or a
**list** of states (vectorised batch; `loss_MW` then per state). `loss_MW`: a number (lost at t = 0) or a list of
`(t_s, MW[, area, H_MWs])` losses. State keys: `units` (`tech`, `MW`, `pmax_MW`, `pmin_MW`, `mva`, `H_s`, `area`,
`governor` {`droop`, `deadband_Hz`, `T_s`, `headroom_frac`, `up_MW`, `enabled`, `LL_A`, `LL_B`}), `load_MW`, `D`
(load damping, % load per % frequency), `ufls` (stages `f_Hz`, `frac` or `MW`, `delay_s`), `ffr` (`MW`,
`trigger_Hz`, `delay_s`, `ramp_s`), `synthetic_inertia` (`H_s`, `MVA`, `MW_max`), `protection` (embedded-generation
relays: `kind` `'rocof'`/`'uf'`, `threshold`, `delay_s`, `window_s`, `MW`), `secondary` (`MW`, `T_s`),
`H_extra_MWs`, `areas`, `ties` (`a`, `b`, `T_MW_per_rad`), `pload_area`.

Returns `t`, `f` (centre-of-inertia Hz), `f_area`, `rocof_Hz_s` (0-0.5 s), `nadir_Hz`, `t_nadir_s`, `settling_Hz`
(mean of the last 10 s), `zenith_Hz`, `inertia_MWs`, `gov_MW` (per unit group), `gov_total_MW`, `ffr_MW`, `si_MW`,
`damping_MW`, `shed_MW`, `ufls` (stage, time, MW), `trips` (protection), `lost_MW`. Output every `dt` (50 Hz by
default; any dt ≥ h).

### `powerflow(network, injections=None, method='dc'|'ac')`

`network`: a `Network`, a MATPOWER ppc dict or a `.m` path. `injections`: `{'Pg_MW': per gen, 'Pd_MW', 'Qd_MVAr'
per bus}`. Returns `Va_deg`, `Vm`, `Pf_MW`/`Pt_MW` (+ `Qf`, `Qt` for AC), `Pg_MW` (slack adjusted), `losses_MW`,
`loading` (flow / rating A), `converged`, `iterations`. Also `Network.ptdf()`, `Network.ybus()`,
`dc_opf(network, Pd_MW=None, cost=None, rate_scale=1)` → `Pg_MW`, `Pf_MW`, `lmp` ($/MWh per bus), `lambda`,
`binding` lines.

### Renewables, demand, generator

```python
from gridsim.renewables import wind_power, farm_curve, hub_wind, air_density, solar_power, fleet_profiles
wind_power(ws_hub, capacity_MW, T_C, p_hPa, rh, specific_power=300)       # farm MW
solar_power(times, lat, lon, ghi, dni=None, dhi=None, temp_air, wind_speed, capacity_MWac, dc_ac=1.3, tracking='single_axis')
fleet_profiles(units, weather_by_point, kind='wind'|'solar')            # all farms of a fleet, nearest weather point
from gridsim.demand import fit_demand, demand_model, btm_pv
params = fit_demand(weather, load_MW); demand_model(weather, params, btm_MW=None)
from gridsim.generator import generate_fleet, generate_states
generate_fleet(seed, peak_MW=None)     # contract fleet DataFrame
generate_states(n, seed, systems=('ERCO', 'GB'))   # (states, losses, labels)
```

## Using it in Phase 2

All calls are single-threaded numpy/scipy (+ highspy if present); set `OMP_NUM_THREADS=1`. Respect
`grid/data/proc/splits.json`: simulate with real *inputs* only from `train_inputs` years (2018, 2020, 2022) or dev
years (2019, 2021) for calibration, and only dev events (4 GB, 5 ERCOT) for frequency calibration.

```python
import gridsim as gs   # pip install -e ".[grid]" from the repo root, or sys.path.insert(0, 'shared/grid/gridsim')

# GridPulse (MW lost + inertia from a frequency trace): random states in one vectorised batch
states, losses, labels = gs.generate_states(20000, seed=1, systems=('ERCO', 'GB'))      # ~30 ms per state
ev = gs.frequency_event(states, losses, t_end=60, dt=0.1, record=False)                  # list of dicts
#   labels.loss_MW, labels.inertia_MWs (targets); ev[i]['t'], ev[i]['f'] (input; resample to 1 s, add real noise
#   from NESO quiet periods); filter labels/ufls if needed.  Or realistic states from a dispatched hour:
res = gs.dispatch(fleet, demand, wind_MW=w, solar_MW=s, imports_MW=imp, fixed_MW={'nuclear': nuc})
st = res.system_state(hour, region='ERCO')          # committed units, H x MVA, headroom incl. scheduled reserve

# DispatchNet (learned dispatcher vs the MILP): random fleets + real input shapes; the MILP is the label
fleet = gs.generate_fleet(seed=7, peak_MW=60000)
lab = gs.dispatch(fleet, demand, wind_cf=wcf, solar_cf=scf, prices={'NG': gas_series, 'SUB': 2.0}, method='milp')
lab.p, lab.u, lab.price, lab.frame()                 # per-cluster MW, units on, $/MWh; method='lp' ~2x faster

# PlantDetective (heat-rate curve, min load, ramps, starts from output): unit-level commitment
lab = gs.dispatch(fleet, demand, wind_cf=wcf, solar_cf=scf, cluster=False, outages='random', seed=3)
#   each cluster is one unit: lab.p[:, j], lab.su[:, j], lab.fuel_MMBtu[:, j]; truth in lab.cf.clusters[j]
#   (pmin, pmax, ru, F0/ihr segments, start cost). Add CAMPD-like measurement noise yourself.

# BatteryPilot: price scenarios with the right daily shape (scarcity spikes are NOT realistic, LIMITS C);
#   physics 1.2 (opt-in) adds the real-time scarcity the day-ahead market cannot see (gridsim.market, below).

# DemandCast: weather -> demand and wind/solar
p = gs.fit_demand(weather_train, load_train); gs.demand_model(weather, p)
gs.fleet_profiles(units, weather_by_point, kind='wind')   # per-farm physics, nearest weather point
```

Fleets for other BAs (CISO, PJM, MISO, SWPP, NYIS, ISNE, BPAT) are in `grid/data/proc/fleet/` and run the same way;
only ERCOT has been checked against reality.

## Physics versions

`$GRIDSIM_PHYSICS` picks the version (read when gridsim is imported; `gridsim.PHYSICS_VERSION`).

| version | what it is |
|---|---|
| `1.0` | as released on 14 Nov 2026 (outage defaults from Texas RE's 2024/2025 reports: an exam leak; kept to reproduce) |
| `1.1` | **the default.** Outage defaults from pre-2023 sources; dispatch and frequency re-calibrated on dev (`data/calibration_v1_1.json`) |
| `1.2` | opt-in (15 Nov 2026): 1.1 plus real-time scarcity the day-ahead market cannot see (`gridsim.market`, parameters `data/rt_scarcity_v1_2.json`, fitted on the training-input years 2018, 2020, 2022 only). Every other output equals 1.1: same outage defaults, same calibration file, and `market` returns zeros under 1.0 and 1.1 (`checks/rt_scarcity.json`, `identity`) |

```python
from gridsim import market                        # GRIDSIM_PHYSICS=1.2, or pass version='1.2'
adder = market.rt_scarcity(times, 'ERCO', seed=3, demand_err=(d - df) / df, da=da)   # $/MWh to add to RT
da_premium = market.expected(times, 'ERCO', da=da)       # what a risk-neutral day-ahead price adds for it
```

The process: an unforecast scarcity event starts in an hour with probability r0 x M[month] x g(z) x k(DA), where z
is the day-ahead demand-forecast miss in standard units (g = exp(0.57 z)) and k = exp(2.28 (rank - 0.5) + 1.32 level)
reads the DA price's own shape (its rank within the local day, and the log of DA over the median of the past week),
each normalised to mean 1. It lasts a geometric number of hours (mean 1.97 h); each of its hours adds $100 + X to the
RT price, X generalised Pareto (xi 0.45, scale $60.7 x exp(1.11 level), capped at $8,900), adjacent hours tied by a
Gaussian copula (rho 0.44). The DA knows when the system is tight and roughly how tight, not whether the spike comes.
Two earlier fits stay in the parameter file (`market.params('ERCO_first_fit')`: clock-hour profile, no DA terms;
`'ERCO_shape_only'`: no size term); BatteryPilot's dev twin test showed both put too little of the spike value in
the DA's dearest hours. ERCOT only; CAISO has no fit (its prices start in Aug 2023). Limits in
[LIMITS.md](LIMITS.md) section I.

## The physics

**Network** (`network.py`). MATPOWER conventions: π-model branches with off-nominal tap and phase shift at the from
end, bus shunts. AC power flow: Newton–Raphson in polar form with a sparse Jacobian (optional generator Q limits).
DC power flow: B′θ = P with phase-shifter injections (MATPOWER `makeBdc`). PTDF from a sparse LU of the reduced B
(single or distributed slack). DC-OPF in PTDF form, quadratic costs, solved by HiGHS (QP); LMP = λ + Σ μ_l PTDF_l,b.

**Fleet** (`fleet.py`). Contract specs, filled from technology defaults (RTS-GMLC for ramp/min up-down/start heat,
EIA averages for heat rates, NREL ATB for VOM, Kundur for inertia; forced and planned outage shares for coal, CC,
CT and gas steam from ERCOT's GADS statistics in Texas RE's 2024/2025 reliability assessments, so that availability
matches the reported equivalent availability factors: coal ~74 %, gas steam ~68 %, CC ~81 %, CT ~86 %). Heat-rate curves:
by default each fossil unit gets its technology's curve shape scaled so the heat rate at 75 % load equals its own
EIA-923 operating heat rate over the three years up to the simulated period (combined-cycle parts pooled per unit;
this moved the ERCOT CC average from an implausible 6.3 to 7.8 MMBtu/MWh). Fuel input F(P) = a + bP + cP² (convex)
is cut into 3 incremental segments; CO2 = fuel × the EPA Part 98 factor of the unit's fuel. Combined-cycle parts are
merged into blocks.

**Dispatch** (`dispatch.py`). Clustered unit commitment (integer number of identical units on per cluster;
Palmintier & Webster 2014): segment and headroom constraints, 10-minute reserve capability, start/stop transitions,
minimum up/down with the history carried across days, ramp limits with start-up/shut-down capability, battery SOC
with round-trip losses and reserve capability, hydro daily energy budget, wind/solar curtailment, load shedding at
$9,000/MWh (ERCOT 2019 cap). Prices are the duals of the energy balance, by default from the LP relaxation of the commitment problem.
A day whose carried history makes it infeasible (e.g. a unit retiring while still inside its minimum up time) is
re-solved without that history and counted (`n_days_relaxed`).

**Frequency** (`frequency.py`). Aggregated swing equation per area,
2E/f0 · dΔf/dt = ΣΔPm + P_FFR + P_SI + P_shed − P_lost − D·P_load·Δf/f0 − P_ties,
with E the kinetic energy of online synchronous machines (reduced when a lost unit carries inertia). Each unit group
has a governor: deadband (no-step), droop gain, valve lag with rate limit and headroom clip, then two lead-lags that
make the turbine: steam reheat (1 + sF_HP T_RH)/(1 + sT_RH), hydro transient droop and water hammer
(1 − sT_w)/(1 + sT_w/2), CCGT steam bottoming (only the gas turbines respond fast). Inverter resources: fast frequency
response (triggered steps with delay and ramp), droop from batteries/curtailed plants, synthetic inertia
(−K·df/dt, filtered and limited). Protection: embedded-generation RoCoF and under-frequency relays. UFLS stages by
interconnection (ERCOT 59.3/58.9/58.5 Hz, GB LFDD from 48.8 Hz). Deadbands by system: GB 0.015 Hz; ERCOT per
BAL-001-TRE-1, 0.034 Hz for steam and hydro and 0.0166 Hz for everything else. Load damping is an effective,
dev-fitted parameter: GB 1.5 %/Hz (9 Aug 2019 first nadir), ERCOT 2.5 %/% (5 ERCOT dev events; it absorbs fast
response the model lacks, see LIMITS 3); other US systems 1 %/% (textbook, not calibrated). Tie lines between areas with synchronising
coefficients. Fixed step 10 ms (exponential integration of the lags), output at 50 Hz, vectorised over a batch.

**Renewables** (`renewables.py`). Wind: hub-height speed from the 10 m/50 m shear of the weather data (or power or
log law), moist-air density, IEC density-equivalent speed, a pitch-regulated turbine curve from the specific power
(W/m², by online year after the LBNL market report) and Cp, smoothed into a farm curve (Gaussian spread of speeds),
wake 10 %, availability 97 %, electrical 2 %, cut-out and cold shutdown. Solar: pvlib solar position, Erbs
decomposition when only GHI is known, Perez transposition, single-axis tracking with backtracking or fixed tilt, SAPM
cell temperature, PVWatts DC (−0.37 %/K) and inverter clipping at the AC rating, 14 % system losses.

**Demand** (`demand.py`). Hour-of-week × day-type shape, seasonal harmonics and trend, heating and cooling
degree-hours on a lagged building temperature (time constant fitted), cooling × humidity (latent load), irradiance,
minus behind-the-meter PV (pvlib). Linear least squares inside a small grid search.

**Generator** (`generator.py`). Random fleets (unit sizes, heat rates, ramp/min up-down, H, governors from real
ranges; wind/solar/battery shares) and random committed system states (demand level, IBR share 0-85 %, spinning
margin covering the largest unit, governor deadbands of GB/ERCOT, battery FFR and droop, synthetic inertia sometimes,
load damping 0.5-2) with labels (inertia, IBR share, headroom, FFR, loss).

## Verification (numbers in `grid/checks/*.json`)

| check (file) | what | result |
|---|---|---|
| `powerflow.json` | AC Newton-Raphson and DC power flow vs PYPOWER 5.1 on case9/14/30/57/118/300, ACTIVSg200, ACTIVSg2000 (2,000 buses); pandapower on case9/30/118; DC-OPF with 8 tightened lines vs PYPOWER rundcopf | AC max |ΔVa| 8e-12°, |ΔVm| 2e-14 pu, |ΔPf| 3e-10 MW; DC |ΔVa| 3e-12°; DC-OPF cost rel. diff 3e-13, LMP max diff $0.013/MWh (ACTIVSg2000), $0.0001 (ACTIVSg200); DC vs AC flows: median error 0.6-5.7 % |
| `network_dispatch.json` | dispatch(..., network=) with PTDF line limits vs the DC-OPF, same congestion, one hour; then 24 h unit commitment on ACTIVSg2000 | cost and LMPs identical (max ΔLMP 1e-13 $/MWh ACTIVSg2000, 1e-14 ACTIVSg200); 24 h MILP: 16 lines monitored lazily, max overload 3e-12 %, 110 s |
| `frequency_analytic.json` | RoCoF = ΔP f0/2E; damping-only steady state; each governor model vs scipy.signal.lsim; closed-form nadir of the 2nd-order system; deadband + headroom steady state; two-area steady state and inter-area mode | RoCoF rel. error 4e-14; nadir vs closed form 0.028 mHz (h = 10 ms); governor traces RMS 0.001-0.30 mHz (hydro largest, non-minimum phase); deadband/headroom Δf_ss -0.4167 vs -0.4167 Hz; inter-area mode 2.802 vs 2.802 rad/s |
| `frequency_andes.json` | aggregate model vs ANDES 2.0 full 4-machine Kundur two-area (GENROU, EXDC2, TGOV1), 200 MW load step | nadir 59.601 vs 59.577 Hz at 2.92 vs 2.84 s; 30 s 59.833 vs 59.822 Hz; RMS 15 mHz |
| `frequency_events.json` | GB 9 Aug 2019 from the ESO technical report (losses, response held/delivered, RoCoF and 49 Hz relays, LFDD); one parameter fitted (load damping, on nadir 1) | D = 1.5 %/Hz; nadir 1 49.11 Hz at 24.5 s (reported 49.1 at 24.5 s); peak before GT1A trip 49.20 Hz (49.2); 350 MW RoCoF trip at 1.2 s; LFDD at 75.6 s (75.9) |
| `frequency_dev_events.json` | the 4 GB and 5 ERCOT DEV events of splits.json (NESO 1 s; ERCOT operations reports); ERCOT state from that day's dispatch, inertia set to ERCOT's value; ERCOT effective damping fitted on these 5 | GB nadir error 20190809 +3 %, 20191203 -316 % (B's loss 1,788 MW looks wrong: the 0.29 Hz drop at 333 GVA s implies ~850 MW), 20190711 -27 %, 20190701 -34 %; 9 Aug RMS vs the 1 s trace 44 mHz over 0-75 s (274 mHz to 100 s, after LFDD the simulation recovers too fast); ERCOT nadir errors +6 %, +16 %, -15 %, -5 %, -0 % after fitting D = 2.5 %/% (before: 41-70 % too deep); dispatch-derived inertia of those hours within -10/+6 % of reported |
| `ercot_dispatch_2019.json` economic | ERCOT 2019 (dev), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price | annual: gas +18.7 %, coal -42.6 %; hourly nMAE gas 19 % (r 0.97), coal 43 % (r 0.79); CO2 -15.1 % vs CAMPD (161 Mt), hourly r 0.93; price median 21.4 vs 20.5 $/MWh, median abs. error 2.6, rank r 0.87, hours ≤ $0 3 % (real 0 %); wall 20.4 min |
| `ercot_dispatch_2019.json` calibrated | ERCOT 2019 (dev), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price | annual: gas -2.2 %, coal +5.6 %; hourly nMAE gas 6 % (r 0.98), coal 14 % (r 0.79); CO2 -1.0 % vs CAMPD (161 Mt), hourly r 0.96; price median 18.6 vs 20.5 $/MWh, median abs. error 2.6, rank r 0.87, hours ≤ $0 2 % (real 0 %); wall 12.8 min |
| `ercot_dispatch_2019.json` merit | ERCOT 2019 (dev), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price | annual: gas +23.6 %, coal -53.6 %; hourly nMAE gas 24 % (r 0.95), coal 54 % (r 0.81); CO2 -29.5 % vs CAMPD (161 Mt), hourly r 0.96; price median 21.5 vs 20.5 $/MWh, median abs. error 3.5, rank r 0.81, hours ≤ $0 0 % (real 0 %); wall 0.0 min |
| `ercot_dispatch_2022.json` economic | ERCOT 2022 (train-input year, not used for calibration), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price | annual: gas -9.6 %, coal +23.1 %; hourly nMAE gas 11 % (r 0.98), coal 26 % (r 0.55); CO2 +0.3 % vs CAMPD (159 Mt), hourly r 0.97; price median 46.5 vs 48.0 $/MWh, median abs. error 7.2, rank r 0.88, hours ≤ $0 0 % (real 0 %); wall 20.9 min |
| `ercot_dispatch_2022.json` calibrated | ERCOT 2022 (train-input year, not used for calibration), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price | annual: gas -11.4 %, coal +27.7 %; hourly nMAE gas 12 % (r 0.99), coal 29 % (r 0.52); CO2 +1.8 % vs CAMPD (159 Mt), hourly r 0.98; price median 46.6 vs 48.0 $/MWh, median abs. error 7.0, rank r 0.88, hours ≤ $0 0 % (real 0 %); wall 17.0 min |
| `ercot_dispatch_2022.json` merit | ERCOT 2022 (train-input year, not used for calibration), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price | annual: gas -10.5 %, coal +25.2 %; hourly nMAE gas 12 % (r 0.98), coal 28 % (r 0.48); CO2 -8.4 % vs CAMPD (159 Mt), hourly r 0.98; price median 45.2 vs 48.0 $/MWh, median abs. error 8.8, rank r 0.84, hours ≤ $0 0 % (real 0 %); wall 0.0 min |
| `ercot_inertia.json` 2019_calibrated | hourly inertia of the simulated commitment vs Texas RE's reported minimum and hours below 150 GW s | min 131 GW s vs 134.6 reported (-2 %); hours < 150 GW s: 127 vs 79; median 278 GW s |
| `ercot_inertia.json` 2022_calibrated | hourly inertia of the simulated commitment vs Texas RE's reported minimum and hours below 150 GW s | min 76 GW s vs 115.0 reported (-34 %); hours < 150 GW s: 949 vs 492; median 242 GW s |
| `renewables.json` 2019 | ERCOT wind and solar fleets from NASA POWER weather vs EIA-930 | wind bias +5.9 % (dev-scaled +0.0 %), hourly r 0.90, nMAE 19 %; solar bias +19.6 %, hourly r 0.98, nMAE 22 % |
| `renewables.json` 2022 | ERCOT wind and solar fleets from NASA POWER weather vs EIA-930 | wind bias +7.8 % (dev-scaled +1.9 %), hourly r 0.91, nMAE 20 %; solar bias +3.6 %, hourly r 0.97, nMAE 15 % |
| `demand.json` | weather-driven ERCOT demand, fit 2018-19, test 2021 (Uri week excluded) and 2022 | MAPE 3.8 % / 4.3 % (bias -3.4 % in 2022); ERCOT's own day-ahead forecast 2.4 / 2.8 % |
| `rt_scarcity.json` (physics 1.2) | the RT scarcity process alone (200 draws per year, driven by the year's real demand-forecast misses and DA prices) vs the real HB_HUBAVG RT - DA spread; fit on 2018/2020/2022, checked on dev 2019 and 2021 | hours with RT - DA > $100 per year: sim mean 106-125 vs real 87 / 81 / 165 (train) and 117 / 104 vs 108 / 99 (dev 2019 / 2021 without Uri; 224 with Uri); share of the spike $ in the day's 3 dearest DA hours: sim 0.50-0.57 vs real 0.53-0.73 (train), 0.57 / 0.51 vs 0.86 / 0.62 (dev; first fit 0.21-0.24); hour-of-day profile corr 0.79-0.89 (train), 0.82 / 0.89 (dev); month profile corr 0.23-0.78 (train), 0.94 / 0.72 (dev) |
| `generator.json` | 2,000 generated states and 40 generated fleets vs real ranges | inertia per MW demand p5/p50/p95 2.1/4.6/7.1 s (ERCOT 2019 min 4.6 s, GB 9 Aug 7.2 s); RoCoF at t=0 = formula to 2e-15; UFLS in 13 % (60 Hz) / 5 % (50 Hz) of random events; heat rates inside the ERCOT EIA-923 spread |

Dev calibration (all in `gridsim/data/calibration.json`, fitted on dev data of `splits.json` only): ERCOT coal
self-commitment + coal offer adder −$6.62/MWh (2019, 12 sample weeks, LP), GB load damping 1.5 %/Hz (9 Aug 2019
first nadir), ERCOT effective damping 2.5 %/% (5 ERCOT dev events). 2022 is a train-input year never used for
fitting, so its rows are the honest out-of-sample dispatch check. The earlier runs with generic availability are kept
in `checks/ercot_dispatch_<year>_v0_generic_availability.json`.

## Speed (single thread, shared 2-core Xeon 2.1 GHz; `checks/speed.json`)

| task | time |
|---|---|
| AC power flow, ACTIVSg2000 (2,000 buses) | 36 ms |
| DC power flow / PTDF build / DC-OPF, ACTIVSg2000 | 4 ms / 0.36 s / 0.68 s |
| one 60 s frequency event (50 Hz output, 10 ms step) | 304 ms |
| batch of 95 events with full generated states, 60 s each | 98 ms per event |
| generate one random system state | 29 ms |
| ERCOT week: merit / LP / MILP | 1.9 / 5.0 / 10.0 s |
| ERCOT solver time per day (median): LP / MILP | 0.43 / 1.08 s |
| ERCOT 2019 full year (8,760 h), wall incl. setup, shared CPU | calibrated 12.8 min, merit 0.0 min, economic 20.4 min |
| ERCOT 2022 full year (8,760 h), wall incl. setup, shared CPU | calibrated 17.0 min, merit 0.0 min, economic 20.9 min |

The ≤ 10 min target for a year of ERCOT dispatch is met by `'lp'` (solver time ~0.43 s/day, ~3 min a year on a free
core) and `'merit'` (~2 s); `'milp'` takes 13-21 min wall here (1.1 s/day solver time, plus model building, on a CPU
shared with other jobs).

## Files

```
gridsim/        the package: network.py, fleet.py, dispatch.py, frequency.py, renewables.py, demand.py, generator.py,
                market.py (physics 1.2 RT scarcity), data/calibration.json (dev-year ERCOT dispatch calibration;
                1.1: calibration_v1_1.json), data/rt_scarcity_v1_2.json (train-year fit, checks/fit_rt_scarcity.py)
refdata/        (git-ignored; `mlratchet data grid` restores it) matpower/ (IEEE + ACTIVSg cases), fuel/ (EIA-923 monthly receipts via PUDL, Henry Hub daily),
                hr/unit_heat_rates.parquet (EIA-923 fuel and generation per generator, via PUDL)
tools/          build_heat_rates.py
demo.py         one ERCOT day + one frequency event -> demo_day.png
.refenv/        isolated venv with pandapower, PYPOWER, PyPSA, ANDES (checks only; git-ignored; pass it as run_all.py --ref)
../checks/      every check script and its JSON; run_all.py re-runs them in order and compares every number with v1: check_powerflow.py,
                check_network_dispatch.py, check_frequency_analytic.py, check_frequency_events.py,
                check_frequency_dev_events.py, check_andes.py, check_renewables.py, check_demand.py, check_generator.py,
                calibrate_ercot.py, check_ercot_dispatch.py, summarize_ercot.py, check_ercot_inertia.py, check_speed.py;
                make_tables.py prints the tables above; out/ has hourly ERCOT runs and figures
```

## Sources and licences

MATPOWER test cases (BSD-3) including the Texas A&M ACTIVSg200/2000 synthetic grids (Birchfield et al. 2017, CC-BY
4.0; taken from the MATPOWER repository because the TAMU site requires a registration form). EIA-930 hourly balancing
authority data, EIA-860/923 (public domain; via GridData's files and Catalyst Cooperative PUDL, CC-BY 4.0). EIA Henry
Hub daily spot price (public domain). EPA CAMPD hourly emissions (public domain; GridData). NASA POWER hourly weather
(public, NASA LaRC). ERCOT day-ahead hub prices (ERCOT public reports; GridData). National Grid ESO "Technical Report
on the events of 9 August 2019" (6 Sep 2019) and Texas RE "Assessment of Reliability Performance" (appendix on
frequency response and inertia), for event and inertia figures. RTS-GMLC (Barrows et al. 2020, generator defaults via
GridData). References: Kundur, *Power System Stability and Control* (1994); IEEE PES-TR1 *Dynamic Models for Turbine-
Governors* (2013); Palmintier & Webster, *IEEE TPWRS* 29(3), 2014 (clustered UC); Morales-España et al., *IEEE TPWRS*
28(4), 2013 (ramping with start-up/shut-down); Nørgaard & Holttinen (2004) multi-turbine power curve; IEC 61400-12-1;
Perez et al. (1990); PVWatts (Dobos 2014). Check libraries: PYPOWER 5.1 (BSD), pandapower 3.5 (BSD), PyPSA (MIT),
ANDES 2.0 (GPL-3, used only as an external check, not imported by gridsim), pvlib (BSD-3), HiGHS (MIT).
