# RoadSim

A vehicle simulator built from the physics, for training networks on. It drives any vehicle (car to Class 8 truck,
gasoline/diesel, hybrid of four kinds, plug-in, battery-electric) over any 1 Hz trip (a real logged route with its
weather, or a test cycle), second by second, and returns every force, power flow and state, plus an **energy ledger**
that must balance on every run.

Part of ML Ratchet ("Trained on physics. Graded on reality."), Phase 1 of the vehicle project. Phase 2 trains
EnergyNet, Detective and Controller on it. Known errors are listed in [LIMITS.md](LIMITS.md); every check is a JSON
file in `road/checks/` with its numbers.

## Quick start

```python
import sys; sys.path.insert(0, 'shared/vehicle/roadsim')   # or: pip install -e . at the repo root
from roadsim import simulate, load_vehicle, make_trip, list_strategies

car = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})   # defaults fill everything else
trip = make_trip('udds', T_amb=-5, wind_speed=6, wind_dir=270, heading=0)   # or a real contract trip dict
out = simulate(car, trip)                     # default strategy for the architecture (A-ECMS here)
out['summary']['mpg'], out['ledger']['closure_rel'], out['fuel_rate'][:10]
list_strategies('hev_powersplit')             # ['powersplit_rule', 'ecms', 'aecms', 'dp', 'mpc']
```

`python3 demo.py` drives one 32-minute cold, windy, hilly commute with all seven powertrains in the same car body and
prints the ledgers (and writes `demo_ledger.png`).

## API

```python
out = simulate(vehicle_spec, trip_inputs, strategy=None, dt=1.0, init_soc=None, init_temps=None, rng=None,
               driver=None, record=True, check=True, ledger_tol=0.005, options=<default physics>)
```

| argument | meaning |
|---|---|
| `vehicle_spec` | spec dict (contract, below), a path to a `.json`, an id in `roadsim/data/vehicles/`, or a compiled `Vehicle`. Specs are compiled once and cached (by content hash), so loops over trips are cheap. |
| `trip_inputs` | dict of 1 Hz arrays. Only `v` [m/s] is required. Optional: `t`, `grade` [rise/run], `elev` [m], `T_amb` [°C], `wind_speed` [m/s], `wind_dir` [deg, FROM], `heading` [deg, travel direction], `rho_air` [kg/m³], `surface` [0 dry asphalt, 1 concrete, 2 wet, 3 gravel, 4 snow], and the extra `solar` [W/m²]. NaNs are interpolated; missing `rho_air` comes from `T_amb` and `elev` (ISA pressure). Real contract trips from `realdata.load_trips` work as they are. |
| `strategy` | a name from `list_strategies(arch)` or `{'strategy': name, 'params': {...}}`; default: the spec's `control.strategy`, else the architecture default. |
| `init_soc` | start SOC (default: HEV middle of window, PHEV/BEV 0.9). |
| `init_temps` | `{'engine', 'cabin', 'battery', 'driveline': °C, 'tire_warm': 0..1, 'eng_on', 'gear'}`. Default: everything soaked at the first ambient temperature, cold tyres (a cold start). For real-trip fragments that start mid-drive, pass warm values. `out['final_state']` has the same keys, so trips can be chained. |
| `rng` | seed or `numpy.random.Generator`: varies the driver's PI gains ±30 % (deterministic if None). |
| `driver` | `{'Kp': 0.5, 'Ki': 0.05, 'a_max', 'a_min'}`. Feed-forward from the road-load model plus PI on the lagged speed error. |
| `record` | False skips the per-second arrays (ledger and summary only); about 10-20 % faster. |
| `options` | physics: omitted = `roadsim.PHYSICS_VERSION` (`'roadsim-physics-v2'` since 13 Nov 2026); `'roadsim-physics-v1'` or `None` = physics v1 exactly; a preset name; or a dict of options (`{'preset': 'roadsim-physics-v2', ...}` changes single values of a preset; a dict without `preset` builds on v1). See [Physics versions and options](#physics-versions-and-options). |

**Returns** a dict. Arrays (float32, length N; quantities over second k→k+1 are stored at index k+1, like a logger's
rate over the last second; states are instantaneous):

| group | keys (units) |
|---|---|
| motion | `t` s, `v` m/s, `v_target`, `a` m/s², `headwind` m/s, `rho_air` kg/m³ |
| road loads (N) | `F_aero` (wind component along heading, yaw-dependent CdA), `F_roll` (Crr with temperature, speed, surface, pressure, tyre warm-up), `F_grade`, `F_inertia` (m_eff·a incl. rotating mass), `F_wheel` (powertrain at the wheels), `F_brake` (friction) |
| powers (W) | `P_wheel`, `P_brake` (friction), `P_regen` (electrical recovered), `aux_power` (at its supply point), `hvac_power`, `trans_loss`, `mot_loss`, `batt_loss` |
| engine | `eng_rpm`, `eng_torque` Nm, `eng_power` W, `fuel_rate` g/s, `eng_on` 0/1, `T_eng` °C, `gear` (1 = first) |
| motors | `mot_rpm`, `mot_torque`, `mot_power` (electrical, W), `gen_rpm`, `gen_torque`, `gen_power` (MG1 / generator) |
| battery | `batt_current` A (+ discharge), `batt_voltage` V, `soc`, `T_batt` °C; cabin `T_cabin` |

`out['ledger']` (kJ over the trip): `fuel` (chemical, LHV), `batt_chem_out` (chemical energy taken from the battery;
`batt_stored_change` = minus that), `eng_loss`, `mot_loss`, `trans_loss` (gearbox, torque converter, clutch slip,
final drive, rotating inertia), `batt_loss` (I²R and RC), `aux` (auxiliaries + HVAC + battery thermal management at
their supply), `aero`, `rolling`, `grade` (net), `brake` (friction), `kinetic_change`, `regen` (info), `residual`,
`closure_rel` = |sources − sinks| / scale, `ok` (closure < 0.5 %), `worst_step_W`.

`out['summary']`: `distance_m`, `duration_s`, `fuel_g`, `fuel_L`, `L_per_100km`, `mpg`, `batt_kWh_out`,
`kWh_per_100km_dc`, `soc_start/end`, trace-miss stats (`trace_rms_mps`, `trace_max_mps`, `trace_frac_over_2mph`),
`wall_s`, `x_realtime`.

Other entry points:

```python
from roadsim import load_vehicle, fill_spec, list_strategies, make_trip, cycle, list_cycles, load_trip, save_trip, sim_to_contract
from roadsim.generator import generate_vehicles      # random, physically consistent vehicles of every class/architecture
specs = generate_vehicles(10000, seed=0)             # or cls='bus', arch='bev'
from roadsim.dp import dp_solve                     # offline DP (used by strategy='dp')
from roadsim.coastdown import coast, fit_abc        # virtual coast-down + J2263-style fit
```

`list_cycles()`: udds, hwfet, us06, sc03, nycc, la92 (EPA), hd_transient (40 CFR 1037 App. A), gem_55 / gem_65 (cruise
over the App. D grade profile; `make_trip('gem_65')` attaches the grade), cbd14 (bus).

## Physics versions and options

A physics version is a named set of option values (`roadsim/options.py`: `DEFAULTS`, `PRESETS`). Every option's
default is the v1 behaviour, so v1 is always one name away and reproduces exactly (`tests/test_roadsim_fingerprint.py`,
171 runs to 1e-9).

| name | what it is |
|---|---|
| `roadsim-physics-v1` | the Phase 1 simulator (13 Nov 2026); the three released models were trained and scored on it and pin it |
| `roadsim-physics-v2` | **the default.** v2 physics calibrated on dev data only, fitted numbers in `roadsim/data/calibration_v2.json` (each with its dev source), decisions and dev numbers in [CALIBRATION.md](CALIBRATION.md), held-out results in [LIMITS.md](LIMITS.md) |
| `roadsim-physics-v2-b1` / `-b2` / `-b3` | the three work packages as built, before calibration (`docs/v2_b1.md`, `v2_b2.md`, `v2_b3.md`) |

```python
simulate(spec, trip)                                                    # physics v2
simulate(spec, trip, options='roadsim-physics-v1')                     # physics v1, exactly
simulate(spec, trip, options={'preset': 'roadsim-physics-v2', 'onroad_aux_W': {}})   # v2, one value changed
simulate(spec, trip, options={'series_direct_drive': True})            # v1 plus one feature
spec['physics'] = {'hvac_model': 'v2'}                                 # per vehicle (preset < spec < options dict)
```

What v2 adds over v1, by group (all behind options; the full list with units is the commented `DEFAULTS` dict):

| group | v2 values (preset) |
|---|---|
| hybrid control | P2 torque converter and start energy, min on 8 s / off 3 s, stay-on above 15 % of engine power, 20 kJ start penalty; Toyota rules (engine on above 45 km/h, warm-up 40 C, catalyst hold 40 s); e:HEV lock-up 22.7 rpm per km/h from 65 km/h, engine drive preferred; P2 brake blending (`regen_share` 0.75); plug-in charge-depleting rules; DP with on/off timers |
| engine and cold | coolant + oil nodes, Vogel oil, enrichment; per-map efficiency tilt for the five conventional gasoline maps (eta 1.04, low-load 450 W/L, transient 0.5 s); converter slip in AT conventional cars; axle and tyre temperature |
| HVAC | two-node cabin (conduction 0.3 of UA), explicit air flows, belt and electric compressors (COP / Carnot 0.25 / 0.50), PTC and heat pump; plug-ins heat from a warm engine first and run it for heat below -10 C; bus accessories, door exchange |
| on-road | road-height wind (log profile, z0 0.3 m), tyre model; on-road extra load per architecture (conventional 700 W, HEV 1,250 W, plug-in and BEV 300 W, road runs only) |
| options kept off in v2 | 10 Hz sub-steps, gusts, drafting, human driver, charge cap, full-load enrichment, battery aging (inputs: `batt_age`, `batt_soh_*`) |

`checks/run_all.py --options NAME` runs every check with a physics version and writes `<check>__NAME.json` (v1: the
plain names) and `scoreboard_NAME.json`; `scoreboard.py --compare roadsim-physics-v1 roadsim-physics-v2` prints the
differences. Exam / held-out results are computed only with `--all-splits`, for a final report.

### Using it at scale (Phase 2)

- One thread is plenty: a conventional car or BEV runs at ~40,000-70,000× real time, a hybrid at ~4,000×, a mixed
  generated fleet at ~10,000× (see the speed check). A million simulated trip-seconds takes a few minutes.
- Compile specs once (`get_vehicle(spec)` caches) and loop over trips; use `record=False` when only totals are needed.
- For real-trip twins: `rd.load_trips(...)` gives contract arrays; pass them straight in. VED trips are mostly fragments
  of longer drives, so pass warm `init_temps` (and a plausible `init_soc`) unless the trip is a cold start.
- `sim_to_contract(out)` renames outputs to the contract's measured-signal names (`fuel_rate`, `engine_rpm`,
  `batt_current`, `batt_voltage`, `soc`, `aux_power`).
- Strategies `dp` and `mpc` look ahead on the trip (they solve on the given speed trace): use them as benchmarks, not as
  causal controllers. The rest are causal.

## Vehicle spec (contract)

All SI. Every field is optional except `class` and `architecture`; `fill_spec` shows the completed spec.

```jsonc
{
 "id": "...", "class": "car|suv|pickup|van|bus|truck_md|truck_hd",
 "architecture": "conv|hev_parallel_p2|hev_series|hev_powersplit|phev_powersplit|phev_series|bev",
 "mass_kg": 1650, "payload_kg": 0, "CdA_m2": 0.62, "Crr": 0.0085, "Crr_speed_coef": 0.004,   // Crr at 20 C, dry asphalt; x (1 + coef * v[m/s])
 "road_load_ABC": {"A_lbf":..,"B_lbf_mph":..,"C_lbf_mph2":..},     // optional; spec_from_abc() turns EPA targets into CdA/Crr
 "wheel_radius_m": 0.32, "final_drive": 3.3, "driveline_eff": 0.975, "rot_mass_frac": 0.025, "yaw_drag_coef": 0.0007,
 "tire": {"K_t": 0.008, "pressure_ratio": 1.0, "warmup_amp": 0.15, "warmup_tau_s": 1200, "temp_effects": true},
 "engine": {"fuel": "gasoline|diesel", "displacement_L": 2.5, "max_power_kW": 150, "bsfc_map": "<map id>",
            "idle_rpm": 650, "max_rpm": null, "eff_scale": 1.0, "idle_stop": false, "inertia_kgm2": null,
            "warmup_model": {"T_hot": 88, "C_J_K": .., "hA_W_K": .., "f_coolant": 0.45, "cold_fric_Nm_per_L": ..}},
 "transmission": {"type": "AT|DCT|CVT|AMT|single|eCVT", "gears": [..], "n_gears": 8, "ratio_range": [2.6, 0.4],
                  "ring_teeth": 78, "sun_teeth": 30, "mg2_ratio": 2.636, "neutral_idle_factor": 0.5},
 "motors": [{"role": "P2|MG1|MG2|traction|generator", "max_power_kW": .., "max_torque_Nm": .., "max_rpm": .., "eff_map": "<map id>", "loss_scale": 1.0}],
 "battery": {"chemistry": "NMC|NCA|LFP|NiMH", "capacity_kWh": .., "nominal_V": .., "R0_ohm": null, "soc_window": [lo, hi],
             "max_power_kW": null, "max_charge_kW": null, "thermal": {"C_J_K":.., "hA_W_K":.., "heater_W":.., "cooler_W":.., "T_heat_on": 0, "T_cool_on": 35}},
 "aux": {"base_W": 350, "HVAC": {"source": "engine|ptc|heat_pump", "UA_W_K": 120, "C_J_K": 9e4, "T_heat_set": 21, "T_cool_set": 22,
          "heat_on_below_C": 16, "cool_on_above_C": 22, "heat_max_W": 6000, "cool_max_W": 5000, "blower_W": 150, "solar_area_m2": 0.45, "enabled": true}},
 "regen": {"blend": "series|parallel|one_pedal", "max_decel_g": 0.3, "min_speed_mps": 2.0, "parallel_share": 0.6},
 "control": {"strategy": "<id>", "params": {}}
}
```

Engine map ids: `gas_na_toyota_a25a`, `gas_na_mazda_20`, `gas_turbo_honda_15`, `gas_turbo_ford_27`, `gas_v6_gm_43`,
`gas_v8_ford_73`, `gas_atk_toyota_tnga_hev_paper` (hybrid default), `gas_atk_toyota_tnga_hev`, `diesel_gm_30`,
`diesel_bmw_n57`, `willans_hd_diesel` (heavy duty), `willans_si`. Motor map ids: `ipm_generic_150` (default),
`bolt_edu_150`, `prius_mg2_60`, `prius_mg1_est`, `sonata_p2_30`. Reference vehicles in `roadsim/data/vehicles/`:
Class 8 sleeper (diesel, BEV), Class 6 box truck, 40 ft bus (diesel, series hybrid, BEV).

## The physics, subsystem by subsystem

**Road loads.** Aero `½ρ CdA (v + w_h)|v + w_h|`, where w_h is the wind component along the heading
(`wind_speed·cos(wind_dir − heading)`); crosswind raises CdA by `yaw_drag_coef·ψ²` (ψ in degrees, wind-averaged drag
practice, SAE J1252; ~7 % at 10° for cars, ~25 % for tractor-trailers). Air density from temperature and pressure
(ideal gas; ISA pressure from elevation). Rolling: `m g cosθ Crr (1 + c_v v)` × ISO 28580 ambient temperature
correction (K_t 0.008 cars, 0.010 trucks) × cold-tyre inflation (ideal gas, set at 20 °C, Crr ∝ p^−0.4) × surface
factor (dry 1, concrete 0.95, wet 1.12, gravel 1.6, snow 2.2; Wong, *Theory of Ground Vehicles*) × tyre warm-up
(+15 % cold, τ = 20 min of driving). Grade `m g sinθ`. Inertia `m_eff a` with wheels/driveline in m_eff, plus engine
and transmission-input inertia in the engaged gear (Gillespie mass-factor practice).

**Driver and dynamics.** Forward-facing. Each second the driver asks for the force that reaches the next target speed
(feed-forward through the same road-load model) plus PI on the lagged speed error; the powertrain delivers what it can;
speed follows from momentum with all work evaluated at the step's mean speed, which makes the ledger close exactly.
When the powertrain is at its limit the step is iterated (damped) until the delivered force and the speed agree.

**Engine.** EPA ALPHA fuel maps (measured benchmarks, NVFEL/NCAT) resized to the spec's rated power at constant BMEP
(and speed-scaled if `max_rpm` is given), stored as fuel power so any fuel works; heavy-duty diesel: Willans-line map
with Chen-Flynn friction, peak BTE ~44.5 %. Idle at the map's idle speed carrying auxiliaries (and, for automatics, the
converter's creep load, halved for neutral-idle). Fuel cut-off whenever the requested torque is at or below the motoring
torque. Cold start: lumped coolant temperature (heat in = 45 % of losses, thermostat at 88 °C, heater core draws from
it); penalty = extra friction torque that follows oil viscosity (Arrhenius fit, friction ∝ √viscosity), fitted on dev
cars (it also stands for catalyst heating and enrichment).

**Transmissions.** AT with torque converter (K-factor / torque-ratio model, stall-equilibrium at full load, lock-up from
2nd gear), DCT/AMT/manual with a slipping launch clutch, CVT (ratio follows the engine's best-efficiency line, converter
launch), single-speed reduction, and the power-split eCVT (Willis equation, MG1 reaction torque, MG2 on the ring).
Losses = mesh efficiency + spin torque `c0 + c1 ω`, the spin part scaled by driveline-oil viscosity (oil temperature
state warmed by its own losses and the ATF cooler). Shift schedules: load/speed rule with hysteresis, kick-down and
overspeed upshift (`conv_rule`), or BSFC-optimal gear (`conv_opt`).

**Motors / inverters.** EPA ALPHA e-machine maps (ORNL/SwRI test data) stored as loss / P_max on normalised
(speed, torque) grids, so any size scales; torque envelope `min(T_max, P_max/ω)`. A machine at exactly zero torque
(free-wheeling MG1, idle P2) keeps 40 % of its zero-torque loss (iron + windage; inverter off).

**Battery.** Pack Thevenin model: OCV(SOC) from half-cell curves (NMC811/graphite, Chen 2020; LFP/graphite, Prada 2013
as in CellSense; NiMH typical), R0 = k V²/Wh (power vs energy cells) × Arrhenius(T) × low-SOC rise, one RC branch
(τ 40 s), voltage-window and temperature power limits, lumped thermal mass with optional heater/chiller (counted in aux).

**Regenerative braking.** Motor torque limit, battery charge acceptance (plus the aux it feeds), stability limit
(`max_decel_g` on the driven axle), fade below `min_speed`. Blending: `series` (regen first), `parallel` (fixed share),
`one_pedal`. Friction brakes take the rest.

**Auxiliaries and HVAC.** Base electrical load (via alternator at 65 % for conventional, DC/DC 93 % for electrified).
Lumped cabin (UA, C, solar, occupants) with heating below 16 °C ambient and cooling above 22 °C: engine coolant heat
(free but limited by coolant temperature; a cold hybrid engine is forced on for heat), PTC (COP 0.95) or heat pump (COP
falling with ambient, ~3 at 10 °C, ~1.4 at −10 °C), A/C with COP falling with condenser temperature (belt-driven on
conventional vehicles).

## Control strategies

| architecture | strategies (default first) |
|---|---|
| conv | `conv_rule` (load/speed shift schedule), `conv_opt` (BSFC-optimal gear), `conv_stopstart` |
| hev_series | `aecms`, `series_thermostat`, `series_power_follower`, `ecms`, `dp`, `mpc` |
| hev_powersplit | `aecms`, `powersplit_rule` (Prius-style, engine on the optimal operating line), `ecms`, `dp`, `mpc` |
| hev_parallel_p2 | `aecms`, `parallel_rule` (electric assist / load levelling), `ecms`, `dp`, `mpc` |
| phev_* | `phev_cdcs` (charge-depleting then charge-sustaining A-ECMS), `phev_blended` (SOC reference falls with distance), plus all HEV strategies |
| bev | `bev_regen_series`, `bev_regen_parallel`, `bev_one_pedal` |

ECMS: minimise fuel power + s × battery chemical power over the candidate controls (Paganelli 2002; SOC-window penalty
after Sciarretta 2004). A-ECMS: s from SOC feedback (Onori, Serrao & Rizzoni 2010). DP: Bellman recursion on
(engine on/off, SOC) with engine start costs, the cost-to-go then used as a closed-loop policy in the forward run.
MPC: the same recursion on a 15 s preview with an ECMS terminal value. Rule-based strategies use hysteresis and minimum
on/off times; hybrids force the engine on for catalyst light-off after a cold start (10 s) and for cabin heat.

## Verification (numbers in `road/checks/*.json`)

| check (file) | what | result |
|---|---|---|
| `cycles_vs_epa.json` | 47 real MY2022-24 EPA test vehicles (22 conventional, 14 HEV, 11 BEV), FTP-75 (cold start, 3 bags, 0.43/0.57 weighting), HWFET, US06 vs the EPA test-car list; 33 held-out | **conventional** FTP median +5.6 % (IQR −4.2/+8.0), HWFET −4.3 % (IQR −5.5/−3.0), US06 +0.2 % (n 5); **power-split HEV** −2.3 % over all cycles (range −10.1/+6.6); **P2 HEV** FTP +12 % (+8/+31), HWFET −6 %, US06 −5 %; **series (Honda)** −4 % FTP, −9.5 % HWFET; **BEV** UDDS −0.5 %, HWFET −1.7 % (range −7.2/+6.0). Cold-start bag1/bag3 ratio: median −0.2 % (IQR −4.2/+1.6, n 20). Extra, uncalibrated: SC03 conventional +12 % (n 2), hybrids −5 % (n 6); 20 °F cold FTP +13 % (n 2) |
| `coastdown.json` | virtual J2263 coast-down (wind, both directions, 10 Hz noise) for all 47 vehicles vs EPA target A/B/C; plus a 5 °C test corrected with SAE J2263 | road-load force error median 1.6 / 0.8 / 0.5 % at 20/40/60 mph (max 5.5 / 2.1 / 1.1 %); 92 % of vehicles inside the 3.1 % spread of repeat EPA targets; 5 °C corrected: +1.1 / +0.5 / +0.35 % (ISO 28580 tyre model vs SAE correction agree within ~1 %) |
| `fastsim_crosscheck.json` | NREL FASTSim 3.1.0 on the same 47 vehicles, UDDS/HWFET/US06 (hot) | with FASTSim's own template components: roadsim higher by +12/+13/+3 % (conv), +8/+1/−7 % (HEV), +26/+11/+15 % (BEV), and FASTSim's templates are the farther from EPA (conv HWFET −14 %, BEV UDDS −20 %). With components matched to roadsim's: **conv −0.3 / −3.1 / −3.0 %, BEV +4.2 / −1.4 / −1.4 %**, HEV +18 / +7 / −4 % (different hybrid controllers: FASTSim's rule vs our A-ECMS) |
| `heavy_duty.json` | Class 8 sleeper (diesel, BEV) on GEM cycles; Class 6 box truck; 40 ft bus (diesel, series hybrid, BEV) on CBD14 and HD transient with HVAC at 10-32 °C | Class 8 diesel 85.8 g CO2/ton-mile (6.25 mpg) vs 40 CFR 1037.106 standards 64.3-75.7 (truck without Phase 2 packages: expected above); Class 6 215 g/ton-mile (driving only) vs 235-265; diesel bus 4.6 mpg CBD14, BEV bus 1.66 kWh/mi vs 1.8-2.15 measured in service (NREL/TP-5400-80022); hybrid bus +76 % on CBD14 (too high), +35 % on HD transient |
| `ledger.json` | 600 stress runs: 300 generated vehicles, all classes/architectures, −25..40 °C, wind to 12 m/s, grades to ±8 %, all surfaces, cold/warm starts | closure median 1.2e-15, **max 5.1e-4 (0.05 %)**, none over 0.5 %, no NaN/inf. Every other check also logs closures (cycle check worst 1.3e-4) |
| `speed.json` | single thread, full recording, Xeon 2.1 GHz | car conventional 65,000x real time (UDDS), BEV 64,000x, power-split A-ECMS 4,300x, P2 5,500x, series 4,800x, PHEV 4,200x, Class 8 54,000x, BEV bus 57,000x; mixed generated fleet 8,100x incl. compile; DP 0.8 s and MPC 1.2 s per UDDS. Target was 200x |
| `strategies_hev.json` | every strategy on UDDS/HWFET, Camry Hybrid (power-split), Elantra Hybrid (P2), Accord Hybrid (series) | see the table below |
| `generator.json` | 400 generated light-duty vehicles vs the real MY2022-24 EPA fleet | hot UDDS / HWFET median: conventional 26.2 / 35.1 mpg (EPA FTP / HWY 25.6 / 38.0), BEV 132 / 114 MPGe (EPA 134 / 125), HEV 48 / 46 (EPA hybrids 59 / 58: generated hybrids too heavy); 0-60 mph median 6.3 s cars, 6.5 SUVs, 5.9 pickups, 10.1 vans, 5.5 BEVs |
| `calibration.json` | dev-only fits | cold friction 45 Nm/L at 20 °C, hybrid factor 0.16, BEV motor loss scale 0.47 |

Strategies (fuel economy corrected for net battery energy, % of DP; DP knows the cycle):

| vehicle | cycle | rule | ECMS | A-ECMS | MPC | DP (mpg) |
|---|---|---|---|---|---|---|
| Camry Hybrid (power-split) | UDDS | 95.4 % | 97.8 % | 97.8 % | 98.4 % | 80.7 |
| | HWFET | 96.3 % | 100.5 %* | 99.5 % | 100.1 %* | 73.3 |
| Elantra Hybrid (P2) | UDDS | 92.2 % | 97.2 % | 98.9 % | 98.9 % | 78.3 |
| | HWFET | 95.1 % | 102.2 %* | 99.8 % | 100.3 %* | 67.7 |
| Accord Hybrid (series) | UDDS | 98.4 % (thermostat) / 97.9 % (power follower) | 97.3 % | 99.2 % | 99.7 % | 69.8 |
| | HWFET | 99.4 % / 99.1 % | 98.9 % | 100.1 %* | 100.3 %* | 59.7 |

\* ended the cycle with more charge than DP (+0.03 to +0.12 SOC); the fixed SOC-correction factor (1/(0.33 x 0.90))
over-credits that charge. Constant-s ECMS does not hold SOC (−0.14 to +0.12 over one cycle); A-ECMS does (|ΔSOC| ≤ 0.06).

Four global constants were fitted on the **dev** third of the EPA vehicles only (`checks/calibrate_dev.py` →
`roadsim/data/calibration.json`): cold-start friction (bag-1/bag-3 ratio of dev conventional cars), its hybrid factor
(dev HEV), BEV drive-unit loss scale (dev BEVs), catalyst warm-up time (fixed at 10 s). Headline errors are on the
held-out vehicles. Re-run everything with `checks/run_all.sh`.

## Files

```
roadsim/          the package: sim.py (loop, ledger), powertrain.py, components.py, strategies.py, dp.py, vehicle.py,
                  generator.py, coastdown.py, trips.py, maps.py, calib.py, data/ (maps.npz, cycles.npz, vehicles/, calibration.json, generator_calib.json)
tools/            build_maps.py (ALPHA .m -> maps), build_cycles.py, build_generator_calib.py, epa_candidates.py
refdata/          downloaded sources (EPA test-car lists, ALPHA packages, eCFR XML, fueleconomy.gov, NREL report via OSTI)
demo.py           one trip, seven powertrains, ledgers + demo_ledger.png
../checks/        every check script and its JSON result; run_all.sh
```

## Sources and licences

US EPA test-car lists (MY2018, 2022-2024) and dynamometer drive schedules (epa.gov, public domain); US EPA ALPHA engine
and e-machine map packages (NVFEL/NCAT, public domain); eCFR 40 CFR Parts 86 and 1037 (cycles, grade profile, GEM
weightings, standards; public domain); fueleconomy.gov vehicles.csv (public domain); NREL/TP-5400-80022 Foothill
Transit Battery Electric Bus Evaluation (2021, via OSTI); NREL FASTSim 3.1.0 (BSD-3, cross-check only); battery
half-cell OCP functions as published in Chen et al. 2020 and Prada et al. 2013 (via PyBaMM parameter sets). Other
references: Heywood, *Internal Combustion Engine Fundamentals* (2018); Gillespie, *Fundamentals of Vehicle Dynamics*
(1992); Wong, *Theory of Ground Vehicles* (2008); ISO 28580; SAE J1263/J2263/J1711/J1634/J1376.
