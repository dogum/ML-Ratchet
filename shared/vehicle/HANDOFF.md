# Vehicle project handoff: what exists, what's missing, and how to finish it

For Greg's code agent, 13 Nov 2026. This is the "make it complete" list for the whole-vehicle work in ML Ratchet
(github.com/dogum/ML-Ratchet). Read this first, then `VEHICLE_BRIEF.md` (the contract), `roadsim/README.md`,
`roadsim/LIMITS.md` and `data/DATA.md`.

## 1. What exists

| Piece | Where | State |
|---|---|---|
| RoadSim simulator (`roadsim`, numpy/scipy) | `roadsim-roaddata-v1.0.zip` → `roadsim/` | v1, verified (section 3) |
| Real data loader (`realdata`), frozen splits, baselines | same zip → `realdata/`, `data/` | v1, two bugs fixed 13 Nov |
| Physics checks and their JSONs | same zip → `checks/` (`run_all.sh`) | pass; numbers in `roadsim/README.md` |
| EnergyNet (trip energy predictor) | `energynet-v1.0.zip`, page https://claude.ai/artifact/UYg6maW3hrMNAG3h8h6jRg | v1.0 released |
| TripDetective (infers the car from a log) | `tripdetective-v1.0.zip`, page https://claude.ai/artifact/2DFuvMBBAwREG9rx5e3tHg | v1.0 released |
| HybridPilot (learned hybrid energy management) | `hybridpilot-v1.0.zip`, page https://claude.ai/artifact/3MU8pXNtGc65HTDnQ8Cgup | v1.0 released |
| Status docs | claude.ai project "Physics Fun": `roadsim-status.md`, `energynet-status.md`, `tripdetective-status.md`, `hybridpilot-status.md` | current |

- Each model zip has a numpy-only inference module, weights, a demo on real exam data, `MODEL_CARD.md`, and `training/` with every script, including the dev-calibration wrappers listed in section 4.
- The HybridPilot demo needs `roadsim`; the other two don't.
- Not in any zip, rebuildable:
  - the processed VED trips, ≈1.2 GB (`data/scripts/`, sources in `data/DATA.md`)
  - the raw downloads
  - the EPA ALPHA map archive, 298 MB (`roadsim/tools/` fetches it; the maps actually used ship inside `roadsim/roadsim/data/`)
  - the simulated training banks

## 2. Greg's original ask vs what was built

Greg asked for every architecture, many hybrid control strategies, subsystem-level models, and real grades, speeds,
dynamics, terrain, tire rolling resistance, weights, aero, wind and temperature, all supported by real data where possible.

| Ask | Status | Notes / gap |
|---|---|---|
| Conventional gasoline and diesel | Done | Light-duty diesel uses the ALPHA-scaled or Willans map; no LD diesel exam data |
| Parallel hybrid (P2) | Done, **weak** | City FTP +8 to +31 %: missing torque converter, start and clutch losses |
| Series hybrid | Done, **partial** | Honda e:HEV direct-drive clutch not modelled (HWFET −9 %) |
| Power-split hybrid | Done, good | −2.3 % median vs EPA; engine runs less than real Toyotas (35 % vs 55 % of time) |
| PHEV (power-split, series/EREV) | Done, **weak on real data** | Mode per trip unknown in VED; simulated PHEVs start the engine for cabin heat too often |
| BEV | Done | Needed one loss-map scale (0.47); Leaf twin uses 1.5× real energy (cabin heat) |
| Mild hybrids (48 V P0/P1), fuel cell, hydrogen ICE | **Not built** | Not asked explicitly; natural extensions |
| Control strategies | Done | Rule, power-follower, thermostat, P2 assist, Prius-style split, ECMS, A-ECMS, MPC, DP, PHEV CD-CS / blended, BEV regen variants, learned (HybridPilot) |
| Engine: fuel maps, warm-up | Done, **partial** | 10 generic ALPHA maps resized; per-engine accuracy ±5–10 %; cold −7 °C too mild |
| Motor / inverter | Done | ALPHA generic IPM maps |
| Transmission | Done, **partial** | AT+TC, DCT, AMT, CVT, single, eCVT; losses generic; shifts lumped at 1 Hz |
| Battery | Done, **partial** | One RC branch, lumped thermal; no aging or capacity fade |
| Regenerative braking | Done, **partial** | Blending with motor and battery limits; stability limit is a fixed deceleration (no tyre model) |
| Real grades | Done | USGS 1 m lidar for Ann Arbor, repeatable to 0.36 %; elsewhere only via EPQS |
| Real speeds | Done | 33,015 real VED trips plus 24 EPA/HD cycles |
| Real dynamics | **Partial** | Longitudinal only, 1 Hz, near-perfect driver; no wheel slip, load transfer or tyre force model |
| Terrain and tire rolling resistance | **Partial** | Crr vs temperature, speed, surface (dry/wet/gravel/snow), inflation and warm-up; snow/gravel only raise Crr; no tyre model per brand or size |
| Different weights per style | Done | Generator draws class- and architecture-specific mass from EPA records; payload supported |
| Real aero | Done | CdA from EPA road-load targets; yaw correction |
| Wind | Done, **partial** | Steady wind per second, airport 10 m wind; no gusts, drafting or terrain sheltering |
| Temperature | Done, **partial** | Air density, Crr, cold start, HVAC, battery thermal; cold and A/C under-predicted |
| Light-duty real data | Done | VED (384 cars), EPA test cars (80,227 tests), fueleconomy.gov |
| Heavy-duty real data | **Weak** | ZTBus (10 trace missions), Mersin bus trip totals, 6 ICCT truck dyno numbers; no public truck trips with traces |

## 3. Verification baseline (don't regress these)

These are from `checks/*.json`. Re-run `checks/run_all.sh` after any physics change.

- **Energy ledger:** closes in all 600 stress runs (max 5e-4).
- **Speed:** about 60,000× real time for conventional/BEV and about 5,000× for HEV, single thread.
- **EPA, 33 held-out vehicles:**
  - conventional FTP +5.6 % / HWFET −4.3 %
  - power-split HEV −2.3 %
  - BEV UDDS −0.5 % / HWFET −1.7 %
  - US06 +0.2 % (n = 5)
- **FASTSim 3.1 with the same components:** conventional within 3 %, BEV within 4 %.
- **Exam numbers per model** are in each status doc and `MODEL_CARD.md`. Record them before changing anything, then re-score.

## 4. Prioritised work list

Each item has a "done when". Every physics change must:
1. bump `PHYSICS_VERSION`;
2. rebuild each model's frozen simulated set;
3. re-score every rung in order and re-take the ratchet decisions;
4. never touch `data/proc/splits.json` (sha256 d917df78…).

### P0: make it a proper part of ML Ratchet
1. **Paths.** Replace every hard-coded `/home/claude/road`, `/home/claude/...` and `/mnt/user-data/...` with one configurable root (env var or `mlratchet` config).
   - Done when `grep -r /home/claude` is empty and all four demos run from a fresh clone.
2. **Package layout.** Put RoadSim and realdata in the repo as a shared module, e.g. `mlratchet/shared/roadsim` and `mlratchet/shared/realdata`. The three models become `models/energynet`, `models/tripdetective`, `models/hybridpilot` with `model.yaml` and `ratchet.yaml` like the other nine.
   - Done when `mlratchet validate <id>` passes for all three and the site builds their pages.
3. **Data rebuild.** One command (`mlratchet data vehicle`) runs `data/scripts/` in order: VED download → stage 1/2 → lidar DEM → weather → EPA → fueleconomy → cycles → HD → baselines. It must check the splits hash at the end.
   - Done when the rebuilt `baselines.json` matches the shipped one.
4. **Licences in DATA_LICENSES.md:**
   - VED: Apache-2.0
   - ZTBus: CC-BY
   - Mersin bus data: CC-BY
   - EPA test-car data, fueleconomy.gov, EPA cycles and ALPHA maps: US government, public domain
   - USGS 3DEP lidar: public domain
   - Iowa Mesonet: free with credit
   - ICCT report numbers: cite
5. **Upstream the dev-calibrated wrappers** (now copies inside each model's `training/`), each behind a flag, then re-run checks:
   - EnergyNet `training/wrappers.py` (`src/` in its working folder):
     - MAF label model: VED's fuel reads ~0.2–0.3 g/s during fuel cut-off.
     - On-road loads: petrol 1380 W + 133 W/°C above 15 °C; hybrid 2831 W + 204 W/°C.
     - PHEV hardware patch: pack power ≥ 1.2 × motor, cell R ÷ 2.5, MG2 torque × 1.8.
   - EnergyNet `training/epa_fix.py`: superseded by the 13 Nov RoadData fix; delete after checking they agree.
   - TripDetective `training/priors.py`: the v3 mass prior matched to VED-era cars (median 1,720 kg vs the generator's 2,200 kg). `training/obdlog.py`: the OBD log emulator (per-PID polling rates, held values, MAF fuel estimate, quantisation).
   - HybridPilot findings:
     - Real Toyota hybrids keep the engine on above ~72 km/h.
     - PHEVs need electric cabin heat.
     - Engine-start penalty.
     - Matched-SOC grading (re-solve DP to the strategy's final SOC). Make this the default grading.

### P1: simulator fidelity (each item has a measured gap)

| Gap | Measured | Likely fix | Calibrate on | Done when |
|---|---|---|---|---|
| P2 hybrids in the city | FTP +8 to +31 % | Torque converter in P2 automatics; engine start and clutch-engagement energy; minimum engine-on time | EPA dev P2 vehicles | Held-out P2 FTP within ±8 % |
| Honda e:HEV | HWFET −9 %, US06 −10 % | Direct-drive clutch above ~70 km/h | EPA dev series HEVs | Within ±6 % |
| Cold weather | 20 °F FTP +10 to +16 % | Enrichment, driveline oil and tyre warm-up at low ambient; cabin heat demand | EPA dev FTP_cold, VED dev winter trips | Within ±6 % on held-out cold FTP |
| A/C on conventional cars | SC03 +9 to +15 % | Compressor and condenser-fan load, hot-soak pull-down | EPA dev SC03 | Within ±6 % |
| Leaf / BEV cabin heat | twin uses 1.5× real energy | Real Leaf heater curve (PTC); logged heater power exists for PHEV/EV in VED | VED dev Leaf | Twin battery energy within ±10 % per trip |
| Toyota HEV engine-on share | 35 % sim vs 55 % real; sim fuel 0.80× logged | Production calibration rules (speed threshold, warm-up hold, catalyst); maybe the MAF estimate counts motoring air | VED dev HEVs | Twin fuel within ±8 %, engine-on share within ±10 points |
| PHEV twins | 2–3.6× logged fuel | Cabin heat from electric heater, CD-mode rules | VED dev PHEVs | Twin total energy within ±15 % |
| Hybrid bus | +76 % vs diesel on CBD14 (literature 10–40 %) | Bus accessory loads, door-open HVAC losses, hybrid sizing | Published bus evaluations | Inside literature range |
| BEV Class 8 | 2.5 kWh/mi vs ~1.5–2.2 | Light-load motor losses on a large machine | Published fleet results | Inside range |
| Per-engine efficiency | ±5–10 % | More ALPHA maps, or fit `eff_scale` from EPA FTP/HWY per family (dev) | EPA dev | Conventional FTP/HWY IQR halved |
| Generated hybrids | ~20 % less efficient than real EPA hybrids | Draw hybrid bodies from efficient car/SUV donors | EPA dev | Generator HEV FE distribution matches EPA |
| City/highway tilt | FTP +5.6 %, HWFET −4.3 % | Low-load losses (idle, converter slip) up; highway driveline/aux down | EPA dev | Both medians within ±3 % |

### P1: real data still missing (blocked here; Greg can download in a browser)
- **Argonne D3 (Downloadable Dynamometer Database).** anl.gov returns 403 from the sandbox. It has 10 Hz dyno tests of modern HEVs, PHEVs and BEVs with fuel, current, voltage, engine speed and temperature. This is the best twin-calibration data available for the P1 table above, especially P2, e:HEV, cold and A/C. Use it as dev and a second exam, split by vehicle.
- **NREL Fleet DNA and bus evaluations.** nrel.gov is blocked. These are real truck and bus drive traces with fuel; they give heavy duty a real exam. Grade by vehicle, never by trip.
- **Full ZTBus set** (1,409 missions, ETH repository returned HTTP 429). Only 10 missions have traces now. Adding the rest grades EnergyNet on buses properly, and lets TripDetective estimate bus mass from passenger counts.
- **More EVs.** VED has only 3 Leafs (2 in the exam). Any public EV trip set with battery current would help EnergyNet's EV bias (−20 %).
- **A modern-fleet trip dataset.** VED is 2017–18 cars. EnergyNet's petrol highway bias (+15 % on EPA) comes from an old-fleet prior; a model-year input plus newer data fixes it.

### P2: physics Greg asked for that is still simplified
1. **Sub-second mode (10 Hz)** for shifts, engine starts, clutch slip and tip-ins. Today it's 1 Hz with lumped events. Needed for drivability metrics and P2 fidelity.
2. **Tyre model.** Slip, load transfer and a friction ellipse, so regen limits come from physics, and snow/gravel/wet change traction, not only Crr. Also tyre size and brand Crr classes (EU label A–E).
3. **Driver model.** Lag, overshoot and aggressiveness variation (VED shows it); eco vs aggressive drivers as an input.
4. **Wind.** Gusts and spectra, drafting, terrain sheltering; downscale airport wind to road height.
5. **Battery aging.** Capacity fade and resistance growth (CellSense's aging model can be reused), plus SOC-window effects.
6. **HVAC.** Humidity/defrost, door openings (buses), seat heaters, preconditioning.
7. **Heavy duty.**
   - AMT shift strategies, engine brake/retarder, trailer aero devices.
   - Payload and passenger variation (ZTBus logs passengers).
   - Idle and PTO, as GEM counts idle.
8. **More architectures (optional):** 48 V mild hybrids (P0/P1), diesel hybrids, fuel-cell EVs, range-extender trucks.

### P2: model next steps (from each model's report)
- **EnergyNet:**
  1. Cold-start calibration from EPA dev FTP_cold.
  2. More dev hybrid and EV twins, or a per-architecture σ model. Hybrids and Leafs lose to baselines (bias +12 % / −20 %), and 68 % bands cover only 47–65 % on VED.
  3. A model-era input.
  4. Real speed profiles for Mersin (it uses synthetic ones now).
  5. Feed ZTBus bus mass.
- **TripDetective:**
  1. Find which v2 prior hurts engine-size estimates and rebuild on v1 plus the v3 mass prior.
  2. Make mass and drag identifiable from the dynamics. Today +800 kg moves the estimate only ~200 kg and CdA barely registers. Options: a physics-regression head that separates mass from engine efficiency; pooled multi-trip training.
  3. Fix BEV heat (Leafs read 460 kg heavy).
  4. Add buses (ZTBus mass vs passengers).
  5. Architecture: it still trails the simple rule (98.9 vs 99.4 %).
- **HybridPilot:**
  1. Rebuild after the P1 twin fixes; the simulator-vs-real gap (~20 %) dwarfs the 1–3 % between strategies.
  2. A PHEV teacher that uses distance-to-go, and a blended-mode test.
  3. RL fine-tuning on matched-SOC fuel.
  4. A causal DP proxy that respects minimum on/off times.
  5. Truck routes beyond cycles.
  6. A drivability score (starts/km, on/off chatter) in the objective.

### P3: product
- **MCP tools**, each returning error bars and the model card's limits:
  - `roadsim.simulate`
  - `energynet.predict`
  - `tripdetective.read`
  - `hybridpilot.compare`
- **Website lab.** One "garage" page where a visitor picks a vehicle and route and sees all three models on the same trip.
- **Share the pages.** The three Artifact pages are private on claude.ai; the repo site should host its own copies.

## 5. Known caveats to keep in the docs
- **VED fuel is estimated.** For 351 of 384 cars it comes from MAF and fuel trims (or load × rpm), not metered. It misses enrichment and can read air during fuel cut-off.
- **Mass is coarse.** VED mass is rounded to 500 lb (±113 kg); make, model and year are hidden; hybrid/PHEV type is inferred from engine size.
- **Trips are warm fragments.** Simulate from the logged speed with warm initial temperatures.
- **Weather is approximate.** T_amb is from the airport station (±2 °C); wind is 10 m airport wind.
- **13 Nov fixes.** EPA cycle labels were swapped and VED mass double-counted 136 kg. Both are fixed in RoadData (see `data/DATA.md`). The three v1.0 models were built before the fixes. EnergyNet graded EPA on corrected labels via its own fix, but its twins and HybridPilot's ran VED cars 136 kg heavy. Re-score after upstreaming.
- **HybridPilot is graded in simulation.** Its exam is real routes driven in RoadSim; no real car ran the policy.
- **Heavy-duty grading is thin.** It rests on 2 trolleybuses, 2 Mersin bus types and 6 truck dyno numbers.

## 6. Definition of "complete" for v2
- All P0 done.
- Every P1 simulator gap inside its "done when".
- D3 and Fleet DNA (or equivalents) added as dev and exam by vehicle.
- All three models re-trained and re-scored on the new physics version, each with a published win/loss table.
- Heavy duty graded on real traces for at least 2 truck types and 3 bus types.
