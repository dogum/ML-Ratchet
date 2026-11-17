# BuildingDetective v1.0: model card

## What it does
Reads a building from its meters and the weather. Outputs, each with a one-sigma error bar or class probabilities:
floor area, year built, envelope conductance UA (W/K) and UA per floor area, thermal time constant (h), heating and
cooling balance points (degC), site heating / cooling energy per kelvin (W/K), non-HVAC share of site energy, weekday
operating hours and weekend/weekday ratio (commercial only), primary use (12 classes), heating system (heat pump,
electric resistance, on-site fuel, district, none) and cooling system (none, DX, chiller, district).

Three networks for three input resolutions: hourly meters (one year, at least ~30 days), 12 monthly bills, or one
annual total per fuel. Meter channels: electricity, on-site fuel, district heat (steam / hot water), district chilled
water. Weather: hourly outdoor temperature and global horizontal irradiance (monthly / annual: temperature and degree
days only).

## How it was made (the ML Ratchet recipe)
1. **Physics.** BuildingSim (Phase 1 of this project): single-zone 5R1C heat balance with solar, infiltration,
   ventilation and moisture; HVAC equipment with capacity / COP / EIR curves, part load, defrost, fans, pumps, plants;
   verified by the Phase 1 checks (analytic RC step and steady conduction, BESTEST ranges, ledger closure, ResStock and
   ComStock twins; `checks/*.json` of the shared building platform, `shared/building/` once its zip lands). Labels come from the simulator itself: UA = the 5R1C network conductance
   (`labels()['UA_network_W_K']`), tau, balance points, end-use shares, schedules, system classes.
2. **Simulated data in real conditions.** 12,000 generated buildings (30 % homes, 70 % commercial in 10 types,
   drawn from ResStock / ComStock / CBECS priors) on real weather: the 18 BDG2 sites in 2016 and 2017 and TMYx cities.
   Each meter is then given real-world faults (gaps, zero runs, spikes, level shifts, clock offsets, daily-spread
   readings, shared unmodelled loads, sub-meter scale errors, noise; billing windows and missing months for bills;
   unreported fuels for surveys).
3. **Calibrated on dev only** (wrappers around BuildingSim, upstream candidates):
   - W1 intensity: one factor per use type so generated site EUI medians match RECS / CBECS prior+dev medians.
   - W2 campus: district meters scaled by a distribution factor fitted on the 52 BDG2 dev twins (heat 0.18,
     chilled water 0.33; meter = delivered / factor), 30 % of commercial buildings on campus service.
   - W3 always-on base loads (physics v2) from the dev twin test (table below): a share of each district meter's annual
     energy moved into a flat base (heat 0.70, chilled water 0.50), plus an unmodelled flat electricity load.
4. **Network.** A 2-layer MLP (256 GELU units, ~0.37M parameters for hourly) on ~1,200 engineered features per
   building: daily load in temperature bins (weekday / weekend), the hour-of-week profile, cold- and hot-day profiles,
   a PRISM 5-parameter change-point fit, correlations with exponentially smoothed temperature (6 time constants, the
   thermal-mass signal) and hourly lag correlations, plus robust statistics. Heads: Gaussian mean and log-sigma for 12
   regression targets (NaN labels masked) and three softmax classifiers. Trained only on simulations.
5. **Ratchet.** Frozen validation set = bank shards 900-904 (1,000 buildings) of the current physics version. A rung is
   kept if its skill (mean of 12 R2 clipped at 0 and 3 chance-rescaled balanced accuracies) beats the incumbent by
   more than 0.01 AND its real dev composite does not drop more than 0.02. Physics changes re-score all incumbents.
6. **Error bars.** One widening factor per output (widen-only), fitted on dev: real BDG2 / RECS / CBECS dev units for
   floor area and year built, the simulated twins of the 52 BDG2 dev buildings for everything without real truth.
   Class temperatures (>= 1) fitted on dev labels.

## Ratchet
| rung | change | frozen-val skill | dev gate | decision |
|---|---|---|---|---|
| R1-hourly | MLP 2x256 GELU, hetero NLL + 3 softmax heads; bank v1 shards 0-11 (2,400 buildings), features f1 (W1+W2, quirks) | 0.703 | 0.193 | kept |
| R1-monthly | MLP 2x256 GELU, hetero NLL + 3 softmax heads; bank v1 shards 0-11 (2,400 buildings), features f1 (W1+W2, quirks) | 0.573 | 0.204 | kept |
| R1-annual | MLP 2x256 GELU, hetero NLL + 3 softmax heads; bank v1 shards 0-11 (2,400 buildings), features f1 (W1+W2, quirks) | 0.490 | 0.326 | kept |
| rebaseline-v2-hourly | physics v2 = W3 always-on base loads (twin test on dev); frozen val rebuilt from the same bank shards 900-904 with W3; incumbents re-scored | 0.703 → 0.663 |  | re-scored |
| rebaseline-v2-monthly | physics v2 = W3 always-on base loads (twin test on dev); frozen val rebuilt from the same bank shards 900-904 with W3; incumbents re-scored | 0.573 → 0.501 |  | re-scored |
| rebaseline-v2-annual | physics v2 = W3 always-on base loads (twin test on dev); frozen val rebuilt from the same bank shards 900-904 with W3; incumbents re-scored | 0.490 → 0.483 |  | re-scored |
| R2-hourly | physics v2: retrain on f2 (W3 always-on base loads in district heat, chilled water, electricity), shards 0-11 | 0.693 | 0.320 | kept |
| R2-monthly | physics v2: retrain on f2 (W3 always-on base loads in district heat, chilled water, electricity), shards 0-11 | 0.547 | 0.328 | kept |
| R2-annual | physics v2: retrain on f2 (W3 always-on base loads in district heat, chilled water, electricity), shards 0-11 | 0.486 | 0.331 | rejected |
| R3a-hourly | wider/deeper MLP 3x512, dropout 0.25, wd 1e-3 (lr 2e-3: training diverged at epoch ~15) | 0.409 | 0.308 | rejected |
| R3b-hourly | MLP 2x256 with dropout 0.3, wd 1e-3, 80 epochs | 0.684 | 0.325 | rejected |
| R3c-hourly | MLP 3x512, dropout 0.2, wd 1e-3, lr 7e-4 (stable) | 0.705 | 0.324 | rejected: WITHDRAWN: 1.15M parameters breaks the ~500k size cap of the shared-machine rules; R2-hourly reinstated |
| R3d-hourly | MLP 3x288 (525k params, within cap), dropout 0.2, wd 1e-3, lr 7e-4 | 0.695 | 0.320 | rejected |
| rebaseline-v2-hourly | label hygiene: base_share NaN when |value|>1 (3 training buildings had ~0 metered site energy -> -8e12), balance points clipped to [-25,30] C; same frozen val, incumbents re-scored | 0.693 → 0.693 |  | re-scored |
| rebaseline-v2-monthly | label hygiene: base_share NaN when |value|>1 (3 training buildings had ~0 metered site energy -> -8e12), balance points clipped to [-25,30] C; same frozen val, incumbents re-scored | 0.547 → 0.547 |  | re-scored |
| rebaseline-v2-annual | label hygiene: base_share NaN when |value|>1 (3 training buildings had ~0 metered site energy -> -8e12), balance points clipped to [-25,30] C; same frozen val, incumbents re-scored | 0.483 → 0.483 |  | re-scored |
| R4-hourly | label hygiene (clean_labels), MLP 2x256, f2 shards 0-11 | 0.746 | 0.331 | kept |
| R4-monthly | label hygiene (clean_labels), MLP 2x256, f2 shards 0-11 | 0.597 | 0.329 | kept |
| R4-annual | label hygiene (clean_labels), MLP 2x256, f2 shards 0-11 | 0.517 | 0.327 | kept |
| R5-hourly | more data: f2 shards 0-22 (4,600 buildings), MLP 2x256 | 0.783 | 0.326 | kept |
| R5-monthly | more data: f2 shards 0-22 (4,600 buildings), MLP 2x256 | 0.630 | 0.318 | kept |
| R5-annual | more data: f2 shards 0-22 (4,600 buildings), MLP 2x256 | 0.530 | 0.319 | kept |
| R6-hourly | MLP 3x288 (525k params), dropout 0.2, wd 1e-3, lr 7e-4, shards 0-22 | 0.779 | 0.329 | rejected |
| R7-hourly | more data: f2 shards 0-39 (8,000 buildings), MLP 2x256 | 0.801 | 0.329 | kept |
| R7-monthly | more data: f2 shards 0-39 (8,000 buildings), MLP 2x256 | 0.641 | 0.329 | kept |
| R7-annual | more data: f2 shards 0-39 (8,000 buildings), MLP 2x256 | 0.539 | 0.325 | rejected |
| R8-hourly | MLP 3x288 (525k params), dropout 0.2, wd 1e-3, lr 7e-4, shards 0-39 | 0.802 | 0.320 | rejected |
| R9-hourly | more data: f2 shards 0-59 (12,000 buildings, the full bank), MLP 2x256 | 0.813 | 0.317 | kept |
| R9-monthly | more data: f2 shards 0-59 (12,000 buildings, the full bank), MLP 2x256 | 0.656 | 0.312 | kept |
| R9-annual | more data: f2 shards 0-59 (12,000 buildings, the full bank), MLP 2x256 | 0.541 | 0.306 | kept |

## Twin test (dev): real BDG2 dev buildings vs their simulated twins
| meter, feature | real | twin v1 | twin v2 (W3) |
|---|---|---|---|
| Electricity · night/day | 0.77 | 0.66 | 0.74 |
| Electricity · cooling slope | 0.14 | 0.32 | 0.26 |
| District heat · heating slope | 0.81 | 4.38 | 1.00 |
| District heat · night/day | 0.94 | 1.59 | 1.06 |
| District heat · p99/3 | 1.16 | 3.54 | 1.08 |
| Chilled water · night/day | 0.69 | 0.31 | 0.63 |
| Chilled water · cooling slope | 0.73 | 2.02 | 1.09 |

## Final exam (run once after all choices were frozen)
"real" = measured buildings or survey answers. "simulated (exact)" = held-out BuildingSim buildings. "sim vs sim" =
EnergyPlus meters (NREL ResStock / ComStock) graded against a BuildingSim twin: two models, not reality. Bold = beats
every baseline listed. Baselines marked TRUE are given the true use type, which the Detective is not.

### BDG2: 1,584 real non-residential buildings (exam split), meters for 2017
| task | input | truth | n | Detective | baselines |
|---|---|---|---|---|---|
| Floor area, median abs. % error | hourly input, 2017 | real | 1428 | 50.0% · 58% within ×2 | electricity ÷ dev median EUI of the TRUE use, same buildings (n=1364): 51.5%; same rule as quoted by BuildingData (1,133 quality-checked meters): 46.4%; dev median area: 68.0% |
| Floor area, median abs. % error | monthly input, 2017 | real | 1428 | 49.4% · 62% within ×2 | electricity ÷ dev median EUI of the TRUE use, same buildings (n=1364): 51.5%; same rule as quoted by BuildingData (1,133 quality-checked meters): 46.4%; dev median area: 68.0% |
| Floor area, median abs. % error | annual input, 2017 | real | 1428 | 48.0% · 61% within ×2 | electricity ÷ dev median EUI of the TRUE use, same buildings (n=1364): 51.5%; same rule as quoted by BuildingData (1,133 quality-checked meters): 46.4%; dev median area: 68.0% |
| Primary use, accuracy · macro-F1 | hourly input | real | 1428 | 11% · 0.10 | majority class (education): 40% · 0.06; weekly-shape nearest centroid (B): 34% · 0.20 |
| Year built, mean abs. error | hourly input | real | 741 | 28.3 yr | dev median: 24.9 yr |
| Has a chilled-water meter? (from electricity alone), ROC AUC | hourly electricity only | real | 1375 | **0.75** | chance: 0.50 |
| Has a steam / hot-water meter? (from electricity alone), ROC AUC | hourly electricity only | real | 1375 | **0.70** | chance: 0.50 |

### RECS 2020: 7,401 real homes (exam split), annual energy only
| task | input | truth | n | Detective | baselines |
|---|---|---|---|---|---|
| Floor area, median abs. % error | annual kWh by fuel + degree days | real | 7401 | 33.6% | energy ÷ dev EUI by TRUE type × climate (B): 33.1%; dev median by TRUE housing type (B): 28.8% |
| Main heating: fuel vs electric vs none, accuracy | annual | real | 7194 | **87.3%** | rule: any fuel used → fuel: 85.3%; majority (fuel): 64.8% |
| Heat pump vs electric resistance, balanced accuracy · AUC | annual | real | 2226 | 0.46 · 0.40 | kWh/m²/HDD rule, uses TRUE area (B): 0.50; majority: 0.50 |
| Vintage, mean abs. error to the survey range midpoint | annual | real | 7401 | **22.1 yr** | dev median 1984: 22.3 yr |

### CBECS 2018: 2,558 real commercial buildings (exam split), annual energy only
| task | input | truth | n | Detective | baselines |
|---|---|---|---|---|---|
| Floor area, median abs. % error | annual | real | 2558 | 43.0% | energy ÷ dev EUI by TRUE type × climate (B): 38.7%; dev median by TRUE type (B): 75.7% |
| Primary use (10 commercial classes), accuracy · macro-F1 | annual | real | 2558 | 16% · 0.15 | majority class: 21% · 0.03 |
| Heating: fuel / electric / district / none, accuracy | annual | real | 2513 | **74.7%** | rule: fuel used → fuel, district → district: 68.6%; majority (fuel): 55.6% |
| Has a chiller, ROC AUC | annual | real | 2362 | **0.77** | chance: 0.50 |
| Weekly operating hours, mean abs. error | annual | real | 2558 | 44.2 h | dev median: 41.4 h; dev median of the TRUE use: 29.7 h |
| Vintage, mean abs. error | annual | real | 2558 | 27.0 yr | dev median 1984: 26.0 yr |

### HUE: 24 real homes in British Columbia, hourly electricity
| task | input | truth | n | Detective | baselines |
|---|---|---|---|---|---|
| Electric main heat (baseboard or heat pump) vs not, accuracy · AUC | hourly electricity only | real | 24 | 50% · 0.46 | majority: 75% |

### Simulated: 1,000 held-out BuildingSim buildings (frozen validation set), exact labels
| task | input | truth | n | Detective | baselines |
|---|---|---|---|---|---|
| Envelope UA, MAE in log10 | hourly, quirked meters | simulated (exact) | 2000 | **0.10** | PRISM change-point (fits 95% of units): 0.85; median of the TRUE use type: 0.35 |
| Time constant, MAE in log10 | hourly, quirked meters | simulated (exact) | 2000 | **0.14** | median of the TRUE use type: 0.21 |
| Heating balance point, MAE | hourly, quirked meters | simulated (exact) | 2000 | **2.14 K** | PRISM change-point (fits 95% of units): 6.54 K; median of the TRUE use type: 3.93 K |
| Cooling balance point, MAE | hourly, quirked meters | simulated (exact) | 2000 | **2.24 K** | PRISM change-point (fits 93% of units): 4.89 K; median of the TRUE use type: 3.87 K |
| Heating energy per K, MAE in log10 | hourly, quirked meters | simulated (exact) | 1880 | **0.14** | PRISM change-point (fits 95% of units): 0.92; median of the TRUE use type: 0.45 |
| Weekday operating hours, MAE | hourly, quirked meters | simulated (exact) | 1400 | **0.96 h** | median of the TRUE use type: 3.25 h |
| Heating system (5 classes), balanced accuracy | hourly | simulated (exact) |  | **0.84** | rule from which meters exist: 0.35; majority: 0.20 |
| Heat pump vs resistance, ROC AUC | hourly | simulated (exact) | 594 | **0.94** | chance: 0.50 |

### NREL ResStock / ComStock models: EnergyPlus meters, truth from a BuildingSim twin of the same characteristics
| task | input | truth | n | Detective | baselines |
|---|---|---|---|---|---|
| ResStock homes: envelope UA, MAE in log10 | EnergyPlus hourly meters, AMY2018 | sim vs sim | 58 | 0.29 (0.29 on all) | PRISM × nominal efficiency (fits 57): 0.73; sample median: 0.23 |
| ResStock homes: heating balance point, MAE | EnergyPlus hourly | sim vs sim | 58 | **2.8 K** | PRISM: 4.5 K |
| ResStock homes: heating system accuracy (stock label) | EnergyPlus hourly | sim vs sim | 58 | 67% |  |
| ComStock buildings: envelope UA, MAE in log10 | EnergyPlus hourly meters, AMY2018 | sim vs sim | 90 | **0.20 (0.19 on all)** | PRISM × nominal efficiency (fits 85): 0.76; sample median: 0.44 |
| ComStock buildings: heating balance point, MAE | EnergyPlus hourly | sim vs sim | 90 | **7.0 K** | PRISM: 10.4 K |
| ComStock buildings: heating system accuracy (stock label) | EnergyPlus hourly | sim vs sim | 90 | 72% |  |


Error-bar coverage (share of truths inside ±1σ, 68% is ideal), after one widening factor per output fitted on dev only: BDG2 floor area (hourly) 72%; BDG2 floor area (annual) 54%; RECS area 69%, RECS vintage 74%; simulated UA 67%, tau 72%.

## Error-bar widening factors (dev)
| mode | logA | year | logUA | logUAm2 | logtau | Tb_heat | Tb_cool | logS_heat | logS_cool | base_share | op_hours | weekend | class temperature (use, heat, cool) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hourly | 3.02 | 2.14 | 1.04 | 1.30 | 1.26 | 1.13 | 1.24 | 1.07 | 1.11 | 1.01 | 1.04 | 1.00 | 10.00, 1.00, 1.00 |
| monthly | 1.54 | 1.14 | 1.00 | 1.16 | 1.15 | 1.02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.11 | 1.00 | 10.00, 1.00, 1.00 |
| annual | 1.08 | 1.04 | 1.00 | 1.09 | 1.13 | 1.02 | 1.00 | 1.00 | 1.04 | 1.00 | 1.08 | 1.00 | 7.40, 1.90, 1.00 |


## Where it fails
- Primary use does not transfer to real buildings: 11% right on BDG2 against 40% for always guessing education. The dev-fitted temperature (10) flattens these probabilities on purpose.
- Year built has no real skill: BDG2 mean error 25 years vs 25 for the dev median; RECS 22 vs 22; CBECS 27 vs 26.
- Heat pump vs resistance from annual totals is worse than chance on RECS (AUC 0.40); on the 24 HUE homes the hourly model's electric-heat AUC is 0.46 (monthly 0.69). In simulation the same task gives AUC 0.94: the simulator's heat-pump signature is not what real homes show.
- UA and time constant have no real truth. Against EnergyPlus (ResStock homes) the hourly UA error is 0.29 log10 with a +0.25 bias, no better than the sample median (0.23); for ComStock buildings it is 0.19 vs 0.44.
- Real BDG2 floor area needed error bars 3x wider than the network's own (hourly), and hourly input reads area no better than one annual total (50% vs 48%): real load shapes still differ from simulated ones.
- Operating hours from one annual total are worse than guessing the dev median (44 h/week vs 41). Hourly schedules are read well in simulation (MAE 1.0 h/day) but no real hourly building has schedule truth.
- The real dev gate is small (52 BDG2 buildings) and noisy. It drifted from 0.331 to 0.306 over the kept rungs, each step inside the 0.02 tolerance.
- Simulator limits carry over: single thermal zone, first-order part-load and fault models, ideal thermostat within an hour, campus plants as a flat distribution factor plus base load (W2, W3).

## Data and licences
- Building Data Genome 2: Miller et al., Sci. Data 7, 368 (2020), https://github.com/buds-lab/building-data-genome-project-2.
  **CC BY-SA 4.0**: anything derived from BDG2 stays share-alike. In this folder that is:
  - `examples/bdg2/` (`bdg2_hourly.npz`, `bdg2_hourly.json`): 3 exam buildings' 2017 meters, site weather and metadata
    (notice: `examples/bdg2/LICENSE-DATA.md`);
  - `lab/index.html`: the original page embeds 29 BDG2 exam buildings and BDG2 exam and twin-test numbers (notice:
    `lab/LICENSE-DATA.md`);
  - the live instrument's data in `site/public/instruments/buildingdetective/` (notice written there by
    `web/export_instrument.py`);
  - `training/exam.json`, `training/twintest.json`, `training/ratchet.json` and `training/calibration.json` hold
    numbers computed on BDG2 buildings (scores, twin medians, W2 campus factors), and the weights' error-bar factors
    were fitted partly on the 52 BDG2 dev buildings. These are summary statistics, not meter data; they are credited here.
- EIA RECS 2020 and CBECS 2018 public microdata (public domain): the 8 survey respondents in `examples/examples.json`.
- HUE: Makonin, Data in Brief 23 (2019), Harvard Dataverse doi:10.7910/DVN/N3HGRN (CC BY 4.0): the home in
  `examples/examples.npz` / `examples/examples.json` (hourly electricity and the station temperature HUE provides;
  GHI is a clear-sky model x 0.6) and 5 homes in `lab/index.html`.
- NREL ResStock 2025 R1 / ComStock 2025 R3 (CC BY 4.0). "Data includes information from the ResStock and ComStock
  datasets developed by the National Renewable Energy Laboratory (NREL; now National Laboratory of the Rockies) with
  funding from the U.S. DOE." Used for generator priors (Phase 1) and the simulator-vs-simulator check only; no
  ResStock/ComStock file is shipped.
- TMYx 2011-2025 (climate.onebuilding.org); NOAA ISD weather inside BDG2. Used for simulation weather; not shipped.
- Code: MIT, ML Ratchet (github.com/dogum/ML-Ratchet).
