# FaultFinder v1.0: model card

## What it does
Reads one local day of an air handler's (AHU) or rooftop unit's (RTU) building-automation trend log and returns:
- **P(fault)**: probability that the unit has a fault that day (calibrated on 20 real dev days);
- **diagnosis**: a probability over 18 fault classes, given a fault;
- a severity estimate for the top class and the day's extra HVAC energy fraction (both trained on simulation only; the energy output is not validated, see Limits).

It was trained only on BuildingSim simulations. No real labelled day was ever a training example.

## Inputs
One day, any sampling from 1 to 15 min, local time. Contract column names (any subset; missing channels are allowed and flagged to the network):

| Column | Unit | Notes |
|---|---|---|
| `T_oa T_ra T_ma T_sa T_zone` | degC | outdoor, return, mixed, supply, zone air |
| `T_sa_sp` (or `T_sa_sp_cool`/`T_sa_sp_heat`) | degC | supply-air setpoint |
| `damper_oa valve_cc valve_hc fan_sa_speed` | 0..1 | controller commands as logged |
| `comp1_status comp2_status` | 0/1 | RTU compressor stages (used as the cooling signal) |
| `fan_sa_power_kW` (or a fan current) | any | scaled by its own daily 95th percentile |
| `fan_sa_status occ` | 0/1 | fan on, occupied |

Data-quality rules: impossible temperatures (< -45 or > 70 degC) and flatlined temperature sensors are treated as missing.

## Channel mapping (BuildingSim trend -> contract -> network)
OAT->T_oa, RAT->T_ra, MAT->T_ma, SAT->T_sa, (VAV reset rule)->T_sa_sp, ZAT->T_zone, OA_damper_pct/100->damper_oa,
cool_cmd_pct/100->valve_cc (RTU: compressor stages), heat_cmd_pct/100->valve_hc, fan_speed_pct/100->fan_sa_speed,
fan_kW->fan_sa_power_kW, fan_status->fan_sa_status, occupied->occ.

## Fault classes
none; oa_damper_stuck; oa_damper_leak; economizer; cc_valve_stuck; cc_valve_leak; hc_valve_stuck; hc_valve_leak;
reheat_valve_leak; cooling_capacity (coil fouling, condenser fouling, refrigerant charge); low_airflow;
sensor_bias_T_sa (incl. SAT setpoint offset); sensor_bias_T_oa; sensor_bias_T_ma; sensor_bias_T_ra; zone_setpoint
(zone sensor bias, setpoint override); schedule; vav_box_stuck; plant_efficiency (chiller/boiler fouling).
These come from BuildingSim's 20 fault types. Of the exam taxonomy's 27 types, 14 map onto a class; the other 13
(exhaust damper, supply duct leak, heating-coil fouling/capacity, return fan, unstable control, reverse-acting valve,
chilled-water sensors, chiller off, lighting schedule, infiltration) cannot be simulated and are graded for
detection only.

## Physics and simulator
BuildingSim (ML Ratchet BuildingSim phase 1): 5R1C zone heat balance (solar, wind/stack infiltration, moisture),
air handlers from fan laws, mixing box, coils, chillers/boilers and DX curves (DOE reference data). Checks
(the shared building platform's `checks/*.json`, `shared/building/checks/` once it lands): RC step response max error 0.003 K vs closed form; steady conduction 2e-9 relative error;
31/36 BESTEST (ASHRAE 140) results inside the 1995 reference ranges; psychrometrics within 0.02 K (p95) of CoolProp;
PMV within 0.003 of pythermalcomfort; 26/26 fault signatures in the expected direction; energy ledger closes to 1e-8.

Fault bank: random BuildingSim buildings with packaged RTUs, VAV chiller/boiler or district-energy VAV air handlers,
17 TMYx US cities, any season. Each 12-day window is simulated healthy and with one fault from day 5 on; every local
day is a sample (faulted days with visible effect >= 1 unit, plus healthy days). Two banks of 38,987 days:
v1 (generic controls) and v3 (dev-calibrated).

## Real-noise injection and dev calibration (20 dev unit-days only)
- **Real-noise model**: 15-min residuals around a 75-min running mean from the real fault-free dev days (residual
  library, random shift and gain), static sensor offsets, sensor drift toward a return/outdoor blend when the fan
  stops, logger gaps, random missing channels, and each exam source's set of logged points.
- **Korean office AHUs** (20 units, unlabelled real BMS data): supply-air tracking-error tail and start/stop
  irregularity (operator jitter 2-3 bins, 15 % of days with a 1-3 h off block).
- **Twin test finding 1 (bank v3)**: BMS logs controller COMMANDS, BuildingSim logs actual positions. FLEXLAB's
  heating valve stuck at 100 % logs 0 %; a 40 % leak logs 0 %. Logged signals for stuck/leaking actuators are rebuilt
  as commands (stuck: healthy command, wound up to 0/1 when the loop cannot reach its target; leak: signal minus leak;
  20 % of units keep the actual position).
- **Twin test finding 2 (noise v4)**: BuildingSim's damper signal equals the outdoor-air fraction; real dampers at
  100 % command mixed only 81-90 % outdoor air (ERS, FLEXLAB) and Nesbitt at 15 % mixed 14-28 %. Logged damper =
  inverse of fraction = a + (b - a) cmd^g, a ~ U(0, 0.15), b ~ U(0.75, 1), g ~ U(0.6, 1.6) (30 % linear). Larger
  static sensor disagreement: mixed-air stratification sd 0.8 K, zone vs return sd 0.9 K (ORNL zone reads 1.5 K
  above return on a normal day).
- Also v3: schedule-shift faults (start/stop 2-4 h off), 24x7 units (25 %), wider VAV minimum (0.2-0.8) and SAT priors.
- **Data finding**: ERS AHU-A outdoor-air temperature reads exactly 0 F all day in summer 2007: flatline rule.

## Network
1-D CNN over 96 x 15-min bins, 35 feature rows (6 temperatures, 5 physical differences MA-RA, SA-MA, SA-SP, OA-RA,
ZONE-RA, estimated outdoor-air fraction, 7 actuator/state signals, time of day, 14 availability masks).
Conv 64-64-128-128 (kernels 5,5,5,3, two max-pools), mean+max pooling, dense 128, heads: 19 class logits, 19
parameters, 1 energy-waste. 160,871 weights. BatchNorm folded into convolutions; numpy == torch to 1e-6; browser JS ==
numpy to 5e-6 on all 217 exam days (top class agrees on 217/217).

## Ratchet (`training/ratchet.json`; normalised in `ratchet.yaml`)
Skill = 0.5 detection AUC + 0.5 diagnosis accuracy (days with a visible fault) on the frozen simulated validation
set; decisions re-taken in order on the newest set (v4). Keep if skill > incumbent + 0.01 AND dev AUC (20 real days)
>= max(0.60, incumbent - 0.05).

| Rung | What | skill v1 | v2 | v3 | v4 | sim AUC v4 | sim diag v4 | dev AUC | dev FPR@0.5 | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| r1_gauss | bank v1 + Gaussian noise | 0.585 | 0.499 | 0.448 | 0.415 | 0.60 | 23 % | 0.60 | 0.91 | kept (first) |
| r2_realnoise | + real-noise model | 0.575 | 0.550 | 0.476 | 0.454 | 0.66 | 25 % | 0.74 | 0.45 | kept |
| r3_calib | + bank v3 (commands logged, operator jitter, schedule shifts) | 0.528 | 0.510 | 0.536 | 0.500 | 0.69 | 31 % | 0.71 | 0.45 | kept |
| r4_damper | + damper characteristic, larger sensor offsets (noise v4) | 0.505 | 0.502 | 0.518 | 0.511 | 0.71 | 32 % | 0.73 | 0.36 | kept |
| r5_bothbanks | banks v1 + v3, 50k days | 0.557 | 0.542 | 0.528 | 0.518 | 0.70 | 33 % | 0.74 | 0.45 | rejected (+0.007 < 0.01) |
| **r6_wide** | 64/128 channels, 10 epochs | 0.539 | 0.535 | 0.556 | **0.543** | 0.73 | 36 % | 0.71 | 0.36 | **kept (final)** |

## Calibration
One temperature T >= 1 on the detection log-odds (and the same T on the diagnosis logits), fitted by log loss on
the 20 dev days: **T = 4.3**. Widen-only rule.

## Final exam (217 real/experimental days, run once after the ladder was closed)

| Units | Method | days (faulty) | AUC [90 % CI] | bal. acc. | TPR | FPR | recall @2 % FA | @5 % | @10 % | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| AHU | **FaultFinder** | 200 (91) | **0.78** [0.72-0.83] | **0.70** | 73 % | 33 % | 23 % | 27 % | 47 % | 0.13 |
| AHU | APAR rules | 200 (91) | 0.69 | 0.65 | 32 % | 2 % | 32 % | 41 % | 45 % | - |
| AHU | majority (no fault) | 200 (91) | - | 0.50 | 0 | 0 | - | - | - | - |
| RTU | FaultFinder | 17 (12) | 0.65 | 0.47 | 75 % | 80 % | 42 % | 42 % | 42 % | 0.17 |
| RTU | rule: fan on while unoccupied | 17 (12) | 0.72 | 0.57 | 33 % | 20 % | 25 % | 25 % | 25 % | - |
| RTU | majority (fault) | 17 (12) | - | 0.50 | 100 % | 100 % | - | - | - | - |
| all | FaultFinder | 217 (103) | 0.77 [0.72-0.83] | 0.69 | 73 % | 35 % | 20 % | 24 % | 45 % | 0.12 |

CI: bootstrap over facility-days. TPR/FPR at calibrated P = 0.5 (rules at their dev-tuned thresholds).

**By source (detection AUC, FaultFinder vs rules):** Iowa ERS lab 0.73 vs 0.73 (101 days); LBNL FLEXLAB 0.93 vs 0.67
(23); **Nesbitt Hall, real occupied building, 0.78 vs 0.54 (76 days, 26 faulty; balanced acc. 0.72 vs 0.57; but FPR
40 % at 0.5 and recall 23 % at 5 % FA)**; ORNL RTU 0.65 vs 0.72 (17).
By season AUC: winter 0.71, transition 0.83, summer 0.75. Severity: high 0.79, low 0.79, unspecified 0.72.

**Diagnosis** (66 faulty days with a simulable fault): top-1 **20 %** (majority-class guess 21 %), top-3 48 %.
Per class top-1/top-3: cc_valve_stuck 50/86 % (14), zone_setpoint 100/100 % (4), hc_valve_stuck 25/25 % (4),
hc_valve_leak 20/40 % (5), sensor_bias_T_sa 0/100 % (3), cooling_capacity 0/100 % (2), oa_damper_stuck 0/29 % (14),
schedule 0/38 % (8), oa_damper_leak 0/0 % (5), cc_valve_leak 0/33 % (3), low_airflow 0/0 % (2), sensor_bias_T_oa
0/0 % (2). By source top-1: ERS 9 %, FLEXLAB 32 %, Nesbitt 6 %, ORNL 50 %. Diagnosis top-1 ECE 0.07.

**Calibration**: 164 of 217 days fall in P 0.4-0.6 (37 % faulty there); 0.6-0.8: 37 days, 78 % faulty; 0.8-1.0: 15
days, 93 % faulty. Brier 0.216 calibrated vs 0.204 uncalibrated: the 20-day temperature over-shrinks.

**Energy output**: on 46 ERS paired days (faulty AHU-A vs healthy AHU-B, air-side energy proxy) Spearman 0.19
between predicted and measured extra energy: not useful; do not rely on `extra_hvac_energy_frac`.

## Limits
1. Diagnosis on real data is about as good as guessing the commonest fault; use the top 3 as a checklist.
2. 33 % false alarms on healthy AHU days at P = 0.5; at strict alarm rates the APAR rules catch more.
3. Probabilities squashed toward 0.5 by a temperature fitted on 20 days (ranking unaffected).
4. RTU: worse than a one-line rule on 17 days.
5. One day at a time; no comparison with the unit's own history. Single-zone simulator with ideal controls, no
   exhaust damper, return fan or duct model.
6. Small exam: 217 days, 5 facilities.

## Data credits and licences
- ASHRAE RP-1312 ERS and Drexel Nesbitt Hall via N. Ghalamsiah (2025), figshare doi:10.6084/m9.figshare.29297999.v3
  (CC BY 4.0); ERS data J. Wen & S. Li, ASHRAE RP-1312 (2011); Nesbitt Y. Chen, PhD thesis, Drexel (2019).
- LBNL fault detection data synthesis (FLEXLAB SZCAV/SZVAV, ORNL RTU, MZVAV-2 ERS), J. Granderson, G. Lin et al.,
  figshare doi:10.6084/m9.figshare.11752740 (CC0).
- Korean office AHU BMS data, S. Wang (2024), figshare doi:10.6084/m9.figshare.25932409.v2 (CC BY 4.0), unlabelled.
- TMYx weather, climate.onebuilding.org (free with attribution).
- APAR rules after J. Schein et al., Energy and Buildings 38 (2006).
No BDG2 data is used by FaultFinder (no share-alike obligation).

## Data and licences (added on import, 17 Nov 2026)
No file in this folder holds Building Data Genome 2 (BDG2) data, so nothing here is CC BY-SA. The shipped data files
and their sources:

| File | Source | Licence |
|---|---|---|
| `examples/flexlab_lbnl_szvav_flexlab_2017-09-11.csv` | LBNL fault detection data synthesis (FLEXLAB SZVAV) | CC0 1.0 |
| `examples/iowa_ers_lbnl_mzvav2_ers_exp_2009-01-19.csv` | LBNL fault detection data synthesis (MZVAV-2 at Iowa ERS) | CC0 1.0 |
| `examples/ornl_rtu_lbnl_rtu_ornl_2018-01-31.csv` | LBNL fault detection data synthesis (ORNL FRP2 RTU) | CC0 1.0 |
| `examples/nesbitt_NES_20180711_AHU-1.csv` | Drexel Nesbitt Hall (Y. Chen, 2019) via N. Ghalamsiah (2025), figshare doi:10.6084/m9.figshare.29297999.v3 | CC BY 4.0 |
| `examples/labels.json` | the dataset labels of those four days | as above |
| `lab/index.html` | all 217 exam days (ASHRAE RP-1312 ERS and Nesbitt via Ghalamsiah, CC BY 4.0; LBNL synthesis, CC0), rounded to 0.01 degC / 0.001, plus BuildingSim lab days (simulated, with one draw of dev noise) | CC BY 4.0 for the Ghalamsiah days, CC0 for the LBNL days; credits in the page footer |
| `site/public/instruments/faultfinder/` (the live instrument) | twelve exam days drawn from the four sources (see `web/export_instrument.py`) | per day as above; the instrument shows each day's source |

The building-status handoff described the fault subset as CC0. The release's own credits (above) say otherwise for
part of it: the LBNL synthesis is CC0, but the ASHRAE RP-1312 ERS and Nesbitt Hall files come from Ghalamsiah's
figshare package under CC BY 4.0, which requires credit (given here, in `model.yaml` and on the page). The Korean
office AHU data (CC BY 4.0) were used only to fit noise statistics; no Korean data ship. The code is MIT.
