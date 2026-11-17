# FaultFinder v1.1: model card

> **v1.1 (17 Nov 2026)** = v1.0's network, frozen, plus 47 input rows that compare the day with the same unit's last
> 1-3 logged days (r8, below). Without reference days it gives v1.0's log-odds; P(fault) differs only by the dev
> temperature (4.2 instead of 4.3). **v1.1 raises false alarms on days with a reference (seen exam: AHU 33 → 56 %).**
> The sections from "Channel mapping" to "Integrity pass" describe v1.0 and keep its exam rows for comparison; v1.1's
> seen rows are in "Round 2" and in `model.yaml`.

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

As released. The integrity pass below re-scores these days (seen) against fair baselines with intervals; its
verdicts replace the wins and losses implied here.

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

## Integrity pass (17 Nov 2026): fair baselines and intervals (the exam is seen)

The exam above was taken once by v1.0. On 17 Nov 2026 it was re-scored from the released numpy weights
(`training/exam.py ../faultfinder_weights.npz`, a numpy path added to `exam.py`): every number in `exam.json`
reproduces to 6e-8 (float32 rounding), with the same dev temperature T = 4.3. Nothing below was used to choose
anything; the plan is in `NOTEBOOK.md`, written before the re-score. Script: `training/integrity.py`.

**Baselines, all fitted on the 20 dev days only.**
- *Rules, textbook*: the v1.0 baseline. APAR with tolerances 1.5 K (temperature), 1.1 K (fan heat), 15-min
  persistence and the released dev threshold; the RTU's fan-on-while-unoccupied rule. The parameterised copy in
  `integrity.py` reproduces `faults_apar.parquet` exactly (max difference 0.0 on all 237 dev and exam days).
- *Rules, dev-tuned*: 54 settings (tolerance 0.75-3 K × fan heat 0.5/1.1/1.7 K × persistence 15/30/60 min) scored by
  AUC on the 18 AHU dev days (8 faulty). Chosen: 1.5 K, fan heat 0.5 K, 15 min (dev AUC 0.775 against 0.731 for the
  textbook setting; a single best setting), threshold 0.037 by dev balanced accuracy. RTU: the same fan rule with the
  midpoint threshold (0.0). The two RTU dev days score the wrong way round (faulty 0.000, healthy 0.002), so neither
  RTU threshold means much.
- *Mixing-box residual*: per unit-day, on 15-min bins with the fan on, the largest dev-standardised of (1) the
  outdoor-air-fraction residual |(MA−RA)/(OA−RA) − f(cmd)| with f(cmd) = 0.88·cmd^1.2 fitted on 416 healthy dev
  bins, (2) the coils-off residual |SA − MA − 1.05 K|, (3) supply-air tracking |SA − SP|. Threshold 3.84 (dev balanced
  accuracy 0.68, dev AUC 0.65). For (2) every logged valve must be ≤ 0.02; where one valve is not logged only the other
  is checked (an implementation detail fixed before scoring). The ORNL RTU logs no outdoor or mixed-air temperature or
  setpoint, so no residual exists there.

**Every graded row.** FF − best: FaultFinder minus the best of the three baselines on that row (the best is picked on
the exam, which favours the baselines). 95 % bootstrap, 2,000 resamples, seed 20261117, over unit-days (the verdict)
and over facility-days (ERS AHU-A/B and Nesbitt AHU-1/2 share a day). A row the dev majority guess wins is a loss
(on the false-alarm rows, never alarming scores 0 %). RTU rows: the two "rules" columns are the fan rule with the
released and the midpoint threshold. Rates in percentage points. \* Report only: thresholds re-set on each resample's
own healthy days.

| Units | Metric | n | FaultFinder | rules, textbook | rules, dev-tuned | mixing box | FF − best [95 % CI, unit-days] | [facility-days] | v1.0 | now |
|---|---|---|---|---|---|---|---|---|---|---|
| All units | AUC | 217 | 0.77 | 0.69 | 0.69 | 0.61 | +0.08 [−0.00, +0.16] | [+0.01, +0.16] | win | tie |
| All units | bal. acc. @0.5 | 217 | 0.69 | 0.65 | 0.69 | 0.63 | −0.00 [−0.08, +0.08] | [−0.07, +0.08] | win | tie |
| All units | false alarms @0.5 (%) | 217 | 35 | 3 | 18 | 14 | +32 [+23, +42] | [+24, +41] | loss | loss |
| All units | recall @0.5 (%) | 217 | 73 | 32 | 55 | 41 | +17 [+5, +29] | [+5, +30] | - | win |
| All units | recall @2 % FA (%)* | 217 | 20 | 30 | 30 | 21 | −10 [−29, +14] | [−24, +11] | loss | tie |
| All units | recall @5 % FA (%)* | 217 | 24 | 39 | 38 | 29 | −15 [−24, +14] | [−24, +14] | loss | tie |
| All units | recall @10 % FA (%)* | 217 | 45 | 43 | 42 | 36 | +2 [−15, +23] | [−14, +23] | win | tie |
| AHU | AUC | 200 | 0.78 | 0.69 | 0.69 | 0.64 | +0.09 [−0.00, +0.18] | [+0.00, +0.17] | win | tie |
| AHU | bal. acc. @0.5 | 200 | 0.70 | 0.65 | 0.68 | 0.66 | +0.02 [−0.08, +0.11] | [−0.06, +0.11] | win | tie |
| AHU | false alarms @0.5 (%) | 200 | 33 | 2 | 17 | 15 | +31 [+23, +41] | [+22, +40] | loss | loss |
| AHU | recall @0.5 (%) | 200 | 73 | 32 | 52 | 46 | +21 [+7, +34] | [+7, +35] | - | win |
| AHU | recall @2 % FA (%)* | 200 | 23 | 32 | 32 | 24 | −9 [−32, +18] | [−22, +15] | loss | tie |
| AHU | recall @5 % FA (%)* | 200 | 27 | 41 | 40 | 33 | −13 [−24, +17] | [−23, +15] | loss | tie |
| AHU | recall @10 % FA (%)* | 200 | 47 | 45 | 44 | 40 | +2 [−17, +26] | [−16, +25] | win | tie |
| RTU (ORNL) | AUC | 17 | 0.65 | 0.72 | 0.72 | – | −0.07 [−0.44, +0.38] | [−0.44, +0.38] | loss | tie |
| RTU (ORNL) | bal. acc. @0.5 | 17 | 0.47 | 0.57 | 0.72 | – | −0.24 [−0.57, +0.20] | [−0.57, +0.20] | loss | loss |
| RTU (ORNL) | false alarms @0.5 (%) | 17 | 80 | 20 | 40 | – | +60 [−20, +100] | [−20, +100] | loss | tie |
| RTU (ORNL) | recall @0.5 (%) | 17 | 75 | 33 | 83 | – | −8 [−25, +0] | [−25, +0] | - | tie |
| RTU (ORNL) | recall @2 % FA (%)* | 17 | 42 | 25 | 25 | – | +17 [−47, +55] | [−47, +55] | win | tie |
| Iowa ERS | AUC | 101 | 0.73 | 0.73 | 0.73 | 0.57 | +0.00 [−0.12, +0.13] | [−0.12, +0.13] | tie | tie |
| Iowa ERS | bal. acc. @0.5 | 101 | 0.65 | 0.65 | 0.70 | 0.60 | −0.05 [−0.18, +0.07] | [−0.17, +0.08] | - | tie |
| Iowa ERS | false alarms @0.5 (%) | 101 | 27 | 2 | 24 | 0 | +27 [+15, +40] | [+16, +38] | - | loss |
| Iowa ERS | recall @0.5 (%) | 101 | 57 | 33 | 63 | 20 | −7 [−26, +12] | [−27, +14] | - | tie |
| Iowa ERS | recall @2 % FA (%)* | 101 | 13 | 33 | 33 | 24 | −20 [−43, +22] | [−43, +22] | - | tie |
| Iowa ERS | recall @5 % FA (%)* | 101 | 20 | 46 | 41 | 26 | −26 [−44, +17] | [−43, +17] | - | tie |
| Iowa ERS | recall @10 % FA (%)* | 101 | 41 | 52 | 46 | 26 | −11 [−37, +13] | [−37, +13] | - | tie |
| LBNL FLEXLAB | AUC | 23 | 0.93 | 0.67 | 0.74 | 0.67 | +0.19 [−0.03, +0.43] | [−0.03, +0.43] | win | tie |
| LBNL FLEXLAB | bal. acc. @0.5 | 23 | 0.85 | 0.76 | 0.69 | 0.57 | +0.09 [−0.22, +0.29] | [−0.22, +0.29] | - | tie |
| LBNL FLEXLAB | false alarms @0.5 (%) | 23 | 25 | 0 | 25 | 75 | +25 [+0, +82] | [+0, +82] | - | loss |
| LBNL FLEXLAB | recall @0.5 (%) | 23 | 95 | 53 | 63 | 89 | +5 [−12, +23] | [−12, +23] | - | tie |
| LBNL FLEXLAB | recall @2 % FA (%)* | 23 | 89 | 63 | 58 | 42 | +26 [+0, +53] | [+0, +53] | - | tie |
| Nesbitt Hall | AUC | 76 | 0.78 | 0.54 | 0.54 | 0.71 | +0.07 [−0.05, +0.19] | [−0.06, +0.22] | win | tie |
| Nesbitt Hall | bal. acc. @0.5 | 76 | 0.72 | 0.57 | 0.58 | 0.68 | +0.05 [−0.08, +0.17] | [−0.08, +0.19] | win | tie |
| Nesbitt Hall | false alarms @0.5 (%) | 76 | 40 | 2 | 8 | 26 | +38 [+24, +52] | [+25, +51] | loss | loss |
| Nesbitt Hall | recall @0.5 (%) | 76 | 85 | 15 | 23 | 62 | +23 [+4, +42] | [+6, +46] | - | win |
| Nesbitt Hall | recall @2 % FA (%)* | 76 | 19 | 15 | 15 | 35 | −15 [−40, +7] | [−42, +8] | - | tie |
| Nesbitt Hall | recall @5 % FA (%)* | 76 | 23 | 19 | 19 | 35 | −12 [−42, +12] | [−41, +14] | win | tie |
| Nesbitt Hall | recall @10 % FA (%)* | 76 | 46 | 23 | 23 | 54 | −8 [−36, +23] | [−35, +21] | - | tie |
| Diagnosis | top-1 (%) | 66 | 20 | always commonest: 21 | – | – | -2 [−17, +14] | [−16, +14] | loss | tie |
| Diagnosis | top-3 (%) | 66 | 48 | always commonest: 55 | – | – | -6 [−23, +9] | [−24, +12] | loss | tie |

**Before and after.** None of v1.0's graded wins survives as a win: AHU and all-units AUC are ties by the unit-day
interval (+0.09 [−0.00, +0.18] and +0.08 [−0.00, +0.16]; the facility-day intervals exclude zero), balanced accuracy
ties the dev-tuned rules (AHU +0.02 [−0.08, +0.11]), Nesbitt ties a mixing-box residual (AUC 0.78 against 0.71),
FLEXLAB ties (+0.19 [−0.03, +0.43]) and every fixed-false-alarm row is a tie. Several losses become ties (RTU AUC and
false alarms, the strict-false-alarm recalls, diagnosis top-1 and top-3). Three new wins: recall at P = 0.5 for all
units (+17 [+5, +29]), AHUs (+21 [+7, +34]) and Nesbitt (+23 [+4, +42]), the other side of the false-alarm losses.
What stands: FaultFinder catches more faults at P = 0.5 and raises far more false alarms; on ranking and balanced
accuracy it is not distinguishable from rules tuned on the same 20 dev days.

**Dev twin test of what the network adds** (20 dev days, no exam data). Anchor: the dev-tuned rule score mapped to a
probability by a two-parameter logistic fit on dev. The network's correction (P − P_rule) against what the truth
needed (truth − P_rule): correlation **+0.07**. Dev AUC 0.71 (network) against 0.71 (rule). Right at the operating
point: network 13 of 20, rule 14 (both 10, network only 3, rule only 4, both wrong 3). The network alone is right on
Nesbitt AHU-2's two faulty days (chilled-water sensor bias, cooling valve stuck; rule score 0) and the ORNL
infiltration day; the rule alone is right on ERS AHU-B's healthy 24 Aug 2007 (network P 0.501, on the threshold), the
ERS exhaust-damper day of 10 May 2008, Nesbitt AHU-1's sensor-bias day and Nesbitt AHU-2's healthy 29 Jul 2017
(network P 0.79). Mean |truth − P| is 0.42 for the anchor and 0.45 for the network.

**Unit habit, dev only** (for the multi-day rung). Day vectors: mean and standard deviation of each feature row over
fan-on bins, standardised over the 20 dev days. For the 10 healthy dev days with another day of the same unit, the
nearest day is the same unit's for 4 (median distance 0.81 to the same unit, 0.82 to other units). The network's
log-odds on the 8 healthy days of the three units with two or more: 38 % of the variance is between units, 62 %
within (one Nesbitt AHU-2 day at +5.6 dominates). Subtracting the mean log-odds of the unit's other healthy dev days
changed dev false alarms from 3 to 4 of 8 and dev AUC on those 13 days from 0.78 to 0.63. On 20 days, "most false
alarms are unit habits" is not supported; two of the four dev false alarms sit at P = 0.501, a calibration effect.

**Detection rate at P = 0.5 by exam fault type** (faulty days): cc valve stuck 100 % (14), OA damper stuck 93 % (14),
schedule 75 % (8), chiller off 83 % (6), EA damper stuck 40 % (5), OA damper leak 60 % (5), hc valve leak 80 % (5),
control unstable 25 % (4), hc valve stuck 100 % (4), zone sensor bias 75 % (4), cc valve leak 100 % (3), hc reduced
capacity 33 % (3), lighting schedule 100 % (3), return-fan fixed speed 33 % (3), SAT sensor bias 100 % (3), air
filter 50 % (2), condenser fouling 0 % (2), hc fouling 0 % (2), return-fan failure 50 % (2), CHW temperature sensor
50 % (2), CHW pressure sensor 100 % (2), OAT sensor bias 50 % (2), supply duct leak 0 % (2), reverse-acting cc valve,
EA damper leak and infiltration 100 % (1 each). Healthy days: 35 % flagged.

## Round 2 rungs (17 Nov 2026): r7 and r7b not kept, r8 released as v1.1

Plan fixed in NOTEBOOK.md before bank v5 was built; record in `training/ratchet_r7.json`. Same rule as the ladder,
judged on a new frozen set v5 (BuildingSim 1.0.1 phase2 wrappers, unit habits fixed per 12-day window, 9,145 samples).

| Rung | skill v5 | dev AUC | dev false alarms (of 11) | decision |
|---|---|---|---|---|
| r6 = v1.0 (released weights) | 0.561 | 0.71 | 4 | incumbent |
| r7b: Platt map on dev instead of T = 4.3 | 0.561 | 0.71 | 1 (leave-one-out) | rejected (skill unchanged by construction) |
| r7_context: + the unit's last 1-3 days, 3-seed mean | 0.558 | 0.66 | 3-6 by seed | rejected (not > 0.571) |
| control: r6's network retrained on bank v5 | 0.554 | 0.68 | 6 | information |
| **r8: v1.0 frozen + reference rows (zero-initialised), 2-seed mean** | **0.572** | 0.71 | 4 | **released as v1.1** (margin +0.0118) |

Reference days help r7 on simulated days that have them (0.563 vs 0.537 with them zeroed), but r7's single-day
accuracy is below v1.0's. Platt on 20 dev days cuts false alarms but also 2 of 6 detections; its leave-one-out Brier
(0.258) is worse than the temperature's (0.246). The fresh exam was not scored.

r8 keeps v1.0's 160,103 weights frozen and trains only the 15,040 first-layer weights of the new rows (AdamW, lr 1e-3
cosine, 3 epochs, bank v5). Single-day mode matches v1.0 to 1.9e-6; numpy matches torch to 3.8e-6. Seed 1 is v1.1,
with the widen-only temperature re-fitted on the dev days and their references: T = 4.2. Only 2 of the 20 dev days
have a reference, so the dev gate could not test the new rows. The release rests on simulation.

**v1.1 on the seen exam (re-scored after the release decision; nothing chosen from it).** References: the same unit's
exam days from the previous 7 days (most recent 3, labels unread); 153 of 217 days have one. Same baselines and
interval rule as the integrity pass.

| Row (seen) | v1.0 | v1.1 | v1.1 minus best baseline, 95 % CI | verdict v1.0 → v1.1 |
|---|---|---|---|---|
| All units AUC | 0.77 | 0.76 | +0.07 [−0.01, +0.14] | tie → tie |
| AHU AUC | 0.78 | 0.78 | +0.09 [−0.00, +0.17] | tie → tie |
| AHU balanced accuracy | 0.70 | 0.65 | −0.03 [−0.11, +0.06] | tie → tie |
| AHU false alarms at P = 0.5 | 33 % | 56 % | +54 points [+44, +64] | loss → loss |
| AHU recall at P = 0.5 | 73 % | 86 % | +34 [+22, +46] | win → win |
| RTU AUC / balanced accuracy | 0.65 / 0.47 | 0.72 / 0.53 | +0.00 [−0.40, +0.44] / −0.18 [−0.55, +0.22] | tie / loss → tie / tie |
| Iowa ERS AUC / balanced accuracy | 0.73 / 0.65 | 0.66 / 0.55 | −0.07 [−0.20, +0.06] / −0.14 [−0.26, −0.02] | tie / tie → tie / loss |
| Iowa ERS false alarms | 27 % | 67 % | +67 points [+54, +80] | loss → loss |
| FLEXLAB AUC | 0.93 | 0.97 | +0.23 [+0.02, +0.48] | tie → win |
| Nesbitt AUC / false alarms | 0.78 / 40 % | 0.79 / 44 % | +0.08 [−0.03, +0.20] / +42 [+28, +56] | tie / loss → tie / loss |
| Diagnosis top-1 / top-3 | 20 / 48 % | 18 / 50 % | −3 [−18, +11] / −5 [−21, +11] | tie → tie |
| ECE / Brier (info) | 0.12 / 0.216 | 0.14 / 0.220 | | |

Cause: on real days with a reference the new rows add a near-constant +0.1 to +0.7 to the detection log-odds (ERS
healthy days +0.71 on average, faulty +0.45). That is a sim-to-real offset the simulated training could not see.

## Limits
1. Diagnosis on real data is about as good as guessing the commonest fault; use the top 3 as a checklist.
2. 33 % false alarms on healthy AHU days at P = 0.5 (rules: 2 %); v1.1 with reference days 56 % (seen exam). On ranking (AUC) and balanced accuracy it is not
   distinguishable from APAR rules tuned on the same 20 dev days (integrity pass).
3. Probabilities squashed toward 0.5 by a temperature fitted on 20 days (ranking unaffected).
4. RTU: worse than a one-line rule on 17 days.
5. History is at most the unit's last 3 days within a week (v1.1), learned on simulated days only; on real days it acts
   mostly as an upward offset (limit 2). Whether false alarms are unit habits is not established on the dev days. Single-zone simulator with ideal controls, no exhaust damper, return fan or duct
   model.
6. Small exam: 217 days, 5 facilities; 95 % intervals on model minus baseline are about ±0.09 AUC (AHU) and ±0.4
   (RTU).

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
| `examples/references/lbnl_rtu_ornl_2018-01-27.csv`, `..._2018-01-28.csv` (v1.1) | LBNL synthesis (ORNL FRP2 RTU): the RTU example's earlier exam days, labels not shipped | CC0 1.0 |
| `examples/references/NES_20180709_AHU-1.csv`, `NES_20180710_AHU-1.csv` (v1.1) | Drexel Nesbitt Hall via Ghalamsiah (as above): the Nesbitt example's earlier exam days, labels not shipped | CC BY 4.0 |
| `site/public/instruments/faultfinder/` reference days (v1.1) | earlier exam days of the instrument's units (the four `examples/references/` files and lab-page exam days) | per day as above |
| `lab/index.html` | all 217 exam days (ASHRAE RP-1312 ERS and Nesbitt via Ghalamsiah, CC BY 4.0; LBNL synthesis, CC0), rounded to 0.01 degC / 0.001, plus BuildingSim lab days (simulated, with one draw of dev noise) | CC BY 4.0 for the Ghalamsiah days, CC0 for the LBNL days; credits in the page footer |
| `site/public/instruments/faultfinder/` (the live instrument) | twelve exam days drawn from the four sources (see `web/export_instrument.py`) | per day as above; the instrument shows each day's source |

The building-status handoff described the fault subset as CC0. The release's own credits (above) say otherwise for
part of it: the LBNL synthesis is CC0, but the ASHRAE RP-1312 ERS and Nesbitt Hall files come from Ghalamsiah's
figshare package under CC BY 4.0, which requires credit (given here, in `model.yaml` and on the page). The Korean
office AHU data (CC BY 4.0) were used only to fit noise statistics; no Korean data ship. The code is MIT.
