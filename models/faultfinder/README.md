<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# FaultFinder <sub>v1.0</sub>

> **Is this air handler faulty today, and which fault is it, from one day of trend log?**

> [!WARNING]
> Research model: it flags days for a person to look at and does not replace commissioning. At P = 0.5 it raises a false alarm on a third of healthy air-handler days, and its fault diagnosis is no better than naming the commonest fault.

*From one day of an air handler's or rooftop unit's building-automation trend log, how likely is a fault, and which fault is it?*

FaultFinder reads one local day of an air handler's (AHU) or rooftop unit's (RTU) building-automation trend log (outdoor, return, mixed, supply and zone air temperatures, the supply-air setpoint, damper and valve commands, fan speed, power and status, occupancy; 1- to 15-min samples, any subset) and returns P(fault), a probability over 18 fault classes, a severity for the top class and the day's extra HVAC energy. It learned only from BuildingSim simulations: 38,987 simulated days of random RTU and VAV buildings in 17 US climates, with real sensor noise and quirks from 20 real dev days. It was examined once on 217 real lab and building days from four sources (Iowa Energy Resource Station, LBNL FLEXLAB, ORNL's rooftop unit and Drexel's Nesbitt Hall).

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/faultfinder/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/faultfinder/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | One local day of an AHU or RTU trend log, 1- to 15-min samples, binned to 96 × 15 min: T_oa, T_ra, T_ma, T_sa, T_sa_sp (or T_sa_sp_cool / T_sa_sp_heat), T_zone (°C); damper_oa, valve_cc, valve_hc, fan_sa_speed (0..1, controller commands as logged); comp1_status / comp2_status for RTUs (0/1); fan_sa_power_kW or a fan current (any unit, scaled by its own daily 95th percentile); fan_sa_status, occ (0/1). Any subset; missing channels are flagged to the network. Temperatures below −45 or above 70 °C, and flatlined temperature sensors, count as missing. |
| **Returns** | P(fault) for the day, calibrated by one temperature (T = 4.3) fitted on 20 real dev days; ranking is unaffected, but values are squashed toward 0.5 (164 of 217 exam days fall in 0.4-0.6)<br>Diagnosis: a probability over 18 fault classes given a fault (top-1 20 %, top-3 48 % on 66 real faulty days)<br>A severity parameter for the top class and the day's extra HVAC energy fraction (simulation-trained only; the energy output is not useful on real data: Spearman 0.19) |
| **Learned from** | BuildingSim, the collection's building simulator (shared building platform, Phase 1): a 5R1C zone heat balance with solar gains, wind and stack infiltration and moisture, and air handlers built from fan laws, a mixing box, cooling and heating coils, chillers and boilers and DX curves from DOE reference data. Twenty fault types are injected into the plant and logged as a BMS-style trend. A FaultFinder wrapper (training/simbank.py, noise.py) makes the log look like a real BMS: commands rather than actuator positions, an installed damper characteristic, larger static sensor offsets and operator start/stop jitter. |
| **Noise** | A real-noise model fitted on the real fault-free dev days only: each channel's 15-min residual around a 75-min running mean while the fan runs (residual library, random circular shift and gain 0.5-1.5), static sensor offsets, sensor drift toward a return/outdoor blend when the fan stops, logger gaps, flatlined sensors and each exam source's set of logged points. Twenty unlabelled Korean office AHUs set the start/stop irregularity and the tail of supply-air tracking error. |
| **Network** | 1-D CNN over 96 × 15-min bins and 35 feature rows (6 temperatures, 5 physical differences MA−RA, SA−MA, SA−SP, OA−RA, zone−RA, the estimated outdoor-air fraction, 7 actuator and state signals, time of day, 14 availability masks): convolutions 64-64-128-128 (kernels 5, 5, 5, 3; two max-pools), mean + max pooling, dense 128, heads for 19 class logits, 19 severity parameters and 1 energy-waste fraction. 160,871 parameters (160,103 after BatchNorm folding). One calibration temperature T = 4.3 on the detection log-odds and the diagnosis logits, fitted on the 20 dev days (widen-only, T >= 1). |
| **Physics also enters** | inputs: Mixing-box and coil physics as feature rows: MA−RA, SA−MA, SA−setpoint, OA−RA, zone−RA, and the outdoor-air fraction (MA−RA)/(OA−RA) from the mixed-air energy balance |
| **Real data's jobs** | backdrop, calibration, gate, exam (never training labels) |
| **Dev set** | BuildingData's frozen fault-run split (data/proc/splits.json, seed 20261116): 20 unit-days, seven at Iowa ERS, three FLEXLAB, two ORNL RTU, eight Nesbitt Hall. They were the only real labelled data used for anything: the noise library, the twin tests, the dev gate and the calibration temperature. |
| **Exam** | Every other gradable run: 217 real or experimental unit-days, 200 AHU days (91 faulty) and 17 RTU days (12 faulty), from Iowa ERS (101), LBNL FLEXLAB (23), ORNL RTU (17) and Nesbitt Hall (76). Run once, after the ladder was closed. |

## Run it

```bash
cd models/faultfinder
pip install numpy pandas
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo faultfinder`.

```python
import pandas as pd
from faultfinder import FaultFinder
ff = FaultFinder()
df = pd.read_csv("examples/nesbitt_NES_20180711_AHU-1.csv")   # contract column names + 'time_local'
r = ff.read_day(df)
print(r["p_fault"], r["diagnosis"][:3])
```

## The exam

Detection (P(fault)) and diagnosis on 217 real days, against the APAR rule set (12 rules, after Schein et al., 2006) for AHUs and a one-line rule (fan on while unoccupied) for RTUs, each with a threshold tuned on the dev days, and the majority guess. Run once with the released rung; re-scoring the 217 exam days embedded in the lab page from the shipped weights in the repository gives the same AUCs and diagnosis rates. Every number here is on real data that never chose anything.

| Metric | FaultFinder | Baselines | |
|---|---|---|---|
| AHU detection AUC (200 days, 91 faulty)<br><sub>90 % interval 0.72-0.83 (bootstrap over facility-days).</sub> | **0.78** | APAR rules (12 rules, dev-tuned) (classical): 0.69 | ✅ wins |
| AHU balanced accuracy at P = 0.5 | **0.7** | APAR rules at the dev-tuned threshold (classical): 0.65<br>Majority guess (no fault) (dumb baseline): 0.5 | ✅ wins |
| AHU false-alarm rate at P = 0.5 (109 healthy days)<br><sub>It catches more faults at 0.5 (73 % against 32 % for the rules) but flags a third of healthy days.</sub> | **33%** | APAR rules at the dev-tuned threshold (classical): 2%<br>Majority guess (no fault) (dumb baseline): 0% | ❌ loses |
| AHU recall at a 2 % false-alarm rate<br><sub>Report only: the threshold for each method is set from the exam's own healthy days, so this is not a deployable operating point.</sub> | **23%** | APAR rule score (classical): 32% | ❌ loses |
| AHU recall at a 5 % false-alarm rate<br><sub>Report only: thresholds set from the exam's own healthy days.</sub> | **27%** | APAR rule score (classical): 41% | ❌ loses |
| AHU recall at a 10 % false-alarm rate<br><sub>Report only: thresholds set from the exam's own healthy days. A margin of two days out of 91.</sub> | **47%** | APAR rule score (classical): 45% | ✅ wins |
| RTU detection AUC (17 ORNL days, 12 faulty) | **0.65** | Rule: fan on while unoccupied (classical): 0.72 | ❌ loses |
| RTU balanced accuracy at P = 0.5 | **0.47** | Rule: fan on while unoccupied (dev-tuned) (classical): 0.57<br>Majority guess (fault) (dumb baseline): 0.5 | ❌ loses |
| RTU false-alarm rate at P = 0.5 (5 healthy days) | **80%** | Rule: fan on while unoccupied (dev-tuned) (classical): 20%<br>Majority guess (fault) (dumb baseline): 100% | ❌ loses |
| RTU recall at 2, 5 and 10 % false alarms (with 5 healthy days, all three mean no false alarm)<br><sub>Report only: thresholds set from the exam's own healthy days; 5 of 12 faulty days against 3 of 12.</sub> | **42%** | Rule: fan on while unoccupied (classical): 25% | ✅ wins |
| All units detection AUC (217 days, 103 faulty)<br><sub>90 % interval 0.72-0.83.</sub> | **0.77** | APAR rules (AHU) and the fan rule (RTU) (classical): 0.69 | ✅ wins |
| All units balanced accuracy at P = 0.5 | **0.69** | Rules at their dev-tuned thresholds (classical): 0.65<br>Majority guess (dumb baseline): 0.5 | ✅ wins |
| All units false-alarm rate at P = 0.5 (114 healthy days) | **35%** | Rules at their dev-tuned thresholds (classical): 3% | ❌ loses |
| All units recall at a 2 % false-alarm rate<br><sub>Report only: thresholds set from the exam's own healthy days.</sub> | **20%** | Rule scores (classical): 30% | ❌ loses |
| All units recall at a 5 % false-alarm rate<br><sub>Report only: thresholds set from the exam's own healthy days.</sub> | **24%** | Rule scores (classical): 39% | ❌ loses |
| All units recall at a 10 % false-alarm rate<br><sub>Report only: thresholds set from the exam's own healthy days.</sub> | **45%** | Rule scores (classical): 43% | ✅ wins |
| Nesbitt Hall, a real occupied building, detection AUC (76 days, 26 faulty) | **0.78** | APAR rules (classical): 0.54 | ✅ wins |
| Nesbitt Hall balanced accuracy at P = 0.5 | **0.72** | APAR rules at the dev-tuned threshold (classical): 0.57<br>Majority guess (no fault) (dumb baseline): 0.5 | ✅ wins |
| Nesbitt Hall false-alarm rate at P = 0.5 (50 healthy days) | **40%** | APAR rules at the dev-tuned threshold (classical): 2% | ❌ loses |
| Nesbitt Hall recall at a 5 % false-alarm rate<br><sub>Report only: thresholds set from the exam's own healthy days; 6 of 26 faulty days against 5.</sub> | **23%** | APAR rule score (classical): 19% | ✅ wins |
| LBNL FLEXLAB detection AUC (23 days, 19 faulty) | **0.93** | APAR rules (classical): 0.67 | ✅ wins |
| Iowa ERS detection AUC (101 days, 46 faulty) | **0.73** | APAR rules (classical): 0.73 | ➖ tie |
| Diagnosis top-1 accuracy (66 faulty days with a simulable fault)<br><sub>Cooling valve stuck 50 % (14 days) and zone sensor or setpoint 100 % (4) work; outdoor-air damper stuck 0 % (14). By source: ERS 9 %, FLEXLAB 32 %, Nesbitt 6 %, ORNL 50 %.</sub> | **20%** | Always the commonest exam class (outdoor-air damper stuck) (dumb baseline): 21% | ❌ loses |
| Diagnosis top-3 accuracy (66 faulty days)<br><sub>The three-class guess (36 of 66 days) was computed at import from the exam's class counts in training/exam.json; the release reported top-3 without a baseline.</sub> | **48%** | Always the three commonest exam classes (damper stuck, cooling valve stuck, schedule) (dumb baseline): 55% | ❌ loses |
| Calibration error of P(fault), ECE in 5 bins (all / AHU / RTU)<br><sub>164 of 217 days fall in P 0.4-0.6 (37 % faulty there); 0.6-0.8: 37 days, 78 % faulty; 0.8-1.0: 15 days, 93 % faulty.</sub> | **0.12 / 0.13 / 0.17** | — | ℹ️ |
| Brier score of P(fault), all 217 days<br><sub>The uncalibrated network scores 0.204: the temperature fitted on 20 dev days over-shrinks.</sub> | **0.216** | — | ℹ️ |
| Detection AUC by season (winter / transition / summer) | **0.71 / 0.83 / 0.75** | — | ℹ️ |
| Detection AUC by fault severity (high / low / unspecified) | **0.79 / 0.79 / 0.72** | — | ℹ️ |
| Extra-HVAC-energy output vs measured, Spearman (46 ERS paired days)<br><sub>Faulty AHU-A against healthy AHU-B on the same day, air-side energy proxy. Not useful; do not rely on extra_hvac_energy_frac.</sub> | **0.19** | — | ℹ️ |

### Where it wins

- Ranking AHU days by fault risk: AUC 0.78 against 0.69 for the APAR rules on 200 real days.
- Nesbitt Hall, a real occupied building: AUC 0.78 against 0.54 for the rules, balanced accuracy 0.72 against 0.57.
- LBNL FLEXLAB: AUC 0.93 against 0.67.
- At a looser 10 % false-alarm rate it catches 47 % of faulty AHU days against 45 % for the rules (report-only thresholds).

### Where it loses

- False alarms: at P = 0.5 it flags 33 % of healthy AHU days (Nesbitt 40 %) against 2 % for the rules.
- At strict false-alarm rates the rules catch more: 32 % against 23 % at 2 %, 41 % against 27 % at 5 % (AHU, report-only thresholds).
- Rooftop unit: AUC 0.65 against 0.72 for the one-line rule 'fan on while unoccupied'; balanced accuracy 0.47, below the majority guess.
- Diagnosis: top-1 20 % against 21 % for always naming the commonest fault; top-3 48 % against 55 % for naming the three commonest.
- Iowa ERS: a tie with the rules (0.73 vs 0.73).

## Limits

- Diagnosis on real data is about as good as guessing the commonest fault. Use the top three as a checklist, not an answer; outdoor-air damper stuck is never top-1 (0 of 14).
- Probabilities are squashed toward 0.5 by a temperature fitted on 20 days (T = 4.3); the ranking is unaffected, and the uncalibrated Brier score is better (0.204 vs 0.216).
- One day at a time: it does not compare a day with the unit's own history, and most real false alarms are unit habits (Nesbitt's AHU runs all night with the chiller off; ORNL's zone sensor reads 1.5 K above return on a normal day).
- Single-zone simulator with ideal controls and no exhaust damper, return fan, duct model, chiller-off or chilled-water sensor faults: 13 of the exam's 27 fault types cannot be diagnosed at all.
- Small exam: 217 days from five facilities; the 90 % interval on AHU AUC is about ±0.06. The fixed-false-alarm recall rows set their thresholds on the exam's own healthy days and are report-only.
- The energy-waste output is not validated (Spearman 0.19 on 46 paired ERS days).

## The ratchet

**Objective:** Skill on the frozen simulated validation set = 0.5 × detection AUC + 0.5 × diagnosis accuracy (faulted days with a visible effect)  
**Rule:** Keep if the skill on the newest frozen simulated set beats the incumbent by more than 0.01 AND the dev AUC on the 20 real dev days is at least max(0.60, incumbent dev AUC − 0.05).

| Rung | Change | Frozen-set score | Decision | Why | Skill v1 | Skill v2 | Skill v3 | Sim AUC v4 | Sim diag v4 | Dev AUC | Dev FPR @0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1_gauss | BuildingSim bank v1 (20 fault types, generic controls) + Gaussian sensor noise + source channel patterns; CNN 48 channels, 8 epochs | 0.415 | ● kept | First rung (incumbent by definition). | 0.585 | 0.499 | 0.448 | 0.6 | 0.23 | 0.6 | 0.91 |
| r2_realnoise | + real-noise and quirk model from the dev fault-free days (residual library, static offsets, fan-off drift) | 0.454 | ● kept | Skill 0.454 > 0.415 + 0.01 and dev AUC 0.74 passes the gate (>= 0.60). | 0.575 | 0.55 | 0.476 | 0.66 | 0.25 | 0.74 | 0.45 |
| r3_calib | + dev-calibrated bank v3: the BMS logs commands, not positions (twin test), operator start/stop jitter and off-blocks, schedule-shift faults, 24×7 units, wider VAV-minimum and supply-air setpoint priors | 0.500 | ● kept | Skill 0.500 > 0.454 + 0.01 and dev AUC 0.71 passes the gate (>= 0.69). | 0.528 | 0.51 | 0.536 | 0.68 | 0.31 | 0.71 | 0.45 |
| r4_damper | + noise v4: installed outdoor-air damper characteristic (logged command vs actual outdoor-air fraction) and larger static sensor disagreement (mixed-air stratification, zone vs return) | 0.511 | ● kept | Skill 0.511 > 0.500 + 0.01 and dev AUC 0.73 passes the gate (>= 0.66). | 0.505 | 0.502 | 0.518 | 0.71 | 0.32 | 0.73 | 0.36 |
| r5_bothbanks | Train on bank v1 (generic controls) + bank v3 (calibrated) together, noise v4, 50k days, 6 epochs | 0.519 | ○ rejected | Skill 0.519 not > 0.511 + 0.01 (dev AUC 0.74). | 0.557 | 0.542 | 0.528 | 0.7 | 0.33 | 0.74 | 0.45 |
| r6_wide | Wider network (64/128 channels) and 10 epochs, bank v3 + noise v4 | 0.543 | **◆ released** | Skill 0.543 > 0.511 + 0.01 and dev AUC 0.71 passes the gate (>= 0.68). Released as v1.0. | 0.539 | 0.535 | 0.556 | 0.73 | 0.36 | 0.71 | 0.36 |
| calib_T | Calibration: one temperature T >= 1 on the detection log-odds (the same T on the diagnosis logits), fitted by log loss on the 20 dev days (widen-only) | — | ● kept | T = 4.3. No effect on ranking (AUC) or the top class; it moves P(fault) toward 0.5. On the exam it over-shrank: Brier 0.216 calibrated vs 0.204 uncalibrated. | – | – | – | – | – | – | – |

<sub>Columns after *Why*: Skill on the frozen sets v1-v3 (v4 is the score), detection AUC and diagnosis accuracy on v4, then the dev gate: dev AUC and dev false-alarm rate at P = 0.5 (20 real dev days).</sub>

Each change of the simulated bank or the noise model got a new frozen validation set (v1 generic bank, v2 real noise, v3 calibrated bank, v4 damper characteristic and larger offsets). Every rung was re-scored on every set, and the keep/reject decisions were re-taken in order on the newest (v4); the earlier decisions on older sets are not in the log. r6's description in ratchet.json says 179k weights; the exported network has 160,871 parameters (160,103 after BatchNorm folding). No exam result was used to choose anything.

Raw per-rung logs are in [`training/`](training/).

## Physics

BuildingSim, the collection's building simulator (shared building platform, Phase 1): a 5R1C zone heat balance with solar gains, wind and stack infiltration and moisture, and air handlers built from fan laws, a mixing box, cooling and heating coils, chillers and boilers and DX curves from DOE reference data. Twenty fault types are injected into the plant and logged as a BMS-style trend. A FaultFinder wrapper (training/simbank.py, noise.py) makes the log look like a real BMS: commands rather than actuator positions, an installed damper characteristic, larger static sensor offsets and operator start/stop jitter.

**Simulator**

- Fault bank: random BuildingSim buildings with packaged RTUs, VAV air handlers with chiller/boiler or DX, or VAV on district energy, in 17 TMYx US cities, any season. Each 12-day window is simulated healthy and with one fault from day 5 on; every local day is a sample (faulted days with a visible effect, plus healthy days). Two banks of 38,987 days: v1 (generic controls) and v3 (dev-calibrated).
- Twenty BuildingSim fault types map onto 18 classes: stuck and leaking outdoor-air dampers, economizer faults, stuck and leaking cooling and heating valves, reheat valve leak, cooling capacity (coil and condenser fouling, refrigerant charge), low airflow, sensor biases (supply, outdoor, mixed, return air), zone setpoint or zone sensor offset, schedule, VAV box stuck, chiller or boiler efficiency.
- Of the exam taxonomy's 27 fault types, 14 map onto a class. The other 13 (exhaust damper, supply duct leak, heating-coil fouling and capacity, return fan, unstable control, reverse-acting valve, chilled-water sensors, chiller off, lighting schedule, infiltration) cannot be simulated and are graded for detection only.
- Dev calibration (bank v3, noise v4) from twin tests on the 20 dev days: the BMS logs controller commands, not positions (FLEXLAB's heating valve stuck at 100 % logs 0 %); a real damper at 100 % command mixes only 81-90 % outdoor air and Nesbitt's at 15 % mixes 14-28 %, so the logged damper is a + (b − a)·cmd^g with a ~ U(0, 0.15), b ~ U(0.75, 1), g ~ U(0.6, 1.6); mixed-air stratification sd 0.8 K and zone vs return sd 0.9 K; operator jitter of 2-3 bins and a 1-3 h off block on 15 % of days; schedule-shift faults; 25 % of units run 24×7.

**Checked against independent references**

- RC step response within 0.003 K of the closed form; steady conduction to 2e-9 relative error; energy ledger closes to 1e-8 (BuildingSim checks)
- 31 of 36 BESTEST (ASHRAE 140) results inside the 1995 reference ranges (the transcribed ranges; the official ASHRAE 140 tables are still wanted)
- Psychrometrics within 0.02 K (p95) of CoolProp; PMV within 0.003 of pythermalcomfort
- 26 of 26 BuildingSim fault signatures move the trend in the expected direction
- Numpy inference matches torch to 9.5e-7 (faultfinder_weights.npz.check.json); the lab page's JS port matches numpy to 5e-6 on all 217 exam days (top class agrees on 217 of 217)

## Help wanted

- LBNL's full fault-detection datasets sit behind a name-and-e-mail form at faultdetection.lbl.gov. They would add a second real RTU exam and more field data; the owner can fill in the form.
- More real, labelled field data from occupied buildings (Nesbitt Hall is the only one): the exam has 217 days from five facilities.
- An EnergyPlus binary for a true reference engine (no PyPI wheel with an engine; GitHub was blocked in the build sandbox) and the official ASHRAE 140 tables for BESTEST.

## Next rungs

- Multi-day context: compare a day with the unit's own recent days, so unit habits (24×7 operation, a chiller off at night) stop reading as faults.
- Simulate the missing air-side and plant faults in BuildingSim (exhaust and return-fan dampers, return fan, duct leak, chiller off, chilled-water sensors) so 13 more exam fault types become diagnosable.
- A two-parameter (Platt) calibration or a larger dev set; one temperature from 20 days over-shrinks.
- Upstream the FaultFinder wrappers into BuildingSim behind flags (log commands, installed damper characteristic, larger static offsets, operator jitter) and re-run the platform checks.

## Files

| Path | What |
|---|---|
| `faultfinder.py`, `pipeline.py` | Inference — numpy/scipy only |
| `faultfinder_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [ASHRAE RP-1312 AHU fault experiments, Iowa Energy Resource Station (J. Wen and S. Li, 2011), as packaged by N. Ghalamsiah (2025)](https://doi.org/10.6084/m9.figshare.29297999.v3) | dev and noise (among the 7 Iowa ERS dev unit-days; residual library), exam (92 days, dataset ers_rp1312) | CC BY 4.0 |
| [Drexel University Nesbitt Hall AHU-1 fault data (Y. Chen, PhD thesis, Drexel, 2019), as packaged by N. Ghalamsiah (2025)](https://doi.org/10.6084/m9.figshare.29297999.v3) | dev (8 unit-days), noise, exam (76 days); examples/nesbitt_NES_20180711_AHU-1.csv | CC BY 4.0 |
| [LBNL fault detection data synthesis: FLEXLAB SZCAV/SZVAV, ORNL Flexible Research Platform RTU, MZVAV-2 at Iowa ERS (J. Granderson, G. Lin et al.)](https://doi.org/10.6084/m9.figshare.11752740) | dev (3 FLEXLAB, 2 ORNL unit-days), noise, exam (23 FLEXLAB, 17 ORNL, 9 MZVAV-2 ERS days); examples/flexlab_*.csv, iowa_ers_*.csv, ornl_rtu_*.csv | CC0 1.0 |
| [Korean office AHU BMS data (S. Wang, 2024), unlabelled](https://doi.org/10.6084/m9.figshare.25932409.v2) | noise (start/stop irregularity, supply-air tracking-error tail); never labelled, never graded | CC BY 4.0 |
| [TMYx typical-year weather for 17 US cities](https://climate.onebuilding.org) | physics (weather for the simulated fault bank) | Cite Lawrie and Crawley, TMYx; the site says All Rights Reserved and states no reuse licence: used for simulation only, not redistributed |
| [BuildingSim building generator priors (NREL ComStock, EIA CBECS) and DOE reference equipment curves, via the shared building platform](https://comstock.nrel.gov) | physics (random buildings and HVAC plant for the fault bank) | ComStock CC BY 4.0; CBECS and DOE data public domain (U.S. Government) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
