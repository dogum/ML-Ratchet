<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# BuildingDetective <sub>v1.0</sub>

> **What kind of building is this, from its meters and the weather?**

> [!WARNING]
> Research model, not an energy audit. On real buildings it beats simple rules only for the heating system and for which meters a campus building has; it loses on primary use (11 % right against 40 % for the majority class), has no skill on year built, and envelope UA and time constant have no real truth at all.

*From a building's energy meters and the local weather, how big, how old, how leaky and how heavy is it, what heats and cools it, and what is it used for?*

BuildingDetective reads a building's meters (electricity, on-site fuel, district heat, district chilled water) and the outdoor weather at one of three resolutions: a year of hourly data, 12 monthly bills, or one annual total per fuel. It returns floor area, year built, envelope conductance UA (W/K and W/m²K), thermal time constant, heating and cooling balance points, heating and cooling energy per kelvin, non-HVAC share, weekday operating hours and weekend ratio, each with a 1-sigma bar, and probabilities for primary use (12 classes), heating system (heat pump, resistance, fuel, district, none) and cooling system (none, DX, chiller, district). It learned only from 12,000 BuildingSim buildings on real 2016-17 weather with emulated meter faults, and was examined once on 1,428 real BDG2 buildings, 7,401 RECS homes, 2,558 CBECS buildings and 24 HUE homes.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/buildingdetective/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/buildingdetective/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Hourly: up to a year (30 days minimum) of hourly mean kW per meter (elec, fuel, dheat, dcool; NaN for missing hours, hour-beginning), outdoor temperature (°C), global horizontal irradiance (W/m²), the local time of each hour and the latitude. Monthly: 12 monthly mean kW per meter plus monthly mean temperature and heating/cooling degree days per day (base 18.33 °C). Annual: kWh per year per fuel plus annual heating and cooling degree days (base 18.33 °C). |
| **Returns** | Floor area (m²) and year built, each with a 1-sigma bar widened on real dev buildings (hourly area ×3.0, monthly ×1.54, annual ×1.08)<br>Envelope UA (W/K and W/m²K), thermal time constant (h), heating and cooling balance points (°C), site heating and cooling energy per kelvin (W/K), non-HVAC share of site energy: 1-sigma bars widened on the simulated twins of the 52 BDG2 dev buildings (no real truth exists)<br>Weekday operating hours (h/day) and weekend/weekday ratio, commercial only (no real hourly truth)<br>Probabilities: primary use (12 classes, dev-fitted temperature 10 for hourly and monthly, 7.4 annual, which flattens them on purpose), heating system (5 classes) and cooling system (4 classes) |
| **Learned from** | BuildingSim, the collection's building simulator (shared building platform, Phase 1): an ISO 13790 5R1C single-zone heat balance with Perez solar gains, LBL infiltration, ventilation and moisture, and 13 HVAC systems built from capacity, COP and EIR curves with part load, defrost, fans, pumps and district plants. Labels come from the simulator itself: UA is the 5R1C network conductance (labels()['UA_network_W_K']), tau, balance points, end-use shares, schedules and system classes. Three wrappers calibrated on dev only (training/wrappers.py, upstream candidates for BuildingSim) set use-type intensities (W1), campus district service (W2) and always-on base loads found by the dev twin test (W3, physics v2). |
| **Noise** | Synthetic meter faults modelled on the BDG2 quirks BuildingData documents (gaps, zero runs, spikes, level shifts, clock offsets, Gator-style daily-spread readings, shared loads, scale errors, AR(1) and day-level multiplicative noise; features.quirk_hourly / quirk_monthly / quirk_annual). No real meter residuals are added; the simulated buildings run in real hourly weather. |
| **Network** | Three MLPs, one per input resolution, each 2 × 256 GELU with a Gaussian head (mean and log-sigma) for 12 regression targets (missing labels masked) and three softmax heads; trained with heteroscedastic NLL plus cross-entropy on simulations only. Hourly: 1,199 engineered features (daily load in 19 temperature bins for weekdays and weekends, the 168-hour week profile, cold- and hot-day profiles, a PRISM 5-parameter change-point fit, correlations with exponentially smoothed temperature at 6 time constants, hourly lag correlations, robust statistics), 0.38 M parameters (384,557). Monthly: 112 features, 106,285 parameters. Annual: 12 features, 80,685 parameters. |
| **Physics also enters** | inputs: A PRISM 5-parameter change-point fit (base load, heating and cooling slopes, balance points), degree days and daily-temperature bins computed from the meters and the weather |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | BuildingData's frozen splits (data/proc/splits.json, seed 20261116): 52 BDG2 buildings, 1,848 RECS and 661 CBECS respondents. They were the only real labelled data used: W1/W2/W3 calibration, the twin test, the dev gate, the error-bar widening and the class temperatures. RECS (9,247) and CBECS (3,217) prior respondents set the generator and W1 priors. |
| **Exam** | Everything else: 1,584 BDG2 exam buildings (1,428 with a usable 2017 electricity year), 7,401 RECS homes, 2,558 CBECS buildings and 24 HUE homes; run once, after all choices were frozen (training/exam.json). Exam features were computed before the exam with no metric looked at. |

## Run it

```bash
cd models/buildingdetective
pip install numpy scipy pandas
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo buildingdetective`.

```python
import numpy as np
from buildingdetective import Detective
det = Detective()                                   # weights/ next to the module
r = det.read_annual({"elec": 9800, "fuel": 14500}, 2900, 600)   # kWh/yr, HDD and CDD (base 18.33 C)
print(r["floor_area_m2"], r["heat"])                # (value, lo, hi) at 1 sigma; class probabilities
# hourly: det.read_hourly({"elec": kw, "fuel": kw_gas}, T_out, GHI, local_DatetimeIndex, lat)
# command line: python pipeline.py hourly meters.csv --tz America/Chicago --lat 41.9
```

## The exam

Real truth only where it exists: floor area, use, year built and which meters exist for BDG2 (2017 meters), survey answers for RECS and CBECS (annual totals), the heating type of HUE homes. Baselines marked TRUE are given the true use type, which the Detective is not. Envelope UA, time constant and balance points have no real truth; their rows are simulator-only (held-out BuildingSim buildings, or EnergyPlus ResStock/ComStock meters graded against a BuildingSim twin) and are info. A margin smaller than about one sampling error of the statistic is called a tie. Every number here is on real data that never chose anything.

| Metric | BuildingDetective | Baselines | |
|---|---|---|---|
| BDG2 floor area, median abs. % error, hourly input (1,428)<br><sub>1.5 points ahead of the like-for-like rule, about one sampling error of a median over ~1,400 buildings; the same rule reaches 46.4 % on BuildingData's quality-checked subset, where the Detective was not scored, and per-building errors were not kept for a paired test. A tie, not a win. 58 % within ×2.</sub> | **50%** | Electricity ÷ dev median EUI of the TRUE use, same buildings (1,364) (classical): 51.5%<br>Same rule as BuildingData quotes it, on its 1,133 quality-checked meters (a different set) (published): 46.4%<br>Dev median floor area (dumb baseline): 68% | ➖ tie |
| BDG2 floor area, median abs. % error, monthly bills (1,428)<br><sub>2.1 points ahead of the like-for-like rule; tie for the same reasons as the hourly row. 62 % within ×2.</sub> | **49.4%** | Electricity ÷ dev median EUI of the TRUE use, same buildings (1,364) (classical): 51.5%<br>Same rule as BuildingData quotes it, on its 1,133 quality-checked meters (a different set) (published): 46.4%<br>Dev median floor area (dumb baseline): 68% | ➖ tie |
| BDG2 floor area, median abs. % error, annual total (1,428)<br><sub>3.5 points ahead of the like-for-like rule, which is given the true use; still behind the 46.4 % the rule gets on the quality-checked subset. Hourly data read area no better than one annual total. 61 % within ×2.</sub> | **48%** | Electricity ÷ dev median EUI of the TRUE use, same buildings (1,364) (classical): 51.5%<br>Same rule as BuildingData quotes it, on its 1,133 quality-checked meters (a different set) (published): 46.4%<br>Dev median floor area (dumb baseline): 68% | ➖ tie |
| BDG2 primary use, accuracy, hourly input (1,428)<br><sub>Macro-F1 0.10 against 0.06 for the majority class and 0.20 for the centroid. Monthly 14 %, annual 9.5 %. Simulated buildings: 88 %. Use does not transfer to real buildings.</sub> | **11.5%** | Majority class (education) (dumb baseline): 40.4%<br>Weekly-shape nearest centroid (BuildingData, 1,082 buildings) (published): 34.4% | ❌ loses |
| BDG2 year built, mean abs. error, hourly input (741 with a year)<br><sub>Monthly 25.2 yr, annual 24.8 yr: no better than the dev median at any resolution.</sub> | **28.3 yr** | Dev median (1969) (dumb baseline): 24.9 yr | ❌ loses |
| BDG2 has a chilled-water meter, from electricity alone, ROC AUC (1,375)<br><sub>P(district cooling) of the hourly network given only the electricity meter; 506 of 1,375 have a chilled-water meter. No classical baseline was run.</sub> | **0.75** | Chance (dumb baseline): 0.5 | ✅ wins |
| BDG2 has a steam or hot-water meter, from electricity alone, ROC AUC (1,375)<br><sub>P(district heat) from electricity alone; 480 of 1,375 have one. No classical baseline was run.</sub> | **0.7** | Chance (dumb baseline): 0.5 | ✅ wins |
| RECS floor area, median abs. % error, annual energy (7,401 homes)<br><sub>84 % within ×2. Both baselines are given the true housing type.</sub> | **33.6%** | Energy ÷ dev EUI by TRUE housing type × climate (BuildingData) (classical): 33.1%<br>Dev median by TRUE housing type (BuildingData) (dumb baseline): 28.8% | ❌ loses |
| RECS main heating, fuel vs electric vs none, accuracy (7,194 homes) | **87.3%** | Rule: any fuel used → fuel (classical): 85.3%<br>Majority (fuel) (dumb baseline): 64.8% | ✅ wins |
| RECS heat pump vs electric resistance, balanced accuracy (2,226 homes)<br><sub>ROC AUC 0.40: worse than chance. In simulation the same task gives AUC 0.94: the simulator's heat-pump signature is not what real homes show.</sub> | **0.46** | kWh/m²/HDD rule with the TRUE floor area (BuildingData) (classical): 0.5<br>Majority (dumb baseline): 0.5 | ❌ loses |
| RECS vintage, mean abs. error to the survey range midpoint (7,401 homes)<br><sub>22.13 vs 22.32 years: 0.2 years, about one sampling error of a mean over 7,401 homes and under 1 % of the error. No real skill on year built.</sub> | **22.1 yr** | Dev median (1984) (dumb baseline): 22.3 yr | ➖ tie |
| CBECS floor area, median abs. % error, annual energy (2,558 buildings) | **43%** | Energy ÷ dev EUI by TRUE type × climate (BuildingData) (classical): 38.7%<br>Dev median by TRUE type (BuildingData) (dumb baseline): 75.7% | ❌ loses |
| CBECS primary use (10 commercial classes), accuracy (2,558)<br><sub>Macro-F1 0.15 against 0.03 for the majority class.</sub> | **16.5%** | Majority class (office) (dumb baseline): 20.8% | ❌ loses |
| CBECS heating, fuel / electric / district / none, accuracy (2,513) | **74.7%** | Rule: fuel used → fuel, district used → district (classical): 68.6%<br>Majority (fuel) (dumb baseline): 55.6% | ✅ wins |
| CBECS has a chiller, ROC AUC (2,362) | **0.77** | Chance (dumb baseline): 0.5 | ✅ wins |
| CBECS weekly operating hours, mean abs. error (2,558)<br><sub>From one annual total. Spearman 0.50.</sub> | **44.2 h** | Dev median (dumb baseline): 41.4 h<br>Dev median of the TRUE use (classical): 29.7 h | ❌ loses |
| CBECS vintage, mean abs. error (2,558) | **27 yr** | Dev median (1984) (dumb baseline): 26 yr | ❌ loses |
| HUE electric main heat (baseboard or heat pump) vs not, accuracy, hourly electricity (24 homes)<br><sub>ROC AUC 0.46 (6 of 24 homes heat electrically). The monthly network does better on the same homes: 75 % (the majority's score), AUC 0.69.</sub> | **50%** | Majority (not electric) (dumb baseline): 75% | ❌ loses |
| Simulator only: envelope UA, MAE in log10, held-out BuildingSim buildings, hourly (2,000)<br><sub>Simulated, exact labels; not reality. Coverage of the 1-sigma bar 67 %.</sub> | **0.1** | PRISM change-point (fits 95 % of units) (classical): 0.85<br>Median of the TRUE use type (dumb baseline): 0.35 | ℹ️ |
| Simulator only: time constant, MAE in log10, held-out BuildingSim buildings, hourly (2,000)<br><sub>Coverage 72 %.</sub> | **0.14** | Median of the TRUE use type (dumb baseline): 0.21 | ℹ️ |
| Simulator only: heating balance point, MAE, held-out BuildingSim buildings, hourly (2,000) | **2.14 K** | PRISM change-point (fits 95 %) (classical): 6.54 K<br>Median of the TRUE use type (dumb baseline): 3.93 K | ℹ️ |
| Simulator only: cooling balance point, MAE, held-out BuildingSim buildings, hourly (2,000) | **2.24 K** | PRISM change-point (fits 93 %) (classical): 4.89 K<br>Median of the TRUE use type (dumb baseline): 3.87 K | ℹ️ |
| Simulator only: heating energy per kelvin, MAE in log10, held-out BuildingSim buildings, hourly (1,880) | **0.14** | PRISM change-point (fits 95 %) (classical): 0.92<br>Median of the TRUE use type (dumb baseline): 0.45 | ℹ️ |
| Simulator only: weekday operating hours, MAE, held-out BuildingSim buildings, hourly (1,400) | **0.96 h** | Median of the TRUE use type (dumb baseline): 3.25 h | ℹ️ |
| Simulator only: heating system (5 classes), balanced accuracy, held-out BuildingSim buildings, hourly | **0.84** | Rule from which meters exist (classical): 0.35<br>Majority (dumb baseline): 0.2 | ℹ️ |
| Simulator only: heat pump vs resistance, ROC AUC, held-out BuildingSim buildings, hourly (594)<br><sub>On real RECS homes the same question gives AUC 0.40.</sub> | **0.94** | Chance (dumb baseline): 0.5 | ℹ️ |
| Simulator vs simulator: ResStock homes, envelope UA, MAE in log10 (EnergyPlus meters, truth from a BuildingSim twin; 58)<br><sub>Two models, not reality. Biased +0.25 log10; coverage 24 %. Monthly input: 0.18.</sub> | **0.29** | PRISM × nominal efficiency (fits 57) (classical): 0.73<br>Sample median (dumb baseline): 0.23 | ℹ️ |
| Simulator vs simulator: ResStock homes, heating balance point, MAE (58) | **2.8 K** | PRISM (fits 57) (classical): 4.5 K<br>Sample median (dumb baseline): 2.7 K | ℹ️ |
| Simulator vs simulator: ResStock homes, heating system accuracy against the stock label (58)<br><sub>Balanced accuracy 0.51; heat pump vs resistance AUC 0.71 (34 homes).</sub> | **67%** | — | ℹ️ |
| Simulator vs simulator: ComStock buildings, envelope UA, MAE in log10 (90)<br><sub>0.20 on the 85 units PRISM fits. Two models, not reality.</sub> | **0.19** | PRISM × nominal efficiency (fits 85) (classical): 0.76<br>Sample median (dumb baseline): 0.44 | ℹ️ |
| Simulator vs simulator: ComStock buildings, heating balance point, MAE (90)<br><sub>7.0 K on the 85 units PRISM fits.</sub> | **7.2 K** | PRISM (fits 85) (classical): 10.4 K<br>Sample median (dumb baseline): 8.6 K | ℹ️ |
| Simulator vs simulator: ComStock buildings, heating system accuracy against the stock label (90)<br><sub>Cooling system 90 %, primary use 37 %.</sub> | **72%** | — | ℹ️ |

Coverage of the 68% error bars on the exam: BDG2 floor area, hourly (1,428) **72%**, BDG2 floor area, monthly (1,428) **63%**, BDG2 floor area, annual (1,428) **54%**, BDG2 year built, hourly (741) **77%**, RECS floor area (7,401) **69%**, RECS vintage (7,401) **74%**, CBECS floor area (2,558) **61%**, Simulated UA, hourly (2,000) **67%**, Simulated time constant, hourly (2,000) **72%**.

### Where it wins

- Heating system from annual totals: 87.3 % right on 7,194 RECS homes (fuel / electric / none) against 85.3 % for 'any fuel used → fuel', and 74.7 % on 2,513 CBECS buildings against 68.6 % for the meter rule.
- Which meters a campus building has, from its electricity alone: ROC AUC 0.75 for a chilled-water meter and 0.70 for a steam or hot-water meter (1,375 BDG2 buildings); a chiller in CBECS from annual totals: AUC 0.77.
- In simulation it reads the envelope: UA within 0.10 log10 against 0.85 for PRISM, balance points within 2.1-2.2 K against 4.9-6.5 K.

### Where it loses

- Primary use does not transfer: 11 % right on BDG2 against 40 % for always guessing education (simulated: 88 %); 16 % on CBECS against 21 % for the majority class.
- Year built has no real skill: BDG2 28 yr (hourly) against 25 yr for the dev median; RECS 22.1 vs 22.3 yr; CBECS 27.0 vs 26.0 yr.
- Heat pump vs resistance fails on real homes: RECS AUC 0.40, HUE hourly AUC 0.46 (50 % right against 75 % for the majority), against 0.94 in simulation.
- Floor area: only level with an energy ÷ intensity rule on BDG2 (48-50 % vs 51.5 %), and behind the rule that knows the true type on RECS (33.6 vs 33.1 %) and CBECS (43.0 vs 38.7 %).
- Weekly operating hours from one CBECS annual total: 44 h off against 41 h for the dev median.

## Limits

- UA, time constant and balance points have no real truth. Against EnergyPlus ResStock homes the hourly UA error is 0.29 log10 with a +0.25 bias, worse than the sample median (0.23).
- Real BDG2 load shapes still differ from simulated ones: hourly floor area needed error bars 3× wider than the network's own, and hourly input reads area no better than one annual total. Real electricity is less weather-sensitive than the twins (PRISM r² 0.07 vs 0.24).
- Single thermal zone, first-order part load, ideal thermostat within the hour; campus plants are a flat distribution factor plus base load (W2, W3).
- The real dev gate is small (52 BDG2 buildings plus the survey dev respondents) and noisy; it drifted from 0.331 to 0.306 over the kept rungs.
- Use probabilities are flattened on purpose (dev-fitted temperature 10): on real buildings they are close to uninformative.
- When every day of the record lies on one side of the best change-point balance point (a home heated all year, three winter months only), the PRISM fit is degenerate and the release's reading depends on floating-point tie-breaking: numpy and the site's JS port can pick different, equally good fits.
- The balance points do not follow the weather the way physics says: with the meters unchanged, shifting the outdoor temperature by ±4 K should move them by ±4 K, but the hourly network moves Lamb_education_Bert's heating balance point by about ±1 K (11.3 / 12.4 / 13.4 °C) and the HUE home's not at all (17.2 / 17.1 / 16.3 °C); the monthly network follows more (5.8 / 10.1 / 13.0 °C). Four exam units, checked in the live instrument on 17 Nov 2026.
- The training bank, frozen set and features need the shared building platform (BuildingSim + BuildingData), which is not in this repository yet; numpy-vs-torch parity of the shipped weights could not be re-checked here (no checkpoints shipped).

## The ratchet

### BuildingDetective v1.0, hourly network

**Objective:** Skill on the frozen simulated validation set (bank shards 900-904: 1,000 BuildingSim buildings; hourly and annual read two meter-quirk draws per building (2,000), monthly one (1,000)): the mean of 12 regression R² (clipped at 0) and 3 class balanced accuracies rescaled so chance = 0  
**Rule:** Keep a rung if its frozen-set skill beats the incumbent of the same network by more than 0.01 AND the real dev composite (all three networks together, the candidate in place of its incumbent) is no more than 0.02 below the composite of the last kept rung of any network. A physics change rebuilds the frozen set from the same shards and re-scores the incumbents.

| Rung | Change | Frozen-set score | Decision | Why | Dev composite | Gate floor (incumbent − 0.02) | Parameters |
|---|---|---|---|---|---|---|---|
| R1 | MLP 2x256 GELU, hetero NLL + 3 softmax heads; bank v1 shards 0-11 (2,400 buildings), features f1 (W1+W2, quirks) (physics v1) | 0.703 | ● kept | first rung for this mode | 0.1928 | – | 384557 |
| rebaseline-v2 | Physics v2 (W3 always-on base loads from the dev twin test): frozen set rebuilt from the same shards 900-904, incumbent R1-hourly re-scored | 0.663 | ● kept | Re-score, not a new network: 0.7034 → 0.6628. The number to beat from here on. | – | – | – |
| R2 | physics v2: retrain on f2 (W3 always-on base loads in district heat, chilled water, electricity), shards 0-11 | 0.693 | ● kept | val skill 0.6934 vs incumbent 0.6628 (delta +0.0306, need > 0.01); dev gate 0.3197 vs 0.3256 (ok) | 0.3197 | 0.3056 | 384557 |
| R3a | wider/deeper MLP 3x512, dropout 0.25, wd 1e-3 (lr 2e-3: training diverged at epoch ~15) | 0.409 | ○ rejected | val skill 0.4091 vs incumbent 0.6934 (delta -0.2843, need > 0.01); dev gate 0.3081 vs 0.3285 (FAIL) | 0.3081 | 0.3085 | 1162797 |
| R3b | MLP 2x256 with dropout 0.3, wd 1e-3, 80 epochs | 0.684 | ○ rejected | val skill 0.6836 vs incumbent 0.6934 (delta -0.0098, need > 0.01); dev gate 0.3247 vs 0.3285 (ok) | 0.3247 | 0.3085 | 384557 |
| R3c | MLP 3x512, dropout 0.2, wd 1e-3, lr 7e-4 (stable) | 0.705 | ○ rejected | Kept by the rule (val skill 0.7047 vs incumbent 0.6934 (delta +0.0112, need > 0.01); dev gate 0.3236 vs 0.3285 (ok)), then withdrawn by hand: 1.16 M parameters breaks the ~500k parameter cap of the shared-machine rules; R2-hourly reinstated. | 0.3236 | 0.3085 | 1162797 |
| R3d | MLP 3x288 (525k params, within cap), dropout 0.2, wd 1e-3, lr 7e-4 | 0.695 | ○ rejected | val skill 0.6948 vs incumbent 0.6934 (delta +0.0014, need > 0.01); dev gate 0.3198 vs 0.3285 (ok) | 0.3198 | 0.3085 | 525069 |
| hygiene | Label hygiene: base_share set to NaN when \|value\| > 1 (3 training buildings had ~0 metered site energy, base_share −8e12), balance points clipped to −25..30 °C; same frozen set v2, incumbent R2-hourly re-scored | 0.693 | ● kept | Re-score, not a new network: 0.6934 → 0.6934. The number to beat from here on. | – | – | – |
| R4 | label hygiene (clean_labels), MLP 2x256, f2 shards 0-11 | 0.746 | ● kept | val skill 0.7457 vs incumbent 0.6934 (delta +0.0523, need > 0.01); dev gate 0.3310 vs 0.3285 (ok) | 0.331 | 0.3085 | 384557 |
| R5 | more data: f2 shards 0-22 (4,600 buildings), MLP 2x256 | 0.783 | ● kept | val skill 0.7834 vs incumbent 0.7457 (delta +0.0376, need > 0.01); dev gate 0.3264 vs 0.3272 (ok) | 0.3264 | 0.3072 | 384557 |
| R6 | MLP 3x288 (525k params), dropout 0.2, wd 1e-3, lr 7e-4, shards 0-22 | 0.779 | ○ rejected | val skill 0.7790 vs incumbent 0.7834 (delta -0.0043, need > 0.01); dev gate 0.3295 vs 0.3192 (ok) | 0.3295 | 0.2992 | 525069 |
| R7 | more data: f2 shards 0-39 (8,000 buildings), MLP 2x256 | 0.801 | ● kept | val skill 0.8010 vs incumbent 0.7834 (delta +0.0177, need > 0.01); dev gate 0.3288 vs 0.3192 (ok) | 0.3288 | 0.2992 | 384557 |
| R8 | MLP 3x288 (525k params), dropout 0.2, wd 1e-3, lr 7e-4, shards 0-39 | 0.802 | ○ rejected | val skill 0.8025 vs incumbent 0.8010 (delta +0.0015, need > 0.01); dev gate 0.3202 vs 0.3289 (ok) | 0.3202 | 0.3089 | 525069 |
| R9 | more data: f2 shards 0-59 (12,000 buildings, the full bank), MLP 2x256 | 0.813 | **◆ released** | val skill 0.8126 vs incumbent 0.8010 (delta +0.0116, need > 0.01); dev gate 0.3166 vs 0.3289 (ok). Released as v1.0 (weights/hourly.npz); the exam was run once after this. | 0.3166 | 0.3089 | 384557 |

<sub>Columns after *Why*: Real dev composite of the rung, the floor it had to clear, and the network's parameter count (training/ratchet.json cfg).</sub>

Raw log: training/ratchet.json (training/ratchet.py). The dev gate is chained across the three networks: each rung is compared with the gate value of the last kept rung of any network, so it drifted from 0.331 (R4-hourly) to 0.306 (R9-annual) over the kept rungs, 0.025 in all, each step inside the 0.02 tolerance; the gate is small and noisy. R3c-hourly was kept by the rule and then withdrawn by hand for breaking the ~500k parameter cap (1.16 M); R2-hourly was reinstated. Label hygiene (base_share undefined for 3 training buildings with ~0 metered site energy, balance points clipped to −25..30 °C) is logged as a re-score of the incumbents on the same frozen set. The gate's area scores are normalised by BuildingData's quoted dumb-baseline errors (64.3 %, 76 %), which BuildingData computed on the exam split; they are fixed constants and no model's exam result was used to choose anything. 525k-parameter 3×288 nets (R3d, R6, R8) were treated as within the ~500k cap.

### BuildingDetective v1.0, monthly network

**Objective:** Skill on the frozen simulated validation set (bank shards 900-904: 1,000 BuildingSim buildings; hourly and annual read two meter-quirk draws per building (2,000), monthly one (1,000)): the mean of 12 regression R² (clipped at 0) and 3 class balanced accuracies rescaled so chance = 0  
**Rule:** Keep a rung if its frozen-set skill beats the incumbent of the same network by more than 0.01 AND the real dev composite (all three networks together, the candidate in place of its incumbent) is no more than 0.02 below the composite of the last kept rung of any network. A physics change rebuilds the frozen set from the same shards and re-scores the incumbents.

| Rung | Change | Frozen-set score | Decision | Why | Dev composite | Gate floor (incumbent − 0.02) | Parameters |
|---|---|---|---|---|---|---|---|
| R1 | MLP 2x256 GELU, hetero NLL + 3 softmax heads; bank v1 shards 0-11 (2,400 buildings), features f1 (W1+W2, quirks) (physics v1) | 0.573 | ● kept | first rung for this mode | 0.2036 | – | 106285 |
| rebaseline-v2 | Physics v2 (W3 always-on base loads from the dev twin test): frozen set rebuilt from the same shards 900-904, incumbent R1-monthly re-scored | 0.501 | ● kept | Re-score, not a new network: 0.5727 → 0.5009. The number to beat from here on. | – | – | – |
| R2 | physics v2: retrain on f2 (W3 always-on base loads in district heat, chilled water, electricity), shards 0-11 | 0.547 | ● kept | val skill 0.5473 vs incumbent 0.5009 (delta +0.0464, need > 0.01); dev gate 0.3285 vs 0.3197 (ok) | 0.3285 | 0.2997 | 106285 |
| hygiene | Label hygiene: base_share set to NaN when \|value\| > 1 (3 training buildings had ~0 metered site energy, base_share −8e12), balance points clipped to −25..30 °C; same frozen set v2, incumbent R2-monthly re-scored | 0.547 | ● kept | Re-score, not a new network: 0.5473 → 0.5473. The number to beat from here on. | – | – | – |
| R4 | label hygiene (clean_labels), MLP 2x256, f2 shards 0-11 | 0.597 | ● kept | val skill 0.5974 vs incumbent 0.5473 (delta +0.0502, need > 0.01); dev gate 0.3291 vs 0.331 (ok) | 0.3291 | 0.3110 | 106285 |
| R5 | more data: f2 shards 0-22 (4,600 buildings), MLP 2x256 | 0.630 | ● kept | val skill 0.6296 vs incumbent 0.5974 (delta +0.0322, need > 0.01); dev gate 0.3183 vs 0.3264 (ok) | 0.3183 | 0.3064 | 106285 |
| R7 | more data: f2 shards 0-39 (8,000 buildings), MLP 2x256 | 0.641 | ● kept | val skill 0.6410 vs incumbent 0.6296 (delta +0.0114, need > 0.01); dev gate 0.3289 vs 0.3288 (ok) | 0.3289 | 0.3088 | 106285 |
| R9 | more data: f2 shards 0-59 (12,000 buildings, the full bank), MLP 2x256 | 0.656 | **◆ released** | val skill 0.6560 vs incumbent 0.6410 (delta +0.0150, need > 0.01); dev gate 0.3123 vs 0.3166 (ok). Released as v1.0 (weights/monthly.npz); the exam was run once after this. | 0.3123 | 0.2966 | 106285 |

<sub>Columns after *Why*: Real dev composite of the rung, the floor it had to clear, and the network's parameter count (training/ratchet.json cfg).</sub>

Same frozen set, rule and chained dev gate as the hourly ladder (see its note). Raw log: training/ratchet.json.

### BuildingDetective v1.0, annual network

**Objective:** Skill on the frozen simulated validation set (bank shards 900-904: 1,000 BuildingSim buildings; hourly and annual read two meter-quirk draws per building (2,000), monthly one (1,000)): the mean of 12 regression R² (clipped at 0) and 3 class balanced accuracies rescaled so chance = 0  
**Rule:** Keep a rung if its frozen-set skill beats the incumbent of the same network by more than 0.01 AND the real dev composite (all three networks together, the candidate in place of its incumbent) is no more than 0.02 below the composite of the last kept rung of any network. A physics change rebuilds the frozen set from the same shards and re-scores the incumbents.

| Rung | Change | Frozen-set score | Decision | Why | Dev composite | Gate floor (incumbent − 0.02) | Parameters |
|---|---|---|---|---|---|---|---|
| R1 | MLP 2x256 GELU, hetero NLL + 3 softmax heads; bank v1 shards 0-11 (2,400 buildings), features f1 (W1+W2, quirks) (physics v1) | 0.490 | ● kept | first rung for this mode | 0.3256 | – | 80685 |
| rebaseline-v2 | Physics v2 (W3 always-on base loads from the dev twin test): frozen set rebuilt from the same shards 900-904, incumbent R1-annual re-scored | 0.483 | ● kept | Re-score, not a new network: 0.4898 → 0.4827. The number to beat from here on. | – | – | – |
| R2 | physics v2: retrain on f2 (W3 always-on base loads in district heat, chilled water, electricity), shards 0-11 | 0.486 | ○ rejected | val skill 0.4863 vs incumbent 0.4827 (delta +0.0036, need > 0.01); dev gate 0.3306 vs 0.3285 (ok) | 0.3306 | 0.3085 | 80685 |
| hygiene | Label hygiene: base_share set to NaN when \|value\| > 1 (3 training buildings had ~0 metered site energy, base_share −8e12), balance points clipped to −25..30 °C; same frozen set v2, incumbent R1-annual re-scored | 0.483 | ● kept | Re-score, not a new network: 0.4827 → 0.4827. The number to beat from here on. | – | – | – |
| R4 | label hygiene (clean_labels), MLP 2x256, f2 shards 0-11 | 0.517 | ● kept | val skill 0.5172 vs incumbent 0.4827 (delta +0.0345, need > 0.01); dev gate 0.3272 vs 0.3291 (ok) | 0.3272 | 0.3091 | 80685 |
| R5 | more data: f2 shards 0-22 (4,600 buildings), MLP 2x256 | 0.530 | ● kept | val skill 0.5298 vs incumbent 0.5172 (delta +0.0126, need > 0.01); dev gate 0.3192 vs 0.3183 (ok) | 0.3192 | 0.2983 | 80685 |
| R7 | more data: f2 shards 0-39 (8,000 buildings), MLP 2x256 | 0.539 | ○ rejected | val skill 0.5387 vs incumbent 0.5298 (delta +0.0089, need > 0.01); dev gate 0.3250 vs 0.3289 (ok) | 0.325 | 0.3089 | 80685 |
| R9 | more data: f2 shards 0-59 (12,000 buildings, the full bank), MLP 2x256 | 0.541 | **◆ released** | val skill 0.5412 vs incumbent 0.5298 (delta +0.0114, need > 0.01); dev gate 0.3061 vs 0.3123 (ok). Released as v1.0 (weights/annual.npz); the exam was run once after this. | 0.3061 | 0.2923 | 80685 |

<sub>Columns after *Why*: Real dev composite of the rung, the floor it had to clear, and the network's parameter count (training/ratchet.json cfg).</sub>

Same frozen set, rule and chained dev gate as the hourly ladder (see its note). Raw log: training/ratchet.json.

Raw per-rung logs are in [`training/`](training/).

## Physics

BuildingSim, the collection's building simulator (shared building platform, Phase 1): an ISO 13790 5R1C single-zone heat balance with Perez solar gains, LBL infiltration, ventilation and moisture, and 13 HVAC systems built from capacity, COP and EIR curves with part load, defrost, fans, pumps and district plants. Labels come from the simulator itself: UA is the 5R1C network conductance (labels()['UA_network_W_K']), tau, balance points, end-use shares, schedules and system classes. Three wrappers calibrated on dev only (training/wrappers.py, upstream candidates for BuildingSim) set use-type intensities (W1), campus district service (W2) and always-on base loads found by the dev twin test (W3, physics v2).

**Simulator**

- Bank: 12,000 generated buildings (30 % homes, 70 % commercial in 10 types) drawn from ResStock, ComStock and CBECS priors, on real 2016-17 weather of 18 BDG2 sites (Peacock left out: its weather does not match its location) and TMYx cities, at 1 h; frozen validation set = bank shards 900-904 (1,000 buildings)
- W1 intensity: one factor per use type so generated site-energy medians match the RECS/CBECS prior and dev medians (e.g. lab 0.42, education 0.55, lodging 1.55)
- W2 campus: district meters scaled by a distribution factor fitted on the 52 BDG2 dev twins (heat 0.18, chilled water 0.33; meter = delivered / factor); 30 % of commercial buildings on campus service
- W3 always-on base loads (physics v2): a share of each district meter's annual energy moved into a flat base (heat 0.70, chilled water 0.50) plus a flat unmodelled electricity load
- Meter emulation on every simulated meter: partial records, gaps, zero runs, spikes, level shifts, clock offsets, daily totals spread over 24 h, shared unmodelled loads, sub-meter scale errors and noise; bills get shifted billing windows (0-15 days) and missing months; annual surveys get unreported fuels

**Checked against independent references**

- BuildingSim Phase 1 (shared building platform checks): steady conduction to 1.8e-9, RC step within 0.0035 K, degree-day ratio 0.999996, ISO 13786 wall amplitude −1.6 % and lag −0.01 h, every energy ledger closes to 1e-8
- BESTEST (ASHRAE 140): 29 of 34 values inside the published ranges, all 20 annual heating and cooling values in (ranges transcribed from the literature)
- ResStock twins (126 homes): site energy median 1.10× (IQR 0.98-1.26), daily r 0.92; ComStock twins (45): 0.86× (aggregate 0.97), daily r 0.89
- Dev twin test (52 BDG2 dev buildings vs simulated twins from their metadata, medians): district-heat heating slope real 0.81 vs twin 4.38 before W3 and 1.00 after; chilled-water night/day real 0.69 vs 0.31 before and 0.63 after; electricity night/day real 0.77 vs 0.66 before and 0.74 after (training/twintest.json)
- The release's JS feature port (training/detective.js) matched Python to 2e-7 on real and simulated buildings; the site's live instrument matches the Python release to 1e-6 on 60 control settings of the 12 example units

## Help wanted

- Real thermostat logs (ecobee Donate Your Data, by application) would give a real exam for time constants and setpoint behaviour; none exists today.
- UK heat-pump field trials (RHPP, Electrification of Heat) need UK Data Service registration; they would test the heat-pump signature that fails on RECS and HUE.
- An EnergyPlus binary for a true reference engine (no PyPI wheel with an engine; GitHub was blocked in the build sandbox) and the official ASHRAE 140 tables for BESTEST.
- Any public building set with measured envelope UA or a co-heating test next to its meters: UA and the time constant have no real truth anywhere in the exam.

## Next rungs

- Physics v3 from the remaining twin gaps: real electricity is much less weather-sensitive than the twins (PRISM r² 0.07 vs 0.24); add a calibrated mix of meters serving unconditioned space or process load.
- Heat-pump realism: backup strips, defrost, sizing and ResStock-like controls fitted on RECS and HUE dev, or WPuQ heat-pump homes as a dev set (done when RECS dev heat pump vs resistance AUC > 0.7).
- Domain-adversarial or twin-consistency training on unlabelled real BDG2 2016 meters, plus a dev-fitted population prior for the use classes.
- Upstream to BuildingSim behind flags: W1 intensity factors, W2 campus factors, W3 base-load shares, and common.clean_labels (base_share undefined when metered site energy is ~0).

## Files

| Path | What |
|---|---|
| `buildingdetective.py`, `features.py`, `common.py`, `pipeline.py` | Inference — numpy/scipy only |
| `weights/hourly.npz`, `weights/monthly.npz`, `weights/annual.npz`, `weights/calibration.json` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Building Data Genome 2 (BDG2): hourly meters and metadata of 1,636 non-residential buildings, 2016-17 (Miller et al., Scientific Data 7, 368, 2020), cleaned by BuildingData](https://github.com/buds-lab/building-data-genome-project-2) | physics (site weather for the bank), calibration (W2 campus factors and W3 base loads from the 52 dev twins), dev gate, error bars (area, year), exam (1,428 buildings with electricity, 2017); examples/bdg2/ (3 exam buildings) | CC BY-SA 4.0 (share-alike): examples/bdg2/, lab/index.html and the instrument data carry the notice |
| [EIA Residential Energy Consumption Survey (RECS) 2020 microdata](https://www.eia.gov/consumption/residential/data/2020/) | physics (W1 intensity prior), dev gate and error bars (1,848 respondents), exam (7,401); examples/examples.json (4 respondents) | Public domain (U.S. Government) |
| [EIA Commercial Buildings Energy Consumption Survey (CBECS) 2018 microdata](https://www.eia.gov/consumption/commercial/data/2018/) | physics (generator prior, W1), dev gate and error bars (661 respondents), exam (2,558); examples/examples.json (4 respondents) | Public domain (U.S. Government) |
| [HUE: The Hourly Usage of Energy dataset for buildings in British Columbia (S. Makonin, Data in Brief 23, 2019)](https://doi.org/10.7910/DVN/N3HGRN) | exam only (24 homes, electric heat or not); examples/examples.* (1 home); 5 homes in lab/index.html | CC BY 4.0 |
| [NREL ResStock 2025 R1 and ComStock 2025 R3 (End-Use Load Profiles, OEDI). "Data includes information from the ResStock and ComStock datasets developed by the National Renewable Energy Laboratory (NREL; now National Laboratory of the Rockies) with funding from the U.S. DOE."](https://data.openei.org/submissions/4520) | physics (BuildingSim generator priors); the simulator-vs-simulator check (58 ResStock homes, 90 ComStock buildings, EnergyPlus meters), report only, never for choosing | CC BY 4.0 |
| [TMYx typical-year weather (climate.onebuilding.org, 2011-2025)](https://climate.onebuilding.org) | physics (weather for part of the bank) | Cite Lawrie and Crawley, TMYx; the site says All Rights Reserved and states no reuse licence: used for simulation only, not redistributed |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
