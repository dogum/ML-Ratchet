# BuildingDetective v1.0: model card

Exam re-scored on 17 Nov 2026 with fair dev-fitted baselines and 95 % intervals: 0 wins, 21 losses, 7 ties on the real rows (section below). The exam is seen.

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

## Exam re-scored with fair baselines (17 Nov 2026; the exam is seen)
v1.0 compared the Detective mostly with rules given the true use type, or with chance, and called three floor-area
margins ties by eye. On 17 Nov 2026 the real rows were re-scored with the released weights (`training/integrity.py`,
log `training/integrity.json`; features rebuilt by `training/integrity_feats.py`). Nothing was chosen from the result; no rung was run.

- **Reproduction.** RECS, CBECS, HUE, BDG2 annual and every v1.0 baseline reproduce `training/exam.json` exactly.
  BDG2 hourly: 24 of 1,428 buildings read differently, median error unchanged (50.0 %). BDG2 monthly: 157 buildings
  differ, area error 49.6 % against 49.4 %. Cause: the PRISM grid fit has exact ties (12 monthly points, or a
  degenerate daily fit) that floating-point noise breaks; a 1e-12 relative perturbation of the meters moves the same
  features by up to 0.75 (monthly) and 14.6 (hourly).
- **Fair baselines, fitted on dev only** (52 BDG2 buildings, 1,848 RECS and 661 CBECS respondents; W1 targets on prior
  + dev as the simulator was tuned), scored on the same exam rows: a dev-fitted line of log area on log electricity;
  ridge-regularised linear and logistic models on the 12 annual features the network gets (L2 = 1 on standardised
  inputs, fixed in advance); a dev site prior for which meters exist; the nearest dev centroid of the 168-hour
  electricity week profile; a PRISM change-point reading of HUE electricity. (TRUE) marks baselines given the true
  use or type, which the Detective is not.
- **Interval rule.** 95 % bootstrap interval on Detective minus each baseline, 2,000 resamples of buildings, homes or
  respondents, seed 20261117, on the rows both can score. The verdict is against the best baseline: win if the
  interval excludes zero on the good side, loss on the bad side, tie if it spans zero.

**Result: 0 wins, 21 losses, 7 ties on 28 real rows** (v1.0: 5 wins, 4 ties, 9 losses on 18). The Detective still
beats the simple meter rules for the heating system and the TRUE-use floor-area rule on monthly and annual BDG2
input, but the best dev-fitted baseline beats it on 21 of 28 rows. The 7 ties are BDG2 year built (monthly and
annual input, against the dev median), CBECS heat-pump presence and the four rows on the 24 HUE homes.

RECS rows use annual input; "hourly / monthly / annual input" on BDG2 means the same 2017 meters at that resolution.

| row | n | Detective | baselines: value; Detective minus baseline [95 % interval] | verdict (v1.0) |
|---|---|---|---|---|
| BDG2 floor area, hourly input | 1,428 | 50.0% | Electricity ÷ dev median EUI of the TRUE use (TRUE): 51.5%; −1.8 [−6.0, +2.0] (n=1,364)<br>Electricity ÷ dev median EUI of all uses: 48.1%; +1.5 [−2.3, +4.7] (n=1,364)<br>Dev-fitted line log10 A = 0.73 log10 kWh − 0.49: 40.9%; +8.8 [+5.9, +11.3] (n=1,364)<br>Dev median floor area: 68.0%; −17.9 [−21.9, −12.7] | **loss** (tie) |
| BDG2 floor area, monthly input | 1,428 | 49.6% | Electricity ÷ dev median EUI of the TRUE use (TRUE): 51.5%; −4.4 [−8.5, −0.4] (n=1,364)<br>Electricity ÷ dev median EUI of all uses: 48.1%; −1.0 [−4.7, +2.0] (n=1,364)<br>Dev-fitted line log10 A = 0.73 log10 kWh − 0.49: 40.9%; +6.2 [+3.2, +9.5] (n=1,364)<br>Dev median floor area: 68.0%; −18.3 [−23.5, −12.8] | **loss** (tie) |
| BDG2 floor area, annual input | 1,428 | 48.0% | Electricity ÷ dev median EUI of the TRUE use (TRUE): 51.5%; −5.9 [−9.3, −2.5] (n=1,364)<br>Electricity ÷ dev median EUI of all uses: 48.1%; −2.5 [−5.2, −0.2] (n=1,364)<br>Dev-fitted line log10 A = 0.73 log10 kWh − 0.49: 40.9%; +4.8 [+2.2, +7.3] (n=1,364)<br>Dev median floor area: 68.0%; −20.0 [−24.4, −15.1] | **loss** (tie) |
| BDG2 floor area, electricity only input | 1,375 | 51.0% | Electricity ÷ dev median EUI of the TRUE use (TRUE): 51.5%; −0.8 [−4.7, +3.1] (n=1,364)<br>Electricity ÷ dev median EUI of all uses: 48.1%; +2.6 [−1.0, +5.6] (n=1,364)<br>Dev-fitted line log10 A = 0.73 log10 kWh − 0.49: 40.9%; +9.8 [+7.3, +12.3] (n=1,364)<br>Dev median floor area: 67.6%; −16.7 [−20.8, −11.4] | **loss** (new) |
| BDG2 primary use, hourly input | 1,428 | 11.5% | Majority class (education): 40.4%; −28.9 [−32.2, −25.6]<br>Nearest dev centroid of the 168-h electricity week profile: 30.8%; −19.4 [−22.2, −16.7] (n=1,375) | **loss** (loss) |
| BDG2 primary use, monthly input | 1,428 | 14.4% | Majority class (education): 40.4%; −26.0 [−29.2, −22.8]<br>Nearest dev centroid of the 168-h electricity week profile: 30.8%; −16.1 [−19.1, −13.1] (n=1,375) | **loss** (new) |
| BDG2 primary use, annual input | 1,428 | 9.5% | Majority class (education): 40.4%; −31.0 [−34.2, −27.7]<br>Nearest dev centroid of the 168-h electricity week profile: 30.8%; −21.3 [−24.1, −18.4] (n=1,375) | **loss** (new) |
| BDG2 year built, hourly input | 741 | 28.3 yr | Dev median (1969): 24.9 yr; +3.4 [+2.2, +4.6]<br>Dev median of the same site: 25.5 yr; +2.8 [+1.3, +4.5] | **loss** (loss) |
| BDG2 year built, monthly input | 741 | 25.3 yr | Dev median (1969): 24.9 yr; +0.3 [−0.4, +1.1]<br>Dev median of the same site: 25.5 yr; −0.2 [−1.4, +1.0] | **tie** (new) |
| BDG2 year built, annual input | 741 | 24.8 yr | Dev median (1969): 24.9 yr; −0.1 [−0.8, +0.6]<br>Dev median of the same site: 25.5 yr; −0.6 [−1.9, +0.6] | **tie** (new) |
| BDG2 has a chilled-water meter, from electricity alone (ROC AUC) | 1,375 | 0.75 | Dev site prior (share of dev buildings at the same site with one): 0.83; −0.077 [−0.112, −0.043]<br>Dev-fitted logistic model on the electricity PRISM and load-shape statistics: 0.75; +0.003 [−0.021, +0.028]<br>Chance: 0.50; +0.251 [+0.222, +0.278] | **loss** (win) |
| BDG2 has a steam or hot-water meter, from electricity alone (ROC AUC) | 1,375 | 0.70 | Dev site prior (share of dev buildings at the same site with one): 0.83; −0.130 [−0.165, −0.094]<br>Dev-fitted logistic model on the electricity PRISM and load-shape statistics: 0.75; −0.052 [−0.083, −0.021]<br>Chance: 0.50; +0.197 [+0.168, +0.228] | **loss** (win) |
| RECS floor area, annual energy | 7,401 | 33.6% | Energy ÷ dev median EUI by TRUE type × climate (BuildingData rule, recomputed) (TRUE): 33.1%; +0.5 [−0.4, +1.5]<br>Energy ÷ W1 target EUI of the TRUE use (prior + dev weighted median) (TRUE): 34.6%; −1.0 [−2.0, +0.1]<br>Dev median by TRUE type (TRUE): 28.8%; +4.9 [+4.2, +5.7]<br>Dev-fitted linear model of log area on the 12 annual features: 29.6%; +4.0 [+3.3, +4.7]<br>Energy ÷ dev median EUI of all respondents: 34.4%; −0.8 [−1.8, +0.3] | **loss** (loss) |
| RECS vintage, annual energy | 7,401 | 22.1 yr | Dev median (1984): 22.3 yr; −0.2 [−0.4, +0.0]<br>Dev-fitted linear model on the 12 annual features: 21.8 yr; +0.3 [+0.1, +0.6] | **loss** (tie) |
| RECS main heating, fuel / electric / none | 7,194 | 87.3% | Rule: fuel used → fuel, district used → district, else electric: 85.3%; +2.0 [+1.2, +2.8]<br>Dev-fitted multinomial logistic model on the 12 annual features: 91.9%; −4.7 [−5.2, −4.1]<br>Majority (fuel): 64.8%; +22.5 [+21.5, +23.5] | **loss** (win) |
| RECS heat pump vs electric resistance, balanced accuracy | 2,226 | 0.46 | kWh/m²/HDD < dev threshold, TRUE floor area (BuildingData rule, recomputed) (TRUE): 0.50; −0.042 [−0.064, −0.023]<br>kWh/HDD < dev threshold (no floor area): 0.49; −0.032 [−0.053, −0.012]<br>Dev-fitted logistic model on the 12 annual features: 0.58; −0.128 [−0.159, −0.100]<br>Majority: 0.50; −0.044 [−0.062, −0.027] | **loss** (loss) |
| RECS heat pump vs electric resistance, ROC AUC | 2,226 | 0.40 | kWh/m²/HDD, TRUE floor area (low → heat pump) (TRUE): 0.45; −0.050 [−0.076, −0.025]<br>kWh/HDD (low → heat pump): 0.38; +0.020 [+0.002, +0.037]<br>Dev-fitted logistic model on the 12 annual features: 0.63; −0.229 [−0.268, −0.192]<br>Chance: 0.50; −0.096 [−0.120, −0.074] | **loss** (new) |
| CBECS floor area, annual energy | 2,558 | 43.0% | Energy ÷ dev median EUI by TRUE type × climate (BuildingData rule, recomputed) (TRUE): 38.7%; +4.2 [+1.9, +6.3] (n=2,525)<br>Energy ÷ W1 target EUI of the TRUE use (prior + dev weighted median) (TRUE): 51.3%; −8.5 [−11.5, −5.6] (n=2,525)<br>Dev median by TRUE type (TRUE): 76.5%; −33.5 [−36.5, −30.5]<br>Dev-fitted linear model of log area on the 12 annual features: 44.9%; −1.8 [−3.9, +0.1]<br>Energy ÷ dev median EUI of all respondents: 48.7%; −5.9 [−8.2, −3.9] (n=2,525) | **loss** (loss) |
| CBECS vintage, annual energy | 2,558 | 27.0 yr | Dev median (1984): 26.0 yr; +1.0 [+0.6, +1.5]<br>Dev-fitted linear model on the 12 annual features: 27.0 yr; +0.0 [−0.4, +0.4] | **loss** (loss) |
| CBECS main heating, fuel / electric / district / none | 2,513 | 74.7% | Rule: fuel used → fuel, district used → district, else electric: 68.6%; +6.1 [+4.6, +7.6]<br>Dev-fitted multinomial logistic model on the 12 annual features: 79.2%; −4.5 [−5.9, −3.2]<br>Majority (fuel): 55.6%; +19.0 [+17.0, +21.1] | **loss** (win) |
| CBECS primary use (10 commercial classes), accuracy | 2,558 | 16.5% | Majority class (office): 20.8%; −4.3 [−6.5, −2.1]<br>Dev-fitted multinomial logistic model on the 12 annual features: 28.4%; −11.9 [−13.9, −9.8] | **loss** (loss) |
| CBECS has a chiller, ROC AUC | 2,362 | 0.77 | Dev-fitted logistic model on the 12 annual features: 0.80; −0.036 [−0.052, −0.020]<br>Chance: 0.50; +0.268 [+0.247, +0.288] | **loss** (win) |
| CBECS has a heat pump, ROC AUC | 2,375 | 0.59 | Dev-fitted logistic model on the 12 annual features: 0.62; −0.031 [−0.069, +0.007]<br>Chance: 0.50; +0.087 [+0.058, +0.114] | **tie** (new) |
| CBECS weekly operating hours, mean abs. error | 2,558 | 44.2 h | Dev median: 41.4 h; +2.8 [+0.8, +5.0]<br>Dev median of the TRUE use (TRUE): 29.7 h; +14.5 [+12.8, +16.4]<br>Dev-fitted linear model on the 12 annual features: 37.4 h; +6.8 [+5.7, +8.0] | **loss** (loss) |
| HUE electric main heat, hourly electricity, accuracy | 24 | 50.0% | Majority (not electric): 75.0%; −25.0 [−54.2, +8.3]<br>PRISM heating share of electricity ≥ 0.20 (threshold fixed before HUE was read): 45.8%; +4.2 [−20.8, +29.2] | **tie** (loss) |
| HUE electric main heat, hourly electricity, ROC AUC | 24 | 0.46 | PRISM heating share of electricity (change-point fit of daily kWh): 0.55; −0.083 [−0.295, +0.137]<br>Chance: 0.50; −0.037 [−0.359, +0.310] | **tie** (new) |
| HUE electric main heat, monthly electricity, accuracy | 24 | 75.0% | Majority (not electric): 75.0%; +0.0 [−20.8, +25.0]<br>PRISM heating share of electricity ≥ 0.20 (threshold fixed before HUE was read): 45.8%; +29.2 [+12.5, +45.8] | **tie** (new) |
| HUE electric main heat, monthly electricity, ROC AUC | 24 | 0.69 | PRISM heating share of electricity (change-point fit of daily kWh): 0.55; +0.148 [−0.042, +0.361]<br>Chance: 0.50; +0.194 [−0.102, +0.432] | **tie** (new) |

### Dev twin test of what the network adds (dev only)
The network's correction to a dev-calibrated rule (network − rule) against what reality needed (truth − rule), on
the dev units. Positive correlation means the corrections point the right way.

| dev set, output, rule | n | correlation (Pearson · Spearman) | rule error | network error |
|---|---|---|---|---|
| BDG2, floor area (log10), electricity ÷ dev EUI of all uses: hourly / monthly / annual | 52 | 0.12 · 0.18 / 0.10 · 0.12 / 0.22 · 0.31 | 0.27 | 0.32 / 0.29 / 0.26 |
| RECS, floor area (log10), energy ÷ dev EUI of all homes | 1,848 | 0.60 · 0.56 | 0.19 | 0.17 |
| RECS, floor area (log10), energy ÷ dev EUI by TRUE type × climate | 1,848 | 0.50 · 0.44 | 0.17 | 0.17 |
| RECS, vintage (yr), dev median | 1,848 | 0.22 · 0.18 | 22.2 | 21.7 |
| RECS, P(fuel heat), meter rule | 1,795 | 0.60 · 0.53 | 0.13 | 0.13 |
| CBECS, floor area (log10), energy ÷ dev EUI of all buildings | 651 | 0.54 · 0.43 | 0.29 | 0.26 |
| CBECS, floor area (log10), energy ÷ dev EUI by TRUE type × climate | 651 | 0.29 · 0.24 | 0.19 | 0.26 |
| CBECS, vintage (yr), dev median | 661 | 0.17 · 0.17 | 23.4 | 24.6 |
| CBECS, P(fuel heat), meter rule | 653 | 0.56 · 0.55 | 0.27 | 0.28 |
| CBECS, weekly hours, dev median of the TRUE use | 661 | 0.43 · 0.39 | 27.0 h | 44.6 h |

The corrections point the right way but are biased and too large. Where the residuals have structure:
- CBECS hours: the network adds a median +37 h/week, +53 to +56 h for offices and schools (it reads them as open far
  longer than they are) and −42 h for healthcare and −32 h for lodging (it misses that they run around the clock).
- BDG2 area, hourly: education needed +0.15 log10 (larger than the rule says) and got −0.16; offices needed −0.02
  and got −0.25.
- RECS area: apartments in 5+ unit buildings needed +0.09 and got +0.28; cold climates (5A-7A) needed −0.05 to
  −0.07 and got −0.13 to −0.18. The network over-reads climate and housing type.
- CBECS area: food sales and food service needed −0.55 and −0.66 (they use far more energy per m² than the median)
  and got −0.10 and −0.17; non-refrigerated warehouses needed +0.42 and got 0.0.

A dev-fitted line or logistic model on the same inputs gets the size of these corrections right, which is why it
wins the exam rows.

### Readiness for later rungs (smoke test on this machine, Windows, 17 Nov 2026)
One 20-building shard for training (shard 0) and one for the frozen set (shard 900), each through `simbank.py`
(2 processes), then `featurize.py ... --w3 --once`, then `train.py --epochs 1` for each network (torch CPU, one
thread). Everything ran; no simulation failed.
- Simulation: 24-25 s per 20 buildings per process (about 1.25 s per building, one year at 1 h).
- Features: 14 s for the first 20 buildings (weather cache cold), 3 s for the next 20.
- Training: one epoch at full-bank size (24,000 hourly rows, 1,199 inputs, 2 × 256) takes 2.0 s; monthly 0.7 s,
  annual 1.2 s. 60 epochs of all three networks: about 4 min. Torch start-up adds about 20 s per run.
- Extrapolated: the 60-shard bank plus the 5-shard frozen set (13,000 buildings) is about 4.5 CPU-hours of
  simulation, 1.5 h on 3 processes; featurizing about 0.2-1 h more on 3 processes.
- Blockers: (1) the ResStock/ComStock samples (`data/proc/stock`, about 1.2 GB) are not unpacked, so `stockcheck.py`
  and the sim-vs-sim rows cannot run; (2) `run_bank.sh` calls `python3` and bash, so on Windows the shards are started
  by hand; (3) the frozen set has to be rebuilt from shards 900-904 and checked against v1.0 (re-score the released
  weights: hourly skill should be 0.813) before any rung is judged on it; (4) training-shard features use time-based
  seeds (`featurize.py`), so a rebuilt training set will not match v1.0's bit for bit; (5) `calibration.json` must be
  copied to `work/models/` first (the wrappers read it there).

## Final exam as released (v1.0, run once after all choices were frozen; real-row verdicts superseded above)
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

### Every simulator output in training/exam.json (not re-run on 17 Nov 2026)
Held-out BuildingSim buildings (frozen set, MAE; cov = share inside the 1-sigma bar) and the simulated twins of the 52 BDG2 dev buildings:

| output | hourly | monthly | annual | PRISM (h / m) | median of TRUE use | dev twins (h / m / a) |
|---|---|---|---|---|---|---|
| floor area, MAE in log10 (n = 2,000) | 0.12 (cov 99 %) | 0.16 | 0.18 | — | 0.44 | 0.11 / 0.15 / 0.16 |
| year built, MAE (n = 2,000) | 19.2 (cov 95 %) | 20.4 | 20.6 | — | 22.0 | 21.7 / 22.4 / 23.4 |
| envelope UA, MAE in log10 (n = 2,000) | 0.10 (cov 67 %) | 0.13 | 0.17 | 0.85 / 0.86 | 0.35 | 0.11 / 0.14 / 0.17 |
| UA per floor area, MAE in log10 (n = 2,000) | 0.11 (cov 73 %) | 0.13 | 0.13 | — | 0.17 | 0.11 / 0.13 / 0.13 |
| time constant, MAE in log10 (n = 2,000) | 0.14 (cov 72 %) | 0.17 | 0.18 | — | 0.21 | 0.14 / 0.17 / 0.18 |
| heating balance point, MAE (n = 2,000) | 2.14 (cov 69 %) | 2.75 | 3.55 | 6.54 / 7.05 | 3.93 | 2.78 / 3.38 / 4.65 |
| cooling balance point, MAE (n = 2,000) | 2.24 (cov 72 %) | 2.95 | 3.82 | 4.89 / 5.29 | 3.87 | 2.99 / 3.45 / 4.62 |
| heating energy per kelvin, MAE in log10 (n = 1,880) | 0.14 (cov 69 %) | 0.16 | 0.21 | 0.92 / 0.93 | 0.45 | 0.19 / 0.17 / 0.21 |
| cooling energy per kelvin, MAE in log10 (n = 1,814) | 0.12 (cov 72 %) | 0.13 | 0.22 | 0.23 / 0.28 | 0.47 | 0.16 / 0.15 / 0.31 |
| non-HVAC share of site energy, MAE (n = 2,000) | 0.07 (cov 67 %) | 0.08 | 0.11 | — | 0.14 | 0.07 / 0.06 / 0.09 |
| weekday operating hours, MAE (n = 1,400) | 0.96 (cov 75 %) | 3.73 | 4.04 | — | 3.25 | 1.08 / 4.33 / 4.80 |
| weekend/weekday ratio, MAE (n = 1,400) | 0.05 (cov 78 %) | 0.24 | 0.27 | — | 0.21 | 0.06 / 0.24 / 0.27 |
| use class, balanced accuracy | 0.87 | 0.58 | 0.40 | — | majority 0.08 | 0.79 / 0.60 / 0.27 |
| heat class, balanced accuracy | 0.84 | 0.76 | 0.65 | — (meter rule 0.35) | majority 0.20 | 0.74 / 0.72 / 0.64 |
| cool class, balanced accuracy | 0.90 | 0.82 | 0.56 | — | majority 0.25 | 0.90 / 0.91 / 0.52 |
| heat pump vs resistance, AUC | 0.94 | 0.83 | 0.65 | — | chance 0.50 | 0.77 / 0.68 / 0.79 |

| stock, output | hourly | monthly | annual | PRISM (h / m) | sample median |
|---|---|---|---|---|---|
| resstock floor area, MAE in log10 (n = 58) | 0.29 (bias +0.26) | 0.17 | 0.20 | — | 0.16 |
| resstock year built, MAE (n = 58) | 22.0 (bias +10.37) | 19.6 | 18.9 | — | 19.1 |
| resstock envelope UA, MAE in log10 (n = 58) | 0.29 (bias +0.25) | 0.18 | 0.20 | 0.73 (n=57) / 0.79 (n=57) | 0.23 |
| resstock UA per floor area, MAE in log10 (n = 58) | 0.18 (bias +0.00) | 0.17 | 0.15 | — | 0.17 |
| resstock time constant, MAE in log10 (n = 58) | 0.21 (bias +0.08) | 0.20 | 0.17 | — | 0.19 |
| resstock heating balance point, MAE (n = 58) | 2.79 (bias -0.12) | 2.41 | 2.54 | 4.45 (n=57) / 4.37 (n=57) | 2.70 |
| resstock cooling balance point, MAE (n = 58) | 2.54 (bias +0.59) | 2.30 | 2.80 | 3.29 (n=52) / 3.38 (n=52) | 2.59 |
| resstock heat class, accuracy · balanced accuracy | 67 % · 0.51 | 67 % · 0.65 | 64 % · 0.50 | — | — |
| resstock heat pump vs resistance, AUC (n = 34) | 0.71 | 0.70 | 0.37 | — | — |
| comstock floor area, MAE in log10 (n = 90) | 0.25 (bias +0.17) | 0.26 | 0.31 | — | 0.57 |
| comstock year built, MAE (n = 90) | 26.7 (bias +6.25) | 24.3 | 22.9 | — | 23.1 |
| comstock envelope UA, MAE in log10 (n = 90) | 0.19 (bias +0.11) | 0.20 | 0.24 | 0.76 (n=85) / 0.76 (n=78) | 0.44 |
| comstock UA per floor area, MAE in log10 (n = 90) | 0.15 (bias -0.05) | 0.15 | 0.13 | — | 0.18 |
| comstock time constant, MAE in log10 (n = 90) | 0.19 (bias +0.04) | 0.19 | 0.17 | — | 0.22 |
| comstock heating balance point, MAE (n = 90) | 7.21 (bias +2.06) | 6.97 | 7.58 | 10.36 (n=85) / 10.34 (n=78) | 8.58 |
| comstock cooling balance point, MAE (n = 90) | 7.35 (bias +1.44) | 7.39 | 7.71 | 8.93 (n=88) / 8.92 (n=87) | 9.18 |
| comstock heat class, accuracy · balanced accuracy | 72 % · 0.58 | 74 % · 0.71 | 71 % · 0.67 | — | — |
| comstock cool class, accuracy · balanced accuracy | 90 % · 0.44 | 84 % · 0.78 | 92 % · 0.74 | — | — |
| comstock use class, accuracy · macro-F1 | 37 % · 0.34 | 40 % · 0.43 | 26 % · 0.25 | — | — |
| comstock heat pump vs resistance, AUC (n = 38) | 0.63 | 0.56 | 0.71 | — | — |

Error-bar coverage (share of truths inside ±1σ, 68% is ideal), after one widening factor per output fitted on dev only: BDG2 floor area (hourly) 72%; BDG2 floor area (annual) 54%; RECS area 69%, RECS vintage 74%; simulated UA 67%, tau 72%.

## Error-bar widening factors (dev)
| mode | logA | year | logUA | logUAm2 | logtau | Tb_heat | Tb_cool | logS_heat | logS_cool | base_share | op_hours | weekend | class temperature (use, heat, cool) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hourly | 3.02 | 2.14 | 1.04 | 1.30 | 1.26 | 1.13 | 1.24 | 1.07 | 1.11 | 1.01 | 1.04 | 1.00 | 10.00, 1.00, 1.00 |
| monthly | 1.54 | 1.14 | 1.00 | 1.16 | 1.15 | 1.02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.11 | 1.00 | 10.00, 1.00, 1.00 |
| annual | 1.08 | 1.04 | 1.00 | 1.09 | 1.13 | 1.02 | 1.00 | 1.00 | 1.04 | 1.00 | 1.08 | 1.00 | 7.40, 1.90, 1.00 |


## Where it fails
- It wins no real exam row against classical baselines fitted on the same dev data (re-scored 17 Nov 2026): a
  ridge-regularised logistic or linear model on the 12 annual numbers beats it on RECS and CBECS heating, CBECS use,
  chillers, RECS heat pump vs resistance, RECS area and CBECS hours; a two-parameter dev line beats it on BDG2 floor
  area (40.9 % against 45.6-49.7 %); a dev site prior beats it on which meters a BDG2 building has (AUC 0.83 against
  0.70-0.75).
- The dev twin test shows its corrections to the dev-calibrated rules point the right way (correlation 0.4-0.6 on
  the survey dev sets, 0.1-0.3 on BDG2) but are biased and too large.
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
  - `training/exam.json`, `training/integrity.json` (the 17 Nov 2026 re-score), `training/twintest.json`,
    `training/ratchet.json` and `training/calibration.json` hold
    numbers computed on BDG2 buildings (scores, twin medians, W2 campus factors), and the weights' error-bar factors
    were fitted partly on the 52 BDG2 dev buildings. These are summary statistics, not meter data; they are credited here.
  - `weights/` (`annual.npz`, `monthly.npz`, `hourly.npz`, `calibration.json`): trained on simulations calibrated on
    BDG2 dev buildings, so treated as an adaptation and licensed **CC BY-SA 4.0** (notice: `weights/LICENSE-DATA.md`;
    decided 17 Nov 2026). The code is MIT.
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
