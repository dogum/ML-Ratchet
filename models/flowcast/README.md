<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# FlowCast <sub>v1.0</sub>

> **Daily streamflow for rivers without a gauge, from weather and basin attributes**

> [!WARNING]
> Every score here uses ERA5-Land forcing (through Caravan), which under-catches mountain and Pacific Northwest precipitation and drifts about 6 % between decades; that caps every method on the page, and published Daymet/NLDAS-driven LSTM scores (NSE about 0.69-0.76) are not comparable. Research model: not for flood warning.

*How much water flows in this river each day, if nobody has ever measured it?*

FlowCast predicts daily streamflow (mm/day and m³/s) with 5-95 % quantiles for a basin that has no gauge, from basin-average ERA5-Land weather and 11 static attributes. It is a 64-unit LSTM (24,710 weights) trained only on 12,000 WaterSim simulations of random basins driven by real ERA5-Land forcing from non-exam years; it never reads streamflow from the basin it predicts. Graded once on 619 real CAMELS-US exam rivers (WY1990-99) and 603 of them again on WY2016-23, it does not beat classical ungauged regionalisation: median NSE 0.341 against 0.372 (and 0.378 against 0.433 on the recent exam). It ties or beats it in arid, regulated and large basins and loses in snowy and humid ones. An optional gauged nowcast mode, fine-tuned to also read the river's own observed flow up to 1-7 days earlier, beats persistence at every lead but not NOAA's National Water Model.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/flowcast/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/flowcast/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Basin-average daily forcing (Caravan / ERA5-Land conventions, UTC days): P (mm/d), PET (mm/d, FAO-56 Penman-Monteith), T_mean, T_min, T_max (°C), SRAD (W/m², net shortwave), vp (Pa), wind (m/s), from 365 days before the first predicted day to one day after the last; 11 static attributes: area (km²), mean/min/max elevation (m), slope (m/km or degrees), forest fraction, soil porosity, soil conductivity (cm/h or mm/h), carbonate fraction, reservoir storage / mean annual flow, irrigated fraction, urban fraction. Seven climate indices are computed from the forcing of the window itself. Nowcast mode also reads the observed daily flow (mm/day) up to `lead` days before each day. |
| **Returns** | Daily mean flow Q (mm/day; Q_m3s in m³/s when the area is given), trained with an NSE-type loss<br>5 / 25 / 50 / 75 / 95 % quantiles (pinball loss), widened about the median by dev-fitted factors: 50 % band × 1.00, 90 % band × 1.08 (raw quantiles in the nowcast mode) |
| **Learned from** | WaterSim, the shared water platform's simulator (shared/water/watersim): four lumped rainfall-runoff models (HBV, GR4J, SAC-SMA, VIC-lite), degree-day or energy-balance snow on elevation bands, unit-hydrograph or Muskingum-Cunge routing, reservoirs and irrigation, driven by real ERA5-Land forcing. FlowCast wraps it, without editing it, in five versioned physics files (training/phys/p0-p4.json); the released rung uses p1, which adds a dev-measured water-yield correction: a simulated basin's true precipitation is ERA5-Land × m with log m = −0.088 + 0.058 elev_km + 0.153 snow share + 0.397 log(P/3 mm) + N(0, 0.146), while the network still reads ERA5-Land. |
| **Noise** | WaterSim's forcing-error model on the network's input (precipitation bias and daily lognormal scatter cv 0.25, snow undercatch, temperature error) and gauge-like error on the target (AR(1) rating error, revisions, low-flow error). No real recordings are laid into the training signals: the forcing itself is real ERA5-Land from non-exam windows of CAMELS basins (exam basins' free-window forcing included, as the platform brief allows). |
| **Network** | One LSTM layer, 64 units, over 11 daily weather features (P/10, log1p P, PET, three temperatures, net shortwave, vapour pressure, wind, day-of-year sine and cosine) plus 18 static features repeated at every step (11 attributes and 7 climate indices computed from the window's forcing: mean P and PET, aridity, snow share of P, high- and low-precipitation day frequencies, mean T). A linear head gives the mean (softplus) and five monotone quantiles (cumulative softplus), all in units of the basin's mean precipitation. 24,710 weights (25,478 for the nowcast copy, which adds three inputs: the observed flow `lead` days earlier, an availability flag and lead/7). Loss: NSE-type on the mean, scaled per sample by the Fu runoff ratio of the forcing (an input-only quantity; r0b), plus pinball loss on the quantiles. 3,000 steps × 32 windows of 730 days (365 scored), Adam lr 2e-3, dropout 0.1, CPU. Inference runs 365-day blocks, each with its own 365-day warm-up. |
| **Physics also enters** | inputs: Seven climate indices computed from the forcing (mean P and PET, aridity, snow share, high/low-precipitation frequencies, mean temperature), and the output is expressed in units of the basin's mean precipitation |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | WaterData's frozen split (shared/water/data/proc/splits.json): 52 CAMELS-US dev basins, WY2001-2014 with a 365-day warm-up from 1999-10-01 (entirely non-exam), Caravan flow, ungauged protocol. Used to fit the p1 water-yield coefficients, to gate every rung and to fit the two band factors |
| **Exam** | Every other CAMELS-US basin in the split: 619 basins on WY1990-99 (main; 600 with NWM) and 603 on WY2016-23 (recent, NWIS flow). Scored once, after r1b was chosen. Three examples (one snowy, one arid, one humid) drawn by seed 20261118 within strata (webdata.py), WY1995-96 |

## Run it

```bash
cd models/flowcast
pip install numpy pandas
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo flowcast`.

```python
import flowcast as fc
m = fc.FlowCast()                          # flowcast_weights.npz next to this file
out = m.predict(forcing_df, attrs, start='1995-10-01', end='1996-09-30')
out[['Q', 'q05', 'q50', 'q95', 'Q_m3s']]   # mm/day (and m3/s)
# gauged nowcast: also reads the observed flow up to `lead` days earlier
n = fc.FlowCast('flowcast_nowcast_weights.npz')
out = n.predict(forcing_df, attrs, '1995-10-01', '1996-09-30', qobs=observed_mm_per_day, lead=3)
```

## The exam

Every exam basin, ungauged: FlowCast never reads the basin's flow. Medians over basins of daily NSE, KGE and flow-duration biases (training/results/exam_r1b_exam_cal.json), against WaterData's baselines on the same basins and days. Per-basin FlowCast scores were not shipped, so a paired bootstrap of the medians is not possible yet; where the shipped file gives the share of basins on which FlowCast beats a baseline, the row gives that share with its exact (Clopper-Pearson) 95 % binomial interval, and an interval wholly below 0.5 is a loss. Gauged GR4J and NWM read information an ungauged model does not have; they are listed with persistence on their own rows. Nowcast rows (gauged protocol) come from results/nowcast_summary.json. All forcing is ERA5-Land: see the disclaimer. Every number here is on real data that never chose anything.

| Metric | FlowCast | Baselines | |
|---|---|---|---|
| Main exam WY1990-99 (619 basins), ungauged, median daily NSE<br><sub>FlowCast's NSE is higher at 211 of 619 basins against U-dev (34.1 %, 95 % CI 30.4-38.0 %, exact binomial) and 202 of 619 against U-k12 (32.6 %, 28.9-36.5 %): loss. Against climatology 499 of 619 (80.6 %, 77.3-83.7 %). NSE < 0 at 14.2 % of basins (U-dev 14.4 %, U-k12 14.1 %). The handoff's 0.341 / 0.372 / 0.385 reproduce.</sub> | **0.341** | NN regionalisation, 12-fold donors (U-k12) (classical): 0.385<br>NN regionalisation, dev donors (U-dev) (classical): 0.372<br>Climatology (day-of-year mean) (dumb baseline): 0.065 | ❌ loses |
| Main exam WY1990-99, against gauged and operational references, median daily NSE<br><sub>Different protocols, listed as accepted losses. FlowCast is better at 128 of 600 basins against NWM (21.3 %, 95 % CI 18.1-24.8 %), 121 of 619 against GR4J (19.5 %, 16.5-22.9 %) and 275 of 619 against persistence (44.4 %, 40.5-48.4 %); exact binomial intervals, all wholly below 0.5. On the 600 NWM basins FlowCast's median is 0.345.</sub> | **0.341** | NOAA National Water Model v3.0 (operational; calibrated on many of these gauges; 600 basins) (published): 0.55<br>Persistence, 1 day (reads yesterday's observed flow) (dumb baseline): 0.475<br>GR4J + snow calibrated per basin on its own gauge (classical): 0.473 | ❌ loses |
| Recent exam WY2016-23 (603 basins), ungauged, median daily NSE<br><sub>Better than U-dev at 205 of 603 basins (34.0 %, 95 % CI 30.2-37.9 %) and than U-k12 at 206 (34.2 %, 30.4-38.1 %), exact binomial: loss. Against climatology 498 of 603 (82.6 %, 79.3-85.5 %). NSE < 0 at 12.3 % (U-dev 11.4 %).</sub> | **0.378** | NN regionalisation, 12-fold donors (U-k12) (classical): 0.443<br>NN regionalisation, dev donors (U-dev) (classical): 0.433<br>Climatology (dumb baseline): 0.053 | ❌ loses |
| Recent exam WY2016-23, against gauged and operational references, median daily NSE<br><sub>Different protocols, listed as accepted losses. Better than NWM at 93 of 600 basins (15.5 %, 95 % CI 12.7-18.6 %) and GR4J at 121 of 603 (20.1 %, 16.9-23.5 %), both losses; against persistence 289 of 603 (47.9 %, 43.9-52.0 %), a tie on that count although the median is lower.</sub> | **0.378** | NOAA National Water Model v3.0 (600 basins) (published): 0.687<br>GR4J + snow calibrated per basin (classical): 0.517<br>Persistence, 1 day (dumb baseline): 0.48 | ❌ loses |
| Main exam, median KGE<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (`mlratchet data water`). Verdict from the medians: the gap to U-dev (0.07) and U-k12 (0.10) is large and in the same direction as the paired NSE count. Gauged GR4J 0.602, NWM 0.596, persistence 0.738. Recent exam: 0.397 against U-dev 0.469, U-k12 0.469, GR4J 0.604, NWM 0.802.</sub> | **0.367** | U-k12 (classical): 0.462<br>U-dev (classical): 0.433<br>Climatology (dumb baseline): -0.054 | ❌ loses |
| Main exam, high-flow volume bias FHV (top 2 % of flows), median<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (`mlratchet data water`). 10 points further from 0 than U-k12 (a loss by the median); against U-dev alone (4 points) it would be a tie. GR4J −18.0 %, NWM −13.0 %. Recent: −32.9 % against U-dev −29.1 %, U-k12 −24.7 %. Low-flow bias FLV +59 % against U-dev +34 % (recent +59 % vs +36 %).</sub> | **-30.9%** | U-k12 (classical): -21%<br>U-dev (classical): -27.3%<br>Climatology (dumb baseline): -80.7% | ❌ loses |
| Main exam, annual daily peaks, median absolute error<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (`mlratchet data water`). Gap 5-7 points against both regionalisations. GR4J 38.0 %, NWM 37.4 %. Recent: 52.7 % against U-dev 47.7 %, U-k12 45.8 %. Peak timing (median days off): 0.85 against 0.82 for both U variants, a tie.</sub> | **52.2%** | U-k12 (classical): 45.5%<br>U-dev (classical): 47.3%<br>Climatology (dumb baseline): 84.5% | ❌ loses |
| Flood peaks (USGS annual instantaneous; annual max daily × 1.38), main exam, median absolute error<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (`mlratchet data water`). 6,048 peaks at 607 basins. Median bias −46 % (the mean is smooth); FlowCast's 95 % quantile as the peak estimate: 63.9 % error, +31 % bias. No regionalisation baseline was scored on instantaneous peaks. Recent (4,513 peaks, 580 basins): 59.7 %, bias −49 %, against own-record 43.5 %, GR4J 47.8 %, NWM daily 34.1 % and NWM hourly 30.6 % (bias −2 %). The 1.38 ratio was measured on dev (447 samples).</sub> | **59.8%** | Own-record median annual peak (gauged, calibration half) (dumb baseline): 42.2%<br>NWM v3.0 daily × 1.38 (published): 42.4%<br>GR4J calibrated × 1.38 (classical): 46.5% | ❌ loses |
| CAMELS-531 subset, main exam, median daily NSE<br><sub>The subset most published CAMELS studies use. No interval for the subset (no paired count shipped for it); the 0.045 gap matches the full set's paired count (34 % of basins). GR4J 0.478, NWM 0.566. Recent: 0.393 against U-dev 0.448, U-k12 0.475.</sub> | **0.357** | U-dev (classical): 0.402<br>U-k12 (classical): 0.401<br>Climatology (dumb baseline): 0.062 | ❌ loses |
| Error bars: share of exam days inside the 90 % band (main, median basin)<br><sub>Nominal 90 %. Averaged over basins 95.0 %; the 50 % band holds 55.5 %. Recent: 97.5 %, 95.1 %, 55.8 %. Median band width (90 %) is 2.9 × the mean flow: the bands are too wide in most basins and too narrow in some dry ones. Widen-only factors (×1.00, ×1.08 fitted on dev) cannot narrow them.</sub> | **97.3%** | — | ℹ️ |
| Snowy basins (snow share > 0.3, 118), main exam, median NSE<br><sub>Breakdown, not graded (no paired count per type). The largest gap: −0.16 against U-dev. Recent (115): 0.41 against U-dev 0.56, NWM 0.70, GR4J 0.70. Only 10 of the 52 dev basins are snowy, and no snow rung was tried.</sub> | **0.4** | U-dev (classical): 0.56<br>GR4J calibrated (gauged) (classical): 0.67<br>NWM v3.0 (published): 0.57 | ℹ️ |
| Arid basins (aridity > 1, 174), main exam, median NSE<br><sub>Breakdown, not graded. Level with U-dev. Recent (171): 0.19 against U-dev 0.20, NWM 0.48, GR4J 0.29.</sub> | **0.21** | U-dev (classical): 0.2<br>GR4J calibrated (gauged) (classical): 0.3<br>NWM v3.0 (published): 0.32 | ℹ️ |
| Humid basins (aridity < 0.6, 128), main exam, median NSE<br><sub>Breakdown, not graded. −0.12 against U-dev. Recent (127): 0.49 against U-dev 0.61, NWM 0.72, GR4J 0.69.</sub> | **0.49** | U-dev (classical): 0.62<br>GR4J calibrated (gauged) (classical): 0.66<br>NWM v3.0 (published): 0.64 | ℹ️ |
| Regulated basins (regulation mid/high, 37), main exam, median NSE<br><sub>Breakdown, not graded; n = 37 is small. +0.05 against U-dev on the main exam, but −0.08 on the recent exam (35 basins: 0.38 against U-dev 0.46, NWM 0.70, GR4J 0.59).</sub> | **0.3** | U-dev (classical): 0.25<br>GR4J calibrated (gauged) (classical): 0.56<br>NWM v3.0 (published): 0.46 | ℹ️ |
| Large basins (> 2,000 km², 48), main exam, median NSE<br><sub>Breakdown, not graded; n = 48. Recent (48): 0.34 against U-dev 0.32, NWM 0.52, GR4J 0.44.</sub> | **0.23** | U-dev (classical): 0.22<br>GR4J calibrated (gauged) (classical): 0.38<br>NWM v3.0 (published): 0.42 | ℹ️ |
| Nowcast (gauged), lead 1 day, main exam (619), median NSE<br><sub>Beats persistence at 405 of 619 basins (65.4 %, 95 % CI 61.5-69.2 %, exact binomial): a win on that count. No per-basin nowcast scores were shipped, so there is no interval against NWM or GR4J: +0.019 over NWM is too small to call, +0.096 over GR4J is large. Verdict tie (NWM). KGE 0.635 against persistence 0.738 (persistence copies the variance). Dev WY2001-14: 0.633 against persistence 0.549. Hindcast: the target days' weather is the observed one.</sub> | **0.569** | NOAA NWM v3.0 (no gauge input at run time; 600 basins) (published): 0.55<br>Persistence, 1 day (dumb baseline): 0.475<br>GR4J calibrated per basin (gauged) (classical): 0.473 | ➖ tie |
| Nowcast (gauged), lead 1 day, recent exam (603), median NSE<br><sub>Beats persistence at 400 of 603 basins (66.3 %, 95 % CI 62.4-70.1 %): a win against persistence. Loses to NWM by 0.096 (no interval: per-basin nowcast scores not shipped; the gap is large). KGE 0.652 against persistence 0.740.</sub> | **0.591** | NOAA NWM v3.0 (600 basins) (published): 0.687<br>GR4J calibrated per basin (gauged) (classical): 0.517<br>Persistence, 1 day (dumb baseline): 0.48 | ❌ loses |
| Nowcast (gauged), lead 3 days, main exam, median NSE<br><sub>Beats persistence at 525 of 619 basins (84.8 %, 95 % CI 81.7-87.6 %). Level with GR4J (+0.001), 0.076 below NWM (no interval; large gap). KGE 0.576 against persistence 0.405. Dev: 0.567 against persistence −0.139. Recent: 0.511 against persistence −0.210 (beats it at 85.2 %, 82.2-88.0 %), NWM 0.687, GR4J 0.517.</sub> | **0.474** | NOAA NWM v3.0 (published): 0.55<br>GR4J calibrated per basin (gauged) (classical): 0.473<br>Persistence, 3 days (dumb baseline): -0.19 | ❌ loses |
| Nowcast (gauged), lead 7 days, main exam, median NSE<br><sub>Beats persistence at 557 of 619 basins (90.0 %, 95 % CI 87.3-92.2 %). Below GR4J by 0.047 and NWM by 0.124 (no interval). KGE 0.548 against persistence 0.265. Dev: 0.520 against −0.515. Recent: 0.466 against persistence −0.458 (beats it at 88.7 %, 85.9-91.1 %), NWM 0.687, GR4J 0.517.</sub> | **0.426** | NOAA NWM v3.0 (published): 0.55<br>GR4J calibrated per basin (gauged) (classical): 0.473<br>Persistence, 7 days (dumb baseline): -0.465 | ❌ loses |

### Where it wins

- Arid basins: level with regionalisation (median NSE 0.21 against 0.20 for U-dev, 174 basins), where flow is a small residual of rain minus evaporation; large basins (0.23 vs 0.22, 48; recent 0.34 vs 0.32). Regulated basins lead on the main exam (0.30 vs 0.25, 37) but trail on the recent one (0.38 vs 0.46). Breakdowns, not graded.
- The gauged nowcast beats persistence at every lead on both exams: at 65-66 % of basins at 1 day and 85-90 % at 3 and 7 days (exact binomial intervals all above 61 %); median NSE 0.569 / 0.474 / 0.426 against 0.475 / −0.19 / −0.47 (1990s).
- Its 90 % band is not too narrow: it holds 97 % of exam days at the median basin.

### Where it loses

- It does not beat classical ungauged regionalisation: median NSE 0.341 against 0.372 (U-dev) and 0.385 (U-k12) on WY1990-99, 0.378 against 0.433 and 0.443 on WY2016-23; FlowCast is better at only 34 % of basins (95 % CI 30-38 %). U-dev borrows GR4J parameters calibrated on real dev gauges, information the network gets only through the simulator.
- Snowy and humid basins: −0.16 and −0.12 NSE against U-dev (0.40 vs 0.56, 0.49 vs 0.62).
- Flood peaks run 46 % low (median absolute error 60 % against 42-47 % for the gauged and operational baselines): the mean of a parameter-uncertain prediction is smooth. Flashier physics fixed peaks on dev but cost NSE (rungs r2, r4).
- KGE and high- and low-flow volume: worse than both regionalisations (KGE 0.367 vs 0.433 and 0.462; FHV −31 % vs −27 and −21 %; FLV +59 % vs +34 and +37 %).
- Bands too wide: the 90 % band is about 2.9 × the mean flow wide at the median basin.
- The nowcast does not beat NOAA's National Water Model except at one day ahead in the 1990s, where it is level (0.569 vs 0.550); KGE at one day is below persistence's (0.635 vs 0.738).

## Limits

- Weather data: ERA5-Land (Caravan conventions, net shortwave) under-catches mountain and Pacific Northwest precipitation (50 of the 671 CAMELS-US basins have a validation-period runoff ratio above 1; WaterData DATA.md gotcha 2) and drifts about 6 % between decades. This caps every method on the exam; published Daymet/NLDAS-forced LSTMs (median NSE about 0.69-0.76 on CAMELS) are not comparable.
- The p1 water-yield correction is a regional regression fitted on the 52 dev basins that also gate the rungs (4 coefficients, partly in-sample). It over-predicts some wet Pacific Northwest basins: South Prairie Creek, WA, in the demo, at 1.76 × the observed volume (regionalisation also fails there: decade NSE −0.44).
- CONUS-like attribute ranges and a daily step only; no hourly flows. WaterSim has no glacier melt (Caravan's glacier_frac is ignored).
- Not a forecast: it simulates flow from observed weather. The nowcast is a hindcast too (observed weather on the target days).
- The climate statics are computed from the forcing of the window being predicted, so a prediction for a short or unusual window carries that window's climate (exam-period forcing, never flow).
- Rung decisions near ±0.01 on the frozen set are noise-level: a 300-basin subset and the full 800 disagreed on r5. A paired bootstrap over the 800 frozen basins (added on import) gives r1b's gain over r0b on p1 as +0.030 (95 % CI +0.001 to +0.052).

## The ratchet

### FlowCast v1.0, ungauged (WaterSim physics versions p0-p4, 18 Nov 2026)

**Objective:** Median daily NSE of the mean prediction against the gauge-like target over 800 frozen simulated basins of the rung's physics version (two fixed 365-day scored windows per basin; training/src/train.py sim_eval). One frozen set per physics version: p0 (WaterSim as shipped), p1 (water yield), p2 (flashier runoff), p3 (forcing error), p4 (p3 + p2).  
**Rule:** Keep if the sim median NSE on the frozen set of the rung's physics version beats the incumbent's on that same set by more than 0.01 AND the dev median NSE (52 real dev basins, WY2001-14, ungauged) is at least 95 % of the incumbent's.

| Rung | Change | Frozen-set score | Decision | Why | Own set | Incumbent on it | Dev NSE | Dev KGE | Dev FHV % | Dev 90% cover | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r0 | Baseline: WaterSim generator as shipped (p0): both attribute priors 50/50, 4 runoff models × 2 snow routines, 25 % Muskingum-Cunge, real ERA5-Land forcing from free windows, default forcing and gauge noise; NSE* loss scaled by the target's own std (Kratzert et al. 2019); LSTM 64, 3,000 steps | 0.441 | ● kept | First rung. Dev NSE 0.390, KGE 0.319, FHV −51 %: the network ran low (twin test: flashiness ratio 0.54 real, 0.51 calibrated twin). | p0 | – | 0.39 | 0.319 | -50.7 | 0.95 | 0.441 | 0.397 | 0.373 | 0.331 | 0.335 |
| r1 | p1 physics: dev-measured water-yield correction (true P = ERA5-Land × m, log m = −0.088 + 0.058 elev_km + 0.153 frac_snow + 0.397 log(P/3) + N(0, 0.146)), old loss | 0.332 | ○ rejected | Sim 0.332 vs r0's 0.397 on p1 (needs > 0.407) and dev 0.331 vs 0.390 (needs ≥ 0.370): both tests failed. It exposed the loss bias fixed in r0b. | p1 | 0.397 | 0.331 | 0.259 | -59.0 | 0.952 | 0.367 | 0.332 | 0.321 | 0.317 | 0.327 |
| r0b | Loss fix: per-sample loss scale from inputs only (the Fu runoff ratio of the forcing's aridity) instead of the target's own std, which down-weighted wet, flashy draws and biased the sim-trained ungauged network low | 0.476 | ● kept | Sim 0.476 vs 0.441 on p0; dev NSE 0.436 vs 0.390, KGE 0.32 → 0.44, FHV −51 → −27 %. Share of dev basins with NSE < 0 rose from 7.7 % to 15.4 %. | p0 | 0.441 | 0.436 | 0.442 | -26.8 | 0.965 | 0.476 | 0.395 | 0.445 | 0.321 | 0.377 |
| r1b | p1 water-yield physics (as r1) with r0b's input-only loss scale | 0.425 | **◆ released** | Sim 0.425 vs r0b's 0.395 on p1; dev 0.425 vs 0.436 (−2.5 %, inside the 5 % gate). Share of dev basins with NSE < 0 fell from 15.4 % to 11.5 %. Released as v1.0. | p1 | 0.395 | 0.425 | 0.424 | -32.3 | 0.951 | 0.446 | 0.425 | 0.446 | 0.335 | 0.38 |
| r2 | p2 flashier runoff: the full dev-calibrated median parameter shift (shrink 1.0, WaterSim ships 0.5) for the four runoff models; simulated Richards-Baker flashiness 0.62× → 1.00× of observed on dev | 0.463 | ○ rejected | Sim 0.463 vs r1b's 0.446 on p2 passed; dev NSE 0.380 vs 0.425 (−10.5 %) failed, although dev KGE rose to 0.490, FHV to −16 % and peak error fell 49 → 42 %. | p2 | 0.446 | 0.38 | 0.49 | -16.2 | 0.944 | 0.404 | 0.369 | 0.463 | 0.23 | 0.393 |
| r3 | p3 larger forcing error for the network input: daily P cv 0.4 (was 0.25), bias sd 0.10 (was 0.08), up to 30 % of each day's rain moved to a neighbouring day (calib_noise.py: default noise left a calibrated twin at NSE 0.88 vs its clean run; reality allows 0.56; p3 gives 0.72) | 0.326 | ○ rejected | Sim 0.326 vs r1b's 0.335 on p3 failed; dev 0.406 vs 0.425 (−4.4 %) passed. | p3 | 0.335 | 0.406 | 0.43 | -38.2 | 0.964 | 0.443 | 0.42 | 0.436 | 0.326 | 0.39 |
| r4 | p4 = p3 forcing error + p2 flashier runoff (flashiness retried with realistic forcing error) | 0.404 | ○ rejected | Sim 0.404 vs r1b's 0.380 on p4 passed; dev NSE 0.378 vs 0.425 (−11 %) failed; dev KGE 0.490, FHV −23 %, peak error 45 %. | p4 | 0.38 | 0.378 | 0.49 | -23.4 | 0.947 | 0.397 | 0.377 | 0.466 | 0.252 | 0.404 |
| r5 | Longer training: r1b fine-tuned for 5,000 more steps (lr 1e-3, cosine, seed 5), same p1 physics | 0.419 | ○ rejected | Sim 0.419 vs r1b's 0.425 on p1 failed (paired bootstrap −0.006, 95 % CI −0.019 to +0.015: noise); dev 0.434 passed. A 300-basin validation during training had said +0.02. | p1 | 0.425 | 0.434 | 0.441 | -28.2 | 0.957 | 0.463 | 0.419 | 0.454 | 0.332 | 0.393 |
| widen | Error bars: one widen-only factor per band, about the median, fitted on the dev predictions of r1b (calib_sigma.py, results/sigma_r1b.json): the smallest factor at which pooled dev coverage reaches the nominal level | — | ● kept | 50 % band × 1.00 (dev coverage 0.502 raw), 90 % band × 1.08 (0.863 → 0.953). No effect on the mean. r0's factors were 1.08 and 1.08. | – | – | – | – | – | – | – | – | – | – | – |

<sub>Columns after *Why*: Sim: the rung's own frozen set and the incumbent's score on it; dev (52 basins, ungauged): median NSE, KGE, high-flow bias FHV %, median 90 % band coverage; then the median sim NSE of the rung re-scored on every frozen set (results/sim_*).</sub>

Each new physics version built a new frozen set, and the incumbent was re-scored on it before deciding (the "vs" numbers in the why column); every rung was also re-scored on every set afterwards (the last five columns). Decisions were not re-taken in order on later sets. On the re-score matrix r1b is the best rung on p1 and p3; r5 beats it on p0 and p2 (0.463, 0.454 vs 0.446) and r4 on p2 and p4. Added on import (18 Nov 2026), from the shipped per-basin sim scores (nse_all): a paired bootstrap of the difference in medians over the 800 frozen basins (2,000 resamples, seed 20261118) gives r0b − r0 on p0 +0.035 (95 % CI +0.009 to +0.060), r1b − r0b on p1 +0.030 (+0.001 to +0.052), r2 − r1b on p2 +0.017 (−0.005 to +0.039), r3 − r1b on p3 −0.010 (−0.026 to +0.016), r4 − r1b on p4 +0.024 (−0.006 to +0.044), r5 − r1b on p1 −0.006 (−0.019 to +0.015), r1 − r0 on p1 −0.065 (−0.088 to −0.039). So the margin of 0.01 sits inside the noise: the handoff's rule note says the same (a 300-basin subset said +0.02 for r5, the full 800 −0.006; the subset run is not in the shipped files). Rule note: the four p1 water-yield coefficients were fitted on the same 52 dev basins that gate the rungs (partly in-sample). configs/smoke.json was a 20-step plumbing test, not a rung.

### FlowCast nowcast mode (gauged, optional, 18 Nov 2026)

**Objective:** None logged. n1 is a single fine-tune of r1b (2,500 steps, lr 1e-3, seed 11, p1 bank) with three extra inputs: the gauge-like flow `lead` days earlier (log scale, in units of mean precipitation), an availability flag (gaps simulated in 30 % of training windows) and lead/7; one network serves leads 1-7.  
**Rule:** No frozen-set comparison and no dev gate were applied: there was no incumbent nowcast. Its only real-data check before the exam was the dev score against persistence at leads 1, 3 and 7 (results/nowcast_summary.json, dev).

| Rung | Change | Frozen-set score | Decision | Why | Dev lead 1 | Persistence 1 | Dev lead 3 | Persistence 3 | Dev lead 7 | Persistence 7 |
|---|---|---|---|---|---|---|---|---|---|---|
| n1 | r1b fine-tuned 2,500 steps with the observed flow up to `lead` (1-7) days earlier, an availability flag and lead/7 as inputs (configs/n1.json) | — | **◆ released** | Released as the optional gauged nowcast weights (flowcast_nowcast_weights.npz). Dev: beats persistence at every lead. | 0.633 | 0.549 | 0.567 | -0.139 | 0.52 | -0.515 |

<sub>Columns after *Why*: Dev (52 basins, WY2001-14) median NSE: nowcast and persistence at the same lead.</sub>

Bands are the raw pinball quantiles (k_widen 1.0, 1.0): no dev widening was fitted for this mode. No sim score for n1 was shipped.

Raw per-rung logs are in [`training/`](training/).

## Physics

WaterSim, the shared water platform's simulator (shared/water/watersim): four lumped rainfall-runoff models (HBV, GR4J, SAC-SMA, VIC-lite), degree-day or energy-balance snow on elevation bands, unit-hydrograph or Muskingum-Cunge routing, reservoirs and irrigation, driven by real ERA5-Land forcing. FlowCast wraps it, without editing it, in five versioned physics files (training/phys/p0-p4.json); the released rung uses p1, which adds a dev-measured water-yield correction: a simulated basin's true precipitation is ERA5-Land × m with log m = −0.088 + 0.058 elev_km + 0.153 snow share + 0.397 log(P/3 mm) + N(0, 0.146), while the network still reads ERA5-Land.

**Simulator**

- Bank: 12,000 runs per physics version plus 800 frozen validation runs. Each run: a random basin (50 % jittered dev-basin attributes, 50 % WaterSim's parametric prior), random parameters with per-parameter scatter, one of four runoff models × two snow routines, 25 % Muskingum-Cunge routing; 5 years of real ERA5-Land forcing from a random CAMELS basin in a non-exam window (1951-1988 or 1999-2015; 1 year spin-up, 4 stored)
- Noise: the network reads forcing after WaterSim's forcing-error model (P bias and daily scatter, snow undercatch, T error); the target is gauge-like flow (AR(1) rating error, revisions, low-flow error)
- p1 water yield (training/src/calib_water.py, results/calib_water.json): on each of the 52 dev basins the attribute-prior simulator was run with ERA5-Land P × m on a grid of m and m* read where the simulated mean flow matched the observed (median of 4 models); fit β = (−0.0884, 0.0583, 0.1535, 0.3971), residual sd 0.146, median m 0.997. m runs from 0.53 (Texas and North Dakota plains) to 1.71 (Olympic Peninsula)
- Tried and not kept: p2 flashier runoff (the full dev prior shift instead of WaterSim's half: simulated Richards-Baker flashiness 0.62× → 1.00× of observed on dev; calib_flash_summary.json); p3 larger forcing error (daily P cv 0.4, bias sd 0.10, 30 % timing jitter: a calibrated GR4J twin under the default noise matched its clean run at NSE 0.88 while reality allows 0.56; with p3 0.72; calib_noise.json); p4 = p3 + p2

**Checked against independent references**

- WaterSim GR4J vs airGR 1.7.9 (CRAN, compiled in an isolated R), help example L0123001: max |ΔQ| 1.0e-7 mm (airGR's single-precision output); NSE 1990-99 0.79882 vs the published 0.7988 (shared/water/checks/reference_models.json)
- WaterSim HBV vs TUWmodel 1.1-1, lumped Vils 1976-2008: snow, soil moisture, ET, rain and melt identical to ≤ 1e-12 mm; runoff NSE 0.9990 (no routing), 0.9998 (MAXBAS 10)
- Water balance, 121 runs over every model × snow × routing and real dev basins: worst closure 1.4e-15 relative, worst daily residual 8.3e-12 mm (balance.json)
- Twin test (r1b, 52 dev basins, twin.py): median NSE of the same prediction against the real record 0.425, against each basin's calibrated twin 0.595, against its prior twin 0.380; flashiness ratio (prediction / target) 0.79 real vs 0.75 calibrated twin: the network reads real and calibrated twin alike. Mixing in the next UTC day gained nothing (no time-zone offset). For r0 the flashiness ratio was 0.54 real vs 0.51 twin, which led to the loss fix and to p2
- numpy release == torch to 1.7e-7 relative on 4 dev basins × 2 years (export_check.json); the site instrument (TypeScript, float16 weights) == numpy on the same weights to 5.3e-8 on 54 cases

## Help wanted

- Per-basin exam scores: the release shipped only medians and win fractions, so the exam rows carry binomial intervals on win counts instead of paired bootstrap intervals on the medians. Rebuilding the real time series (`mlratchet data water`: a 24.8 GB Caravan stream plus about 385k NWM range requests) and re-scoring the released weights would fix that.
- Daymet or NLDAS daily forcing for the CAMELS basins in Caravan's format, to separate method skill from ERA5-Land bias.

## Next rungs

- A snow rung driven by twins on the 10 snowy dev basins: the full dev snow shift (TT −0.66 °C, SFCF +0.11, CFMAX) and an elevation-dependent ERA5-Land temperature bias. Snow is the largest exam gap (−0.16 against U-dev).
- Per-basin water yield learned from attributes with uncertainty, instead of the regional regression; or ERA5-Land snow water equivalent and soil-moisture states as inputs.
- Peaks without losing NSE: a separate high-flow head, the q75/q90 annual maximum as the peak estimate, or the flashier physics with a larger network.
- Capacity: 128 units with longer training, or an ensemble of seeds, judged on larger frozen sets (decisions near ±0.01 are noise).
- Narrower bands: a calibration that can also narrow (a rule change) or a log-space quantile head.
- Re-run every exam with Daymet or NLDAS forcing to separate method skill from ERA5-Land bias; out-of-domain exams on CAMELS-GB, -CL, -BR and -AUS (attributes already in shared/water).

## Files

| Path | What |
|---|---|
| `flowcast.py`, `pipeline.py` | Inference — numpy/scipy only |
| `flowcast_weights.npz`, `flowcast_nowcast_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Caravan v1.6 (Kratzert et al. 2023, Scientific Data 10:61): CAMELS-US basins, ERA5-Land forcing, streamflow, HydroATLAS attributes](https://zenodo.org/records/14673536) | physics (forcing for every simulated basin, from non-exam windows), dev (52 basins: water-yield fit, dev gate, band widening), exam (forcing and flow of 619 basins, WY1990-99 and WY2016-23); examples/ excerpts | CC BY 4.0 (cite Caravan and each source dataset) |
| [CAMELS-US (Newman et al. 2015, HESS 19:209; Addor et al. 2017, HESS 21:5293), through Caravan](https://gdex.ucar.edu/dataset/camels.html) | dev, exam (basin set, attributes, the CAMELS-531 subset) | Free use with citation (NCAR/UCAR); redistributed in Caravan under CC BY 4.0 |
| [ERA5-Land (Muñoz-Sabater et al. 2021, ESSD 13:4349), Copernicus Climate Change Service, through Caravan](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | physics, dev, exam (all forcing) | Copernicus licence (free, attribution; not share-alike) |
| [HydroATLAS / BasinATLAS (Linke et al. 2019, Scientific Data 6:283), through Caravan](https://www.hydrosheds.org/hydroatlas) | dev, exam (static attributes) | CC BY 4.0 |
| [USGS NWIS daily values and annual peak flows](https://waterdata.usgs.gov/nwis) | exam (recent-period flow WY2016-23 and the instantaneous flood peaks), dev | Public domain (U.S. Government) |
| [NOAA National Water Model v3.0 retrospective (CONUS), at the CAMELS gauges](https://registry.opendata.aws/nwm-archive/) | exam (operational baseline), dev reference | Public domain (U.S. Government) |
| [WaterData baselines (shared/water/waterdata): regionalisation U-dev and U-k12, GR4J + snow calibrated per gauge, climatology, persistence](https://github.com/dogum/ML-Ratchet/tree/main/shared/water) | exam (baselines), dev reference | MIT (code); scores derived from the CC BY 4.0 / public-domain data above |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
