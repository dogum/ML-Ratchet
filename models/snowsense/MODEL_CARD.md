# SnowSense v1.0 model card

SnowSense estimates daily snow water equivalent (SWE) at a snow-pillow station from that station's own daily
precipitation and air temperature. It also gives each water year's peak SWE, peak date and melt-out date. Every
output has an error bar. It was trained only on simulations and graded once on 621 real SNOTEL exam stations.

Part of ML Ratchet ("Trained on physics. Graded on reality."), Watershed Phase 2, project 2. Imported into the
repository on 18 Nov 2026 (`models/snowsense/`); the numbers below are the shipped ones, checked against the shipped
JSONs in `training/` (see NOTEBOOK.md for what could and could not be checked).

> **Weather-data caveat.** SnowSense reads the station's own gauge and thermometer, not gridded ERA5-Land, so the
> ERA5-Land under-catch of mountain and Pacific Northwest precipitation (50 CAMELS basins with runoff ratio > 1) and
> its ~6 % drift between decades, which cap FlowCast and BasinDetective, do not apply to this exam. Gauge
> under-catch of windblown snow does, and is the likely cause of the -9 % peak bias. Gridded forcing is untested,
> and literature numbers from gridded-forcing models are not comparable with these.

## What it does

| | |
|---|---|
| Input | Daily `P` (mm, the gauge total for the local day), `T_mean`, `T_min`, `T_max` (deg C), station latitude and elevation. Gaps are allowed: missing P counts as 0, missing T is interpolated, and both are flagged to the network. |
| Never used | Any SWE or snow-depth measurement, at any station, as input. |
| Output, daily | SWE median and the 68 % and 90 % bands (mm, end of day). SNOTEL reports SWE at the start of the day, so compare day D with the pillow on day D+1. |
| Output, per water year | Peak SWE, peak date and melt-out date (first day after the peak below 5 mm, as in `waterdata.metrics.melt_out`), each with a 68 % and 90 % interval. |
| Window | Each water year is read from 1 August of the previous year to 1 October (427 days). The network starts every window with no snow knowledge. |
| Speed | About 0.05 s per station-year in numpy. |

## Physics and simulator

- **Snow physics:** WaterSim's two snow routines (Phase 1, agent A), one band, station forcing, Hargreaves radiation
  from the daily temperature range, as in A's SNOTEL check.
  - degree-day (HBV-light): verified exact against a 5-day hand calculation (0.0 mm) and identical to TUWmodel to
    <= 1e-12 mm (`shared/water/checks/analytic.json`, `reference_models.json`);
  - daily energy balance (Walter et al. 2005 / UEB style): net energy terms exact against hand arithmetic, mass
    6e-15 mm (`shared/water/checks/analytic.json`).
  Each simulated station-winter uses one routine, chosen 50/50.
- **Weather:** real daily P and T of the 33 development (dev) SNOTEL stations that have usable records, water years
  up to 2010 only (2011-2025 is the dev gate). Six dev stations (1059_CO, 461_UT, 510_MT, 560_WY, 573_NV, 647_OR;
  fixed seed) are held out of training and supply the frozen simulated validation sets. Each run shifts the climate:
  P x e^N(0, 0.3), T + N(0, 1.5) deg C, elevation moved with the lapse rate, latitude +- 1 deg.
- **Measurement error (real-noise rung):** the simulated snowpack feels the "true" weather; the network reads the
  gauge record. Wet-day P x lognormal(sd 0.25), 15 % of wet days slip part of their total to the next or previous
  day, temperature error AR(1) 0.6 deg C plus a station bias N(0, 0.5) deg C, and injected gaps in the network input.
- **Parameters (calibrated on dev, recipe step 5):** both routines calibrated on each dev station's own SWE, water
  years <= 2010, with wider bounds than WaterSim's tables (`calib_prior.py`; median dev NSE 0.92 degree-day, 0.95
  energy balance). Training draws from a multivariate normal fitted to the 36 parameter sets per routine, in log
  space where positive. Means: degree-day TT 0.99 C, TTI 0.83 C, CFMAX 3.6 mm/C/d, SFCF 1.01; energy balance TT
  0.91 C, SFCF 1.04, ALB_MIN 0.50, ALB_DECAY 2.4 d, TURB 0.17, GHEAT 0.32 MJ/m2/d.
- **Plausibility filter:** simulated windows with more than 5 mm of SWE on 1 August are dropped (some cold-shifted
  runs grew packs for years; SNOTEL pillows almost never carry snow through summer).

## Network

- 2-layer GRU, hidden size 64, input layer 10 -> 64 with tanh, head 64 -> 5. 50,949 parameters.
- Input channels per day: log1p(P)/2, T_mean/10, T_min/10, T_max/10, sin and cos of day of year, P-gap flag,
  T-gap flag, (elevation - 2200)/800, (latitude - 43)/8.
- Output: five quantiles (5, 16, 50, 84, 95 %) of z = sqrt(SWE)/10, built monotone (median plus cumulative
  softplus offsets), trained with a pinball loss on the water-year days. SWE = 100 max(z, 0)^2.
- Training: AdamW, one-cycle schedule, batch 128, one CPU thread. Release rung R6: 48,000 simulated station-winters
  (two banks of physics p4), fine-tuned through R1 -> R2 -> R3 -> R4 -> R5 -> R6.
- numpy inference reproduces torch to 1.2e-6 in z (5e-4 mm SWE); the page's JS reproduces numpy to 0.005 mm
  (`results/export_check.json`, `results/js_check.json` in the sandbox; not shipped). In this repository the
  browser port (`site/src/instruments/snowsense/net.ts`) reproduces the numpy model on float16 weights to 9e-15
  relative over 54 recorded and what-if cases (`site/tests/parity.test.ts`).

## Ratchet

Primary sim metric: median per-series NSE on the frozen simulated validation set (1,600 windows from the six
held-out stations' weather; one set per physics version: val_p1..val_p4). A rung is kept if it beats the incumbent by
more than 0.01 on the newest set AND the 38 dev stations (WY2011-2025) get no more than 5 % worse in (1 - NSE).
Every rung was re-scored on every set; decisions are re-taken in order on the newest (val_p4). The exam column was
filled in once, after the ladder was finished; nothing was chosen on it.

| Rung | Change | sim p1 | sim p2 | sim p3 | sim p4 | dev NSE | exam NSE | Decision |
|---|---|---|---|---|---|---|---|---|
| R1 | Baseline: both snow routines, WaterSim parameter ranges (uniform), network input = the forcing that drives the physics | 0.715 | 0.715 | 0.698 | 0.744 | 0.728 | 0.734 | kept (first) |
| R2 | Gauge error between what the snowpack feels and what the network reads; injected gaps | 0.728 | 0.734 | 0.717 | 0.769 | 0.805 | 0.765 | kept |
| R3 | Dev-fitted parameter prior (x1.3 inflation) | 0.725 | 0.707 | 0.747 | 0.810 | 0.880 | 0.854 | kept |
| R4 | Quantile head + pinball loss instead of a Gaussian (fixes the late melt-out of a Gaussian mean over many zero days) | 0.735 | 0.730 | 0.778 | 0.822 | 0.882 | 0.886 | kept |
| R5 | Dev prior as fitted (inflation 1.0): R4's bands were too wide on dev | 0.700 | 0.695 | 0.793 | 0.831 | 0.887 | 0.883 | **rejected** (0.831 not > 0.822 + 0.01) |
| R6 | Second p4 bank (48k windows), 6 more epochs from R5 | 0.726 | 0.725 | 0.785 | 0.839 | 0.888 | **0.888** | kept, **released** |

Twin test (dev stations, WY2011-2025, `twin.py`): each dev station's calibrated energy-balance twin (calibrated on
WY <= 2010) scores NSE 0.917 against the real pillow (median). With R1 the network read the twin at NSE 0.80 and real
at 0.73, with peaks 21 % below the twin: the parameter prior, not the network, was the gap (-> R3). With R3 the
network melted out 10 days later than the twin (-> R4). With R6: net vs twin 0.914, net vs real 0.888, peak bias -4 %
vs twin and -9 % vs real, melt-out -2 d vs twin.

## Exam (621 SNOTEL exam stations, WY2011-2025, B's protocol, medians over stations)

| Method | Protocol | NSE | RMSE mm | Peak SWE abs err % | Peak bias % | Melt-out MAE d |
|---|---|---|---|---|---|---|
| **SnowSense v1.0** | weather only | **0.888** | **51.7** | **17.5** | -9.0 | **8.3** |
| Energy balance, WaterSim defaults | weather only | 0.853 | 57.2 | 23.7 | -20.0 | 9.1 |
| Degree-day, WaterSim defaults | weather only | 0.691 | 80.0 | 28.5 | -26.4 | 21.9 |
| Climatology (own 2001-2010 SWE, B) | uses station SWE | 0.701 | 84.6 | 28.2 | +2.1 | 11.4 |
| Degree-day calibrated per station (B; gauged ceiling) | uses station SWE | 0.925 | 41.2 | 12.5 | -3.6 | 6.1 |

- SnowSense has the higher NSE at 75 % of stations against the default energy balance, 84 % against climatology and
  34 % against the per-station calibrated degree-day. Stations with NSE < 0.5: 8.4 % (energy balance 9.5 %,
  calibrated 2.3 %).
- Median error in the date of peak SWE: 3 days. 9,071 station-years.
- By region (NSE: SnowSense / default EB / calibrated DD): Central Rockies (233) 0.910 / 0.859 / 0.939; Northern
  Rockies (159) 0.921 / 0.908 / 0.943; Pacific (153) 0.845 / 0.798 / 0.900; Southwest (31) 0.776 / 0.704 / 0.811;
  Great Basin (24) 0.831 / 0.765 / 0.906; Alaska (21) 0.871 / 0.808 / 0.857.
- By elevation: < 1500 m 0.787 / 0.761 / 0.884; 1500-2200 m 0.897 / 0.877 / 0.918; 2200-2800 m 0.902 / 0.857 / 0.936;
  > 2800 m 0.910 / 0.867 / 0.942.

### Reproduction re-score and dev-tuned physics (18 Nov 2026, seen exam)

Run on the rebuilt SNOTEL series (`training/rescore_exam.py`, `training/rescore_summary.py`; per-station scores in
`training/rescore_exam_stations.csv`, summary in `training/rescore_summary.json`). The released numpy weights, scored
through the shipped exam path, reproduce the shipped exam exactly on the same 621 stations (NSE 0.888, RMSE 51.7 mm,
peak error 17.5 %, melt-out 8.3 d, coverage 78 / 76 / 86 %). WaterSim's default baselines re-score to the shipped
medians, and WaterData's climatology and calibrated degree-day, re-run from scratch, match `snow_scores.parquet` at
every station (largest NSE difference 0).

Two baselines were added: WaterSim's energy balance and degree-day with **dev-tuned** parameters, frozen in
`training/devtuned_params.json` before scoring and run once, on the same inputs SnowSense reads. Energy balance:
the medians of the dev stations' own calibrations (`twin_r6.json`): TT 1.06 C, SFCF 1.05, ALB_MIN 0.52,
ALB_DECAY 1.40 d, TURB 0.199, GHEAT 0.060 MJ/m2/d. Degree-day: the dev prior's means (`prior_p4.json`; per-station
medians were not shipped): TT 0.99 C, TTI 0.82 C, CFMAX 3.57 mm/C/d, SFCF 1.01. Every other parameter comes from
WaterSim's default and attribute-transfer layer, as in the calibrations and the default baselines. They are labelled
"dev-tuned (added on import, seen exam)": the exam had been seen, but nothing in them was chosen on it.

| Method | Protocol | NSE | RMSE mm | Peak abs err % | Peak bias % | Melt-out MAE d | Melt-out bias d |
|---|---|---|---|---|---|---|---|
| **SnowSense v1.0** | weather only | 0.888 | 51.7 | 17.5 | -9.0 | 8.3 | -2.8 |
| Energy balance, dev-tuned | weather only | 0.895 | 49.7 | 16.2 | -4.3 | 8.6 | +0.3 |
| Degree-day, dev-tuned | weather only | 0.864 | 56.3 | 18.2 | -7.5 | 9.7 | -3.7 |
| Energy balance, WaterSim defaults | weather only | 0.853 | 57.2 | 23.7 | -20.0 | 9.1 | -4.7 |
| Degree-day, WaterSim defaults | weather only | 0.691 | 80.0 | 28.5 | -26.4 | 21.9 | -21.5 |
| Climatology (own 2001-2010 SWE) | reads station SWE | 0.701 | 84.6 | 28.2 | +2.1 | 11.4 | +3.5 |
| Degree-day calibrated per station | reads station SWE | 0.925 | 41.2 | 12.5 | -3.6 | 6.1 | -0.3 |

Intervals: paired bootstrap over the 621 stations, 10,000 resamples, seed 20261118, 95 % percentile interval of
SnowSense minus the baseline in the median (for bias rows, in |median bias|; negative is better for SnowSense except
NSE). A baseline is beaten only if the interval excludes zero; a row is a win only if every baseline on it is beaten,
a loss if any baseline beats SnowSense.

| Row | SnowSense | Baseline median: difference [95 % interval] | Verdict |
|---|---|---|---|
| NSE, weather-only + climatology | 0.888 | EB default 0.853: +0.035 [+0.025, +0.047]; DD default 0.691: +0.197 [+0.174, +0.228]; **EB dev-tuned** 0.895: -0.007 [-0.018, +0.001]; DD dev-tuned 0.864: +0.024 [+0.012, +0.037]; climatology 0.701: +0.187 [+0.166, +0.201] | tie |
| NSE, gauged ceiling | 0.888 | DD calibrated 0.925: -0.037 [-0.050, -0.026] | loss |
| RMSE mm, weather-only + climatology | 51.7 | EB default 57.2: -5.5 [-7.8, -3.4]; DD default 80.0: -28.4 [-32.9, -24.1]; **EB dev-tuned** 49.7: +2.0 [-0.7, +4.1]; DD dev-tuned 56.3: -4.6 [-7.1, -2.3]; climatology 84.6: -32.9 [-37.5, -28.7] | tie |
| RMSE mm, gauged ceiling | 51.7 | DD calibrated 41.2: +10.5 [+7.1, +13.1] | loss |
| Peak abs err %, weather-only + climatology | 17.5 | EB default 23.7: -6.2 [-7.4, -5.1]; DD default 28.5: -11.0 [-12.5, -9.2]; **EB dev-tuned** 16.2: +1.3 [+0.3, +2.0]; DD dev-tuned 18.2: -0.7 [-1.6, +0.4]; climatology 28.2: -10.7 [-12.2, -8.7] | loss |
| Peak abs err %, gauged ceiling | 17.5 | DD calibrated 12.5: +5.0 [+4.0, +5.9] | loss |
| Melt-out MAE d, weather-only + climatology | 8.3 | EB default 9.1: -0.8 [-1.4, -0.2]; DD default 21.9: -13.5 [-15.4, -12.5]; **EB dev-tuned** 8.6: -0.3 [-0.8, +0.1]; DD dev-tuned 9.7: -1.4 [-2.0, -0.7]; climatology 11.4: -3.1 [-3.7, -2.5] | tie |
| Melt-out MAE d, gauged ceiling | 8.3 | DD calibrated 6.1: +2.2 [+1.6, +2.7] | loss |
| Peak bias % (abs. median), weather-only + climatology | -9.0 | EB default -20.0: -11.0 [-12.3, -9.8]; DD default -26.4: -17.5 [-19.4, -15.3]; **EB dev-tuned** -4.3: +4.7 [+3.8, +5.7]; DD dev-tuned -7.5: +1.4 [-0.4, +2.4]; climatology 2.1: +6.8 [+4.5, +9.4] | loss |
| Peak bias % (abs. median), gauged ceiling | -9.0 | DD calibrated -3.6: +5.4 [+3.7, +7.0] | loss |
| Melt-out bias d (abs. median), weather-only + climatology | -2.8 | EB default -4.7: -1.9 [-2.6, -1.2]; DD default -21.5: -18.7 [-20.2, -17.6]; **EB dev-tuned** 0.3: +2.5 [+1.2, +3.5]; DD dev-tuned -3.7: -0.9 [-1.4, +0.0]; climatology 3.5: -0.7 [-1.7, +0.5] | loss |
| Melt-out bias d (abs. median), gauged ceiling | -2.8 | DD calibrated -0.3: +2.5 [+1.6, +3.5] | loss |
| Share with NSE < 0.5 | 8.4 % | EB default 9.5 %: -1.1 [-3.2, +0.8]; EB dev-tuned 10.5 %: -2.1 [-3.9, -0.3]; DD dev-tuned 10.0 %: -1.6 [-3.4, +0.2]; climatology 17.9 %: -9.5 [-12.7, -6.3]; DD calibrated 2.3 %: +6.1 [+3.9, +8.4] | loss |

**What this means.** Against WaterSim with default parameters SnowSense wins clearly, as shipped. Against the same
energy balance with parameters tuned on the dev stations it ties on NSE (0.888 vs 0.895, -0.007 [-0.018, +0.001]),
RMSE and melt-out timing, and loses on peak SWE (error 17.5 vs 16.2 %, bias -9.0 vs -4.3 %). The tuned energy balance
has the higher NSE at 55 % of stations (Clopper-Pearson 41.3-49.3 % for SnowSense). This is the WakeNet and StormNet
lesson again: dev-tuned physics turns the win into a tie, and the network adds nothing over it on the median.
SnowSense still fails less often (NSE < 0.5 at 8.4 % vs 10.5 %) and is far better below 1,500 m (0.787 vs 0.704,
+0.083 [+0.041, +0.159]), where the tuned energy balance struggles with thin, wet packs; the tuned physics is better at
2,200-2,800 m (-0.019 [-0.034, -0.008]). By region the differences to the tuned EB are within their intervals.

Clopper-Pearson win fractions (cross-check, higher NSE): default EB 74.6 % [70.9, 77.9]; default DD 87.3 %
[84.4, 89.8]; dev-tuned EB 45.2 % [41.3, 49.3]; dev-tuned DD 66.8 % [63.0, 70.5]; climatology 84.2 % [81.1, 87.0];
calibrated DD 33.8 % [30.1, 37.7].

"WaterSim defaults" are the parameters the exam path resolves for a station given only latitude and elevation:
WaterSim's table defaults plus its attribute-transfer layer (energy balance CANOPY 0.535 instead of 1.0; degree-day
TT -0.33 C, CFMAX 3.77, SFCF 1.054), identical at every station.

## Error bars and coverage

One widen-only factor per output was fitted on the dev stations (`calib_sigma.py`). All three are 1.0 because the
dev intervals already covered at least 68 % (daily 72 %, peak 76 %, melt-out 88 %).

| Output | 68 % interval, exam coverage | 90 % interval, exam coverage |
|---|---|---|
| Daily SWE (snow days) | 78 % | 96 % |
| Peak SWE (station-years) | 76 % | 93 % |
| Melt-out date | 86 % | 95 % |

The bars are conservative, not tight: the widen-only rule cannot narrow them.

## Rule notes

Reported as written in the Watershed handoff (section 6), the parts about SnowSense:

- Dev data were used both to fit physics and to gate rungs (partly in-sample): SnowSense's parameter prior was fitted
  on the dev stations' SWE up to WY2010, and the gate scored the same stations on WY2011-2025. The exam was untouched.
- No exam SWE was used for training. (SnowSense's training weather came from dev stations only, WY <= 2010.)
- R6 was initialised from the rejected R5 weights; only the keep decision compares against the incumbent R4.
- The release demo was run on 3 exam stations before the final exam, for code testing only. Nothing was chosen on it.
- One training process peaked at about 1.34 GB, above the 1.2 GB sandbox guidance.
- The dev-tuned baselines and the reproduction re-score (18 Nov 2026) were run on an exam that had been seen; the
  dev-tuned parameters were frozen from dev data before scoring, and nothing in the model changed on the result.
- Ratchet noise: on val_p3, R5 would have been kept and R6 rejected; on val_p4 the reverse. Margins of 0.01 are near
  the noise of these frozen sets. The exam columns in the ratchet table were filled in after the ladder finished and
  were never used to choose.

## Limits

1. The gauged ceiling is still better (0.925 vs 0.888). With ten years of a station's own SWE, calibrate it.
2. Peak SWE is low by 9 % (median). About half of that is the network averaging over unknown snow parameters; the
   rest is snow the physics does not make (wind drift, undercatch beyond the prior).
3. Low and warm stations are weakest (< 1500 m: NSE 0.79; Southwest 0.78): rain-on-snow, thin intermittent packs.
4. No snow knowledge at 1 August; perennial snow is outside the training data.
5. Driven by the station's own gauge and thermometer. Gridded forcing (ERA5-Land, PRISM, Daymet) is untested. SNOTEL
   temperature sensors changed network-wide in the 2000s (Oyler et al. 2015), so period-dependent biases are likely.
6. The simulated validation sets use six dev stations' weather; the dev prior was fitted on 36 stations' calibrations.

## Data and licences

- USDA NRCS SNOTEL daily SWE, precipitation, temperature (public domain; NRCS National Water and Climate Center report
  generator), via the Watershed WaterData package (B) with frozen splits (seed 20261118): 39 dev / 772 exam stations;
  621 exam stations scored.
- WaterSim (A): HBV-light degree-day snow (Bergstrom 1976; Seibert & Vis 2012); daily energy balance after Walter et
  al. (2005) and Tarboton & Luce (1996, UEB); Hargreaves & Samani (1985) radiation; Brutsaert (1975) longwave.
- Baselines: B's `snow_scores.parquet` (climatology, calibrated degree-day), now at
  `shared/water/data/proc/baselines/snow_scores.parquet`; its per-station table reproduces every calibrated-degree-day
  number in `final_summary.json` (all regions, elevation bands and the share below NSE 0.5) exactly.
- None of these sources is share-alike, so the weights are MIT like the rest of the repository.

## Files

`snowsense.py` (numpy model), `snowsense_weights.npz` (weights + widen factors), `pipeline.py` (CSV in, CSV out),
`demo.py` (prints to the terminal only), `examples/` (three exam stations, a fixed random draw, WY2016-2025; sources in
`examples/README.md`), `training/` (every script used: simbank, calib_prior, train, evaluate, ratchet, twin,
calib_sigma, final_exam, export, webdata, make_examples, and the page's JS port `snowsense.js` with `check_js.cjs`; the
shipped result JSONs ratchet.json, final_summary.json, exam_by_rung.json, twin_r6.json, k_r6.json, prior_p3.json,
prior_p4.json; `_paths.py` maps the sandbox paths to this repository), `web/` (the live-instrument exporter and the lab
builder), `lab/index.html` (the offline lab, rebuilt from the site's instrument), `NOTEBOOK.md` (the build notebook).
