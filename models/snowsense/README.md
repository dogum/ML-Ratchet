<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# SnowSense <sub>v1.0</sub>

> **Daily snow water equivalent at a snow pillow, from the station's own weather**

*How much water is stored as snow at this mountain station today, and when will it peak and melt out?*

SnowSense estimates daily snow water equivalent (SWE) at a SNOTEL snow-pillow station from that station's own daily precipitation and air temperature, with 68 % and 90 % bands, plus each winter's peak SWE, peak date and melt-out date. It never reads a snow measurement. It learned only from WaterSim snow simulations (degree-day and energy balance) driven by real dev-station weather up to 2010. On 621 SNOTEL exam stations (water years 2011-2025) it scores median NSE 0.888 against 0.853 for WaterSim's energy balance with default parameters, and it has the higher NSE at 75 % of stations (95 % interval 71-78 %). A degree-day model calibrated on ten years of each station's own snow still does better (0.925; SnowSense ahead at 34 % of stations). The fairest weather-only baseline, the energy balance with the dev-calibrated parameters, has not been run on the exam yet.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/snowsense/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/snowsense/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Daily precipitation P (mm, the gauge total for the local day), T_mean, T_min and T_max (deg C) from the station itself, plus its latitude and elevation. Each water year is read from 1 August of the previous year to 1 October (427 days). Gaps are allowed: missing P counts as 0, missing T is interpolated, and both are flagged to the network. Never any SWE or snow depth. |
| **Returns** | Daily SWE median (mm, end of day) with 68 % and 90 % bands (exam coverage of snow days 78 % and 96 %)<br>Per water year: peak SWE (mm) with 68 % / 90 % intervals (exam coverage 76 % / 93 %)<br>Per water year: peak date, and melt-out date (first day after the peak below 5 mm) with 68 % / 90 % intervals (exam coverage 86 % / 95 %) |
| **Learned from** | Point snowpack mass and energy balance. WaterSim's two snow routines run snow-only at the station (one band, no runoff model): the HBV-light degree-day routine (rain/snow partition over TT +- TTI/2, snowfall correction, melt CFMAX (T - TT), refreeze, liquid water holding) and a daily single-layer energy balance after Walter et al. (2005) and the Utah Energy Balance model (net shortwave with an ageing albedo, Brutsaert long-wave with a cloud correction, bulk-transfer sensible and latent heat with a stability reduction, ground and rain heat, cold content, refreeze, sublimation). Radiation comes from the daily temperature range (Hargreaves-Samani). |
| **Noise** | Real weather, not synthetic: the simulated snowpacks are driven by the dev stations' own recorded P and T. From rung R2 on, the snowpack feels a 'true' weather that differs from the gauge record the network reads: wet-day P x lognormal(sd 0.25), 15 % of wet days slip part of their total to the next or previous day, temperature error AR(1) 0.6 deg C plus a station bias N(0, 0.5) deg C, and injected data gaps in the network input (P set to 0, T interpolated, both flagged). |
| **Network** | Input layer 10 -> 64 (tanh), a 2-layer GRU with 64 hidden units, and a head 64 -> 5 giving five monotone quantiles (5, 16, 50, 84, 95 %) of z = sqrt(SWE)/10, trained with a pinball loss on the water-year days; 50,949 parameters. Input channels per day: log1p(P)/2, T_mean/10, T_min/10, T_max/10, sin and cos of day of year, P-gap and T-gap flags, (elevation - 2200)/800, (latitude - 43)/8. Released rung R6: 48,000 simulated station-winters (two physics-p4 banks), fine-tuned R1 -> R6, AdamW one-cycle, batch 128, one CPU thread. numpy-only inference, about 0.05 s per station-year. |
| **Physics also enters** | only through the simulated training data |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | WaterData's frozen SNOTEL split (shared/water/data/proc/splits.json, seed 20261118): 39 dev stations, drawn per state, systematically by elevation, only where the HUC8 holds no exam CAMELS gauge. Training weather: dev stations up to WY2010, six of them (1059_CO, 461_UT, 510_MT, 560_WY, 573_NV, 647_OR) held out for the frozen simulated sets. Parameter prior: dev SWE up to WY2010. Dev gate, widen factors and twin test: the 38 dev stations with at least 5 complete water years in 2011-2025. |
| **Exam** | The 621 of 772 exam stations that WaterData scored (at least 5 complete calibration and scoring years), water years 2011-2025, end-of-day SWE against the pillow's next-morning reading (WaterData's protocol). Scored once, after the ratchet had chosen the release on simulated and dev data only. |

## Run it

```bash
cd models/snowsense
pip install numpy pandas
python demo.py        # about 3 s
```

or from anywhere in the repo: `mlratchet demo snowsense`.

```python
import pandas as pd
from snowsense import SnowSense
df = pd.read_csv("examples/349_MT_SNTL.csv", parse_dates=["date"]).set_index("date")
out = SnowSense().predict(df, lat=46.414, elev_m=2185)
print(out["annual"][["peak_swe_mm", "peak_lo68", "peak_hi68", "melt_out_date"]])
```

## The exam

Medians over 621 exam stations of each station's 2011-2025 scores, against WaterSim's two snow routines with default parameters (weather only, like SnowSense), a climatology and a degree-day model calibrated per station (both read the station's own 2001-2010 SWE). Intervals: exact binomial (Clopper-Pearson) 95 % intervals on the shipped station win fractions (n = 621) where they exist; otherwise none, because per-station model scores were not shipped and the SNOTEL series are not on disk. The fairest weather-only baseline, the energy balance with the dev-calibrated parameters, was never run on the exam. Every number here is on real data that never chose anything.

| Metric | SnowSense | Baselines | |
|---|---|---|---|
| Daily SWE, median NSE, against weather-only physics and climatology<br><sub>Higher NSE than the default energy balance at 463 of 621 stations (74.6 %, Clopper-Pearson 95 % interval 70.9-77.9 %, entirely above 50 %) and than climatology at 523 (84.2 %, 81.1-87.0 %). Against the default degree-day no win fraction was shipped; the median gap (0.888 vs 0.691) is large. Open: the energy balance with the dev-calibrated parameters (prior_p4.json means TURB 0.17, ALB_DECAY 2.4 d, GHEAT 0.32) is the fairer baseline and was never run on the exam (it needs the SNOTEL series: mlratchet data water). On dev the per-station calibrated EB twin reaches 0.917, so as with WakeNet and StormNet this win may become a tie.</sub> | **0.888** | Energy balance, WaterSim defaults (weather only) (classical): 0.853<br>Degree-day, WaterSim defaults (weather only) (classical): 0.691<br>Climatology, station's own WY2001-2010 SWE (WaterData) (dumb baseline): 0.701 | ✅ wins |
| Daily SWE, median NSE, against the gauged ceiling (degree-day calibrated on each station's own SWE)<br><sub>Different protocol: the baseline reads ten years of the station's own SWE, SnowSense none. SnowSense has the higher NSE at 210 of 621 stations (33.8 %, Clopper-Pearson 95 % interval 30.1-37.7 %, entirely below 50 %).</sub> | **0.888** | Degree-day calibrated per station on WY2001-2010 SWE (WaterData; reads the station's snow) (classical): 0.925 | ❌ loses |
| Daily SWE, median RMSE, against weather-only physics and climatology<br><sub>At one station both methods are scored on the same observed days, so lower RMSE is the same event as higher NSE: the NSE win fractions carry over (default energy balance 74.6 %, 95 % interval 70.9-77.9 %; climatology 84.2 %, 81.1-87.0 %). Same open item: dev-tuned energy balance not run.</sub> | **51.7 mm** | Energy balance, WaterSim defaults (weather only) (classical): 57.2 mm<br>Degree-day, WaterSim defaults (weather only) (classical): 80 mm<br>Climatology, station's own WY2001-2010 SWE (WaterData) (dumb baseline): 84.6 mm | ✅ wins |
| Daily SWE, median RMSE, against the gauged ceiling<br><sub>Same station win fraction as NSE: SnowSense lower at 33.8 % of stations (95 % interval 30.1-37.7 %).</sub> | **51.7 mm** | Degree-day calibrated per station (reads the station's snow) (classical): 41.2 mm | ❌ loses |
| Peak SWE, median absolute error, against weather-only physics and climatology<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (mlratchet data water). Verdict from the medians: 6.2 points (26 %) below the best baseline, and the default energy balance's peaks are biased -20 % against SnowSense's -9 %.</sub> | **17.5%** | Energy balance, WaterSim defaults (classical): 23.7%<br>Degree-day, WaterSim defaults (classical): 28.5%<br>Climatology, station's own WY2001-2010 SWE (dumb baseline): 28.2% | ✅ wins |
| Peak SWE, median absolute error, against the gauged ceiling<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (mlratchet data water). Gap 5.0 points (40 % larger error).</sub> | **17.5%** | Degree-day calibrated per station (reads the station's snow) (classical): 12.5% | ❌ loses |
| Peak SWE, median bias<br><sub>No interval: per-unit scores not shipped. Climatology and the calibrated degree-day, both reading the station's own SWE, are closer to zero. Against the weather-only physics the low bias is halved. Peaks low by 9 % are the main miss: about half is the network averaging over unknown snow parameters (twin: -4 % vs the calibrated twin), the rest snow the physics does not make (drift, gauge under-catch).</sub> | **-9%** | Energy balance, WaterSim defaults (classical): -20%<br>Degree-day, WaterSim defaults (classical): -26.4%<br>Climatology, station's own WY2001-2010 SWE (dumb baseline): 2.1%<br>Degree-day calibrated per station (reads the station's snow) (classical): -3.6% | ❌ loses |
| Melt-out date, median absolute error, against weather-only physics and climatology<br><sub>No interval: per-unit scores not shipped; needs the data rebuild (mlratchet data water). 0.8 days better than the default energy balance is too small to call from medians alone; clearly better than the default degree-day and climatology.</sub> | **8.3 days** | Energy balance, WaterSim defaults (classical): 9.1 days<br>Degree-day, WaterSim defaults (classical): 21.9 days<br>Climatology, station's own WY2001-2010 SWE (dumb baseline): 11.4 days | ➖ tie |
| Melt-out date, median absolute error, against the gauged ceiling<br><sub>No interval: per-unit scores not shipped. Gap 2.2 days (36 % larger error).</sub> | **8.3 days** | Degree-day calibrated per station (reads the station's snow) (classical): 6.1 days | ❌ loses |
| Melt-out date, median bias<br><sub>No interval: per-unit scores not shipped. Closest to zero among the weather-only methods; the calibrated degree-day is closer. R4, before the dev prior was narrowed, had -0.3 days on the exam (reported after the fact).</sub> | **-2.8 days** | Energy balance, WaterSim defaults (classical): -4.7 days<br>Degree-day, WaterSim defaults (classical): -21.5 days<br>Climatology, station's own WY2001-2010 SWE (dumb baseline): 3.5 days<br>Degree-day calibrated per station (reads the station's snow) (classical): -0.3 days | ❌ loses |
| Peak date, median absolute error (9,071 station-years)<br><sub>No baseline peak-date errors were shipped.</sub> | **3 days** | — | ℹ️ |
| Share of stations with NSE below 0.5<br><sub>No paired interval. 52 vs 59 of 621 stations against the default energy balance is a tie; the calibrated degree-day fails at 14 stations.</sub> | **8.4%** | Energy balance, WaterSim defaults (classical): 9.5%<br>Climatology, station's own WY2001-2010 SWE (computed from WaterData's per-station table) (dumb baseline): 17.9%<br>Degree-day calibrated per station (reads the station's snow) (classical): 2.3% | ❌ loses |
| By region, median NSE: Central Rockies (233 stations)<br><sub>Breakdown, not graded. No interval.</sub> | **0.91** | Energy balance, defaults (classical): 0.859<br>Degree-day calibrated per station (classical): 0.939 | ℹ️ |
| By region, median NSE: Northern Rockies (159)<br><sub>Breakdown, not graded. No interval.</sub> | **0.921** | Energy balance, defaults (classical): 0.908<br>Degree-day calibrated per station (classical): 0.943 | ℹ️ |
| By region, median NSE: Pacific (153)<br><sub>Breakdown, not graded. No interval.</sub> | **0.845** | Energy balance, defaults (classical): 0.798<br>Degree-day calibrated per station (classical): 0.9 | ℹ️ |
| By region, median NSE: Southwest (31)<br><sub>Breakdown, not graded. Weakest region: thin, intermittent packs. No interval.</sub> | **0.776** | Energy balance, defaults (classical): 0.704<br>Degree-day calibrated per station (classical): 0.811 | ℹ️ |
| By region, median NSE: Great Basin (24)<br><sub>Breakdown, not graded. No interval.</sub> | **0.831** | Energy balance, defaults (classical): 0.765<br>Degree-day calibrated per station (classical): 0.906 | ℹ️ |
| By region, median NSE: Alaska (21)<br><sub>Breakdown, not graded. The only region where SnowSense's median is above the calibrated degree-day's; 21 stations, no interval.</sub> | **0.871** | Energy balance, defaults (classical): 0.808<br>Degree-day calibrated per station (classical): 0.857 | ℹ️ |
| By elevation, median NSE: below 1,500 m (104 stations)<br><sub>Breakdown, not graded. Weakest band: rain-on-snow and thin packs. No interval.</sub> | **0.787** | Energy balance, defaults (classical): 0.761<br>Degree-day calibrated per station (classical): 0.884 | ℹ️ |
| By elevation, median NSE: 1,500-2,200 m (174)<br><sub>Breakdown, not graded. No interval.</sub> | **0.897** | Energy balance, defaults (classical): 0.877<br>Degree-day calibrated per station (classical): 0.918 | ℹ️ |
| By elevation, median NSE: 2,200-2,800 m (210)<br><sub>Breakdown, not graded. No interval.</sub> | **0.902** | Energy balance, defaults (classical): 0.857<br>Degree-day calibrated per station (classical): 0.936 | ℹ️ |
| By elevation, median NSE: above 2,800 m (133)<br><sub>Breakdown, not graded. No interval. The shipped median equals the Central Rockies median to 16 digits; both groups have an odd count, so each median is one station, plausibly the same one.</sub> | **0.91** | Energy balance, defaults (classical): 0.867<br>Degree-day calibrated per station (classical): 0.942 | ℹ️ |
| 90 % bands, exam coverage: daily SWE / peak SWE / melt-out<br><sub>Nominal 90 %. Widen-only factors fitted on dev are all 1.0 (dev raw 68 % coverage: daily 72 %, peak 76 %, melt-out 88 %), so the bars are conservative.</sub> | **95.6 / 93.3 / 94.6 %** | — | ℹ️ |

Coverage of the 68% error bars on the exam: Daily SWE, snow days **77.7%**, Peak SWE, station-years **76.3%**, Melt-out date, station-years **85.6%**.

### Where it wins

- Beats every weather-only baseline that was run: median NSE 0.888 against 0.853 for WaterSim's default energy balance (higher at 75 % of stations, 95 % interval 71-78 %) and 0.691 for the default degree-day; peak SWE error 17.5 % against 23.7 %.
- Beats climatology built from each station's own 2001-2010 snow, without reading any snow (higher NSE at 84 % of stations, interval 81-87 %).
- Its error bars hold on the exam: the 68 % bars cover 78 % of snow days, 76 % of peaks and 86 % of melt-out dates.

### Where it loses

- Loses to a degree-day model calibrated on ten years of each station's own SWE: median NSE 0.888 against 0.925, ahead at only 34 % of stations (interval 30-38 %); peak error 17.5 against 12.5 %; melt-out 8.3 against 6.1 days.
- Peaks are 9 % low (median); climatology (+2 %) and the calibrated degree-day (-4 %) are less biased.
- Melt-out MAE ties the default energy balance (8.3 against 9.1 days, no interval).

## Limits

- The fairest weather-only baseline was not run: WaterSim's energy balance with the dev-calibrated parameters (TURB 0.17, ALB_DECAY 2.4 d, GHEAT 0.32) needs the SNOTEL series, which are not in the repository. With WakeNet and StormNet such dev-tuned physics turned wins into ties; the energy-balance win here is a win over default physics until it runs.
- Weather data: SnowSense reads the station's own gauge and thermometer, not ERA5-Land, so the ERA5-Land under-catch and decade drift that cap FlowCast and BasinDetective do not apply here; but gauge under-catch of windblown snow does, and is the likely cause of the -9 % peak bias. Gridded forcing (ERA5-Land, PRISM, Daymet) is untested, and SNOTEL temperature sensors changed network-wide in the 2000s (Oyler et al. 2015).
- Low and warm stations are weakest: median NSE 0.787 below 1,500 m and 0.776 in the Southwest (rain-on-snow, thin intermittent packs).
- Every water year starts on 1 August with no snow knowledge; perennial snow is outside the training data (simulated windows with snow on 1 August were dropped).
- The dev prior was fitted on dev SWE up to WY2010 and the dev gate uses the same stations in WY2011-2025, so the gate is partly in-sample. The exam was untouched.
- No per-station exam scores were shipped, so only the station win fractions have intervals.

## The ratchet

**Objective:** Median per-series NSE of daily SWE (water-year days) on the frozen simulated validation set: 1,600 windows simulated from the weather of six held-out dev stations, one set per physics version (val_p1..val_p4)  
**Rule:** Keep if the rung beats the incumbent by more than 0.01 median NSE on the newest frozen set (val_p4) AND (1 - NSE_dev) <= 1.05 (1 - NSE_dev of the incumbent), dev NSE being the median over 38 dev stations, WY2011-2025.

| Rung | Change | Frozen-set score | Decision | Why | val_p1 | val_p2 | val_p3 | Dev NSE | Dev melt-out MAE d | Dev 68% daily cover | Exam NSE (after) | Exam melt-out MAE d (after) | Exam 68% daily cover (after) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | Baseline (physics p1): WaterSim degree-day and energy balance 50/50, parameters uniform over WaterSim's ranges, the network reads the forcing that drives the physics (no measurement error); Gaussian head | 0.744 | ● kept | First rung. | 0.715 | 0.715 | 0.698 | 0.728 | 12.5 | 0.712 | 0.734 | 11.7 | 0.776 |
| R2 | Physics p2: gauge error between what the snowpack feels and what the network reads (wet-day P x lognormal 0.25, one-day timing slips, T AR(1) 0.6 C + station bias 0.5 C) and injected data gaps; fine-tuned from R1 | 0.769 | ● kept | val_p4 0.769 > 0.744 + 0.01; dev 0.805 vs 0.728 passes. | 0.728 | 0.734 | 0.717 | 0.805 | 10.9 | 0.783 | 0.765 | 12.7 | 0.824 |
| R3 | Physics p3: parameter prior fitted on the dev stations (both routines calibrated on each dev station's own SWE up to WY2010 with wider bounds; multivariate normal x1.3); fine-tuned from R2 | 0.810 | ● kept | val_p4 0.810 > 0.769 + 0.01; dev 0.880 vs 0.805 passes. Twin test: R1 had read the twins at 0.80 but real at 0.73 with peaks 21 % below the twin; the prior, not the network, was the gap (twin_r1 not shipped). | 0.725 | 0.707 | 0.747 | 0.88 | 12.8 | 0.877 | 0.854 | 14.3 | 0.865 |
| R4 | Quantile head: five quantiles of sqrt(SWE) with a pinball loss instead of a Gaussian, so the many snow-free days stop dragging the median melt-out late (same p3 bank; fine-tuned from R3) | 0.822 | ● kept | val_p4 0.822 > 0.810 + 0.01; dev 0.882 vs 0.880 passes. R3 had melted out 10 days after its twins (twin_r3 not shipped). | 0.735 | 0.73 | 0.778 | 0.882 | 8.0 | 0.839 | 0.886 | 8.3 | 0.836 |
| R5 | Physics p4: the dev prior used as fitted (inflation 1.3 -> 1.0), because R4's dev bands were too wide (68 % bands held 84 % of snow days, 93 % of melt-outs) and its peaks 11 % low; new frozen set val_p4; fine-tuned from R4 | 0.831 | ○ rejected | val_p4 0.831 not > 0.822 + 0.01 (dev 0.887 would have passed). On val_p3 it would have been kept (0.793 vs 0.778). | 0.7 | 0.695 | 0.793 | 0.887 | 8.3 | 0.758 | 0.883 | 8.5 | 0.772 |
| R6 | More simulated data: a second p4 bank with a new seed (48,000 windows in all), 6 more epochs, initialised from the rejected R5's weights | 0.839 | **◆ released** | val_p4 0.839 > 0.822 + 0.01 over the incumbent R4; dev 0.888 vs 0.882 passes. Released as v1.0. Rule note: initialised from R5, which the ratchet had rejected; on val_p3 it would have fallen short (0.785 vs 0.778). | 0.726 | 0.725 | 0.785 | 0.888 | 8.0 | 0.763 | 0.888 | 8.3 | 0.777 |

<sub>Columns after *Why*: Frozen simulated sets val_p1..val_p3 (val_p4 is `sim`); dev (38 dev stations, WY2011-2025, median NSE, melt-out MAE, 68 % daily coverage); then the real exam (621 stations, WY2011-2025) scored for every rung after the ladder was finished (training/exam_by_rung.json): after-the-fact columns, never used to choose.</sub>

Every rung was re-scored on every frozen set; the decisions were re-taken in order on val_p4. val_p3 existed before R2 was decided, so R2 was first decided on it; under the re-take rule everything now stands on val_p4. Near the margin the sets disagree: on val_p3 R5 would have been kept (+0.015 over R4) and R6 rejected (+0.007 over R4); on val_p4 R4 itself passes R3 by only 0.002 above the margin. Rule notes (18 Nov 2026, from the build handoff): the dev prior (p3, p4) was fitted on the dev stations' own SWE up to WY2010 and the gate uses the same stations in WY2011-2025, so the gate is partly in-sample; the exam was untouched. R6 was initialised from the rejected R5's weights (only the decision compares against the incumbent R4). The release demo was run on three exam stations before the final exam, to test code only; nothing was chosen on it. One training process peaked at about 1.34 GB, above the 1.2 GB sandbox guidance. The exam columns were filled in once, after the ladder was finished: R4, R5 and R6 lie within 0.005 exam NSE, and R5 and R6 mainly narrowed the bars (daily 68 % coverage 84 % -> 78 %).

Raw per-rung logs are in [`training/`](training/).

## Physics

Point snowpack mass and energy balance. WaterSim's two snow routines run snow-only at the station (one band, no runoff model): the HBV-light degree-day routine (rain/snow partition over TT +- TTI/2, snowfall correction, melt CFMAX (T - TT), refreeze, liquid water holding) and a daily single-layer energy balance after Walter et al. (2005) and the Utah Energy Balance model (net shortwave with an ageing albedo, Brutsaert long-wave with a cloud correction, bulk-transfer sensible and latent heat with a stability reduction, ground and rain heat, cold content, refreeze, sublimation). Radiation comes from the daily temperature range (Hargreaves-Samani).

**Simulator**

- WaterSim v1 (shared/water/watersim), model='none', one band at the station; each simulated station-winter uses one routine, 50/50
- Weather: real daily P and T of the dev SNOTEL stations, water years up to 2010 only; six dev stations held out for the frozen simulated sets. Climate shifts P x e^N(0, 0.3), T + N(0, 1.5) deg C with elevation moved by the lapse rate, latitude +- 1 deg
- Snow parameters (p3, p4): a multivariate normal fitted to per-station calibrations of both routines on 36 dev stations' own SWE, water years up to 2010 (training/prior_p4.json; means TT 0.99 C, CFMAX 3.6 mm/C/d, SFCF 1.01 for degree-day; TT 0.91 C, ALB_DECAY 2.4 d, TURB 0.17, GHEAT 0.32 MJ/m2/d for energy balance)
- Plausibility filter: simulated windows with more than 5 mm of SWE on 1 August are dropped (cold-shifted runs grew perennial packs)

**Checked against independent references**

- Degree-day routine against a 5-day hand calculation: 0.0 mm; identical to TUWmodel 1.1-1 to <= 1e-12 mm (shared/water/checks/analytic.json, reference_models.json)
- Energy-balance terms exact against hand arithmetic; mass closes to 6e-15 mm (shared/water/checks/analytic.json)
- Against SNOTEL pillows at 33 dev stations, WY2011-2025 (shared/water/checks/snotel.json): degree-day NSE 0.700 with default parameters and 0.892 calibrated per station; energy balance 0.824 default and 0.930 calibrated
- Twin test, 38 dev stations, WY2011-2025 (training/twin_r6.json): each station's calibrated energy-balance twin scores NSE 0.917 against its pillow; the released network reads the twin at 0.914 and the pillow at 0.888

## Help wanted

- The SNOTEL daily series (data/proc/snotel) are not in the repository; rebuild them with `mlratchet data water` (shared/water/data/scripts/fetch_snotel.py, build_snotel.py) to re-score the exam per station and run the dev-tuned physics baseline.

## Next rungs

- Run the dev-tuned energy balance (prior_p4.json means) on the 621 exam stations once the SNOTEL series are rebuilt, and pair it station by station with SnowSense (the decisive fair comparison).
- Peak under-catch: a wind- or exposure-dependent snowfall-correction (SFCF) prior, or a drift term.
- Rain-on-snow and thin packs below 1,500 m: sub-daily melt-refreeze, or rain energy in the degree-day routine.
- A gridded ERA5-Land band mode so SnowSense can feed FlowCast (the SnowSense -> FlowCast test was not done).
- An initial-snow input for stations that keep snow through summer; checks across the SNOTEL temperature-sensor change; more Alaska stations.

## Files

| Path | What |
|---|---|
| `snowsense.py`, `pipeline.py` | Inference — numpy/scipy only |
| `snowsense_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [USDA NRCS SNOTEL daily SWE, precipitation and air temperature](https://wcc.sc.egov.usda.gov/reportGenerator/) | noise and physics (dev stations' weather up to WY2010 drives the simulations), dev (36 dev stations' SWE up to WY2010 for the parameter prior; 38 dev stations WY2011-2025 for the gate, the twin test and the widen factors), exam (621 exam stations WY2011-2025; examples/ holds three of them, WY2016-2025) | US public domain (USDA NRCS National Water and Climate Center) |
| [WaterData snow baselines (climatology and per-station calibrated degree-day, Watershed Phase 1)](https://github.com/dogum/ML-Ratchet/tree/main/shared/water/waterdata) | exam (baselines, data/proc/baselines/snow_scores.parquet) | Derived from US public-domain SNOTEL data; code MIT |
| [WaterSim v1 snow routines (HBV-light after Bergstrom 1976 and Seibert & Vis 2012; energy balance after Walter et al. 2005 and Tarboton & Luce 1996; Hargreaves & Samani 1985; Brutsaert 1975)](https://github.com/dogum/ML-Ratchet/tree/main/shared/water/watersim) | physics (simulator and the uncalibrated exam baselines) | MIT (code in this repository) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
