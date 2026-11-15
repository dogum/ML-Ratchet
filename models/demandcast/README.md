<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# DemandCast <sub>v1.0</sub>

> **Tomorrow's hourly electricity demand, wind and solar for a US grid, with error bars.**

*How much electricity will this grid use tomorrow, hour by hour, and how much will its wind and solar farms make?*

DemandCast forecasts the next day's 24 hourly values of electricity demand, wind output and solar output for 8 US balancing authorities (ERCOT, CAISO, PJM, MISO, SPP, ISO-NE, NYISO, BPA), each with a 95% band. The forecast is issued at 10:00 local standard time the day before, like a grid operator's own. Three small networks correct a building-stock heat-transfer model of demand and wind- and solar-farm physics on the real fleets, reading the last week of metered output against the physics. They learned only from 182,500 simulated region-days. On 2023-2025 with the weather that actually happened (an upper bound) demand is 2.38% off against 3.41% for the grids' own forecasts; with real archived forecast weather (2024, as released) it is 2.92% against 3.20% and wins in 4 of 8 grids; a second look over 2024-2025 gives 3.05% against 3.59%, but 2.92% against 2.70% without CAISO and SPP, whose published forecasts are flawed. Solar loses to persistence.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/demandcast/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/demandcast/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Weather for the target day and the past week (population-weighted temperature, humidity, irradiance; hub-height wind, temperature, pressure and humidity at the wind points; irradiance at the solar points), the calendar (DST-aware hour, weekday / Saturday / Sunday-holiday), the grid's dev-fitted building-stock parameters and real EIA-860 wind and solar fleet, and the last 168 h of metered EIA-930 demand, wind and solar before issue |
| **Returns** | Hourly demand (MW) for the 24 hours of the next day, with a log-scale sigma (widened × 1.14 on dev) and a 95% band<br>Hourly wind and solar output as a capacity factor of the fleet online at issue (MW = CF × fleet MW), with sigma (wind × 1.00, solar × 1.34 on dev) and a 95% band; solar is 0 at night |
| **Learned from** | Demand is a building-stock thermal model (training/buildings.py): conductance × the stock-averaged hinge of a Normal thermostat spread around an envelope-lagged sol-air temperature (fast and slow RC response, internal and window gains, night setback), latent cooling load, resistance / heat-pump / non-electric heating and air conditioning with COPs that fall with outdoor temperature, occupancy schedules from the calendar, daylight lighting, a water-heating season term and rooftop PV (pvlib). Wind and solar are GridSim's farm physics on every EIA-860 farm with its real online dates: moist-air density, IEC density-equivalent speed, the pitch-regulated farm curve from specific power, losses, and pvlib with tracking and clipping per weather point. |
| **Noise** | Real residuals of building-model fits to each grid's demand in the training-input years 2018, 2020 and 2022 (whole days, local-hour aligned); wind and solar weather-error widths sized to the dev-year physics mismatch |
| **Network** | Three MLPs (two hidden layers of 512, SiLU), inputs standardised: demand 329 inputs / 456k parameters, wind 164 / 372k, solar 156 / 368k; Gaussian negative log-likelihood, AdamW, one-cycle LR, 40 epochs; numpy equals torch to 3e-7 |
| **Physics also enters** | inputs: Building-stock demand components (base, lighting, heating, cooling, rooftop PV) for tomorrow and the past week; wind and solar fleet physics with its spread at ±10% wind and ±20% irradiance<br>output: Demand: predicts log(demand / building-stock physics); wind and solar: a correction added to the physics capacity factor |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 2019 and 2021 for all 8 grids (GridData's frozen split, content sha256 895b4e9a…, seed 20261113): building and wind/solar calibration, sigma widening, and the rung gate (30 seeded days per grid). Each dev year is described by the other dev year's fit. Training-input years 2018, 2020 and 2022 give only weather and residual noise. |
| **Exam** | Every day of 2023-2025 for the 8 grids: 209,340 demand hours with actual weather; 66,079 hours of 2024 with Open-Meteo forecast weather. Run once on the final model, never used for choosing. Second look with forecast weather, 2024-2025 (135,700 hours), run once on 14 Nov 2026 from the shipped weights; it chose nothing. |

## Run it

```bash
cd models/demandcast
pip install numpy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo demandcast`.

```python
import numpy as np
from demandcast import DemandCast
E = dict(np.load("examples/exam_days.npz"))        # 16 real exam days, inputs as the pipeline built them
m = DemandCast()
d = m.predict_demand(E["xd"], E["pd_total"])       # hourly MW, log sigma, 95% band
w = m.predict_cf("w", E["xw"], E["pw"])            # wind capacity factor, 95% band
print(E["ba"][0], E["day"][0], d["mw"][0].round(), E["d_true"][0])
```

## The exam

Every day of 2023-2025, forecast at 10:00 local standard time the day before for the 24 hours of the next day. Demand error is the mean absolute percentage error over hours where every method has a value; wind and solar error is total absolute error over total output. The headline uses the weather that actually happened, an upper bound: the grids' own forecasts had to use forecast weather. A second run uses real archived day-ahead forecasts, reachable only for 2024 before the archive's rate limit. Raw results: training/exam_r8_wide.json. Re-scored in the repository from the shipped weights (training/rescore_release.py): every actual-weather number reproduced; with the archive fetched in full (14 Nov 2026), every forecast-weather number of the released 2024 row reproduced too (training/rescore_forecast.py release2024, 338 numbers). A second look with forecast weather over 2024-2025 was then run once (rescore_forecast.py full) and is reported next to the released row; 2023 cannot be graded this way because the archive holds no humidity or radiation forecasts before 2024. Every number here is on real data that never chose anything.

| Metric | DemandCast | Baselines | |
|---|---|---|---|
| Demand, 8 grids pooled, actual weather (209,340 h)<br><sub>Upper bound: actual weather. Bias 0.0%. Beats the grid's own forecast in 6 of 8 grids.</sub> | **2.38%** | Grid's own day-ahead forecast (published): 3.41%<br>Physics alone (building model) (classical): 5.02%<br>Persistence (same hour yesterday) (dumb baseline): 4.9%<br>Same hour last week (dumb baseline): 7.95% | ✅ wins |
| Demand, 8 grids pooled, real forecast weather (2024, 66,079 h)<br><sub>Open-Meteo previous-runs day-1 forecasts at the top population points. Same days with actual weather: 2.44%. The pooled win rests on CAISO, whose published forecast is misaligned by 1 h (3.73 vs 7.00).</sub> | **2.92%** | Grid's own day-ahead forecast (published): 3.2%<br>Physics alone (building model) (classical): 5.12%<br>Persistence (same hour yesterday) (dumb baseline): 4.79%<br>Same hour last week (dumb baseline): 8.12% | ✅ wins |
| Grids where demand beats the grid's own forecast, real forecast weather (2024)<br><sub>Wins CAISO 3.73 vs 7.00, PJM 2.84 vs 3.57, MISO 2.15 vs 2.80, NYISO 2.57 vs 3.28. Loses ERCOT 2.98 vs 2.05, SPP 2.74 vs 2.22, ISO-NE 3.93 vs 2.79, BPA 2.44 vs 1.92.</sub> | **4 of 8** | — | ↔️ mixed |
| Demand, 8 grids pooled, real forecast weather, 2024-2025 (second look, run once on 14 Nov 2026; 135,700 h)<br><sub>Open-Meteo previous-runs day-1 forecasts at the shipped fetch list's points (top 4 population, 3 wind, 3 solar per grid; the released 2024 row used up to 8 population points). 2024 2.97% vs 3.20%, 2025 3.13% vs 3.97%. Same days with actual weather: 2.48%. Bands 71 / 93%. The pooled win rests on CAISO (misaligned by 1 h) and SPP (6.5% high in 2025): without them see the next row.</sub> | **3.05%** | Grid's own day-ahead forecast (published): 3.59%<br>Physics alone (building model) (classical): 5.88%<br>Persistence (same hour yesterday) (dumb baseline): 4.91%<br>Same hour last week (dumb baseline): 8.16% | ✅ wins |
| Demand, 6 grids pooled without CAISO and SPP, real forecast weather, 2024-2025 (second look; 101,874 h)<br><sub>The grids whose published forecasts have no known defect: their own forecasts beat DemandCast with forecast weather.</sub> | **2.92%** | Grid's own day-ahead forecast (published): 2.7% | ❌ loses |
| Grids where demand beats the grid's own forecast, real forecast weather, 2024-2025 (second look)<br><sub>Wins CAISO 3.92 vs 7.58, PJM 2.71 vs 3.50, MISO 2.15 vs 2.58, SPP 2.96 vs 5.01, NYISO 2.84 vs 3.24. Loses ERCOT 3.33 vs 2.26, ISO-NE 4.18 vs 2.65, BPA 2.33 vs 1.96.</sub> | **5 of 8** | — | ↔️ mixed |
| Wind, 8 grids pooled, real forecast weather, 2024-2025 (second look; 131,024 h)<br><sub>Same days with actual weather: 19.2%. Bands 65 / 91%. BPA 61.1%, ISO-NE 38.6%, CAISO 37.2%; ERCOT 26.0%, MISO 23.5%, SPP 23.5%.</sub> | **26.7%** | Physics alone (real fleet, dev scale) (classical): 39%<br>Persistence (same hour yesterday) (dumb baseline): 42.9%<br>Dev-year CF climatology × fleet (dumb baseline): 40.6% | ✅ wins |
| Solar, 7 grids pooled, real forecast weather, 2024-2025 (second look; 119,348 h)<br><sub>Same days with actual weather: 32.5%. CAISO 72.2% (persistence 10.5%), BPA 51.7%, PJM 36.1%; ERCOT 11.6%, MISO 13.4%.</sub> | **34.7%** | Physics alone (real fleet, dev scale) (classical): 39.2%<br>Persistence (same hour yesterday) (dumb baseline): 18.9%<br>Dev-year CF climatology × fleet (dumb baseline): 41.1% | ❌ loses |
| Demand ERCOT, actual weather<br><sub>Upper bound. By year 2.14 / 2.14 / 2.47 (grid 2.60 / 2.16 / 2.46). With forecast weather (2024): 2.98 vs 2.05, a loss. Bands 68 / 92%.</sub> | **2.25%** | Grid's own day-ahead forecast (published): 2.4%<br>Physics alone (classical): 6.04%<br>Persistence (dumb baseline): 4.56%<br>Same hour last week (dumb baseline): 8.36% | ✅ wins |
| Demand CAISO, actual weather<br><sub>Upper bound. CAISO's published forecast is misaligned by 1 h in EIA-930, which flatters DemandCast. Bands too narrow: 46 / 74%.</sub> | **3.37%** | Grid's own day-ahead forecast (published): 6.86%<br>Physics alone (classical): 6%<br>Persistence (dumb baseline): 4.9%<br>Same hour last week (dumb baseline): 6.46% | ✅ wins |
| Demand PJM, actual weather<br><sub>Upper bound. With forecast weather (2024): 2.84 vs 3.57, still a win.</sub> | **2.1%** | Grid's own day-ahead forecast (published): 3.49%<br>Physics alone (classical): 3.24%<br>Persistence (dumb baseline): 4.95%<br>Same hour last week (dumb baseline): 8.32% | ✅ wins |
| Demand MISO, actual weather<br><sub>Upper bound. With forecast weather (2024): 2.15 vs 2.80, still a win.</sub> | **1.88%** | Grid's own day-ahead forecast (published): 2.71%<br>Physics alone (classical): 2.98%<br>Persistence (dumb baseline): 4.17%<br>Same hour last week (dumb baseline): 6.98% | ✅ wins |
| Demand SPP, actual weather<br><sub>Upper bound. SPP's own forecast ran 6.5% high in 2025 (7.68%); in 2024 it beat DemandCast (2.32 vs 2.54), and with forecast weather 2.22 vs 2.74.</sub> | **2.41%** | Grid's own day-ahead forecast (published): 4.31%<br>Physics alone (classical): 5.64%<br>Persistence (dumb baseline): 4.32%<br>Same hour last week (dumb baseline): 8.21% | ✅ wins |
| Demand ISO-NE, actual weather<br><sub>Loses every year even with actual weather (2.77 / 3.13 / 3.10 vs 2.40 / 2.77 / 2.52).</sub> | **3%** | Grid's own day-ahead forecast (published): 2.56%<br>Physics alone (classical): 5.51%<br>Persistence (dumb baseline): 7.27%<br>Same hour last week (dumb baseline): 10.64% | ❌ loses |
| Demand NYISO, actual weather<br><sub>Upper bound. With forecast weather (2024): 2.57 vs 3.28, still a win.</sub> | **1.97%** | Grid's own day-ahead forecast (published): 2.97%<br>Physics alone (classical): 5.16%<br>Persistence (dumb baseline): 5.34%<br>Same hour last week (dumb baseline): 8.22% | ✅ wins |
| Demand BPA, actual weather<br><sub>With forecast weather (2024): 2.44 vs 1.92.</sub> | **2.1%** | Grid's own day-ahead forecast (published): 2%<br>Physics alone (classical): 5.58%<br>Persistence (dumb baseline): 3.69%<br>Same hour last week (dumb baseline): 6.42% | ❌ loses |
| Wind, 8 grids pooled, actual weather (Σ\|err\| / Σ output)<br><sub>BPA 50.1% and CAISO 32.7% (50 km reanalysis in complex terrain); ERCOT 17.8%, MISO 16.0%. With forecast weather (2024, 7,632 h, ERCOT only): 31.2% vs physics 47.9%, persistence 42.6%.</sub> | **19.2%** | Physics alone (real fleet, dev scale) (classical): 22.7%<br>Persistence (same hour yesterday) (dumb baseline): 43.5%<br>Dev-year CF climatology × fleet (dumb baseline): 41.4% | ✅ wins |
| Solar, 7 grids pooled, actual weather (Σ\|err\| / Σ output)<br><sub>EIA-930 solar timestamps moved between dev and exam (CAISO -1 h in 2019/2021, 0 in 2023, +1 h in 2024-25; PJM in 2025), so the dev-calibrated shift is wrong in the exam: CAISO 59.4% (persistence 11.4%), BPA 51.8% (30.5%). Good in ERCOT 10.6%, MISO 10.4%, ISO-NE 16.4%. NYISO reports no solar.</sub> | **29.9%** | Physics alone (real fleet, dev scale) (classical): 36.3%<br>Persistence (same hour yesterday) (dumb baseline): 19.1%<br>Dev-year CF climatology × fleet (dumb baseline): 39.2% | ❌ loses |
| Texas heat, Aug 2023 (ERCOT, 744 h)<br><sub>Peak over-predicted: 88.5 GW vs 85.4 GW metered.</sub> | **1.96%** | Grid's own day-ahead forecast (published): 1.59%<br>Physics alone (classical): 3.03% | ❌ loses |
| Eastern heat dome, Jun 2024 (PJM)<br><sub>MISO 1.56 vs 4.01, NYISO 1.72 vs 2.73, ISO-NE 2.97 vs 3.38 in the same week.</sub> | **1.98%** | Grid's own day-ahead forecast (published): 4.5%<br>Physics alone (classical): 2.32% | ✅ wins |
| Arctic outbreak, Jan 2025 (ISO-NE)<br><sub>PJM in the same outbreak: 1.83 vs 4.17.</sub> | **4.78%** | Grid's own day-ahead forecast (published): 1.27%<br>Physics alone (classical): 8.5% | ❌ loses |
| Winter Storm Uri, Feb 2021 (ERCOT; a dev year, not exam)<br><sub>DemandCast forecast about 85 GW of demand; ERCOT served 69 GW under load shedding, so the metered 'truth' is curtailed.</sub> | **14.5%** | Grid's own day-ahead forecast (published): 18.6%<br>Physics alone (classical): 21.1% | ℹ️ |
| Demand error by temperature band, actual weather (7 bands, -10 to 50 °C)<br><sub>Flat across temperature; above 35 °C (1,211 h) 2.73% vs the grids' 2.55%. Below -10 °C (913 h) it rises to 3.25% (grids 3.33%).</sub> | **2.2-2.7%** | Grid's own day-ahead forecast (published): 2.6-3.9% | ℹ️ |

Coverage of the 68% error bars on the exam: Demand, pooled (95% band: 89%) **64%**, Demand, CAISO (95%: 74%) **46%**, Demand, forecast weather 2024 (95%: 94%) **72%**, Demand, forecast weather 2024-2025, second look (95%: 93%) **71%**, Wind, pooled (95%: 94%) **71%**, Solar, daylight hours (95%: 84%) **65%**.

### Where it wins

- Demand with actual weather: 2.38% against 3.41% for the grids' own day-ahead forecasts, and ahead in 6 of 8 grids (an upper bound).
- With real forecast weather (2024): 2.92% against 3.20% pooled, and ahead in CAISO, PJM, MISO and NYISO.
- Second look with real forecast weather, 2024-2025 (run once, 14 Nov 2026): wind 26.7% against 39.0% for the physics alone and 42.9% for persistence.
- Wind: 19.2% against 22.7% for the physics alone and 43.5% for persistence.
- Heat waves and cold snaps the grids missed in PJM: heat dome Jun 2024 1.98% vs 4.50%; Arctic outbreak Jan 2025 1.83% vs 4.17%. From -10 °C to 35 °C the error stays at 2.2-2.7% (grids 2.9-3.9%); below -10 °C (913 h) 3.25% vs 3.33%.

### Where it loses

- With real forecast weather (2024) it loses to ERCOT, SPP, ISO-NE and BPA's own forecasts (ERCOT 2.98% vs 2.05%).
- Second look with real forecast weather, 2024-2025: 3.05% vs 3.59% pooled, but without CAISO and SPP (flawed published forecasts) 2.92% vs the grids' 2.70%; behind ERCOT (3.33% vs 2.26%), ISO-NE (4.18% vs 2.65%) and BPA (2.33% vs 1.96%). Solar 34.7% vs persistence 18.9%.
- Solar loses to persistence: 29.9% vs 19.1%; CAISO 59.4% and BPA 51.8%, because EIA-930's solar timestamps moved by an hour between dev and exam.
- ISO-NE (3.00% vs 2.56%) and BPA (2.10% vs 2.00%) lose even with actual weather; ISO-NE's Jan 2025 cold snap 4.78% vs 1.27%.
- Texas heat, Aug 2023: 1.96% vs ERCOT's 1.59%, with the peak 3% too high. Above 35 °C (1,211 h) 2.73% vs the grids' 2.55%.

## Limits

- The headline uses actual weather and is an upper bound. With real forecast weather DemandCast wins in 4 of 8 grids in 2024 (released) and 5 of 8 over 2024-2025 (second look); the pooled win rests on CAISO and SPP. The archive holds no humidity or radiation forecasts before 2024, so 2023 cannot be graded with forecast weather.
- Solar fails where EIA-930 timestamps changed (CAISO, PJM in 2025); the dev-calibrated 1 h shift is wrong in the exam. BPA's tiny solar fleet is badly served.
- Bands are too narrow on real data: demand 64% / 89% inside the 68% / 95% bands (CAISO 46% / 74%).
- Wind is limited by 50 km reanalysis in complex terrain (BPA 50%, CAISO 33%); the network trims the physics error by about 15%.
- It needs the last week of metered output: physics alone is about 5% off (load growth since 2019-21). Building parameters are effective (several fits sit on bounds), so the heating / cooling split is indicative.
- Over-predicts heat-wave peaks by about 3%.

## The ratchet

**Objective:** Mean over demand, wind and solar of skill = 1 - error / physics-alone error on the frozen simulated validation set val_v3 (4,000 region-days)  
**Rule:** Keep a rung if the objective beats the incumbent by more than 0.01 AND the dev gate's mean relative error gets no more than 2% worse

| Rung | Change | Frozen-set score | Decision | Why | Dev demand MAPE (%) | Dev wind (%) | Dev solar (%) | Dev demand 68% coverage |
|---|---|---|---|---|---|---|---|---|
| r1_base | Three MLPs (256-256), 39k region-days, simulator v1 (first-guess drift and error widths) | 0.333 | ● kept | First rung. | 2.06 | 21.39 | 15.82 | 0.665 |
| r2_simv2 | Simulator v2: drift, residual noise and wind/solar error widths calibrated so simulated physics errors match dev; half the solar fleets report near full capacity | 0.339 | ○ rejected | Objective 0.3385 vs 0.3330 (needs +0.01); dev errors +10.1% (solar broke on real data: night-time negative EIA output against near-constant simulated features). | 1.98 | 21.46 | 21.14 | 0.677 |
| r3_robust | Rung 2 + robust inputs: negative night output clamped to 0, wind/solar input scales floored at 0.02 | 0.339 | ○ rejected | Objective 0.3393 vs 0.3330, +0.006 (needs +0.01); dev errors -2.4%. | 1.98 | 21.35 | 15.28 | 0.677 |
| r4_more | Rung 3 inputs on both simulator banks (v2 + v1, 79k region-days), 384-384 | 0.348 | ● kept | Objective 0.3484 vs 0.3330; dev errors -2.3%. | 1.98 | 21.36 | 15.38 | 0.737 |
| r5_simv3 | Simulator v3 (the twin test found residual noise missing in ~60% of simulated hours), 52k region-days, 384-384 | 0.348 | ○ rejected | Objective 0.3483 vs 0.3484, no gain; dev errors +0.2%. Its v3 banks were kept for later rungs. | 1.99 | 21.84 | 15.01 | 0.648 |
| r6_all | All four banks (v3 ×2, v2, v1; 131k region-days), 384-384 | 0.356 | ○ rejected | Objective 0.3558 vs 0.3484, +0.007 (needs +0.01); dev errors -0.1%. | 1.95 | 21.45 | 15.52 | 0.691 |
| r7_hist | Full 168 h history of real-vs-physics error as extra inputs, 102k v3 region-days, 384-384 | 0.355 | ○ rejected | Objective 0.3552 vs 0.3484 (needs +0.01); dev errors +8.3% (gate +2%), demand 2.27% and its 68% coverage fell to 45%. | 2.27 | 22.43 | 16.17 | 0.451 |
| r8_wide | All six banks (v3 ×4, v2, v1; 180k region-days), base features, 512-512 (largest within the 500k-parameter budget) | 0.362 | **◆ released** | Objective 0.3617 vs 0.3484; dev errors +0.9% (inside the 2% gate). Released as v1.0. | 2.0 | 21.44 | 15.6 | 0.651 |

<sub>Columns after *Why*: Dev gate errors (5,760 demand hours, 5,759 wind, 5,040 solar), never the exam.</sub>

The first rungs were scored on earlier frozen sets when taken (r4 scored 0.408 on val_v2); every rung was re-scored on val_v3 after the simulator changed, and those are the numbers here (training/ratchet.json). Dev calibration steps outside the network: per-grid building fits on 2019/2021, one wind and solar scale per grid, the CAISO and PJM 1 h solar shift, simulator v2 widths matched to dev, the v3 noise fix from the twin test.

Raw per-rung logs are in [`training/`](training/).

## Physics

Demand is a building-stock thermal model (training/buildings.py): conductance × the stock-averaged hinge of a Normal thermostat spread around an envelope-lagged sol-air temperature (fast and slow RC response, internal and window gains, night setback), latent cooling load, resistance / heat-pump / non-electric heating and air conditioning with COPs that fall with outdoor temperature, occupancy schedules from the calendar, daylight lighting, a water-heating season term and rooftop PV (pvlib). Wind and solar are GridSim's farm physics on every EIA-860 farm with its real online dates: moist-air density, IEC density-equivalent speed, the pitch-regulated farm curve from specific power, losses, and pvlib with tracking and clipping per weather point.

**Simulator**

- 182,500 simulated region-days in 6 banks (simulator v1, v2 and four v3 banks), each region-year driven by one focus grid's real weather from a training-input year (2018, 2020, 2022), perturbed: temperature offset N(0, 2 K) and anomaly scale 0.85-1.2, humidity × 0.85-1.15, cloudiness × 0.85-1.1, wind × 0.9-1.1
- Random true building stocks (80% around one grid's dev fit, 20% broad uniform); the network is told descriptor parameters drifted from the truth, as a fit 2-4 years old would be
- Wind truth: reanalysis × exp(point bias + correlated AR(1) error); solar: a cloud error; random losses, specific-power and hub shifts, reporting scale and curtailment; the real fleet at a 2018-2022 date with random per-group scaling
- Half of the samples carry synthetic day-ahead forecast-weather errors (the forecast-weather flag input)

**Checked against independent references**

- Envelope RC lag vs an ODE solver 3.2e-7 K; step response vs closed form 3.6e-15 K (training/checks/physics.json)
- Stock-averaged thermostat hinge vs a 1e6-draw Monte Carlo 0.0035 K; degree-day limit exact (1,080,000 vs 1,080,000 MWh)
- Specific humidity vs PsychroLib (ASHRAE formulas) 0.2% (T >= 0 °C); heat-pump COP 3.26 at 8.3 °C and 1.94 at -8.3 °C, AC COP 3.5 at 35 °C (typical rating points)
- Tabulated wind farm curve vs GridSim's farm_curve: mean 0.0005 CF (max 0.038 away from the 25 m/s cut-out); GridSim wind and solar vs ERCOT EIA-930 (phase 1): hourly r 0.90 and 0.97-0.98
- Dev calibration (2019, 2021): building model fitted per grid, in-sample 2.2-3.5% MAPE; twin test re-simulated each grid's dev year with the full simulator (training/checks/twin_r8_wide.json)

## Help wanted

- Day-ahead forecast weather before 2024. Open-Meteo's Previous Runs API rate-limited the first build (HTTP 429 after 1-25 year-requests per half hour); on 14 Nov 2026 the shared build machine fetched all 220 year-requests with no refusal. The archive holds humidity, radiation and wind forecasts only from 2024 (temperature from 2022), so 2023 cannot be graded with forecast weather and the dev years 2019 and 2021 have no real forecast errors. An archive of day-ahead forecasts for 2019-2023 is wanted.

## Next rungs

- Simulate random 0 / ±1 h reporting lags so the network reads the timestamp convention from the week of history, instead of a fixed dev shift; solar should then stop failing in CAISO and PJM.
- Real forecast-error noise in training: the 2022 Open-Meteo temperature library is now fetched and built (training/make_tfc.py; day-ahead error 1.3-2.0 K RMS per grid). A rung that trains on it would be judged on dev.
- Better bands (64% inside the 68% band): per-grid noise-scale variety in the simulator, or a dev-fitted conformal widening.
- Higher-resolution weather for wind in complex terrain (BPA, CAISO).
- Upstream to GridSim: the building-stock demand model (1.7-3.0% cross-year on dev, replacing GridSim's demand regression) and the tabulated farm curve with fleet grouping; to GridData: flags for EIA-930 solar timestamp changes and negative night solar, and an archived forecast-weather table.

## Files

| Path | What |
|---|---|
| `demandcast.py`, `pipeline.py` | Inference — numpy/scipy only |
| `demandcast_weights.npz`, `demandcast_meta.json`, `calib/demand_dev.json`, `calib/renew_dev.json` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [EIA-930 Hourly Electric Grid Monitor (demand, day-ahead demand forecast, generation by fuel), via GridData](https://www.eia.gov/electricity/gridmonitor/) | noise (training-input years), dev (2019, 2021), exam (2023-2025), truth; also the input week before each forecast | Public domain (U.S. Government) |
| [EIA-860 / EIA-923 generator fleets (wind and solar farms, dates, hub heights, tracking), via Catalyst Cooperative PUDL](https://catalyst.coop/pudl/) | physics (real fleets) | EIA data public domain; PUDL CC BY 4.0 |
| [NASA POWER hourly (MERRA-2 meteorology, CERES irradiance) at GridData's weather points](https://power.larc.nasa.gov/) | physics (weather) | Free use, cite the NASA LaRC POWER Project |
| [ASOS station weather via the Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml) | physics (population-weighted temperature and humidity) | Free use with credit |
| [Open-Meteo Previous Runs API (archived day-ahead forecasts, day-1 lead)](https://open-meteo.com/en/docs/previous-runs-api) | exam (forecast-weather mode: 2024 as released; 2024-2025 second look, 14 Nov 2026); noise (2022 temperature forecast errors, not yet used in training) | CC BY 4.0 (free non-commercial use) |
| [EIA-861 service territories and US Census population (weather-point weights), via GridData](https://www.eia.gov/electricity/data/eia861/) | physics (weights) | Public domain (U.S. Government) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
