<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# DemandCast <sub>v1.2</sub>

> **Tomorrow's hourly electricity demand, wind and solar for a US grid, with error bars.**

*How much electricity will this grid use tomorrow, hour by hour, and how much will its wind and solar farms make?*

DemandCast forecasts the next day's 24 hourly values of electricity demand, wind output and solar output for 8 US balancing authorities (ERCOT, CAISO, PJM, MISO, SPP, ISO-NE, NYISO, BPA), each with a 95% band. The forecast is issued at 10:00 local standard time the day before, like a grid operator's own. Three small networks correct a building-stock heat-transfer model of demand and wind- and solar-farm physics on the real fleets, reading the last week of metered output against the physics. They learned only from 180,000 simulated region-days. v1.2 (27 Nov 2026) repairs inputs the simulator never varied (twilight solar, missing demand hours, 2 h solar offsets): solar is now 10.6% off against 19.1% for persistence (v1.1: 16.1%), but demand got worse. On 2023-2025 with the weather that actually happened (an upper bound) demand is 2.49% off against 3.41% for the grids' own forecasts (v1.1 2.38%); with real archived forecast weather 2024-2025 it is 3.14% against 3.59%, but 2.98% against 2.70% without CAISO and SPP, whose published forecasts are flawed.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/demandcast/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/demandcast/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Weather for the target day and the past week (population-weighted temperature, humidity, irradiance; hub-height wind, temperature, pressure and humidity at the wind points; irradiance at the solar points), the calendar (DST-aware hour, weekday / Saturday / Sunday-holiday), the grid's dev-fitted building-stock parameters and real EIA-860 wind and solar fleet, and the last 168 h of metered EIA-930 demand, wind and solar before issue |
| **Returns** | Hourly demand (MW) for the 24 hours of the next day, with a log-scale sigma (widened × 1.34 on dev) and a 95% band<br>Hourly wind and solar output as a capacity factor of the fleet online at issue (MW = CF × fleet MW), with sigma (wind × 1.00, solar × 1.12 on dev) and a 95% band; solar is 0 at night and is moved onto the reported clock (the reporting offset, -2 to +2 h, read at issue) |
| **Learned from** | Demand is a building-stock thermal model (training/buildings.py): conductance × the stock-averaged hinge of a Normal thermostat spread around an envelope-lagged sol-air temperature (fast and slow RC response, internal and window gains, night setback), latent cooling load, resistance / heat-pump / non-electric heating and air conditioning with COPs that fall with outdoor temperature, occupancy schedules from the calendar, daylight lighting, a water-heating season term and rooftop PV (pvlib). Wind and solar are GridSim's farm physics on every EIA-860 farm with its real online dates: moist-air density, IEC density-equivalent speed, the pitch-regulated farm curve from specific power, losses, and pvlib with tracking and clipping per weather point. |
| **Noise** | Real residuals of building-model fits to each grid's demand in the training-input years 2018, 2020 and 2022 (whole days, local-hour aligned); wind and solar weather-error widths sized to the dev-year physics mismatch |
| **Network** | Three MLPs (two hidden layers of 512, SiLU), inputs standardised: demand 329 inputs / 456k parameters, wind 164 / 372k, solar 156 / 368k; Gaussian negative log-likelihood, AdamW, one-cycle LR, 40 epochs; numpy equals torch to 3e-7. v1.1 adds no learned parameters: before the solar network, a two-term least-squares fit of the 168 h reported-minus-physics error on the physics diurnal profile and its slope reads the reporting offset (-1, 0 or +1 h, counted from each grid's dev-year value); the history is put back on the physics clock and the forecast is moved by the offset. v1.2 (rung r10, 27 Nov 2026): the same three architectures retrained on 180,000 region-days of simulator v5 (base features, 40 epochs); offsets read up to ±2 h; the reported solar of the last 24 h is set to 0 at hours the physics calls dark before the network reads it; count-like inputs (flags, missing-hour fractions) have a standardisation floor of 0.05; standardised inputs are clipped at \|z\| 8 |
| **Physics also enters** | inputs: Building-stock demand components (base, lighting, heating, cooling, rooftop PV) for tomorrow and the past week; wind and solar fleet physics with its spread at ±10% wind and ±20% irradiance<br>output: Demand: predicts log(demand / building-stock physics); wind and solar: a correction added to the physics capacity factor |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | 2019 and 2021 for all 8 grids (GridData's frozen split, content sha256 895b4e9a…, seed 20261113): building and wind/solar calibration, sigma widening, and the rung gate (30 seeded days per grid). Each dev year is described by the other dev year's fit. Training-input years 2018, 2020 and 2022 give only weather and residual noise. |
| **Exam** | Every day of 2023-2025 for the 8 grids: 209,340 demand hours with actual weather; 66,079 hours of 2024 with Open-Meteo forecast weather. Run once on the final model, never used for choosing. Second look with forecast weather, 2024-2025 (135,700 hours), run once on 14 Nov 2026 from the shipped weights; it chose nothing. Run once per version (v1.0 14 Nov, v1.1 15 Nov, v1.2 27 Nov 2026, each after its release choice was committed); the 27 Nov 2026 re-score of v1.1 with z_clip=8 (v1.1.1) was a dated supplement that changed no weights. |

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

Every day of 2023-2025, forecast at 10:00 local standard time the day before for the 24 hours of the next day. Demand error is the mean absolute percentage error over hours where every method has a value; wind and solar error is total absolute error over total output. The headline uses the weather that actually happened, an upper bound: the grids' own forecasts had to use forecast weather. A second run uses real archived day-ahead forecasts: 2024 as first released, and a second look over 2024-2025; 2023 cannot be graded this way because the archive holds no humidity or radiation forecasts before 2024. The rows are v1.2's (27 Nov 2026, the audit repair rung r10), run once after the ratchet promoted it and its weights were committed (training/repair.py exam; training/exam_v1_2.json holds v1.1, v1.1.1 and v1.2 side by side). Each note keeps v1.1 (the r8 network; its demand and wind are v1.0's first exam, training/exam_r8_wide.json; solar from 15 Nov 2026) and v1.1.1 (the same network with inputs clipped at |z| 8, a dated supplementary re-score of 27 Nov 2026, training/rescore_zclip.py). Every number here is on real data that never chose anything.

| Metric | DemandCast | Baselines | |
|---|---|---|---|
| Demand, 8 grids pooled, actual weather (209,340 h)<br><sub>Upper bound: actual weather. Bias +0.1%. Beats the grid's own forecast in 5 of 8 grids (ERCOT is now a narrow loss, 2.41 vs 2.40). v1.1: 2.38% (6 of 8); v1.1.1: 2.32% (6 of 8). v1.2 is worse on demand than v1.1.1 in every grid.</sub> | **2.49%** | Grid's own day-ahead forecast (published): 3.41%<br>Physics alone (building model) (classical): 5.02%<br>Persistence (same hour yesterday) (dumb baseline): 4.9%<br>Same hour last week (dumb baseline): 7.95% | ✅ wins |
| Demand, 8 grids pooled, real forecast weather (2024, 66,079 h)<br><sub>Open-Meteo previous-runs day-1 forecasts at the top population points (the release's 2024 cache). Same days with actual weather: 2.50%. The pooled win rests on CAISO, whose published forecast is misaligned by 1 h (3.75 vs 7.00). v1.1: 2.92%; v1.1.1: 2.83%.</sub> | **3%** | Grid's own day-ahead forecast (published): 3.2%<br>Physics alone (building model) (classical): 5.12%<br>Persistence (same hour yesterday) (dumb baseline): 4.79%<br>Same hour last week (dumb baseline): 8.12% | ✅ wins |
| Grids where demand beats the grid's own forecast, real forecast weather (2024)<br><sub>Wins CAISO 3.75 vs 7.00, PJM 2.46 vs 3.57, MISO 2.10 vs 2.80, NYISO 2.76 vs 3.28. Loses ERCOT 3.38 vs 2.05, SPP 3.02 vs 2.22, ISO-NE 4.01 vs 2.79, BPA 2.50 vs 1.92. v1.1 and v1.1.1: the same 4 of 8.</sub> | **4 of 8** | — | ↔️ mixed |
| Demand, 8 grids pooled, real forecast weather, 2024-2025 (second look; 135,700 h)<br><sub>Open-Meteo previous-runs day-1 forecasts at the shipped fetch list's points (top 4 population, 3 wind, 3 solar per grid). 2024 3.04% vs 3.20%, 2025 3.24% vs 3.97%. Same days with actual weather: 2.58%. Bands 70 / 92%. The pooled win rests on CAISO (misaligned by 1 h) and SPP (6.5% high in 2025): without them see the next row. v1.1: 3.05% (run once 14 Nov 2026); v1.1.1: 2.96%.</sub> | **3.14%** | Grid's own day-ahead forecast (published): 3.59%<br>Physics alone (building model) (classical): 5.88%<br>Persistence (same hour yesterday) (dumb baseline): 4.91%<br>Same hour last week (dumb baseline): 8.16% | ✅ wins |
| Demand, 6 grids pooled without CAISO and SPP, real forecast weather, 2024-2025 (second look; 101,874 h)<br><sub>The grids whose published forecasts have no known defect: their own forecasts beat DemandCast with forecast weather. v1.1: 2.92%; v1.1.1: 2.84%.</sub> | **2.98%** | Grid's own day-ahead forecast (published): 2.7% | ❌ loses |
| Grids where demand beats the grid's own forecast, real forecast weather, 2024-2025 (second look)<br><sub>Wins CAISO 4.01 vs 7.58, PJM 2.50 vs 3.50, MISO 2.11 vs 2.58, SPP 3.22 vs 5.01, NYISO 3.02 vs 3.24 (2025 alone 3.26 vs 3.21). Loses ERCOT 3.56 vs 2.26, ISO-NE 4.26 vs 2.65, BPA 2.43 vs 1.96. v1.1 and v1.1.1: the same 5 of 8.</sub> | **5 of 8** | — | ↔️ mixed |
| Wind, 8 grids pooled, real forecast weather, 2024-2025 (second look; 131,024 h)<br><sub>Same days with actual weather: 19.3%. Bands 65 / 91%. BPA 61.8%, ISO-NE 38.6%, CAISO 36.9%; ERCOT 26.3%, MISO 23.7%, SPP 23.6%. v1.1: 26.7%; v1.1.1: 26.7%.</sub> | **26.9%** | Physics alone (real fleet, dev scale) (classical): 39%<br>Persistence (same hour yesterday) (dumb baseline): 42.9%<br>Dev-year CF climatology × fleet (dumb baseline): 40.6% | ✅ wins |
| Solar, 7 grids pooled, real forecast weather, 2024-2025 (second look; 119,348 h)<br><sub>CAISO 12.1% (persistence 10.5%), PJM 11.5% (27.4%), BPA 21.7% (29.2%), SPP 28.1% (28.6%); ERCOT 11.7%, MISO 10.8%, ISO-NE 18.5%. Same days with actual weather: 10.5%. Bands 74 / 92%. v1.1: 18.2% (CAISO 28.3%, BPA 51.9%, SPP 44.4%), a 0.7-point win; v1.1.1: 18.1%; v1.0: 34.7%.</sub> | **12%** | Physics alone (real fleet, dev scale) (classical): 39.2%<br>Persistence (same hour yesterday) (dumb baseline): 18.9%<br>Dev-year CF climatology × fleet (dumb baseline): 41.1% | ✅ wins |
| Demand ERCOT, actual weather<br><sub>Upper bound. By year 2.32 / 2.41 / 2.50 (grid 2.60 / 2.16 / 2.46). With forecast weather (2024): 3.38 vs 2.05. Bands 69 / 93%. v1.1: 2.25 (a win); v1.1.1: 2.17.</sub> | **2.41%** | Grid's own day-ahead forecast (published): 2.4%<br>Physics alone (classical): 6.04%<br>Persistence (dumb baseline): 4.56%<br>Same hour last week (dumb baseline): 8.36% | ❌ loses |
| Demand CAISO, actual weather<br><sub>Upper bound. CAISO's published forecast is misaligned by 1 h in EIA-930, which flatters DemandCast. Bands too narrow: 49 / 76%. v1.1: 3.37; v1.1.1: 3.24.</sub> | **3.44%** | Grid's own day-ahead forecast (published): 6.86%<br>Physics alone (classical): 6%<br>Persistence (dumb baseline): 4.9%<br>Same hour last week (dumb baseline): 6.46% | ✅ wins |
| Demand PJM, actual weather<br><sub>Upper bound. With forecast weather (2024): 2.46 vs 3.57, still a win. v1.1: 2.10; v1.1.1: 1.91.</sub> | **2.01%** | Grid's own day-ahead forecast (published): 3.49%<br>Physics alone (classical): 3.24%<br>Persistence (dumb baseline): 4.95%<br>Same hour last week (dumb baseline): 8.32% | ✅ wins |
| Demand MISO, actual weather<br><sub>Upper bound. With forecast weather (2024): 2.10 vs 2.80, still a win. v1.1: 1.88; v1.1.1: 1.79.</sub> | **1.84%** | Grid's own day-ahead forecast (published): 2.71%<br>Physics alone (classical): 2.98%<br>Persistence (dumb baseline): 4.17%<br>Same hour last week (dumb baseline): 6.98% | ✅ wins |
| Demand SPP, actual weather<br><sub>Upper bound. SPP's own forecast ran 6.5% high in 2025 (7.68%); in 2024 it beat DemandCast (2.32 vs 2.77), and with forecast weather 2.22 vs 3.02. v1.1: 2.41; v1.1.1: 2.41.</sub> | **2.6%** | Grid's own day-ahead forecast (published): 4.31%<br>Physics alone (classical): 5.64%<br>Persistence (dumb baseline): 4.32%<br>Same hour last week (dumb baseline): 8.21% | ✅ wins |
| Demand ISO-NE, actual weather<br><sub>Loses every year even with actual weather (3.03 / 3.37 / 3.56 vs 2.40 / 2.77 / 2.52). Bands 55 / 83%. v1.1: 3.00; v1.1.1: 3.00.</sub> | **3.32%** | Grid's own day-ahead forecast (published): 2.56%<br>Physics alone (classical): 5.51%<br>Persistence (dumb baseline): 7.27%<br>Same hour last week (dumb baseline): 10.64% | ❌ loses |
| Demand NYISO, actual weather<br><sub>Upper bound. With forecast weather (2024): 2.76 vs 3.28, still a win. v1.1: 1.97; v1.1.1: 1.97.</sub> | **2.17%** | Grid's own day-ahead forecast (published): 2.97%<br>Physics alone (classical): 5.16%<br>Persistence (dumb baseline): 5.34%<br>Same hour last week (dumb baseline): 8.22% | ✅ wins |
| Demand BPA, actual weather<br><sub>With forecast weather (2024): 2.50 vs 1.92. v1.1: 2.10; v1.1.1: 2.10.</sub> | **2.14%** | Grid's own day-ahead forecast (published): 2%<br>Physics alone (classical): 5.58%<br>Persistence (dumb baseline): 3.69%<br>Same hour last week (dumb baseline): 6.42% | ❌ loses |
| Wind, 8 grids pooled, actual weather (Σ\|err\| / Σ output)<br><sub>BPA 50.2% and CAISO 32.2% (50 km reanalysis in complex terrain); ERCOT 18.0%, MISO 15.9%. With forecast weather (2024, 7,632 h, ERCOT only): 32.0% vs physics 47.9%, persistence 42.6%. v1.1: 19.2%; v1.1.1: 19.2%.</sub> | **19.3%** | Physics alone (real fleet, dev scale) (classical): 22.7%<br>Persistence (same hour yesterday) (dumb baseline): 43.5%<br>Dev-year CF climatology × fleet (dumb baseline): 41.4% | ✅ wins |
| Solar, 7 grids pooled, actual weather (Σ\|err\| / Σ output)<br><sub>183,858 h. CAISO 11.2% (persistence 11.4%), PJM 9.3% (28.0%), BPA 20.8% (30.5%), SPP 27.0% (29.1%); ERCOT 10.5%, MISO 9.3%, ISO-NE 13.7%: ahead of persistence in every grid, CAISO by 0.2 points. Bands 72 / 90%. v1.1: 16.1% (CAISO 24.9%, BPA 52.0%, SPP 41.5%, all three behind persistence); v1.1.1: 16.1%; v1.0: 29.9%. NYISO reports no solar.</sub> | **10.6%** | Physics alone (real fleet, dev scale) (classical): 36.3%<br>Persistence (same hour yesterday) (dumb baseline): 19.1%<br>Dev-year CF climatology × fleet (dumb baseline): 39.2% | ✅ wins |
| Texas heat, Aug 2023 (ERCOT, 744 h)<br><sub>Peak over-predicted: 88.8 GW vs 85.4 GW metered. v1.1: 1.96%.</sub> | **2.08%** | Grid's own day-ahead forecast (published): 1.59%<br>Physics alone (classical): 3.03% | ❌ loses |
| Eastern heat dome, Jun 2024 (PJM)<br><sub>MISO 1.38 vs 4.01, NYISO 1.70 vs 2.73, ISO-NE 3.44 vs 3.38 (a loss) in the same week. v1.1: PJM 1.98%, ISO-NE 2.97%.</sub> | **2.37%** | Grid's own day-ahead forecast (published): 4.5%<br>Physics alone (classical): 2.32% | ✅ wins |
| Arctic outbreak, Jan 2025 (ISO-NE)<br><sub>PJM in the same outbreak: 2.32 vs 4.17. v1.1: ISO-NE 4.78%, PJM 1.83%.</sub> | **4.49%** | Grid's own day-ahead forecast (published): 1.27%<br>Physics alone (classical): 8.5% | ❌ loses |
| Winter Storm Uri, Feb 2021 (ERCOT; a dev year, not exam)<br><sub>DemandCast forecast about 89 GW of demand; ERCOT served 69 GW under load shedding, so the metered 'truth' is curtailed. v1.1: 14.5% (85 GW).</sub> | **16.9%** | Grid's own day-ahead forecast (published): 18.6%<br>Physics alone (classical): 21.1% | ℹ️ |
| Demand error by temperature band, actual weather (7 bands, -10 to 50 °C)<br><sub>Above 35 °C (1,211 h) 2.94% vs the grids' 2.55%; 30-35 °C 2.82% vs 3.12%. Below -10 °C (913 h) 3.35% (grids 3.33%). v1.1: 2.2-2.7% (2.73% above 35 °C).</sub> | **2.3-2.9%** | Grid's own day-ahead forecast (published): 2.6-3.9% | ℹ️ |

Coverage of the 68% error bars on the exam: Demand, pooled (95% band: 89%) **64%**, Demand, CAISO (95%: 76%) **49%**, Demand, forecast weather 2024 (95%: 93%) **72%**, Demand, forecast weather 2024-2025, second look (95%: 92%) **70%**, Wind, pooled (95%: 94%) **72%**, Solar, daylight hours (95%: 90%; v1.1 69 / 86%) **72%**.

### Where it wins

- Solar, v1.2 (27 Nov 2026): 10.6% against 19.1% for persistence with actual weather and 12.0% against 18.9% with real forecast weather 2024-2025; ahead of persistence in every grid (v1.1: 16.1% and 18.2%, behind persistence in CAISO, BPA and SPP).
- Demand with actual weather: 2.49% against 3.41% for the grids' own day-ahead forecasts, ahead in 5 of 8 grids (an upper bound).
- With real forecast weather (2024): 3.00% against 3.20% pooled, and ahead in CAISO, PJM, MISO and NYISO.
- Wind: 19.3% against 22.7% for the physics alone and 43.5% for persistence; with real forecast weather 2024-2025 26.9% against 39.0% and 42.9%.
- Heat waves and cold snaps the grids missed in PJM and MISO: heat dome Jun 2024 PJM 2.37% vs 4.50%, MISO 1.38% vs 4.01%; Arctic outbreak Jan 2025 PJM 2.32% vs 4.17%.

### Where it loses

- v1.2 is worse than v1.1 on demand: 2.49% vs 2.38% with actual weather (v1.1.1 2.32%), 3.14% vs 3.05% with forecast weather 2024-2025. The dev gate showed it (dev demand 2.05% vs 1.98%, +3%), but the gate pools demand, wind and solar, and solar's dev gain (13.1% vs 15.7%) carried the rung.
- With real forecast weather (2024) it loses to ERCOT, SPP, ISO-NE and BPA's own forecasts (ERCOT 3.38% vs 2.05%).
- Second look with real forecast weather, 2024-2025: 3.14% vs 3.59% pooled, but without CAISO and SPP (flawed published forecasts) 2.98% vs the grids' 2.70%; behind ERCOT (3.56% vs 2.26%), ISO-NE (4.26% vs 2.65%) and BPA (2.43% vs 1.96%).
- ERCOT (2.41% vs 2.40%), ISO-NE (3.32% vs 2.56%) and BPA (2.14% vs 2.00%) lose even with actual weather; ISO-NE's Jan 2025 cold snap 4.49% vs 1.27%.
- Texas heat, Aug 2023: 2.08% vs ERCOT's 1.59%, with the peak 4% too high. Above 35 °C (1,211 h) 2.94% vs the grids' 2.55%.
- Solar in SPP (27.0%) and BPA (20.8%) is still far worse than in the other grids, and CAISO beats persistence by only 0.2 points.

## Limits

- The headline uses actual weather and is an upper bound. With real forecast weather DemandCast wins in 4 of 8 grids in 2024 (released) and 5 of 8 over 2024-2025 (second look); the pooled win rests on CAISO and SPP. The archive holds no humidity or radiation forecasts before 2024, so 2023 cannot be graded with forecast weather.
- The pooled dev gate let solar decide: r10 passed with dev errors -4.8% (solar -16%, wind -1%, demand +3%), and the exam shows the same split. A per-output gate (no output worse by more than 2%), fixed before the next ladder, would have rejected r10 and kept v1.1.1's demand.
- Solar timestamps: v1.2 reads whole-hour offsets (-2 to +2 h) from the week before issue, counted from each grid's dev-year value. A change is picked up only once the week before issue shows it, and CAISO's dev-year error profile sits about half an hour off (r_hat 0.46), so some CAISO days near that boundary flip. On val_v5 the reader gets 80% of 2 h offsets right (32 of 40 at +2, 55 of 80 at -2), and a misread offset costs solar error.
- Bands are too narrow on real data: demand 64% / 89% inside the 68% / 95% bands (CAISO 49% / 76%, ISO-NE 55% / 83%).
- Wind is limited by 50 km reanalysis in complex terrain (BPA 50%, CAISO 32%); the network trims the physics error by about 15%.
- It needs the last week of metered output: physics alone is about 5% off (load growth since 2019-21). Building parameters are effective (several fits sit on bounds), so the heating / cooling split is indicative.
- Over-predicts heat-wave peaks by about 4%.
- Inputs the simulator never varied (audit, 25 Nov 2026), repaired in v1.2: reported solar at hours the physics calls dark is now set to 0 before the network reads it (training and inference), simulator v5 adds twilight output, -2 to +2 h offsets and 1-48 h gaps in the demand history, count-like inputs have a standardisation floor of 0.05, and inputs are clipped at |z| 8 (from v1.1.1). In v1.1 the twilight input pinned two BPAT exam days at the 1.2 cap and missing demand hours reached |z| 286; on the 201 actual-weather exam rows with missing history hours demand error was 5.96% (v1.1) and 2.58% with the clip (v1.1.1).

## The ratchet

### DemandCast

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

### DemandCast v1.1 (15 Nov 2026)

**Objective:** Same objective on the frozen simulated set val_v4 (4,000 region-days, simulator v4, seed 404): v3 plus a solar reporting offset of -1 / 0 / +1 h (15 / 70 / 15 %) per region-year  
**Rule:** Same rule: keep a rung if the objective beats the incumbent by more than 0.01 AND the dev gate's mean relative error gets no more than 2% worse

| Rung | Change | Frozen-set score | Decision | Why | Dev demand MAPE (%) | Dev wind (%) | Dev solar (%) | Solar skill on val_v4 |
|---|---|---|---|---|---|---|---|---|
| r8_wide | The release v1.0 re-scored on val_v4 (fixed dev-year solar shift; no offset read) | 0.333 | ● kept | Incumbent. On val_v4 its solar error is 14.5% on regions reported on time and 36.2% / 31.8% on regions reported 1 h early / late. | 2.0 | 21.44 | 15.6 | 0.386 |
| r9a | r8 + the solar reporting offset read at issue from the 168 h history (least squares of the reported-minus-physics error on the physics profile and its slope; -1 / 0 / +1 h), history put back on the physics clock, forecast moved by the offset | 0.375 | ○ rejected | Objective 0.3752 vs 0.3329 and dev +1.1% (inside the gate), but on dev it moved 201 of 730 CAISO days by an hour: CAISO's dev-year profile reads +0.46 h from the physics clock. Replaced by its dev-calibrated form r9b before the release choice (same frozen-set score). | 2.0 | 21.44 | 16.11 | 0.513 |
| r9b | r9a with the offset counted from each grid's dev-year reference (median r_hat over 2019 and 2021: CAISO 0.46 h, the others 0.04-0.19 h) | 0.375 | **◆ released** | Objective 0.3752 vs 0.3329 (+0.042); dev errors +0.2% (solar 15.60 -> 15.68%). On val_v4 the offset is read right on 3,886 of 4,000 region-days. Released as v1.1; solar sigma scale re-fitted on dev (1.34 -> 1.37). | 2.0 | 21.44 | 15.68 | 0.513 |

<sub>Columns after *Why*: Dev gate errors (5,760 demand hours, 5,759 wind, 5,040 solar), never the exam.</sub>

The measurement model changed (EIA-930 solar timestamps move between years: in 2018-2022, outside the exam, 9 of 35 BA-years are 1 h off and CAISO moved from -1 h in the dev years to 0 in 2022), so a new frozen set was built. Only the release r8 could be re-scored on it: the r1-r7 checkpoints are not in the repository. Rungs scored from the shipped weights with training/solar_align.py; raw log training/ratchet.json.

### DemandCast v1.2 (27 Nov 2026, audit repair)

**Objective:** Same objective on the frozen simulated set val_v5 (4,000 region-days, simulator v5, seed 505): v4 plus -2 to +2 h solar offsets, twilight output at dark edge hours, and 1-48 h gaps in the demand history  
**Rule:** Same rule: keep a rung if the objective beats the incumbent by more than 0.01 AND the dev gate's mean relative error gets no more than 2% worse

| Rung | Change | Frozen-set score | Decision | Why | Dev demand MAPE (%) | Dev wind (%) | Dev solar (%) | val_v5 demand, gap samples (%) |
|---|---|---|---|---|---|---|---|---|
| r9b | v1.1 as released, re-scored on val_v5 | 0.349 | ● kept | Re-scored only. Demand 11.2% on val_v5 samples with a history gap (2.3% without); solar 33.7% / 32.8% on regions reported 2 h early / late. Replaced as incumbent by v1.1.1 without a ratchet decision: the clip changes no weights and was made the default after its dated re-score. | 2.0 | 21.44 | 15.68 | 11.18 |
| r9b_zclip8 | v1.1.1: the r8 network with standardised inputs clipped at \|z\| 8 | 0.391 | ● kept | Incumbent. The clip takes the gap samples from 11.2% to 2.6% demand error; solar unchanged. | 1.98 | 21.44 | 15.67 | 2.59 |
| r10_repair | Simulator v5 banks (180,000 region-days, train_v5a-c); solar offsets read to ±2 h, history aligned and zeroed where the physics is dark (training and inference); count-like inputs floored at sd 0.05; inputs clipped at \|z\| 8; three 512-512 MLPs on base features, 40 epochs, as r8 | 0.405 | **◆ released** | Objective 0.4054 vs 0.3908 (+0.015); dev errors -4.8% (solar 15.67 -> 13.11%, wind 21.44 -> 21.23%, demand 1.98 -> 2.05%, +3%). Released as v1.2; exam run once after the choice was committed (285eedd). The pooled gate let solar carry a demand loss, which the exam repeats. | 2.05 | 21.23 | 13.11 | 2.36 |

<sub>Columns after *Why*: Dev gate errors (5,760 demand hours, 5,759 wind, 5,040 solar), never the exam; and the demand error on val_v5's 295 samples with a history gap.</sub>

The audit (25 Nov 2026) found real inputs simulator v4 never produced: reported solar at hours the physics calls dark, 2 h solar offsets, and missing hours in the demand history (miss_frac had a simulated sd of ~0, so one missing hour reached |z| 286). So a new frozen set val_v5 was built before any training, and the incumbents re-scored on it. The incumbent is v1.1.1 (the r8 network with inputs clipped at |z| 8, the default since the dated re-score of 27 Nov 2026). One rung was trained, fixed before training. On val_v4 the three score 0.3752 (v1.1), 0.3751 (v1.1.1) and 0.3774 (r10). Log: training/ratchet.json (training/repair.py).

Raw per-rung logs are in [`training/`](training/).

## Physics

Demand is a building-stock thermal model (training/buildings.py): conductance × the stock-averaged hinge of a Normal thermostat spread around an envelope-lagged sol-air temperature (fast and slow RC response, internal and window gains, night setback), latent cooling load, resistance / heat-pump / non-electric heating and air conditioning with COPs that fall with outdoor temperature, occupancy schedules from the calendar, daylight lighting, a water-heating season term and rooftop PV (pvlib). Wind and solar are GridSim's farm physics on every EIA-860 farm with its real online dates: moist-air density, IEC density-equivalent speed, the pitch-regulated farm curve from specific power, losses, and pvlib with tracking and clipping per weather point.

**Simulator**

- 182,500 simulated region-days in 6 banks (simulator v1, v2 and four v3 banks), each region-year driven by one focus grid's real weather from a training-input year (2018, 2020, 2022), perturbed: temperature offset N(0, 2 K) and anomaly scale 0.85-1.2, humidity × 0.85-1.15, cloudiness × 0.85-1.1, wind × 0.9-1.1
- Random true building stocks (80% around one grid's dev fit, 20% broad uniform); the network is told descriptor parameters drifted from the truth, as a fit 2-4 years old would be
- Wind truth: reanalysis × exp(point bias + correlated AR(1) error); solar: a cloud error; random losses, specific-power and hub shifts, reporting scale and curtailment; the real fleet at a 2018-2022 date with random per-group scaling
- Half of the samples carry synthetic day-ahead forecast-weather errors (the forecast-weather flag input)
- Simulator v4 (15 Nov 2026, frozen set val_v4 only): the simulated solar is reported 1 h early, on time or 1 h late (15 / 70 / 15 %), constant over a region-year; in the non-exam years 2018-2022, 9 of 35 EIA-930 solar BA-years are 1 h off
- Simulator v5 (27 Nov 2026, the v1.2 training banks train_v5a-c and frozen set val_v5): v4 with -2 to +2 h solar offsets (3 / 12 / 70 / 12 / 3 %), output at the dark edge hours next to daylight in 30% of region-years (2-15% of the neighbouring hour), and a 1-48 h gap in the 168 h demand history of 8% of samples

**Checked against independent references**

- Envelope RC lag vs an ODE solver 3.2e-7 K; step response vs closed form 3.6e-15 K (training/checks/physics.json)
- Stock-averaged thermostat hinge vs a 1e6-draw Monte Carlo 0.0035 K; degree-day limit exact (1,080,000 vs 1,080,000 MWh)
- Specific humidity vs PsychroLib (ASHRAE formulas) 0.2% (T >= 0 °C); heat-pump COP 3.26 at 8.3 °C and 1.94 at -8.3 °C, AC COP 3.5 at 35 °C (typical rating points)
- Tabulated wind farm curve vs GridSim's farm_curve: mean 0.0005 CF (max 0.038 away from the 25 m/s cut-out); GridSim wind and solar vs ERCOT EIA-930 (phase 1): hourly r 0.90 and 0.97-0.98
- Dev calibration (2019, 2021): building model fitted per grid, in-sample 2.2-3.5% MAPE; twin test re-simulated each grid's dev year with the full simulator (training/checks/twin_r8_wide.json)

## Help wanted

- Day-ahead forecast weather before 2024. Open-Meteo's Previous Runs API rate-limited the first build (HTTP 429 after 1-25 year-requests per half hour); on 14 Nov 2026 the shared build machine fetched all 220 year-requests with no refusal. The archive holds humidity, radiation and wind forecasts only from 2024 (temperature from 2022), so 2023 cannot be graded with forecast weather and the dev years 2019 and 2021 have no real forecast errors. An archive of day-ahead forecasts for 2019-2023 is wanted.

## Next rungs

- A per-output dev gate (each of demand, wind and solar no more than 2% worse), fixed before the next ladder; then a demand rung that keeps v1.1.1's demand network with v1.2's solar, judged by that gate (the exam chooses nothing, so this has to come from a new ladder).
- Real forecast-error noise in training: the 2022 Open-Meteo temperature library is now fetched and built (training/make_tfc.py; day-ahead error 1.3-2.0 K RMS per grid). A rung that trains on it would be judged on dev.
- Better bands (64% inside the 68% band): per-grid noise-scale variety in the simulator, or a dev-fitted conformal widening.
- Half-hour solar offsets (CAISO's dev reference sits at 0.46 h), and higher-resolution weather for wind in complex terrain (BPA, CAISO).
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
