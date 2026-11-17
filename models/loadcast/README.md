<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# LoadCast <sub>v1.0</sub>

> **Hourly energy per building meter from weather, calendar and metadata, with 90 % bands**

*How much electricity, chilled water, steam, hot water or gas will this building use each hour next year?*

LoadCast forecasts a non-residential building's hourly energy on one meter (electricity, chilled water, steam, hot water or gas) for any span of hours, from weather, the local calendar and what is known about the building, with a 90 % band for every hour. Cold mode reads metadata only (use, floor area, which meters exist, climate); warm mode also reads 2 weeks, 8 weeks or a year of the meter's own history and corrects a time-of-week-and-temperature (TOWT) model fitted on it. It learned only from 1,600 BuildingSim buildings simulated under the real 2016-17 weather of the 19 BDG2 sites, with real dev-meter residuals added as noise. It was examined once on 1,502 real BDG2 exam building-meters, forecasting local 2017 from 2016.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/loadcast/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/loadcast/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Hourly UTC weather (T_out °C, dewpoint °C, GHI W/m², wind m/s) over the forecast window, the history window and 7 days before each; time zone and latitude (holidays for the US, Canada, UK and Ireland); use type (10 classes), floor area (m²), optional year built, the meter type and which meters the building has, optional HVAC class and operating hours; optionally the meter's own hourly kWh history (warm mode). |
| **Returns** | Hourly mean load (kW) on the chosen meter for any span of hours<br>A 90 % band (lo90, hi90) and sigma for every hour, widened by one dev-fitted factor per mode, history length and meter type (×1.0 to ×7.75; ×30, the cap, for 2-week chilled water and gas); raw bands with calibrated=False |
| **Learned from** | The shared BuildingSim simulator (ML Ratchet building platform, Phase 1): an ISO 13790 5R1C single-zone heat balance with multinode walls, solar gains (pvlib), infiltration, ventilation and moisture; DX, heat-pump, chiller (DOE-2 EIR), boiler and district plants; VAV and unitary air sides; domestic hot water; driven by real weather. LoadCast's wrappers (training/lcsim.py, no edit to the shared package) add country holiday calendars, a BDG2-style spec maker (the meters a building has decide its plant), operating knobs and a per-meter calibration y = a·sim + b·A fitted on the dev buildings. |
| **Noise** | Real noise from 85 BDG2 dev building-meters with good data in both years: the hourly residual of an in-sample regression (TOWT + 24 h mean temperature + dewpoint + solar), divided by the year mean — occupancy, campus breaks, events, meter quirks. One residual series is added to each simulated meter on the same local calendar dates, with a real 2017/2016 level change (whole year or a step), occasional quantisation and the real gap pattern. The bank holds the dev buildings' 2017 residuals too. |
| **Network** | Two MLPs, cold (64 inputs) and warm (83 inputs): 3 hidden layers of 192 SiLU units, dropout 0.1, Gaussian head (mean, log sigma); 177,604 parameters (cold 86,978, warm 90,626). Inputs per hour: 38 metadata, 10 weather and 16 calendar features; warm adds 19 history features (TOWT and change-point predictions from the history, time-of-week profile, history level and shape statistics). Cold predicts kW / (A × I_ref), warm kW / history mean. Loss: weighted MSE on the mean (weight 1 / ybar², floored) + 0.1 × Gaussian NLL for sigma; AdamW, cosine schedule, 30 epochs of fresh random hours, per-row metadata jitter (log area ±0.1, year ±6, climate ±0.5 K / ±8 %). |
| **Physics also enters** | only through the simulated training data |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | BuildingData's frozen split (shared/building/data/proc/splits.json, seed 20261116): 52 BDG2 dev buildings. 2016 for calibration; 85 dev building-meters' 2016 and 2017 residuals in the noise bank; 84 dev building-meters forecasting 2017 from 2016 for the dev gate and the widening factors (only 5 hot water and 2 gas meters) |
| **Exam** | Every other BDG2 building-meter with good data in both years (q_ok): 1,502 (856 electricity, 323 chilled water, 225 steam, 72 hot water, 26 gas), forecasting local 2017 with the actual 2017 weather from 2016 history |

## Run it

```bash
cd models/loadcast
pip install numpy scipy pandas
python demo.py        # about 2 s
```

or from anywhere in the repo: `mlratchet demo loadcast`.

```python
import loadcast as lc
m = lc.LoadCast()                       # loadcast_weights.npz next to this file
fc = m.forecast(weather, tz='America/New_York', lat=40.3, use='education', area_m2=9000,
                meter='chilledwater', meters_present=['electricity', 'chilledwater', 'steam'],
                start='2017-01-01', end='2018-01-01',
                history=None)           # or a Series of the meter's own hourly kWh (UTC) -> warm mode
fc[['kW', 'lo90', 'hi90']]
```

## The exam

Every BDG2 exam building-meter, forecast for local 2017 with the actual 2017 weather, against baselines BuildingData fitted on 2016 (TOWT, a G14 change-point model) and a dumb median guess fitted on dev. Median hourly CV(RMSE) (ASHRAE Guideline 14; LoadCast and the guess use p = 0, the fitted baselines n − p). Run once, after R3 was chosen; the full table (2- and 8-week histories, NMBE, G14 pass rates, daily and monthly totals, nowcasts) is in MODEL_CARD.md. Every number here is on real data that never chose anything.

| Metric | LoadCast | Baselines | |
|---|---|---|---|
| Electricity, a year of history (856 meters), median hourly CV(RMSE)<br><sub>By 0.3 points; G14 pass rate 70 % vs 71 % (change point) and 70 % (TOWT), so the build notes call it a tie. 90 % band covers 92 % of hours. Persistence of yesterday's real reading (a nowcast, not a forecast from 2016) gets 18.6 %.</sub> | **18.8%** | Change-point model fitted on 2016 (classical): 19.1%<br>TOWT fitted on 2016 (classical): 19.6% | ✅ wins |
| Chilled water, a year of history (323 meters), median hourly CV(RMSE)<br><sub>Beats TOWT, loses to the change-point model by 0.5 points. 90 % band covers 92 %. Persistence 24 h (nowcast) 40.3 %.</sub> | **42.9%** | Change-point model fitted on 2016 (classical): 42.4%<br>TOWT fitted on 2016 (classical): 45.7% | ❌ loses |
| Steam, a year of history (225 meters), median hourly CV(RMSE)<br><sub>G14 pass rate 10 % vs 25 % (TOWT). 90 % band covers 88 %.</sub> | **45.5%** | TOWT fitted on 2016 (classical): 39.8%<br>Change-point model fitted on 2016 (classical): 40.5% | ❌ loses |
| Hot water, a year of history (72 meters), median hourly CV(RMSE)<br><sub>With 8 weeks of history LoadCast gets 86.5 %. 90 % band covers 92 %.</sub> | **94.5%** | Change-point model fitted on 2016 (classical): 82.6%<br>TOWT fitted on 2016 (classical): 83.1% | ❌ loses |
| Gas, a year of history (26 meters), median hourly CV(RMSE)<br><sub>90 % band covers 73 %. Only 2 gas meters in dev.</sub> | **110.5%** | TOWT fitted on 2016 (classical): 95.7%<br>Change-point model fitted on 2016 (classical): 95.8% | ❌ loses |
| Electricity, no history (856 meters), median hourly CV(RMSE)<br><sub>Median NMBE +24 % (forecast too low). 90 % band covers 88 %.</sub> | **59.3%** | Median-intensity × median-shape guess per use and meter, fitted on dev (dumb baseline): 53.8% | ❌ loses |
| Chilled water, no history (323 meters), median hourly CV(RMSE)<br><sub>Median NMBE +47 % (about half the real load). 90 % band covers 90 %.</sub> | **104.1%** | Median-intensity × median-shape guess per use and meter, fitted on dev (dumb baseline): 101.6% | ❌ loses |
| Steam, no history (225 meters), median hourly CV(RMSE)<br><sub>Still underpredicts: median NMBE +59 %. 90 % band covers 84 %.</sub> | **102.8%** | Median-intensity × median-shape guess per use and meter, fitted on dev (dumb baseline): 111.6% | ✅ wins |
| Hot water, no history (72 meters), median hourly CV(RMSE)<br><sub>Median NMBE +58 %. 90 % band covers 81 %.</sub> | **134%** | Median-intensity × median-shape guess per use and meter, fitted on dev (dumb baseline): 207.6% | ✅ wins |
| Gas, no history (26 meters), median hourly CV(RMSE)<br><sub>90 % band covers 74 %.</sub> | **181.5%** | Median-intensity × median-shape guess per use and meter, fitted on dev (dumb baseline): 190.4% | ✅ wins |

### Where it wins

- With a year of history, electricity: median CV(RMSE) 18.8 % against 19.1 % for the change-point model and 19.6 % for TOWT, both fitted on the same 2016 year (856 meters); a 0.3-point margin with the same G14 pass rate, 70 %.
- With no history, steam, hot water and gas beat the dev-fitted median guess: 102.8 vs 111.6 %, 134.0 vs 207.6 % and 181.5 vs 190.4 %.
- Its error bars: the 90 % band covers 88-92 % of hours with a year of history on electricity, chilled water, steam and hot water, and 81-90 % with no history on every meter type but gas (74 %).
- Daily totals with a year of history: electricity 14.1 % against 17.2 % for the change-point model (chilled water 32.4 vs 32.4 %).

### Where it loses

- With a year of history, chilled water (42.9 vs 42.4 %), steam (45.5 vs 39.8 %), hot water (94.5 vs 82.6 %) and gas (110.5 vs 95.7 %) lose to the classical models fitted on 2016.
- With no history, electricity (59.3 vs 53.8 %) and chilled water (104.1 vs 101.6 %) lose to a median guess, and thermal meters come out about half the real load (median NMBE +47 % chilled water, +59 % steam).
- Two weeks of December history cannot forecast summer chilled water: CV 87.3 %, and the dev widening for that case hit its ×30 cap.
- Not a nowcaster: yesterday's real reading (18.6 % on electricity) is as good as any 2016-based forecast.

## Limits

- Non-residential campus buildings at BDG2's 19 sites, trained under their 2016-17 weather; other climates, years and residential buildings are extrapolation.
- BuildingSim is single zone and its campus district energy runs about 3× low; 52 dev buildings' a, b calibration closes only part of that gap, empirically rather than physically.
- The CV²-weighted loss pulls unknown magnitudes low; R4 tried a median-aimed weight and failed both tests.
- The dev set is small: 5 hot water and 2 gas meters. The noise bank holds the dev buildings' 2017 residuals, so dev-gate numbers are a little optimistic (the exam is unaffected).
- Short-history modes were trained on year-end histories; 2-week and 8-week bands are unusable for chilled water and gas when the history is winter.
- BDG2 timestamps were mixed (local DST vs standard) and are fixed to UTC; Peacock's weather does not match its stated location; four sites were located by temperature matching.

## The ratchet

**Objective:** Mean over meter types (n >= 5) of (cold skill + warm skill) / 2 on the frozen simulated set v1_val (300 BuildingSim buildings of the v1 calibrated physics, seed 99, real dev noise seed 12345; 666 building-meters scored). Cold skill = 1 − median CV(RMSE) / median CV of the set's own median-intensity × median-shape guess; warm skill = 1 − median CV / median CV of TOWT on the same 2016 history.  
**Rule:** Keep if the objective beats the incumbent by more than 0.01 AND the dev score (mean of cold and warm median CV(RMSE) over 84 dev building-meters, 2016 → 2017) is no more than 2 % (relative) worse than the incumbent's.

| Rung | Change | Frozen-set score | Decision | Why | Cold skill | Warm skill | Dev cold CV | Dev warm CV | Dev score | Dev vs incumbent |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 | Generic Phase 1 BuildingSim physics (v0 bank, 800 buildings), real dev noise, two MLPs 3 × 192, 30 epochs | -0.174 | ● kept | First rung. | -0.253 | -0.094 | 88.6 | 31.2 | 59.9 | – |
| r1 | Dev-calibrated physics (v1 bank: knobs and meter a, b drawn from the 52 dev fits), 800 buildings, otherwise as R0 | -0.118 | ● kept | Frozen objective −0.118 vs −0.174; dev score 54.9 vs 59.9. | -0.187 | -0.049 | 79.3 | 30.5 | 54.9 | −8% |
| r2 | R1 + anti-memorisation: per-row jitter of the area, year and climate metadata, dropout 0.1, weight decay 1e-3 | -0.018 | ● kept | Frozen objective −0.018 vs −0.118; dev score 51.9 vs 54.9. | -0.071 | 0.035 | 73.8 | 30.0 | 51.9 | −6% |
| obj-unfloored | Re-definition before R3: pooled cold and warm skills no longer floored at 0 (rung.py --redecide); every rung re-decided in order | — | ● kept | Definition change, not a model. Re-decided, R0-R2 are kept (pooled objective −0.215, −0.155, −0.067). The floored values and the decisions under them were overwritten and are not in the log. | – | – | – | – | – | – |
| obj-per-meter | Re-definition after R2: skills computed per meter type and averaged (pooled medians mixed meter types); every rung re-scored on v1_val in order (rung.py --rescore) | — | ● kept | Definition change, not a model. R0-R2 stayed kept; the values above are re-scores under this definition. | – | – | – | – | – | – |
| r3 | R2 on the full calibrated bank (1,600 buildings instead of 800) | 0.013 | **◆ released** | Frozen objective 0.014 vs −0.018; dev score 49.2 vs 51.9. Released as v1.0 (R4 and R5 were rejected). | -0.016 | 0.043 | 70.6 | 27.9 | 49.2 | −5% |
| r4 | Cold loss weight 1 / ybar^0.5 instead of 1 / ybar² (aims the unknown magnitude at the median, not about half of it); warm net = R3's | -0.026 | ○ rejected | Frozen objective −0.026 vs 0.014 (needs +0.01); dev score 52.2 vs 49.2 (dev gate failed). | -0.095 | 0.043 | 76.6 | 27.9 | 52.2 | +6% |
| r5 | Real noise matched by use type (75 % of residual series from a dev building of the same use: campus calendars by use); cold loss weight as R3 | 0.006 | ○ rejected | Frozen objective 0.006 vs 0.014 (needs +0.01); dev score 51.7 vs 49.2 (dev gate failed). | -0.022 | 0.035 | 74.5 | 28.8 | 51.7 | +5% |
| widen | Error bars: one widen-only factor per mode, history length and meter type: the smallest factor (1.0 to 30, the cap) at which the dev building-meters' mean coverage of the 90 % band, forecasting 2017 from 2016, reaches 90 % (training/exam.py fit_widen; widen_r3.json) | — | ● kept | Cold ×2.0 electricity, ×2.9 chilled water, ×7.75 steam, ×1.0 hot water, ×1.5 gas; warm (a year) ×1.0-1.2; 8 weeks ×1.0-1.65; 2 weeks ×1.25-4.0, and ×30 (the cap) for chilled water and gas. No effect on the forecasts. | – | – | – | – | – | – |

<sub>Columns after *Why*: Skills on the frozen set v1_val, then dev (84 dev building-meters, 2016 → 2017): median cold and warm CV(RMSE) %, the dev score and its ratio to the incumbent's.</sub>

Rule note (from the build handoff; no exam result chose anything): the frozen-set objective changed twice before R3 — floored pooled skills, then unfloored pooled skills, then per meter type (pooled medians mixed meter types and could rank a model that is better on every meter as worse) — and every rung was re-scored and re-decided in order each time; the values here are the final definition (training/ratchet.json keeps the pooled values as frozen_history; the floored values were overwritten). R0-R2 are kept under the last two definitions; their decisions under the first are not recorded. The real-noise bank holds the dev buildings' 2017 residuals, so the dev-gate numbers are slightly optimistic; the exam is unaffected. The card gives the frozen set as 650 building-meters; ratchet.json scores 666.

Raw per-rung logs are in [`training/`](training/).

## Physics

The shared BuildingSim simulator (ML Ratchet building platform, Phase 1): an ISO 13790 5R1C single-zone heat balance with multinode walls, solar gains (pvlib), infiltration, ventilation and moisture; DX, heat-pump, chiller (DOE-2 EIR), boiler and district plants; VAV and unitary air sides; domestic hot water; driven by real weather. LoadCast's wrappers (training/lcsim.py, no edit to the shared package) add country holiday calendars, a BDG2-style spec maker (the meters a building has decide its plant), operating knobs and a per-meter calibration y = a·sim + b·A fitted on the dev buildings.

**Simulator**

- Generator: BuildingSim specs from ComStock / CBECS priors; use mix like BDG2; meter sets drawn from the dev buildings (80 %) or a fixed list; half the floor areas from the dev log-area distribution
- 1,600 buildings at the 19 BDG2 sites under their real 2016-2017 hourly weather; each building draws its knobs and a, b from a dev donor (60 % same use), jittered ×exp(N(0, 0.2))
- Knobs: HVAC for electricity-only buildings, operating hours (default / extended / 24-7), outdoor air ×1 or ×2.5, VAV minimum flow 0.3 or 0.8 with 12.8 °C supply air
- Dev calibration on 52 BDG2 dev buildings (2016 only): best knobs 24-7 in 22 of 52, extended hours in 9, outdoor air ×2.5 in 37; median electricity a = 0.43 with b = 6.3 W/m² (real campus electricity is flatter than simulated)
- Simulated annual energy / real, generic twins → best knobs (dev medians): electricity 0.97 → 0.96, chilled water 0.29 → 0.47, steam 0.29 → 0.87, hot water 0.18 → 0.57, gas 0.14 → 0.14; hourly CV(RMSE) after the a, b fit (in sample, 2016): electricity 62 → 24 %, chilled water 93 → 53 %, steam 113 → 62 %, hot water 171 → 79 %

**Checked against independent references**

- BuildingSim Phase 1: steady conduction vs closed form to 1.8e-9; RC step response within 0.0035 K; degree-day ratio 0.999996; ISO 13786 wall amplitude −1.6 %, lag −0.01 h; every energy ledger closes to 1e-8
- BESTEST: 29 of 34 values inside the published ranges, all 20 annual heating and cooling values in (ranges transcribed from the literature, not the official ASHRAE 140 tables)
- Twins: ResStock homes (126) site energy median 1.10× (IQR 0.98-1.26), daily r 0.92; ComStock buildings (45) 0.86× (aggregate 0.97), daily r 0.89
- Real BDG2 dev twins from metadata alone (median annual sim/real): electricity 0.98 (daily r 0.34), chilled water 0.28, steam 0.29, hot water 0.15 — the campus district-energy gap LoadCast's calibration only partly closes
- Twin test (dev, R3): median cold CV(RMSE) real vs twin: electricity 62.5 vs 51.8 % (per-building r 0.86), chilled water 101 vs 99 % (r 0.96), steam 104 vs 108 % (r 0.95); warm 22.3 vs 17.8, 43.2 vs 41.7, 61.7 vs 59.9 % (r ≈ 0)
- numpy == torch to 1.5e-6 on real dev rows; the lab page's JavaScript port == Python to 3e-4 (weather stored at 0.01 °C); the site instrument == numpy to 4e-6 on 25 reference weeks

## Help wanted

- An EnergyPlus binary for a true reference engine: there is no PyPI wheel with an engine and GitHub was blocked in the build sandbox, so BESTEST ranges and ResStock/ComStock twins stand in.
- The official ASHRAE Standard 140 (BESTEST) result tables: BuildingSim's ranges were transcribed from the literature.
- A residential exam: large open smart-meter sets (IDEAL homes, Low Carbon London) are open but were skipped for size.
- More campus thermal meters with plant metadata (which plant serves which building, seasonal shutdowns): the dev set has 5 hot water and 2 gas meters.

## Next rungs

- Cold magnitude: a median-aimed log-magnitude head plus the shape network, or a/b calibrated by use type with more dev-like buildings (R4's fix improved coverage but not CV).
- A heating-meter physics rung: seasonal plant shutdowns, steam distribution losses, plants serving several buildings.
- Warm mode trained on 4-12 week histories from every season, falling back to cold when the history is out of season.
- Upstream to BuildingSim behind flags: the US/CA/UK/IE holiday calendars, the BDG2-style spec maker, campus defaults (24/7 in 22 of 52, outdoor air ×2.5, VAV minimum flow), a per-meter district scale and base load, and a bounded-memory mode for long bank runs.

## Files

| Path | What |
|---|---|
| `loadcast.py`, `pipeline.py` | Inference — numpy/scipy only |
| `loadcast_weights.npz` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Building Data Genome 2 (BDG2), Miller et al. 2020, Scientific Data 7:368 (cleaned by BuildingData)](https://github.com/buds-lab/building-data-genome-project-2) | dev (52 buildings for calibration, the dev gate and the widening), noise (85 dev building-meters' residuals), exam (1,502 building-meters), physics (site weather and locations) | CC BY-SA 4.0 (share-alike; examples/bdg2/, lab/ and the instrument data are BDG2-derived) |
| [NOAA Integrated Surface Database station weather, as shipped with BDG2](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | physics (simulator weather), dev, exam (forecast inputs) | Public domain (U.S. Government); redistributed within BDG2 under CC BY-SA 4.0 |
| [Iowa Environmental Mesonet ASOS archive (weather gap fill, by BuildingData)](https://mesonet.agron.iastate.edu/request/download.phtml) | physics, dev, exam (gap fill) | Free use with credit (Iowa State University) |
| [NASA POWER hourly irradiance (by BuildingData)](https://power.larc.nasa.gov) | physics, dev, exam (GHI) | Free use with credit (NASA Langley Research Center POWER Project) |
| [NREL ResStock and ComStock (OEDI)](https://data.openei.org) | physics (BuildingSim generator priors; twin checks) | CC BY 4.0 ("Data includes information from the ResStock™ / ComStock™ datasets developed by the National Renewable Energy Laboratory") |
| [EIA Commercial Buildings Energy Consumption Survey (CBECS) 2018](https://www.eia.gov/consumption/commercial/) | physics (BuildingSim generator priors) | Public domain (U.S. Government) |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
