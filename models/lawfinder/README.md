<!-- Generated from model.yaml and ratchet.yaml by `mlratchet readme --write`. Edit those, not this file. -->
# LawFinder <sub>v1.0</sub>

> **What does each simulator believe, in one line, and is it true?**

*What readable law does each shared simulator imply, and does that law hold on the platform's real exam?*

LawFinder runs symbolic regression on simulated banks from WaterSim, FireSim, GridSim, BuildingSim and RoadSim and returns short equations with error bars: the long-term runoff ratio of a basin, the next-day growth of a wildfire, GB system inertia, the GB frequency nadir after a trip, the annual energy intensity of a single-family home, and a car's energy per km on a trip. Each law is a rung on a ratchet of complexity levels and physics versions, chosen on a frozen simulated set and a small real dev set, then examined once on the platform's real exam beside the textbook law (its constant fitted on the same dev set), the collection's network and a dumb baseline.

[Interactive lab](https://dogum.github.io/ML-Ratchet/models/lawfinder/lab/) · [Project page](https://dogum.github.io/ML-Ratchet/models/lawfinder/) · [Model card](MODEL_CARD.md) · [Lab notebook](NOTEBOOK.md) · [The method](https://github.com/dogum/ML-Ratchet/blob/main/docs/method.md)

| | |
|---|---|
| **Reads** | Per law: water - aridity PET/P, snow fraction, P-PET seasonal synchrony, soil storage index; fire - perimeter (km), daily 10 m wind (m/s), 100-h dead fuel moisture (%), fuel group; grid inertia - demand (GW), inverter-based share; grid nadir - MW lost, inertia (GVA s), demand; homes - HDD and CDD (base 18 C), floor area, year built; road - mean speed, positive kinetic energy per metre, net and absolute grade, mass, CdA, Crr, ambient temperature, hybrid and electric flags. |
| **Returns** | Long-term runoff ratio: runoff ratio Q/P with a 68 % bar (frozen-set residual sd, dev-widened)<br>Daily fire growth: log10(new area km² + 1 cell) with a 68 % bar (frozen-set residual sd, dev-widened)<br>GB system inertia: system inertia E with a 68 % bar (frozen-set residual sd, dev-widened)<br>GB frequency nadir: frequency drop to the nadir f0 - f_min with a 68 % bar (frozen-set residual sd, dev-widened)<br>Home energy signature: site energy per floor area with a 68 % bar (frozen-set residual sd, dev-widened)<br>Trip energy per km: energy per km, fuel at LHV or DC battery (MJ/km), with a 68 % bar (frozen-set residual sd, dev-widened) |
| **Learned from** | No new physics: the collection's shared simulators are used unchanged (WaterSim v1, FireSim v1, GridSim 1.1 via GridPulse's GB state sampler, BuildingSim 1.0/1.4, RoadSim physics v1/v2). Each law is a statement about what one simulator implies. |
| **Noise** | Platform noise models as shipped: WaterSim forcing error on the inputs the law sees; FireSim runs on real HRRR; GridSim events measured as 1 s means; BuildingSim weather shifts; RoadSim drives the true weather (ambient + N(0, 1.5) C, wind x U(0.4, 0.9)) while the law reads the observed one. No real labels in training. |
| **Network** | No network: symbolic regression (lawsearch, numpy/scipy; PySR-style regularised evolution with Levenberg-Marquardt constants and a hall of fame per complexity). Operators + - * / ^ exp log sqrt inv sq; complexity = node count, up to 25. |
| **Physics also enters** | inputs: Every law input is a physical quantity (aridity, perimeter, inertia, degree days) |
| **Real data's jobs** | backdrop, calibration, gate, error bars, exam (never training labels) |
| **Dev set** | Each platform's frozen dev split, copied in splits.json: 52 CAMELS basins; 40 fires (791 fire-days); 4 GB events of 2019; RECS dev single-family homes; 14 VED dev cars (1,143 trips) and 1,178 EPA dev tests (seeded draw) |
| **Exam** | Each platform's exam split: 619 CAMELS basins (WY1990-99); FireData exam fire-days 2012-2024; 410 GB events (inertia) and the loss-labelled ones (nadir), 2018-2025; RECS exam single-family homes; 22,877 VED exam trips and 52,522 EPA exam tests (EnergyNet's exam sets) |

## Run it

```bash
cd models/lawfinder
pip install numpy
python demo.py        # about 1 s
```

or from anywhere in the repo: `mlratchet demo lawfinder`.

```python
from lawfinder import LawFinder
lf = LawFinder()
mu, sigma = lf["water"].predict(phi=0.8, fs=0.1, sync=-0.2, S=0.14)
```

## The exam

Each law run once on its platform's real exam by a separate script, beside the textbook law tuned on dev, the collection's network and a dumb baseline. Every number here is on real data that never chose anything.

| Metric | LawFinder | Baselines | |
|---|---|---|---|
| Long-term runoff ratio: mean \|error\| of Q/P (619 real exam units)<br><sub>Released law w2-c13: Q/P = 1.84·(δ + 3.733)^(0.3644·f_s - φ - 0.2988). WaterSim's own law (w2-c3): 0.338/φ.</sub> | **0.129** | WaterSim's law (w2-c3, simplest within 0.02 of best sim R2) (classical): 0.14<br>Budyko-Fu, w = 2.6 (WaterData baseline) (classical): 0.141<br>Budyko-Fu, w = 2.38 fitted on dev (classical): 0.146<br>Budyko-Fu, w = 2.23 fitted on WaterSim (classical): 0.154<br>Dev median (0.433) (dumb baseline): 0.211<br>FlowCast v1.0 (network, ungauged) (published): 0.112 | ❌ loses |
| Daily fire growth: mean \|log10 ratio\| of new area (38599 real exam units)<br><sub>Released law f2-c3: log₁₀ G = -0.5585/P. FireSim's own law (f2-c6): 0.6979·log(P) - 0.7956. FireCast has growth-day rows only (see the next row).</sub> | **1.15** | FireSim's law (f2-c6) (classical): 1.8<br>Rothermel ROS x perimeter, k = 0.0152 (dev) (classical): 1.06<br>Rothermel ROS x perimeter, k = 0.175 (FireSim) (classical): 1.34<br>FireData best baseline: yesterday's growth from the active line (classical): 0.671<br>FireData clim (dumb: dev median growth by size) (dumb baseline): 1.37<br>Persistence (no growth) (dumb baseline): 1.4 | ❌ loses |
| Daily fire growth, growth days only (FireData's convention; 25105 fire-days)<br><sub>FireCast is scored on growth days only, so this is the row where it can be compared.</sub> | **0.693** | FireSim's law (f2-c6) (classical): 1.13<br>Rothermel ROS x perimeter, k = 0.0152 (dev) (classical): 0.87<br>Rothermel ROS x perimeter, k = 0.175 (FireSim) (classical): 0.771<br>FireData best baseline: yesterday's growth from the active line (classical): 0.849<br>FireData clim (dumb: dev median growth by size) (dumb baseline): 2.07<br>Persistence (no growth) (dumb baseline): 2.15<br>FireCast v1.0 (network, threshold area) (published): 0.757 | ✅ wins |
| GB system inertia: MdAPE of inertia (410 real exam units)<br><sub>Released law i1-c7: E = D·(6.603 - 5.617·s). GridSim's own law (i1-c7): D·(6.603 - 5.617·s).</sub> | **33%** | GridSim's law (i1-c7) (classical): 33%<br>7.00 s x demand (dev fit) (classical): 15.1%<br>GridPulse v1.0 (network: 60 s trace + demand) (published): 16.4%<br>Dev median inertia (dumb baseline): 24% | ❌ loses |
| GB frequency nadir: MAE of the nadir depth (148 real exam units)<br><sub>Released law g1-c12: Δf = (ΔP + 2)^2/(52940·E - 3.541e+6)^0.2084. GridSim's own law (g2-c3): 0.4334·ΔP.</sub> | **0.06 Hz** | GridSim's law (g2-c3) (classical): 0.106 Hz<br>Swing equation + ramp response, k L^2/E (k = 108, dev) (classical): 0.173 Hz<br>GridPulse v1.0 nadir forecast after 5 s of trace (context: reads the trace) (published): 0.0799 Hz<br>Dev median drop (dumb baseline): 0.129 Hz | ✅ wins |
| Home energy signature: MdAPE of annual site energy per m2 (6019 real exam units)<br><sub>Released law b2-c13: EUI = -H·Y + 2022·H + 186.5/1.926^A. BuildingSim's own law (b2-c9): 2021·H - Y·(H - 0.03403).</sub> | **39.3%** | BuildingSim's law (b2-c9) (classical): 39.7%<br>Degree-day signature 85 + 22.9 HDD + 17.4 CDD (dev) (classical): 35.6%<br>Degree-day signature 23 + 61.2 HDD + 15.6 CDD (BuildingSim) (classical): 45.1%<br>BuildingData-style: log energy ~ log area + HDD + CDD (dev) (classical): 26.6%<br>Dev median EUI (dumb baseline): 33.4% | ❌ loses |
| Trip energy per km: MdAPE, petrol cars, VED trips (14,469 real exam units)<br><sub>Bias -18.9 %, coverage of the 68 % bar 62 %. EnergyNet's row re-computed from its stored predictions (exam run 1 mis-scaled it; training/exam_road_errata.py).</sub> | **19.3%** | RoadSim's law (p2-c10) (classical): 14%<br>Road-load / eta, eta per powertrain fitted on dev (classical): 20%<br>Road-load / eta, eta per powertrain fitted on RoadSim (classical): 20.6%<br>EnergyNet v2.1 (network) (published): 8.64%<br>Window label x distance (RoadData) (classical): 13%<br>MOVES-like VSP bins (RoadData) (classical): 10.3%<br>Class median (RoadData) (dumb baseline): 22.8%<br>Dev median per powertrain (dumb baseline): 22.4% | ❌ loses |
| Trip energy per km: MdAPE, hybrids, VED trips (8,241 real exam units)<br><sub>Bias -7.1 %, coverage of the 68 % bar 89 %. EnergyNet's row re-computed from its stored predictions (exam run 1 mis-scaled it; training/exam_road_errata.py).</sub> | **16.8%** | RoadSim's law (p2-c10) (classical): 40.4%<br>Road-load / eta, eta per powertrain fitted on dev (classical): 15.7%<br>Road-load / eta, eta per powertrain fitted on RoadSim (classical): 15.7%<br>EnergyNet v2.1 (network) (published): 13.4%<br>Window label x distance (RoadData) (classical): 16.2%<br>MOVES-like VSP bins (RoadData) (classical): 15.3%<br>Class median (RoadData) (dumb baseline): 17.2%<br>Dev median per powertrain (dumb baseline): 19% | ❌ loses |
| Trip energy per km: MdAPE, electric (2 Nissan Leafs), VED trips (167 real exam units)<br><sub>Bias -42.2 %, coverage of the 68 % bar 94 %. EnergyNet's row re-computed from its stored predictions (exam run 1 mis-scaled it; training/exam_road_errata.py).</sub> | **43.3%** | RoadSim's law (p2-c10) (classical): 21.9%<br>Road-load / eta, eta per powertrain fitted on dev (classical): 21.7%<br>Road-load / eta, eta per powertrain fitted on RoadSim (classical): 17.3%<br>EnergyNet v2.1 (network) (published): 23.2%<br>Window label x distance (RoadData) (classical): 21.5%<br>MOVES-like VSP bins (RoadData) (classical): 19%<br>Class median (RoadData) (dumb baseline): 27.1%<br>Dev median per powertrain (dumb baseline): 28% | ❌ loses |
| Trip energy per km: MdAPE, petrol cars, EPA tests (46,876 real exam units)<br><sub>Bias +10.6 %, coverage of the 68 % bar 81 %.</sub> | **17.3%** | RoadSim's law (p2-c10) (classical): 14.5%<br>Road-load / eta, eta per powertrain fitted on dev (classical): 25%<br>Road-load / eta, eta per powertrain fitted on RoadSim (classical): 24.3%<br>EnergyNet v2.1 (network) (published): 7.49%<br>Dev median per powertrain (dumb baseline): 25.3% | ❌ loses |
| Trip energy per km: MdAPE, hybrids, EPA tests (4,154 real exam units)<br><sub>Bias +5.8 %, coverage of the 68 % bar 98 %.</sub> | **13.1%** | RoadSim's law (p2-c10) (classical): 22.3%<br>Road-load / eta, eta per powertrain fitted on dev (classical): 19.6%<br>Road-load / eta, eta per powertrain fitted on RoadSim (classical): 19.4%<br>EnergyNet v2.1 (network) (published): 7.82%<br>Dev median per powertrain (dumb baseline): 21.9% | ❌ loses |
| Trip energy per km: MdAPE, battery-electric, EPA tests (1,492 real exam units)<br><sub>Bias +42.2 %, coverage of the 68 % bar 100 %.</sub> | **43.1%** | RoadSim's law (p2-c10) (classical): 26.8%<br>Road-load / eta, eta per powertrain fitted on dev (classical): 11.9%<br>Road-load / eta, eta per powertrain fitted on RoadSim (classical): 12.9%<br>EnergyNet v2.1 (network) (published): 6.27%<br>Dev median per powertrain (dumb baseline): 15.8% | ❌ loses |

Coverage of the 68% error bars on the exam: Long-term runoff ratio **82.9%**, Daily fire growth **69.1%**, GB system inertia **91.2%**, GB frequency nadir **100%**, Home energy signature **69.3%**, Trip energy per km (VED trips) **71.7%**, Trip energy per km (EPA tests) **82.6%**.

## Limits

- <b>Water:</b> FlowCast, the collection's ungauged network, beats the law (MAE 0.112 vs 0.129 on 619 basins). The law beats Budyko-Fu (0.141-0.154). In snowy basins (f_s > 0.3, 88 basins) every method is poor (law 0.22). All rows share ERA5-Land forcing, which under-catches mountain precipitation: 50 basins carry more water out than ERA5 says falls in.
- <b>Fire:</b> FireSim's own law over-predicts real next-day growth a median 36 times (bias +1.56 decades). The released law survives only because the dev gate cut it down to a saturating curve. Over all fire-days it loses to FireData's yesterday's-growth baseline (1.15 vs 0.67 mean |log₁₀|); on growth days it is the best row (0.69 vs FireCast 0.76). FireSim has no fire persistence (stalls, burnout), the platform's known gap. Slope was not an input: real exam fires have no landscape.
- <b>Grid inertia:</b> GridSim's raw GB fleets commit about 4.5 s of inertia per MW of demand where NESO's outturn shows about 7, so the law is 33 % low (MdAPE) and loses to 7 s × demand (15.1 %) and to the dev median (24 %). Its ranking is the best of all rows (ρ 0.975).
- <b>Grid nadir:</b> the loss labels are noisy (REMIT/FUELINST matches); 148 events. The dev gate had only 4 events, one of them (9 Aug 2019) a cascade. The 68 % bar (±0.31 Hz) is far too wide: it covers 100 % of exam events.
- <b>Homes:</b> the law loses to the dumb dev median (MdAPE 39.3 % vs 33.4 %) and to a log-linear fit on dev (26.6 %). BuildingSim's homes are about 1.4 times too energy-intensive at physics 1.0; 1.4 fixes the median but not the spread. RECS energy is EIA's billing-based estimate, and home-to-home variation (occupants, setpoints) is not in the inputs.
- <b>Road:</b> the law loses to EnergyNet everywhere and, on real VED trips, to RoadData's MOVES-like and window-label baselines (18.6 % against 11.8 % and 14.1 %; EnergyNet 10.1 %). It beats road-load on EPA petrol and hybrid tests (17.3 % and 13.1 % against 25.0 % and 19.6 %) and loses to it on electric cars: it reads the 2 VED Leafs 42 % low and EPA BEVs 42 % high, one exponent for every electric car. RoadSim's own shorter law (p2-c10) beats the released one on VED petrol (14.0 %) and on EPA (15.1 %); the dev gate (14 dev cars, 1,178 EPA dev tests) picked the longer law. Real CdA and Crr are class defaults on VED (EPA: from the road-load coefficients); VED fuel is MAF-estimated for most cars. Exam run 1 mis-scaled EnergyNet's VED rows (units); they were re-computed from its stored predictions.
- <b>Method:</b> one search per physics version, time-boxed at 4-7 minutes on a shared core; another seed can give a different front. Dev sets are small (4 grid events, 40 fires, 52 basins), so the gate is coarse. The road law (RoadSim trip energy) was added on 22 Nov 2026, after the other four.
- <b>Road, electric cars and grade:</b> the released law's energy falls as the road gets steeper. On one battery-electric drive (same speed trace, start charge and seed) it gives 0.512, 0.408 and 0.324 MJ/km at -2 %, 0 and +2 % net grade, where RoadSim's battery energy is 0.009, 0.332 and 0.659 MJ/km (audit, 25 Nov 2026). At e = 1 the exponent's base is below one, so the sign holds for every BEV. Six of the 21 logged p2 candidates rise with grade; all six missed the pooled promotion margin. A sign-checked search (27 Nov 2026) found no replacement that passed the ratchet.
- <b>Input ranges:</b> each law is fitted inside its bank's input ranges (var_info in lawfinder_laws.json) and extrapolates freely outside them. The homes law's heating slope turns negative for homes built after 2020 (+1.87 at 2020, -3.13 at 2025, kWh/m² per 1000 K·d). Chaining the two grid laws, 615 of 2,356 inertia predictions on a regular grid fall below the nadir law's 90 GVA·s floor. Since 27 Nov 2026 predict() warns outside the declared ranges (strict=True raises). One of the demo's real grid events is outside them (a loss above 2 GW).

## The ratchet

### LawFinder Long-term runoff ratio (WaterSim)

**Objective:** R^2 of runoff ratio Q/P on the frozen simulated set  
**Rule:** Keep a rung if its frozen-set R^2 beats the incumbent by more than 0.01 AND its dev error is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; after a physics change every rung is re-scored on the new frozen set and every decision re-taken.

| Rung | Change | Frozen-set score | Decision | Why | Dev error | Exam mean |error| of Q/P |
|---|---|---|---|---|---|---|
| r0 | A constant (the simulator mean): Q/P = 0.4408 | -0.001 | ● kept | First rung (the incumbent to beat). | 0.2 | 0.212 |
| w1-c3 | Physics w1, complexity 3: Q/P = 0.3948^H | 0.477 | ● kept | Frozen-set skill 0.477 > -0.001 + 0.01, dev error 0.1607 vs 0.2 (within 5%). | 0.161 | 0.165 |
| w1-c5 | Physics w1, complexity 5: Q/P = 0.224^H + 0.1578 | 0.486 | ○ rejected | Frozen-set skill 0.486 does not beat the incumbent's 0.477 by more than 0.01. | 0.156 | 0.159 |
| w1-c7 | Physics w1, complexity 7: Q/P = 0.4091^H - 0.09574·A | 0.547 | ● kept | Frozen-set skill 0.547 > 0.477 + 0.01, dev error 0.1399 vs 0.1607 (within 5%). | 0.14 | 0.147 |
| w1-c8 | Physics w1, complexity 8: Q/P = 0.1741^H + 0.1757·exp(C) | 0.502 | ○ rejected | Frozen-set skill 0.502 does not beat the incumbent's 0.547 by more than 0.01. | 0.144 | 0.151 |
| w1-c9 | Physics w1, complexity 9: Q/P = 0.3925^H - 0.1019·A + 0.1019·C | 0.563 | ● kept | Frozen-set skill 0.563 > 0.547 + 0.01, dev error 0.1319 vs 0.1399 (within 5%). | 0.132 | 0.142 |
| w1-c10 | Physics w1, complexity 10: Q/P = exp(-0.9964/(0.6993^A/H + C)) | 0.529 | ○ rejected | Frozen-set skill 0.529 does not beat the incumbent's 0.563 by more than 0.01. | 0.128 | 0.141 |
| w1-c11 | Physics w1, complexity 11: Q/P = 0.2083^H - 0.05287·A + 0.05287·exp(exp(C)) | 0.549 | ○ rejected | Frozen-set skill 0.549 does not beat the incumbent's 0.563 by more than 0.01. | 0.131 | 0.138 |
| w1-c13 | Physics w1, complexity 13: Q/P = 0.271^(H + Y) - 0.05446·A + 0.05446·exp(exp(C)) | 0.540 | ○ rejected | Frozen-set skill 0.540 does not beat the incumbent's 0.563 by more than 0.01. | 0.136 | 0.144 |
| w1-c15 | Physics w1, complexity 15: Q/P = 0.3414^(H + Y) - 0.07412·A + 0.07412·Y + 0.07412·exp(2·C) | 0.551 | ○ rejected | Frozen-set skill 0.551 does not beat the incumbent's 0.563 by more than 0.01. | 0.133 | 0.142 |
| w1-c16 | Physics w1, complexity 16: Q/P = 0.3244^(H + Y) - 0.07336·A + 0.07336·sqrt(Y) + 0.07336·exp(2·C) | 0.549 | ○ rejected | Frozen-set skill 0.549 does not beat the incumbent's 0.563 by more than 0.01. | 0.133 | 0.142 |
| w1-c17 | Physics w1, complexity 17: Q/P = 0.3948^(H + Y + 0.164) - 0.08257·A + 0.08257·exp(2·C) + 0.008992 | 0.532 | ○ rejected | Frozen-set skill 0.532 does not beat the incumbent's 0.563 by more than 0.01. | 0.136 | 0.146 |
| w1-c18 | Physics w1, complexity 18: Q/P = 0.3708^H - 0.1101·A + 0.2202·sqrt(C) - 0.1101·Y + 0.1101·log(H) | 0.515 | ○ rejected | Frozen-set skill 0.515 does not beat the incumbent's 0.563 by more than 0.01. | 0.13 | 0.146 |
| w1-c20 | Physics w1, complexity 20: Q/P = 0.2982^H - 0.0972·A + 0.0972·C - 0.1944·Y + 0.0972·exp(sqrt(C)) + 0.0972·log(H) | 0.532 | ○ rejected | Frozen-set skill 0.532 does not beat the incumbent's 0.563 by more than 0.01. | 0.133 | 0.145 |
| w1-c21 | Physics w1, complexity 21: Q/P = 0.3198^H - 0.09998·A + 0.09998·C - 0.09998·sqrt(Y) - 0.09998·Y + 0.09998·exp(sqrt(C)) + 0.09998·log(H) | 0.533 | ○ rejected | Frozen-set skill 0.533 does not beat the incumbent's 0.563 by more than 0.01. | 0.132 | 0.145 |
| w1-c22 | Physics w1, complexity 22: Q/P = 0.3026^H - 0.09795·A + 0.09795·sqrt(C) - 0.09795·sqrt(Y) - 0.09795·Y + 0.09795·exp(sqrt(C)) + 0.09795·log(H) | 0.534 | ○ rejected | Frozen-set skill 0.534 does not beat the incumbent's 0.563 by more than 0.01. | 0.132 | 0.145 |
| w1-c24 | Physics w1, complexity 24: Q/P = 0.3383^H - 0.1025·A + 0.1025·sqrt(C) - 0.1025·sqrt(Y) - 0.1025·Y + 0.1025·exp(sqrt(C)) + 0.1025·log(H) - 0.03702 | 0.534 | ○ rejected | Frozen-set skill 0.534 does not beat the incumbent's 0.563 by more than 0.01. | 0.132 | 0.145 |
| w2-c3 | Physics w2, complexity 3: Q/P = 0.338/H | 0.526 | ○ rejected | Frozen-set skill 0.526 does not beat the incumbent's 0.563 by more than 0.01. | 0.14 | 0.14 |
| w2-c7 | Physics w2, complexity 7: Q/P = (A + 3.129)^(C - H) | 0.564 | ○ rejected | Frozen-set skill 0.564 does not beat the incumbent's 0.563 by more than 0.01. | 0.143 | 0.142 |
| w2-c9 | Physics w2, complexity 9: Q/P = (A + Y + 2.92)^(C - H) | 0.571 | ○ rejected | Frozen-set skill 0.571 does not beat the incumbent's 0.563 by more than 0.01. | 0.145 | 0.144 |
| w2-c11 | Physics w2, complexity 11: Q/P = 1.318·(A + 3.951)^(0.3666·C - H) | 0.604 | ○ rejected | Frozen-set skill 0.604 beats 0.563, but the dev error 0.1404 is more than 5% worse than 0.1319. | 0.14 | 0.138 |
| w2-c13 | Physics w2, complexity 13: Q/P = 1.84·(A + 3.733)^(0.3644·C - H - 0.2988) | 0.609 | **◆ released** | Frozen-set skill 0.609 > 0.563 + 0.01, dev error 0.13 vs 0.1319 (within 5%). | 0.13 | 0.129 |
| w2-c15 | Physics w2, complexity 15: Q/P = 1.698·(A + Y + 3.41)^(0.4044·C - H - 0.2736) | 0.612 | ○ rejected | Frozen-set skill 0.612 does not beat the incumbent's 0.609 by more than 0.01. | 0.133 | 0.131 |
| w2-c17 | Physics w2, complexity 17: Q/P = (228.9·A + 231.3)^(0.144·C - 0.144·H - 0.07124)/H^0.2794 | 0.605 | ○ rejected | Frozen-set skill 0.605 does not beat the incumbent's 0.609 by more than 0.01. | 0.118 | 0.122 |
| w2-c19 | Physics w2, complexity 19: Q/P = (28.5·A + 31.42)^(C/4 - H/4 - 0.06725)/sqrt(H^0.4947 + Y) | 0.608 | ○ rejected | Frozen-set skill 0.608 does not beat the incumbent's 0.609 by more than 0.01. | 0.121 | 0.123 |
| w2-c22 | Physics w2, complexity 22: Q/P = (18.19·A + 21.43)^(C/4 - H/4 - 0.0992)/sqrt(H^((Y + 0.6795)^2) + Y) | 0.613 | ○ rejected | Frozen-set skill 0.613 does not beat the incumbent's 0.609 by more than 0.01. | 0.122 | 0.124 |
| w2-c23 | Physics w2, complexity 23: Q/P = (125.3·A + 126.3)^(0.1637·C - 0.1637·H - 0.06041)/sqrt(H^(Y + 0.469) + Y) | 0.611 | ○ rejected | Frozen-set skill 0.611 does not beat the incumbent's 0.609 by more than 0.01. | 0.121 | 0.124 |

<sub>Columns after *Why*: Dev error (the gate), then the real exam metric reported after the fact (never used to choose).</sub>

### LawFinder Daily fire growth (FireSim)

**Objective:** R^2 of log10(new area km² + 1 cell) on the frozen simulated set  
**Rule:** Keep a rung if its frozen-set R^2 beats the incumbent by more than 0.01 AND its dev error is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; after a physics change every rung is re-scored on the new frozen set and every decision re-taken.

| Rung | Change | Frozen-set score | Decision | Why | Dev error | Exam mean |log10 ratio| of new area |
|---|---|---|---|---|---|---|
| r0 | A constant (the simulator mean): log₁₀ G = 0.4762 | -0.243 | ● kept | First rung (the incumbent to beat). | 1.39 | 1.38 |
| f1-c3 | Physics f1, complexity 3: log₁₀ G = log(H)/2 | 0.355 | ○ rejected | Frozen-set skill 0.355 beats -0.243, but the dev error 2.062 is more than 5% worse than 1.393. | 2.06 | 2.02 |
| f1-c4 | Physics f1, complexity 4: log₁₀ G = log(H) - 1.156 | 0.655 | ○ rejected | Frozen-set skill 0.655 beats -0.243, but the dev error 2.338 is more than 5% worse than 1.393. | 2.34 | 2.22 |
| f1-c5 | Physics f1, complexity 5: log₁₀ G = log(sqrt(H) - 0.4081) | 0.776 | ○ rejected | Frozen-set skill 0.776 beats -0.243, but the dev error 1.953 is more than 5% worse than 1.393. | 1.95 | 1.9 |
| f1-c6 | Physics f1, complexity 6: log₁₀ G = 0.6681·log(H) - 0.6139 | 0.803 | ○ rejected | Frozen-set skill 0.803 beats -0.243, but the dev error 1.946 is more than 5% worse than 1.393. | 1.95 | 1.88 |
| f1-c7 | Physics f1, complexity 7: log₁₀ G = log(H^0.6117/A^0.3059) | 0.791 | ○ rejected | Frozen-set skill 0.791 beats -0.243, but the dev error 1.799 is more than 5% worse than 1.393. | 1.8 | 1.72 |
| f1-c9 | Physics f1, complexity 9: log₁₀ G = log((H - 0.1185)^0.5978/A^0.2989) | 0.825 | ○ rejected | Frozen-set skill 0.825 beats -0.243, but the dev error 1.772 is more than 5% worse than 1.393. | 1.77 | 1.69 |
| f1-c10 | Physics f1, complexity 10: log₁₀ G = -0.02859·A + log(sqrt(H) - 0.296) | 0.820 | ○ rejected | Frozen-set skill 0.820 beats -0.243, but the dev error 1.782 is more than 5% worse than 1.393. | 1.78 | 1.71 |
| f1-c11 | Physics f1, complexity 11: log₁₀ G = -0.03161·A + 0.03161·log((sqrt(H) - 0.2817)^32.34) | 0.817 | ○ rejected | Frozen-set skill 0.817 beats -0.243, but the dev error 1.792 is more than 5% worse than 1.393. | 1.79 | 1.72 |
| f1-c13 | Physics f1, complexity 13: log₁₀ G = 1.033·0.9502^A + 1.033·log(0.3897·sqrt(H) - 0.1091) | 0.820 | ○ rejected | Frozen-set skill 0.820 beats -0.243, but the dev error 1.782 is more than 5% worse than 1.393. | 1.78 | 1.71 |
| f1-c14 | Physics f1, complexity 14: log₁₀ G = -0.03343·A + log(1.014^Y·sqrt(H) - 0.3117) | 0.803 | ○ rejected | Frozen-set skill 0.803 beats -0.243, but the dev error 1.762 is more than 5% worse than 1.393. | 1.76 | 1.69 |
| f1-c15 | Physics f1, complexity 15: log₁₀ G = 0.9481^A + 0.01894·Y + log(0.3843·sqrt(H) - 0.1181) | 0.805 | ○ rejected | Frozen-set skill 0.805 beats -0.243, but the dev error 1.744 is more than 5% worse than 1.393. | 1.74 | 1.68 |
| f1-c17 | Physics f1, complexity 17: log₁₀ G = 0.9667^(A - 8.841) + 0.01897·Y + log(0.2645·sqrt(H) - 0.08) | 0.804 | ○ rejected | Frozen-set skill 0.804 beats -0.243, but the dev error 1.746 is more than 5% worse than 1.393. | 1.75 | 1.68 |
| f1-c21 | Physics f1, complexity 21: log₁₀ G = -1.021^A + log(2.49·sqrt(H) - 0.736) + 0.03637·Y/1.131^A | 0.802 | ○ rejected | Frozen-set skill 0.802 beats -0.243, but the dev error 1.766 is more than 5% worse than 1.393. | 1.77 | 1.7 |
| f1-c23 | Physics f1, complexity 23: log₁₀ G = -1.021^A + 9.705e+68·Y/(A^80.2 + 4.269e+70) + log(2.486·sqrt(H) - 0.7324) | 0.801 | ○ rejected | Frozen-set skill 0.801 beats -0.243, but the dev error 1.764 is more than 5% worse than 1.393. | 1.76 | 1.7 |
| f1-c24 | Physics f1, complexity 24: log₁₀ G = -1.122^(sqrt(A)) + 3.787e+50·Y/(A^59.07 + 1.725e+52) + log(2.993·sqrt(H) - 0.8962) | 0.803 | ○ rejected | Frozen-set skill 0.803 beats -0.243, but the dev error 1.75 is more than 5% worse than 1.393. | 1.75 | 1.68 |
| f1-c25 | Physics f1, complexity 25: log₁₀ G = 0.9599^A + 1.754e+68·Y/(A^79.16 + 7.97e+69) + log(2.058·sqrt(H) - 0.6206) - 1.731 | 0.803 | ○ rejected | Frozen-set skill 0.803 beats -0.243, but the dev error 1.749 is more than 5% worse than 1.393. | 1.75 | 1.68 |
| f2-c3 | Physics f2, complexity 3: log₁₀ G = -0.5585/H | 0.408 | **◆ released** | Frozen-set skill 0.408 > -0.243 + 0.01, dev error 1.173 vs 1.393 (within 5%). | 1.17 | 1.15 |
| f2-c4 | Physics f2, complexity 4: log₁₀ G = 0.6305 - 1/H | 0.701 | ○ rejected | Frozen-set skill 0.701 beats 0.408, but the dev error 1.406 is more than 5% worse than 1.173. | 1.41 | 1.39 |
| f2-c5 | Physics f2, complexity 5: log₁₀ G = log(sqrt(H) - 0.4312) | 0.780 | ○ rejected | Frozen-set skill 0.780 beats 0.408, but the dev error 1.946 is more than 5% worse than 1.173. | 1.95 | 1.9 |
| f2-c6 | Physics f2, complexity 6: log₁₀ G = 0.6979·log(H) - 0.7956 | 0.826 | ○ rejected | Frozen-set skill 0.826 beats 0.408, but the dev error 1.866 is more than 5% worse than 1.173. | 1.87 | 1.8 |
| f2-c8 | Physics f2, complexity 8: log₁₀ G = 0.5834·log(H - 0.1998) - 0.5702 | 0.837 | ○ rejected | Frozen-set skill 0.837 beats 0.408, but the dev error 1.764 is more than 5% worse than 1.173. | 1.76 | 1.71 |
| f2-c11 | Physics f2, complexity 11: log₁₀ G = 0.5559·log(H/(sqrt(A)/sqrt(H) + 1.698)) | 0.845 | ○ rejected | Frozen-set skill 0.845 beats 0.408, but the dev error 1.773 is more than 5% worse than 1.173. | 1.77 | 1.71 |
| f2-c12 | Physics f2, complexity 12: log₁₀ G = 0.1925·log((H^3.244 - 0.02073)/A) - 0.3732 | 0.850 | ○ rejected | Frozen-set skill 0.850 beats 0.408, but the dev error 1.712 is more than 5% worse than 1.173. | 1.71 | 1.64 |
| f2-c13 | Physics f2, complexity 13: log₁₀ G = 0.5763·log((H - 0.1917)/(sqrt(A)/sqrt(C) + 1.427)) | 0.855 | ○ rejected | Frozen-set skill 0.855 beats 0.408, but the dev error 1.688 is more than 5% worse than 1.173. | 1.69 | 1.64 |
| f2-c15 | Physics f2, complexity 15: log₁₀ G = 0.4934·log(H/(sqrt(A)/sqrt(C) + 0.3745)) - 0.4934/sqrt(H) | 0.857 | ○ rejected | Frozen-set skill 0.857 beats 0.408, but the dev error 1.641 is more than 5% worse than 1.173. | 1.64 | 1.6 |
| f2-c16 | Physics f2, complexity 16: log₁₀ G = -0.4719/H^0.608 + 0.4719·log(H/(sqrt(A)/sqrt(C) + 0.4127)) | 0.858 | ○ rejected | Frozen-set skill 0.858 beats 0.408, but the dev error 1.622 is more than 5% worse than 1.173. | 1.62 | 1.58 |
| f2-c17 | Physics f2, complexity 17: log₁₀ G = 0.8326·log((H - 0.1903)^0.6851/(sqrt(A)/sqrt(C) + 1.661)) + 0.3238 | 0.857 | ○ rejected | Frozen-set skill 0.857 beats 0.408, but the dev error 1.655 is more than 5% worse than 1.173. | 1.65 | 1.61 |

<sub>Columns after *Why*: Dev error (the gate), then the real exam metric reported after the fact (never used to choose).</sub>

### LawFinder GB system inertia (GridSim)

**Objective:** R^2 of system inertia E on the frozen simulated set  
**Rule:** Keep a rung if its frozen-set R^2 beats the incumbent by more than 0.01 AND its dev error is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; after a physics change every rung is re-scored on the new frozen set and every decision re-taken.

| Rung | Change | Frozen-set score | Decision | Why | Dev error | Exam MdAPE of inertia |
|---|---|---|---|---|---|---|
| r0 | A constant (the simulator mean): E = 98.48 | -0.001 | ● kept | First rung (the incumbent to beat). | 0.6 | 53.1 |
| i1-c3 | Physics i1, complexity 3: E = 4.232·H | 0.716 | ● kept | Frozen-set skill 0.716 > -0.001 + 0.01, dev error 0.3962 vs 0.5996 (within 5%). | 0.396 | 34.1 |
| i1-c5 | Physics i1, complexity 5: E = H·(4.654 - C) | 0.796 | ● kept | Frozen-set skill 0.796 > 0.716 + 0.01, dev error 0.3913 vs 0.3962 (within 5%). | 0.391 | 34.1 |
| i1-c6 | Physics i1, complexity 6: E = 6.38·H·exp(-C) | 0.946 | ● kept | Frozen-set skill 0.946 > 0.796 + 0.01, dev error 0.3753 vs 0.3913 (within 5%). | 0.375 | 33.0 |
| i1-c7 | Physics i1, complexity 7: E = H·(6.603 - 5.617·C) | 0.964 | **◆ released** | Frozen-set skill 0.964 > 0.946 + 0.01, dev error 0.3688 vs 0.3753 (within 5%). | 0.369 | 33.0 |
| i1-c9 | Physics i1, complexity 9: E = (1.177 - C)·(5.403·H + 7.339) | 0.966 | ○ rejected | Frozen-set skill 0.966 does not beat the incumbent's 0.964 by more than 0.01. | 0.369 | 32.8 |
| i1-c11 | Physics i1, complexity 11: E = (C - 1.104)·(-H·(C + 5.642) - 8.104) | 0.966 | ○ rejected | Frozen-set skill 0.966 does not beat the incumbent's 0.964 by more than 0.01. | 0.364 | 32.5 |
| i1-c12 | Physics i1, complexity 12: E = (23.22·H + 31.88)·(log(4.729 - C) - 1.283) | 0.966 | ○ rejected | Frozen-set skill 0.966 does not beat the incumbent's 0.964 by more than 0.01. | 0.365 | 32.6 |
| i1-c15 | Physics i1, complexity 15: E = (5.803·H·log(4.259 - C) + 11.03)·log(2.115 - C) | 0.966 | ○ rejected | Frozen-set skill 0.966 does not beat the incumbent's 0.964 by more than 0.01. | 0.365 | 32.6 |
| i1-c22 | Physics i1, complexity 22: E = (23.32·H + 31.75)·(log(4.741 - C) - 1.287) | 0.965 | ○ rejected | Not finite on the frozen set or on dev. | – | 33.9 |

<sub>Columns after *Why*: Dev error (the gate), then the real exam metric reported after the fact (never used to choose).</sub>

### LawFinder GB frequency nadir (GridSim)

**Objective:** R^2 of frequency drop to the nadir f0 - f_min on the frozen simulated set  
**Rule:** Keep a rung if its frozen-set R^2 beats the incumbent by more than 0.01 AND its dev error is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; after a physics change every rung is re-scored on the new frozen set and every decision re-taken.

| Rung | Change | Frozen-set score | Decision | Why | Dev error | Exam MAE of the nadir depth |
|---|---|---|---|---|---|---|
| r0 | A constant (the simulator mean): Δf = 0.4049 | -0.140 | ● kept | First rung (the incumbent to beat). | 0.086 | 0.163 |
| g1-c3 | Physics g1, complexity 3: Δf = 0.3074·H | -0.071 | ● kept | Frozen-set skill -0.071 > -0.140 + 0.01, dev error 0.03707 vs 0.08596 (within 5%). | 0.0371 | 0.0725 |
| g1-c5 | Physics g1, complexity 5: Δf = H/C^0.2191 | -0.049 | ○ rejected | Frozen-set skill -0.049 beats -0.071, but the dev error 0.0517 is more than 5% worse than 0.03707. | 0.0517 | 0.0733 |
| g1-c7 | Physics g1, complexity 7: Δf = 1.848^H/C^0.3211 | -0.008 | ○ rejected | Frozen-set skill -0.008 beats -0.071, but the dev error 0.04237 is more than 5% worse than 0.03707. | 0.0424 | 0.0613 |
| g1-c9 | Physics g1, complexity 9: Δf = (H + 1.984)^2/log(C)^1.959 | -0.004 | ○ rejected | Frozen-set skill -0.004 beats -0.071, but the dev error 0.04475 is more than 5% worse than 0.03707. | 0.0447 | 0.0614 |
| g1-c12 | Physics g1, complexity 12: Δf = (H + 2)^2/(52940·C - 3.541e+6)^0.2084 | -0.005 | **◆ released** | Frozen-set skill -0.005 > -0.071 + 0.01, dev error 0.0387 vs 0.03707 (within 5%). | 0.0387 | 0.06 |
| g1-c13 | Physics g1, complexity 13: Δf = (H + 0.3614)/(A·(C - 126.2) + 1474)^0.1698 | -0.016 | ○ rejected | Frozen-set skill -0.016 does not beat the incumbent's -0.005 by more than 0.01. | 0.0427 | 0.0614 |
| g2-c3 | Physics g2, complexity 3: Δf = 0.4334·H | 0.123 | ○ rejected | Frozen-set skill 0.123 beats -0.005, but the dev error 0.09859 is more than 5% worse than 0.0387. | 0.0986 | 0.106 |
| g2-c4 | Physics g2, complexity 4: Δf = exp(-0.6625/H) | 0.156 | ○ rejected | Frozen-set skill 0.156 beats -0.005, but the dev error 0.1733 is more than 5% worse than 0.0387. | 0.173 | 0.159 |
| g2-c5 | Physics g2, complexity 5: Δf = 0.3263·H + 0.1486 | 0.160 | ○ rejected | Frozen-set skill 0.160 beats -0.005, but the dev error 0.1377 is more than 5% worse than 0.0387. | 0.138 | 0.138 |
| g2-c6 | Physics g2, complexity 6: Δf = 7.511·sqrt(H)/sqrt(C) | 0.199 | ○ rejected | Frozen-set skill 0.199 beats -0.005, but the dev error 0.1396 is more than 5% worse than 0.0387. | 0.14 | 0.178 |
| g2-c7 | Physics g2, complexity 7: Δf = 0.3208·H + 33.36/C | 0.203 | ○ rejected | Frozen-set skill 0.203 beats -0.005, but the dev error 0.121 is more than 5% worse than 0.0387. | 0.121 | 0.146 |
| g2-c8 | Physics g2, complexity 8: Δf = 3.222·H^0.703/C^0.3515 | 0.202 | ○ rejected | Frozen-set skill 0.202 beats -0.005, but the dev error 0.1282 is more than 5% worse than 0.0387. | 0.128 | 0.145 |
| g2-c10 | Physics g2, complexity 10: Δf = 8.534·H^0.6483/(A^0.3242·C^0.3242) | 0.197 | ○ rejected | Frozen-set skill 0.197 beats -0.005, but the dev error 0.1221 is more than 5% worse than 0.0387. | 0.122 | 0.141 |
| g2-c11 | Physics g2, complexity 11: Δf = 4.763·H^0.7074/(A^0.2593·C^0.2593) | 0.197 | ○ rejected | Frozen-set skill 0.197 beats -0.005, but the dev error 0.1227 is more than 5% worse than 0.0387. | 0.123 | 0.135 |
| g2-c14 | Physics g2, complexity 14: Δf = 6.854·H^0.7053/(A^0.2033·(C + 33.67)^0.3526) | 0.200 | ○ rejected | Frozen-set skill 0.200 beats -0.005, but the dev error 0.1237 is more than 5% worse than 0.0387. | 0.124 | 0.138 |

<sub>Columns after *Why*: Dev error (the gate), then the real exam metric reported after the fact (never used to choose).</sub>

### LawFinder Home energy signature (BuildingSim)

**Objective:** R^2 of site energy per floor area on the frozen simulated set  
**Rule:** Keep a rung if its frozen-set R^2 beats the incumbent by more than 0.01 AND its dev error is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; after a physics change every rung is re-scored on the new frozen set and every decision re-taken.

| Rung | Change | Frozen-set score | Decision | Why | Dev error | Exam MdAPE of annual site energy per m2 |
|---|---|---|---|---|---|---|
| r0 | A constant (the simulator mean): EUI = 225.8 | -0.171 | ● kept | First rung (the incumbent to beat). | 0.702 | 70.6 |
| b1-c3 | Physics b1, complexity 3: EUI = 88.91·H | 0.068 | ● kept | Frozen-set skill 0.068 > -0.171 + 0.01, dev error 0.6166 vs 0.7021 (within 5%). | 0.617 | 66.7 |
| b1-c4 | Physics b1, complexity 4: EUI = 154.6·sqrt(H) | 0.067 | ○ rejected | Frozen-set skill 0.067 does not beat the incumbent's 0.068 by more than 0.01. | 0.623 | 69.3 |
| b1-c5 | Physics b1, complexity 5: EUI = H·(2060 - Y) | 0.400 | ● kept | Frozen-set skill 0.400 > 0.068 + 0.01, dev error 0.6139 vs 0.6166 (within 5%). | 0.614 | 62.3 |
| b1-c6 | Physics b1, complexity 6: EUI = sqrt(H)·(2127 - Y) | 0.397 | ○ rejected | Frozen-set skill 0.397 does not beat the incumbent's 0.400 by more than 0.01. | 0.577 | 64.9 |
| b1-c7 | Physics b1, complexity 7: EUI = -H·(Y - 2029) + 91.76 | 0.471 | ● kept | Frozen-set skill 0.471 > 0.400 + 0.01, dev error 0.4942 vs 0.6139 (within 5%). | 0.494 | 54.1 |
| b1-c8 | Physics b1, complexity 8: EUI = (2044 - Y)·(H + 1/A) | 0.410 | ○ rejected | Frozen-set skill 0.410 does not beat the incumbent's 0.471 by more than 0.01. | 0.516 | 51.3 |
| b1-c10 | Physics b1, complexity 10: EUI = (2031 - Y)·(H + 1/A) + 48.46 | 0.439 | ○ rejected | Frozen-set skill 0.439 does not beat the incumbent's 0.471 by more than 0.01. | 0.459 | 47.2 |
| b1-c11 | Physics b1, complexity 11: EUI = -H·(13.86·A + Y - 2054) + 93.27 | 0.468 | ○ rejected | Frozen-set skill 0.468 does not beat the incumbent's 0.471 by more than 0.01. | 0.443 | 46.8 |
| b1-c13 | Physics b1, complexity 13: EUI = -(Y - 2033)·(H - log(A) + 0.5975) + 73.88 | 0.472 | ○ rejected | Frozen-set skill 0.472 does not beat the incumbent's 0.471 by more than 0.01. | 0.441 | 46.7 |
| b1-c14 | Physics b1, complexity 14: EUI = (sqrt(A) - 3.227)·(0.3258·H + 0.5516)·(Y - 2062) | 0.471 | ○ rejected | Frozen-set skill 0.471 does not beat the incumbent's 0.471 by more than 0.01. | 0.435 | 45.8 |
| b1-c15 | Physics b1, complexity 15: EUI = -(Y - 2034)·(H + 0.02081·Y - log(A) - 39.13) | 0.484 | ● kept | Frozen-set skill 0.484 > 0.471 + 0.01, dev error 0.4625 vs 0.4942 (within 5%). | 0.462 | 47.5 |
| b1-c16 | Physics b1, complexity 16: EUI = (-8.678·A - Y + 2045)·(H - 0.5712 + 1/A) + 83.55 | 0.446 | ○ rejected | Frozen-set skill 0.446 does not beat the incumbent's 0.484 by more than 0.01. | 0.415 | 43.7 |
| b1-c17 | Physics b1, complexity 17: EUI = -(7.063·A + Y - 2042)·(H - log(A) + 0.5505) + 83.24 | 0.455 | ○ rejected | Frozen-set skill 0.455 does not beat the incumbent's 0.484 by more than 0.01. | 0.411 | 43.0 |
| b1-c18 | Physics b1, complexity 18: EUI = (H - 0.5722 + 1/A)·(-8.617·A + C - Y + 2045) + 82.05 | 0.446 | ○ rejected | Frozen-set skill 0.446 does not beat the incumbent's 0.484 by more than 0.01. | 0.415 | 43.7 |
| b1-c19 | Physics b1, complexity 19: EUI = -(3.535·A + Y - 2039)·(H + 0.02296·Y - log(A) - 43.3) | 0.469 | ○ rejected | Frozen-set skill 0.469 does not beat the incumbent's 0.484 by more than 0.01. | 0.449 | 46.5 |
| b1-c21 | Physics b1, complexity 21: EUI = -(4.905·A + Y - 2039)·(H + 0.01541·Y - log(A) - 28.96) + 41 | 0.473 | ○ rejected | Frozen-set skill 0.473 does not beat the incumbent's 0.484 by more than 0.01. | 0.423 | 44.2 |
| b1-c22 | Physics b1, complexity 22: EUI = -(3.586·A + Y - log(C) - 2041)·(H + 0.02184·Y - log(A) - 41.02) | 0.472 | ○ rejected | Frozen-set skill 0.472 does not beat the incumbent's 0.484 by more than 0.01. | 0.44 | 46.0 |
| b1-c24 | Physics b1, complexity 24: EUI = -(3.702·A + Y - 3.211·log(C) - 2045)·(H + 0.01858·Y - log(A) - 34.9) | 0.471 | ○ rejected | Frozen-set skill 0.471 does not beat the incumbent's 0.484 by more than 0.01. | 0.437 | 44.7 |
| b1-c25 | Physics b1, complexity 25: EUI = -(H + log(1.019^Y/A) - 35.21)·(Y + log(A^6.681/C^3.308) - 2041) | 0.470 | ○ rejected | Frozen-set skill 0.470 does not beat the incumbent's 0.484 by more than 0.01. | 0.434 | 44.5 |
| b2-c3 | Physics b2, complexity 3: EUI = 72.42·H | 0.224 | ○ rejected | Frozen-set skill 0.224 does not beat the incumbent's 0.484 by more than 0.01. | 0.518 | 55.4 |
| b2-c5 | Physics b2, complexity 5: EUI = 5712 - 2.801·Y | 0.493 | ○ rejected | Frozen-set skill 0.493 does not beat the incumbent's 0.484 by more than 0.01. | 0.415 | 41.9 |
| b2-c6 | Physics b2, complexity 6: EUI = 2954 - 0.0007108·Y^2 | 0.493 | ○ rejected | Frozen-set skill 0.493 does not beat the incumbent's 0.484 by more than 0.01. | 0.416 | 42.0 |
| b2-c7 | Physics b2, complexity 7: EUI = H·(2044 - Y) | 0.559 | ○ rejected | Frozen-set skill 0.559 beats 0.484, but the dev error 0.5451 is more than 5% worse than 0.4625. | 0.545 | 53.9 |
| b2-c9 | Physics b2, complexity 9: EUI = 2021·H - Y·(H - 0.03403) | 0.648 | ● kept | Frozen-set skill 0.648 > 0.484 + 0.01, dev error 0.4063 vs 0.4625 (within 5%). | 0.406 | 39.7 |
| b2-c11 | Physics b2, complexity 11: EUI = (A·H·(2026 - Y) + 78.25)/A | 0.680 | ● kept | Frozen-set skill 0.680 > 0.648 + 0.01, dev error 0.4029 vs 0.4063 (within 5%). | 0.403 | 39.9 |
| b2-c13 | Physics b2, complexity 13: EUI = -H·Y + 2022·H + 186.5/1.926^A | 0.694 | **◆ released** | Frozen-set skill 0.694 > 0.680 + 0.01, dev error 0.3924 vs 0.4029 (within 5%). | 0.392 | 39.3 |
| b2-c14 | Physics b2, complexity 14: EUI = -H·Y + 2021·H - 59.16·log(A) + 95.05 | 0.695 | ○ rejected | Frozen-set skill 0.695 does not beat the incumbent's 0.694 by more than 0.01. | 0.385 | 39.1 |
| b2-c15 | Physics b2, complexity 15: EUI = -0.8643·H·Y + 1755·H + 192.7/1.991^A | 0.701 | ○ rejected | Frozen-set skill 0.701 does not beat the incumbent's 0.694 by more than 0.01. | 0.368 | 37.0 |
| b2-c16 | Physics b2, complexity 16: EUI = -0.8635·H·Y + 1753·H - 61.85·log(A) + 94.23 | 0.701 | ○ rejected | Frozen-set skill 0.701 does not beat the incumbent's 0.694 by more than 0.01. | 0.37 | 37.1 |
| b2-c17 | Physics b2, complexity 17: EUI = -0.0002195·H·Y^2 + 904.1·H - 61.74·log(A) + 94.2 | 0.701 | ○ rejected | Frozen-set skill 0.701 does not beat the incumbent's 0.694 by more than 0.01. | 0.37 | 37.2 |
| b2-c18 | Physics b2, complexity 18: EUI = 5.884e-7·H·((Y - 1925)^2 - 11710)^2 - 58.67·log(A) + 94.82 | 0.697 | ○ rejected | Frozen-set skill 0.697 does not beat the incumbent's 0.694 by more than 0.01. | 0.382 | 38.4 |
| b2-c20 | Physics b2, complexity 20: EUI = 5.621e-7·H^0.918·((Y - 1924)^2 - 12800)^2 - 58.73·log(A) + 86.8 | 0.697 | ○ rejected | Frozen-set skill 0.697 does not beat the incumbent's 0.694 by more than 0.01. | 0.395 | 39.2 |
| b2-c21 | Physics b2, complexity 21: EUI = -H^2 + 4.636e-7·H·((Y - 1923)^2 - 13640)^2 - 59.15·log(A) + 89.72 | 0.696 | ○ rejected | Frozen-set skill 0.696 does not beat the incumbent's 0.694 by more than 0.01. | 0.388 | 39.0 |
| b2-c22 | Physics b2, complexity 22: EUI = 2.479e-6·H·((Y - 1936)^2 - 4945)^2 + 19.79·H - 59.32·log(A) + 92.7 | 0.698 | ○ rejected | Frozen-set skill 0.698 does not beat the incumbent's 0.694 by more than 0.01. | 0.37 | 36.1 |
| b2-c23 | Physics b2, complexity 23: EUI = 54.54·sqrt(H) + 2.395e-6·H·((Y - 1936)^2 - 5057)^2 - 59.69·log(A) + 57.83 | 0.695 | ○ rejected | Frozen-set skill 0.695 does not beat the incumbent's 0.694 by more than 0.01. | 0.378 | 36.9 |
| b2-c24 | Physics b2, complexity 24: EUI = -H·(H - 24.7) + 2.447e-6·H·((Y - 1936)^2 - 4981)^2 - 59.64·log(A) + 88.01 | 0.697 | ○ rejected | Frozen-set skill 0.697 does not beat the incumbent's 0.694 by more than 0.01. | 0.375 | 36.4 |

<sub>Columns after *Why*: Dev error (the gate), then the real exam metric reported after the fact (never used to choose).</sub>

### LawFinder Trip energy per km (RoadSim)

**Objective:** R^2 of energy per km (fuel at LHV or DC battery) on the frozen simulated set  
**Rule:** Keep a rung if its frozen-set R^2 beats the incumbent by more than 0.01 AND its dev error is no more than 5% worse. Rungs = Pareto-front laws in order of complexity, then physics versions; after a physics change every rung is re-scored on the new frozen set and every decision re-taken.

| Rung | Change | Frozen-set score | Decision | Why | Dev error | Exam MdAPE of energy per km |
|---|---|---|---|---|---|---|
| r0 | A constant (the simulator mean): E = 2.532 | -0.013 | ● kept | First rung (the incumbent to beat). | 0.282 | 30.8 |
| p1-c2 | Physics p1, complexity 2: E = exp(D) | 0.058 | ○ rejected | Frozen-set skill 0.058 beats -0.012, but the dev error 0.3118 is more than 5% worse than 0.282. | 0.312 | 35.3 |
| p1-c3 | Physics p1, complexity 3: E = 2.752 - e | 0.228 | ○ rejected | Frozen-set skill 0.228 beats -0.012, but the dev error 0.3082 is more than 5% worse than 0.282. | 0.308 | 29.9 |
| p1-c4 | Physics p1, complexity 4: E = 2.998·exp(-e) | 0.345 | ○ rejected | Frozen-set skill 0.345 beats -0.012, but the dev error 0.3525 is more than 5% worse than 0.282. | 0.353 | 29.5 |
| p1-c5 | Physics p1, complexity 5: E = M·(1.377 - e) | 0.474 | ● kept | Frozen-set skill 0.474 > -0.012 + 0.01, dev error 0.2526 vs 0.282 (within 5%). | 0.253 | 29.4 |
| p1-c7 | Physics p1, complexity 7: E = M·exp(2·K - 2·e) | 0.567 | ● kept | Frozen-set skill 0.567 > 0.474 + 0.01, dev error 0.1884 vs 0.2526 (within 5%). | 0.188 | 27.5 |
| p1-c9 | Physics p1, complexity 9: E = M·(-e - 0.5278·h + 1.503) | 0.554 | ○ rejected | Frozen-set skill 0.554 does not beat the incumbent's 0.567 by more than 0.01. | 0.227 | 21.5 |
| p1-c10 | Physics p1, complexity 10: E = M·exp(-(K - e)·(h - 2.293)) | 0.658 | ● kept | Frozen-set skill 0.658 > 0.567 + 0.01, dev error 0.1684 vs 0.1884 (within 5%). | 0.168 | 21.6 |
| p1-c11 | Physics p1, complexity 11: E = M·exp(-2·(K - e)·(h - 1.27)) | 0.696 | ● kept | Frozen-set skill 0.696 > 0.658 + 0.01, dev error 0.1601 vs 0.1684 (within 5%). | 0.16 | 17.9 |
| p1-c12 | Physics p1, complexity 12: E = M·exp((1.618 - h)·(-e + v^(-0.5426))) | 0.717 | ○ rejected | Frozen-set skill 0.717 beats 0.696, but the dev error 0.1716 is more than 5% worse than 0.1601. | 0.172 | 19.2 |
| p1-c13 | Physics p1, complexity 13: E = M·exp(-2·e + 2·(v + 0.0096)^(-h - 0.6261)) | 0.727 | ○ rejected | Frozen-set skill 0.727 beats 0.696, but the dev error 0.1685 is more than 5% worse than 0.1601. | 0.169 | 18.0 |
| p1-c14 | Physics p1, complexity 14: E = M·exp((0.9649 - h)·(-2·e + 2/v^0.6057)) | 0.728 | ● kept | Frozen-set skill 0.728 > 0.696 + 0.01, dev error 0.1674 vs 0.1601 (within 5%). | 0.167 | 18.0 |
| p1-c17 | Physics p1, complexity 17: E = M·exp((0.3353·sqrt(R) - h)·(-2·e + 2/v^0.6288)) | 0.754 | ● kept | Frozen-set skill 0.754 > 0.728 + 0.01, dev error 0.1753 vs 0.1674 (within 5%). | 0.175 | 18.7 |
| p1-c18 | Physics p1, complexity 18: E = M·exp((-2·e + 2/(v + 3.136)^0.6499)·(sqrt(K)·sqrt(R) - h)) | 0.763 | ○ rejected | Frozen-set skill 0.763 does not beat the incumbent's 0.754 by more than 0.01. | 0.165 | 18.0 |
| p1-c19 | Physics p1, complexity 19: E = M·exp((-2·e + 2/(v + 0.9515)^0.6285)·(K + 0.09243·R - h)) | 0.749 | ○ rejected | Frozen-set skill 0.749 does not beat the incumbent's 0.754 by more than 0.01. | 0.178 | 19.9 |
| p1-c20 | Physics p1, complexity 20: E = M·exp((-2·e + 2/(v + 2.013)^0.5779)·(0.7864·sqrt(K)·sqrt(R) - h)) | 0.764 | ○ rejected | Frozen-set skill 0.764 does not beat the incumbent's 0.754 by more than 0.01. | 0.159 | 18.1 |
| p1-c21 | Physics p1, complexity 21: E = M·exp((-2·e + 2/(v + 1.821)^0.5851)·(2.403·K + 0.06157·R - h)) | 0.764 | ○ rejected | Frozen-set skill 0.764 does not beat the incumbent's 0.754 by more than 0.01. | 0.165 | 18.5 |
| p1-c23 | Physics p1, complexity 23: E = M·exp((-2·e + 2/(v + 1.48)^0.5951)·(0.06983·G + 2·K + 0.06983·R - h)) | 0.767 | **◆ released** | Frozen-set skill 0.767 > 0.754 + 0.01, dev error 0.168 vs 0.1753 (within 5%). | 0.168 | 18.6 |
| p1-c25 | Physics p1, complexity 25: E = (M - 0.03307)·exp((-2·e + 2/(v + 1.555)^0.5865)·(0.07119·G + 2·K + 0.07119·R - h)) | 0.766 | ○ rejected | Frozen-set skill 0.766 does not beat the incumbent's 0.767 by more than 0.01. | 0.171 | 19.0 |
| p2-c2 | Physics p2, complexity 2: E = sqrt(R) | 0.102 | ○ rejected | Frozen-set skill 0.102 does not beat the incumbent's 0.767 by more than 0.01. | 0.335 | 29.1 |
| p2-c3 | Physics p2, complexity 3: E = 2.91 - e | 0.239 | ○ rejected | Frozen-set skill 0.239 does not beat the incumbent's 0.767 by more than 0.01. | 0.338 | 29.5 |
| p2-c4 | Physics p2, complexity 4: E = 1/(e + 0.3088) | 0.371 | ○ rejected | Frozen-set skill 0.371 does not beat the incumbent's 0.767 by more than 0.01. | 0.386 | 30.9 |
| p2-c5 | Physics p2, complexity 5: E = M/(e + 0.6681) | 0.448 | ○ rejected | Frozen-set skill 0.448 does not beat the incumbent's 0.767 by more than 0.01. | 0.256 | 27.6 |
| p2-c6 | Physics p2, complexity 6: E = sqrt(K)/(e + 0.131) | 0.419 | ○ rejected | Frozen-set skill 0.419 does not beat the incumbent's 0.767 by more than 0.01. | 0.231 | 28.6 |
| p2-c7 | Physics p2, complexity 7: E = (M - h)/(e + 0.5706) | 0.561 | ○ rejected | Frozen-set skill 0.561 does not beat the incumbent's 0.767 by more than 0.01. | 0.288 | 23.7 |
| p2-c8 | Physics p2, complexity 8: E = (M - h)/(e + 0.7497)^2 | 0.635 | ○ rejected | Frozen-set skill 0.635 does not beat the incumbent's 0.767 by more than 0.01. | 0.249 | 23.1 |
| p2-c9 | Physics p2, complexity 9: E = K·(M - h)/(e + 0.1086) | 0.623 | ○ rejected | Frozen-set skill 0.623 does not beat the incumbent's 0.767 by more than 0.01. | 0.247 | 25.4 |
| p2-c10 | Physics p2, complexity 10: E = sqrt(K)·(M - h)/(e + 0.2383) | 0.711 | ○ rejected | Frozen-set skill 0.711 does not beat the incumbent's 0.767 by more than 0.01. | 0.152 | 20.9 |
| p2-c11 | Physics p2, complexity 11: E = sqrt(K)·(M - h)/(exp(e) - 0.7619) | 0.711 | ○ rejected | Frozen-set skill 0.711 does not beat the incumbent's 0.767 by more than 0.01. | 0.147 | 21.0 |
| p2-c12 | Physics p2, complexity 12: E = sqrt(K)·(M - h)^0.8128/(e + 0.2077) | 0.721 | ○ rejected | Frozen-set skill 0.721 does not beat the incumbent's 0.767 by more than 0.01. | 0.153 | 17.0 |
| p2-c14 | Physics p2, complexity 14: E = sqrt(K)·(M - h)^(v^(-0.1335))/(e + 0.1979) | 0.733 | ○ rejected | Frozen-set skill 0.733 does not beat the incumbent's 0.767 by more than 0.01. | 0.149 | 15.4 |
| p2-c15 | Physics p2, complexity 15: E = sqrt(K)·(M - h)^(log(v)^(-0.3821))/(e + 0.1991) | 0.736 | ○ rejected | Frozen-set skill 0.736 does not beat the incumbent's 0.767 by more than 0.01. | 0.147 | 15.3 |
| p2-c16 | Physics p2, complexity 16: E = ((M - h)^(v^(-0.4844)) - 0.6371)·exp(K)/(e + 0.2089) | 0.737 | ○ rejected | Frozen-set skill 0.737 does not beat the incumbent's 0.767 by more than 0.01. | 0.208 | 17.9 |
| p2-c17 | Physics p2, complexity 17: E = ((M - h)^(v^(-0.2933)) - 0.4207)·exp(2·K)/(e + 0.6392)^2 | 0.741 | ○ rejected | Frozen-set skill 0.741 does not beat the incumbent's 0.767 by more than 0.01. | 0.178 | 17.5 |
| p2-c18 | Physics p2, complexity 18: E = sqrt(K)·(M - h)^(v^((G - 1.132)/R))/(e + 0.1985) | 0.741 | ○ rejected | Frozen-set skill 0.741 does not beat the incumbent's 0.767 by more than 0.01. | 0.144 | 15.2 |
| p2-c20 | Physics p2, complexity 20: E = sqrt(K)·(-K + (M - h)^(v^((G - 1.551)/R)))/(e + 0.1639) | 0.750 | ○ rejected | Frozen-set skill 0.750 does not beat the incumbent's 0.767 by more than 0.01. | 0.155 | 16.1 |
| p2-c21 | Physics p2, complexity 21: E = sqrt(K)·(-K + (M - h)^(v^((G - 1.594)/R)))/sqrt(e + 0.02657) | 0.752 | ○ rejected | Frozen-set skill 0.752 does not beat the incumbent's 0.767 by more than 0.01. | 0.158 | 15.9 |
| p2-c22 | Physics p2, complexity 22: E = sqrt(K)·sqrt(M)·(-K + (M - h)^(v^((G - 2.647)/R)))/(e + 0.2163) | 0.763 | ○ rejected | Frozen-set skill 0.763 does not beat the incumbent's 0.767 by more than 0.01. | 0.129 | 17.6 |
| p2-c24 | Physics p2, complexity 24: E = sqrt(K)·(-K + (1.483·M - 1.483·h)^(v^((G - 2.132)/(M + R))))/(e + 0.2169) | 0.764 | ○ rejected | Frozen-set skill 0.764 does not beat the incumbent's 0.767 by more than 0.01. | 0.138 | 14.6 |
| p2-c25 | Physics p2, complexity 25: E = sqrt(K)·sqrt(M)·(-K·h + (M - h)^(v^((2·G - 4.154)/R)))/(e + 0.2193) | 0.749 | ○ rejected | Frozen-set skill 0.749 does not beat the incumbent's 0.767 by more than 0.01. | 0.126 | 14.6 |
| p2s-c2 | Physics p2, sign-checked search (27 Nov 2026), complexity 2: E = sqrt(R) | 0.102 | ○ rejected | Frozen-set skill 0.102 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.335 | not run (exam closed) |
| p2s-c3 | Physics p2, sign-checked search (27 Nov 2026), complexity 3: E = 2.91 - e | 0.239 | ○ rejected | Frozen-set skill 0.239 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.338 | not run (exam closed) |
| p2s-c5 | Physics p2, sign-checked search (27 Nov 2026), complexity 5: E = M/(e + 0.6681) | 0.448 | ○ rejected | Frozen-set skill 0.448 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.256 | not run (exam closed) |
| p2s-c6 | Physics p2, sign-checked search (27 Nov 2026), complexity 6: E = M/(e + 0.8112)^2 | 0.508 | ○ rejected | Frozen-set skill 0.508 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.221 | not run (exam closed) |
| p2s-c7 | Physics p2, sign-checked search (27 Nov 2026), complexity 7: E = M·exp(2·K - 2·e) | 0.567 | ○ rejected | Frozen-set skill 0.567 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.188 | not run (exam closed) |
| p2s-c8 | Physics p2, sign-checked search (27 Nov 2026), complexity 8: E = M·exp(2.209·K - 2.209·e) | 0.573 | ○ rejected | Frozen-set skill 0.573 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.185 | not run (exam closed) |
| p2s-c9 | Physics p2, sign-checked search (27 Nov 2026), complexity 9: E = M·exp(2.282·K - 2·e) | 0.578 | ○ rejected | Frozen-set skill 0.578 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.17 | not run (exam closed) |
| p2s-c11 | Physics p2, sign-checked search (27 Nov 2026), complexity 11: E = M·exp(2·K^h·(K - e)) | 0.632 | ○ rejected | Frozen-set skill 0.632 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.178 | not run (exam closed) |
| p2s-c12 | Physics p2, sign-checked search (27 Nov 2026), complexity 12: E = M·exp(0.1969^h·(2.716·K - e)) | 0.651 | ○ rejected | Frozen-set skill 0.651 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.156 | not run (exam closed) |
| p2s-c13 | Physics p2, sign-checked search (27 Nov 2026), complexity 13: E = M·exp(2·K^h·(1.474·K - e)) | 0.725 | ○ rejected | Frozen-set skill 0.725 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.141 | not run (exam closed) |
| p2s-c15 | Physics p2, sign-checked search (27 Nov 2026), complexity 15: E = G + M·exp(2·K^h·(1.465·K - e)) | 0.725 | ○ rejected | Frozen-set skill 0.725 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.159 | not run (exam closed) |
| p2s-c17 | Physics p2, sign-checked search (27 Nov 2026), complexity 17: E = (M - 0.009655·T)·exp(2·K^h·(1.633·K - e)) | 0.737 | ○ rejected | Frozen-set skill 0.737 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.13 | not run (exam closed) |
| p2s-c19 | Physics p2, sign-checked search (27 Nov 2026), complexity 19: E = (M + 0.27·R/v)·exp(-2·K^h·(-1.142·K + e)) | 0.757 | ○ rejected | Frozen-set skill 0.757 does not beat the incumbent's 0.767 by more than 0.01. Sign check passed on val_p2. | 0.137 | not run (exam closed) |

<sub>Columns after *Why*: Dev error (the gate), then the real exam metric reported after the fact (never used to choose).</sub>

Raw per-rung logs are in [`training/`](training/).

## Physics

No new physics: the collection's shared simulators are used unchanged (WaterSim v1, FireSim v1, GridSim 1.1 via GridPulse's GB state sampler, BuildingSim 1.0/1.4, RoadSim physics v1/v2). Each law is a statement about what one simulator implies.

**Simulator**

- WaterSim: 16,000 generated basins (two physics versions), 4 runoff models, 2 snow routines, real ERA5-Land forcing from free windows, 10-year runs; w2 adds FlowCast's dev-fitted ERA5 water-yield multiplier
- FireSim: 1,680 multi-day fires on guarded real dev LANDFIRE windows or parametric landscapes, real dev HRRR weather; f2 uses the dev-calibrated ROS factors
- GridSim: generate_states GB fleets (inertia law); GridPulse's GB sampler p0/p1 for trips, measured like GridData's catalogue (1 s means, 10 s fall <= -0.15 Hz)
- BuildingSim: 2,600 single-family homes from the ResStock-bootstrapped generator, TMYx weather shifted by U(-3, 3) C; b2 = physics 1.4
- RoadSim: 16,000 trips (two physics versions) of light-duty generator vehicles (petrol, three hybrid kinds, battery-electric) on RoadData's train_noise VED routes (real Ann Arbor routes, lidar grade, airport weather) and on dynamometer schedules; p2 = physics v2 (dev-calibrated on-road loads and maps)

**Checked against independent references**

- lawsearch recovers a planted Budyko-Fu law with 1 % noise to the noise floor (MSE 1.0e-4 vs 1.0e-4)
- Every simulator's own Phase 1 checks apply (shared/*/checks)
- Release evaluator (numpy) equals the search engine to machine precision; the page's JS equals numpy (training/results/js_parity.json; road law: 6e-16 on 1,500 points, js_parity_road.json)

## Help wanted

- Daymet/NLDAS forcing would separate law skill from ERA5-Land bias (WaterData help_wanted)

## Next rungs

- Fire: add a persistence module to FireSim (stall and burnout probability by fuel and weather) and re-run the search; the law ladder shows exactly where FireSim's growth departs from reality.
- Grid inertia: rescale GridSim's GB fleet to NESO's inertia per MW (the GridPulse wrapper's fix) as physics i2, and re-take the ladder; the shape term (1 − 0.85 s) should survive.
- Water: a recession law −dQ/dt = a Q^b on the same banks, and a Daymet/NLDAS exam to separate law skill from ERA5-Land bias.
- Road: give the law the car's engine size or label fuel economy, cold start and trip length as inputs (RoadData has them), and widen the generator toward lighter cars.
- Road: a powertrain-aware grade term. The sign-checked search (27 Nov 2026, rungs p2s-*) found no law that reads grade with the right sign; its best, p2s-c19, ignores grade and scored R^2 0.757 against 0.767, so it was rejected and p1-c23 stays. Options for the owner: an EV dev check as a rule, or a search seeded with a road-load grade term per powertrain.
- Homes: add heating fuel and system as inputs (a law per fuel) and occupants; RECS dev shows the fuel matters most.

## Files

| Path | What |
|---|---|
| `lawfinder.py` | Inference — numpy/scipy only |
| `lawfinder_laws.json` | Weights |
| `demo.py`, `examples/` | Demo on real exam examples (fixed random draw, not picked by result) |
| `MODEL_CARD.md` | Full model card: every number, every caveat |
| `model.yaml`, `ratchet.yaml` | Structured card and ladder (this README, the CLI and the site are generated from them) |
| `training/` | Every script that built it, as run; paths via `training/_paths.py` |
| `lab/index.html` | The original interactive page, self-contained (the network runs in the browser) |
| `NOTEBOOK.md` | Lab notebook: the build story, dead ends and sandbox gotchas, as recorded |

## Data and credits

| Source | Used for | Licence / terms |
|---|---|---|
| [Caravan / CAMELS-US (ERA5-Land forcing, streamflow)](https://zenodo.org/records/7944025) | dev, exam, physics (free forcing windows) | CC BY 4.0 |
| [USGS NWIS daily values](https://waterdata.usgs.gov) | dev, exam | Public domain |
| [FEDS-MTBS fire progression; MTBS; LANDFIRE; HRRR; gridMET](https://zenodo.org/records/20187963) | dev, exam, physics | CC BY 4.0 / US public data |
| [NESO 1 s frequency, outturn inertia, generation mix; Elexon REMIT/B1610 labels via GridData](https://www.neso.energy/data-portal) | dev, exam, physics (train-year operating points) | NESO Open Data Licence; Elexon BMRS licence |
| [EIA RECS 2020](https://www.eia.gov/consumption/residential/) | dev, exam | Public domain |
| [TMYx weather (climate.onebuilding.org)](https://climate.onebuilding.org) | physics | Free with attribution |
| [Vehicle Energy Dataset (VED) via RoadData (USGS 3DEP lidar grades, Iowa Mesonet ASOS weather)](https://github.com/gsoh/VED) | dev, exam, physics (train_noise routes) | Apache-2.0 (VED); U.S. public domain (3DEP); free use with credit (IEM) |
| [US EPA test-car data and drive cycles via RoadData](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy) | dev, exam, physics (cycles) | Public domain |

Please cite the data providers when you use this model. Data stays under its providers' terms; see [DATA_LICENSES.md](../../DATA_LICENSES.md).

---

Part of **[ML Ratchet](https://github.com/dogum/ML-Ratchet)** — physics-first models that are trained only on simulations, kept only when they ratchet up, and graded on real data they never saw.
