# TreeTwin + ConformalRatchet — Results

Pre-registered cross-platform method experiment (`PREREG.md`, frozen SHA-256 `352d3edf570a…`). Zero real training labels entered any GBDT; every real exam split was opened once.

## 1. Point Accuracy: Physics-Trained GBDT vs Released Neural Incumbent

| Platform | Exam units (rows) | Metric (lower better) | Incumbent MLP / DeepONet | GBDT-Plain | GBDT-Mono | Mono − Inc (95% CI) | Parity vs LGB |
|---|---:|---|---:|---:|---:|---|---:|
| **BuildingDetective (RECS 2020)** | 7401 (7,401) | log10_floor_area | 33.622 | 45.297 | **44.959** | +11.338 [+10.073, +12.532] (loss) | `0.0e+00` |
| **AirfoilNet (UIUC LSAT drag)** | 71 (5,565) | ln_cd | 0.166 | 0.179 | **0.184** | +0.018 [+0.005, +0.030] (loss) | `0.0e+00` |
| **EnergyNet (VED real trips)** | 350 (25,793) | log_energy_ratio | 12.965 | 15.412 | **15.679** | +2.714 [+0.421, +5.313] (loss) | `0.0e+00` |

## 2. Uncertainty Calibration: Pooled Widen-Only vs Mondrian Conformal (Nominal 68.3%)

Fitted strictly on real Dev residuals (widen-only `>= 1.0`), graded once across held-out exam strata.

| Platform & Learner | Raw cov (worst gap) | Pooled widen cov (worst gap) | Mondrian raw cov (worst gap) | Mondrian shrunk cov (worst gap) | Mean width: Pooled -> Shrunk |
|---|---|---|---|---|---|
| **BuildingDetective (RECS 2020)** (Incumbent) | 64.8% (`-17.7 pp`) | 66.8% (`-15.3 pp`) | 67.5% (`-5.1 pp`) | **67.4% (`-5.7 pp`)** | 0.2179 -> 0.2214 |
| **BuildingDetective (RECS 2020)** (GBDT-Mono) | 35.1% (`-51.0 pp`) | 66.1% (`-16.3 pp`) | 66.6% (`-6.8 pp`) | **67.0% (`-5.9 pp`)** | 0.3675 -> 0.3464 |
| **AirfoilNet (UIUC LSAT drag)** (Incumbent) | 41.2% (`-48.2 pp`) | 73.4% (`-21.6 pp`) | 75.0% (`-5.1 pp`) | **74.7% (`-4.7 pp`)** | 0.2542 -> 0.2695 |
| **AirfoilNet (UIUC LSAT drag)** (GBDT-Mono) | 49.8% (`-24.1 pp`) | 63.0% (`-9.8 pp`) | 64.0% (`-11.5 pp`) | **64.0% (`-11.1 pp`)** | 0.2029 -> 0.2069 |
| **EnergyNet (VED real trips)** (Incumbent) | 46.0% (`-43.7 pp`) | 64.4% (`-31.1 pp`) | 59.1% (`-26.4 pp`) | **59.4% (`-27.0 pp`)** | 0.1830 -> 0.1605 |
| **EnergyNet (VED real trips)** (GBDT-Mono) | 46.6% (`-26.2 pp`) | 59.4% (`-14.4 pp`) | 62.6% (`-6.2 pp`) | **62.5% (`-5.8 pp`)** | 0.1782 -> 0.1819 |

## 3. Exact TreeSHAP Twin-Test Gap Rankings (Real Dev vs Simulator)

Rebuilt 27 Nov 2026 (`dev_rebuild_2026-11-27.json`, dev only) after the PR #19 review: the first version
grouped EnergyNet features by the wrong indices, filed AirfoilNet's tmax under trailing-edge thickness, and
ran BuildingSim with constant 3 m/s wind. The tables below come from forests rebuilt on Windows with the same
scripts and seeds (close to, not identical with, the forests the exam scored). The score files keep the
original tables as they were run.

### BuildingDetective (RECS 2020) (`input_shift_paired_twins`, units: % floor-area shift)

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Electricity (kWh/yr) | -2.17 | -48.79 | -46.62 | 51.96 | 64.2% |
| 2 | Fuel gas/oil (kWh/yr) | +10.07 | +8.09 | -1.97 | 16.52 | 20.4% |
| 3 | Heating degree days | -12.35 | -6.98 | +5.37 | 10.08 | 12.4% |
| 4 | Cooling degree days | -0.27 | -0.49 | -0.22 | 2.39 | 2.9% |
| 5 | District meters | +0.00 | +0.00 | +0.00 | 0.00 | 0.0% |

### AirfoilNet (UIUC LSAT drag) (`forward_residual_gap`, units: % drag shift)

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Angle of attack (alpha) | +12.45 | +12.40 | -0.06 | 11.29 | 41.4% |
| 2 | Mid-chord thickness (25-75% c) | -1.98 | -2.03 | -0.05 | 9.28 | 34.0% |
| 3 | Mid/aft camber | +4.75 | +4.87 | +0.12 | 2.40 | 8.8% |
| 4 | Reynolds number (log10 Re) | -10.91 | -10.95 | -0.04 | 2.33 | 8.5% |
| 5 | Leading-edge thickness (0-25% c) | +0.84 | +0.93 | +0.09 | 1.41 | 5.2% |
| 6 | Trailing-edge thickness (75-100% c) | +1.59 | +1.60 | +0.01 | 0.58 | 2.1% |

### EnergyNet (VED real trips) (`forward_residual_gap`, units: % energy correction)

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Powertrain, labels & year | +8.04 | +8.01 | -0.04 | 3.89 | 28.9% |
| 2 | Ambient temperature | -2.85 | -2.79 | +0.06 | 3.32 | 24.7% |
| 3 | Duration & idle | +0.11 | -0.01 | -0.11 | 2.20 | 16.4% |
| 4 | Speed & stops | -3.13 | -3.08 | +0.06 | 1.55 | 11.5% |
| 5 | Rolling work | +0.19 | +0.18 | -0.01 | 0.92 | 6.8% |
| 6 | Grade work | +0.51 | +0.53 | +0.02 | 0.68 | 5.0% |

## 4. Cross-Family Disagreement as a Label-Free Gap Detector (`|GBDT-Mono − Incumbent|`)

| Platform | Spearman r vs Incumbent |err| | p-value | Top-10% disagreement error enrichment | Spearman r vs GBDT-Mono |err| |
|---|---:|---:|---:|---:|
| **BuildingDetective (RECS 2020)** | +0.081 | 0.00e+00 | 1.09× | +0.374 |
| **AirfoilNet (UIUC LSAT drag)** | +0.229 | 0.00e+00 | 1.90× | +0.303 |
| **EnergyNet (VED real trips)** | +0.157 | 0.00e+00 | 2.11× | +0.256 |

## 5. Dated supplementary exam: BuildingDetective (RECS 2020), `2026-11-27-wind`

GBDT arms retrained on BuildingSim banks with the real hourly wind and site elevation (the first banks used a constant 3 m/s and sea level; PR #19 review). Same scripts, seeds and PREREG; the incumbent is unchanged. Scored once (`scores/buildingdetective_2026-11-27-wind.json`, PREREG sha `352d3edf570a…`, scored 2026-11-27T18:40:35Z). Sections 1, 2 and 4 above keep the first exam; this table puts the two side by side.

| Row | First exam (24 Nov) | Supplementary |
|---|---:|---:|
| Simulated validation, GBDT-Plain | 29.72 | 29.14 |
| Simulated validation, GBDT-Mono | 30.58 | 29.74 |
| Simulated validation, incumbent (reference) | 59.13 | 60.11 |
| Exam floor-area MdAPE %, Incumbent | 33.62 | 33.62 |
| Exam floor-area MdAPE %, GBDT-Plain | 45.30 | 46.03 |
| Exam floor-area MdAPE %, GBDT-Mono | 44.96 | 45.78 |
| Mono − Inc (95% CI) | +11.34 [+10.07, +12.53] | +12.15 [+10.99, +13.14] |
| Plain − Inc (95% CI) | +11.68 [+10.44, +12.79] | +12.41 [+11.28, +13.57] |
| Mono − Plain (95% CI) | -0.34 [-0.80, +0.25] | -0.26 [-0.86, +0.18] |
| Stratum `cold_elec` (n 298): Inc / Plain / Mono | 63.4 / 33.6 / 34.8 | 63.4 / 33.7 / 35.8 |
| Stratum `cold_fuel` (n 2544): Inc / Plain / Mono | 31.8 / 56.9 / 57.6 | 31.8 / 58.3 / 58.7 |
| Stratum `mixed_elec` (n 652): Inc / Plain / Mono | 47.1 / 31.6 / 30.5 | 47.1 / 32.1 / 31.5 |
| Stratum `mixed_fuel` (n 2004): Inc / Plain / Mono | 31.4 / 46.4 / 46.0 | 31.4 / 47.0 / 46.0 |
| Stratum `warm_elec` (n 742): Inc / Plain / Mono | 37.6 / 29.6 / 28.3 | 37.6 / 30.4 / 28.6 |
| Stratum `warm_fuel` (n 1161): Inc / Plain / Mono | 30.4 / 34.6 / 34.9 | 30.4 / 34.1 / 35.4 |
| GBDT-Mono coverage, raw (worst under-gap) | 35.1% (`-51.0 pp`) | 37.0% (`-48.5 pp`) |
| GBDT-Mono coverage, pooled_widen (worst under-gap) | 66.1% (`-16.3 pp`) | 66.6% (`-13.8 pp`) |
| GBDT-Mono coverage, mondrian_raw (worst under-gap) | 66.6% (`-6.8 pp`) | 66.5% (`-6.0 pp`) |
| GBDT-Mono coverage, mondrian_shrunk (worst under-gap) | 67.0% (`-5.9 pp`) | 66.8% (`-6.0 pp`) |
| Disagreement Spearman r vs Incumbent abs. error (top-10% enrichment) | +0.081 (1.09×) | +0.076 (1.07×) |
| Disagreement Spearman r vs GBDT-Mono abs. error (top-10% enrichment) | +0.374 (1.33×) | +0.397 (1.37×) |
