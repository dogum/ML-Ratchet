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

### BuildingDetective (RECS 2020) (`input_shift_paired_twins`, units: % floor-area shift)

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Electricity (kWh/yr) | -2.57 | -49.10 | -46.53 | 52.16 | 64.6% |
| 2 | Fuel gas/oil (kWh/yr) | +9.36 | +8.63 | -0.73 | 16.33 | 20.2% |
| 3 | Heating degree days | -12.11 | -6.83 | +5.27 | 9.74 | 12.1% |
| 4 | Cooling degree days | +0.28 | -0.12 | -0.40 | 2.50 | 3.1% |
| 5 | District meters | +0.00 | +0.00 | +0.00 | 0.00 | 0.0% |

### AirfoilNet (UIUC LSAT drag) (`forward_residual_gap`, units: % drag shift)

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Angle of attack (alpha) | +13.24 | +13.19 | -0.05 | 10.98 | 38.3% |
| 2 | Mid-chord thickness (25-75% c) | -1.61 | -1.58 | +0.03 | 8.68 | 30.2% |
| 3 | Mid/aft camber | +5.27 | +5.42 | +0.15 | 4.66 | 16.2% |
| 4 | Reynolds number (log10 Re) | -9.68 | -9.75 | -0.07 | 2.92 | 10.2% |
| 5 | Leading-edge thickness (0-25% c) | -0.21 | -0.13 | +0.08 | 0.72 | 2.5% |
| 6 | Trailing-edge thickness (75-100% c) | +1.32 | +1.33 | +0.01 | 0.44 | 1.5% |

### EnergyNet (VED real trips) (`forward_residual_gap`, units: % energy correction)

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean |real − twin| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Model year | +9.09 | +9.09 | +0.00 | 3.94 | 28.2% |
| 2 | Ambient temperature | -2.93 | -2.81 | +0.13 | 3.04 | 21.8% |
| 3 | Accessory load | +0.88 | +0.79 | -0.09 | 2.34 | 16.7% |
| 4 | Stop density | -3.54 | -3.53 | +0.01 | 1.95 | 14.0% |
| 5 | Rolling work | -0.84 | -0.91 | -0.07 | 0.93 | 6.7% |
| 6 | Mass | +7.47 | +7.47 | -0.01 | 0.67 | 4.8% |

## 4. Cross-Family Disagreement as a Label-Free Gap Detector (`|GBDT-Mono − Incumbent|`)

| Platform | Spearman r vs Incumbent |err| | p-value | Top-10% disagreement error enrichment | Spearman r vs GBDT-Mono |err| |
|---|---:|---:|---:|---:|
| **BuildingDetective (RECS 2020)** | +0.081 | 0.00e+00 | 1.09× | +0.374 |
| **AirfoilNet (UIUC LSAT drag)** | +0.229 | 0.00e+00 | 1.90× | +0.303 |
| **EnergyNet (VED real trips)** | +0.157 | 0.00e+00 | 2.11× | +0.256 |
