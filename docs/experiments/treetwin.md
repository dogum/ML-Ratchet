# TreeTwin + ConformalRatchet: physics-trained trees, exact TreeSHAP twin diagnostics, and Mondrian conformal calibration

*Method experiment M2, 24 Nov 2026. Scripts, frozen pre-registration and raw results: [`experiments/treetwin/`](../../experiments/treetwin/). Core modules: [`mlratchet/gbm.py`](../../mlratchet/gbm.py), [`mlratchet/calibrate.py`](../../mlratchet/calibrate.py), [`mlratchet/twintest.py`](../../mlratchet/twintest.py), and browser runtime [`site/src/lib/nn/trees.ts`](../../site/src/lib/nn/trees.ts).*

ML Ratchet's model catalogue had two absent tabular/uncertainty families (`gbm` and `conformal`) and three open method items on `docs/roadmap.md`: a reusable twin-test harness, per-stratum $\sigma$ widening, and feature-level real-vs-twin attribution. On three tabular exams across three platforms (**BuildingDetective RECS 2020**, **AirfoilNet UIUC LSAT drag**, and **EnergyNet VED real trips** — `38,759` real exam rows across `7,822` unseen units), we tested:

1. **Physics-trained GBDTs (`GBDT-Plain` and `GBDT-Mono`):** trained strictly on physics simulations (zero real training labels), tuned only on frozen simulated validation splits, exported to pure numpy (`mlratchet.gbm.TreeEnsemble`, `0.0` max diff vs LightGBM) and TypeScript (`1e-12` parity), and graded against each platform's released neural network (**Incumbent P**).
2. **Mondrian conformal calibration (`mlratchet.calibrate.MondrianConformal`):** widen-only (`>= 1.0`) per-stratum calibration with empirical-Bayes shrinkage (`shrink_k=10, min_n=5`) fitted strictly on real dev residuals, compared against ML Ratchet's current single pooled `widen_factor`.
3. **Exact path-dependent TreeSHAP twin diagnostics (`mlratchet.twintest`):** Lundberg et al. (2020) Algorithm 2 in pure numpy, run on real dev units vs simulator twins — in zero-label paired input-shift mode for inverse models (`shap_twin_diff` on 300 BuildingSim RECS twins) and residual-gap decomposition mode for forward models (`shap_residual_gap` on EnergyNet and AirfoilNet).
4. **Cross-family disagreement (`mlratchet.twintest.disagreement_detector`):** whether `|GBDT-Mono - Incumbent P|` predicts real exam error without labels at inference.

Everything was pre-registered and frozen in [`experiments/treetwin/PREREG.md`](../../experiments/treetwin/PREREG.md) (`prereg_log.json`, SHA-256 `352d3edf570a5adb741bc8250a31079123150cd204092a4b61924adb5bfde7ba`) before any exam split was opened. Every exam was scored once.

## Verdict

- **Point accuracy (GBDT vs released neural incumbent): 0 wins, 0 ties, 3 losses on pooled exam headlines.** On smooth physics manifolds (EnergyNet's 151 road-load work integrals, AirfoilNet's 48 airfoil coordinates) and against BuildingDetective's residential-calibrated MLP, pure simulation-trained trees lost the overall exam headline on all three platforms (`+2.71 pp` on EnergyNet, `+0.018` ln Cd MAE on AirfoilNet, `+11.34 pp` on BuildingDetective). Yet trees beat the released neural incumbents in specific weak strata — every electric-heated RECS climate band (`28.3–34.8%` vs `37.6–63.4%`), unseen EnergyNet EVs (`17.7–19.4%` vs `23.2%`), and AirfoilNet `lowRe_highA` (`GBDT-Plain` `0.261` vs `0.290`) — making them a strong complementary arm and mixture-of-experts candidate, not a drop-in replacement.
- **Monotone physics constraints (`GBDT-Mono` vs `GBDT-Plain`): neutral on 2 exams, small loss on 1.** Monotone constraints tied on BuildingDetective (`-0.34 pp`, 95% CI `[-0.80, +0.25]`) and EnergyNet (`+0.27 pp`, `[-0.39, +0.95]`), and lost slightly on AirfoilNet (`+0.0052`, `[+0.0014, +0.0090]`) specifically in `lowRe_highA`, where thicker airfoils delay leading-edge stall and have *lower* near-stall drag than thin sections — violating a global `+1` thickness constraint.
- **Mondrian conformal calibration (nominal 68.3%): clear win on all 3 released neural incumbents (and 5 of 6 platform x learner pairs).** Per-stratum `MondrianConformal` (`shrink_k=10, widen_only=True`) cut worst-stratum exam under-coverage from `15.3 pp -> 5.7 pp` on BuildingDetective, `21.6 pp -> 4.7 pp` on AirfoilNet (`4.6x` reduction), and `31.1 pp -> 27.0 pp` on EnergyNet Incumbent (and `14.4 pp -> 5.8 pp` on EnergyNet `GBDT-Mono`) with nearly unchanged or narrower mean interval widths.
- **Exact TreeSHAP twin diagnostics: turns the twin test into a ranked physical bug list.** On 300 paired RECS dev twins with **zero dev labels**, `shap_twin_diff` immediately isolated `Electricity (kWh/yr)` (`-46.5%` floor-area shift, `64.6%` of total twin gap) as the dominant error in uncalibrated BuildingSim. On EnergyNet's 327 freezing-weather VED dev trips (`T < 0 C`), real trips burn `+9.59%` more energy than RoadSim predicts, and exact TreeSHAP isolates `+6.38%` directly to `Ambient temperature` (`+7.84%` real vs `+1.46%` twin), replacing the placeholder chart on the Model families page with real measurements.
- **Cross-family disagreement (`|GBDT-Mono - Incumbent|`):** correlated positively and significantly (`p < 1e-10`) with real exam error on all three exams without labels; top-10% disagreement cases carried `2.11x` higher real error on EnergyNet, `1.90x` on AirfoilNet, and `1.09x` on BuildingDetective (`r = +0.374` vs `GBDT-Mono` error).

## What this does and does not show

- **It shows** that pure-numpy + TypeScript GBDTs (`0.0` parity vs LightGBM, `<= 2.6e-14` TreeSHAP efficiency error, `47k–368k` stored parameters) fit cleanly inside ML Ratchet's inference budget and turn the twin test from manual curve plotting into an exact ranked feature attribution.
- **It shows** that per-stratum Mondrian conformal widening with empirical-Bayes shrinkage is an immediate, zero-retraining upgrade over a single pooled widening factor whenever sim-to-real error varies by regime (climate/fuel, Reynolds/angle, powertrain).
- **It does not show** that GBDTs trained on *identical* banks always lose to MLPs. Here the released neural incumbents trained on larger or dev-calibrated banks (BuildingDetective's released MLP trained on a residential-calibrated wrapper, which is why it scores `33.6%` on RECS exam and `59.1%` on our uncalibrated BuildingSim validation split; AirfoilNet's DeepONet trained on ~50x more `mfoil` points; EnergyNet's 5-MLP ensemble trained on ~3x more RoadSim vehicles). Even on equal simulated validation banks, `GBDT-Plain` beat AirfoilNet's incumbent (`0.153` vs `0.158`) and BuildingDetective's (`29.7%` vs `59.1%`), while losing on EnergyNet (`15.4%` vs `11.5%`).
- **It does not bypass small-dev limits.** On EnergyNet, EV coverage remains under-covered (`41.3%`) even after Mondrian calibration because the frozen dev split contains only one Nissan Leaf (`258` trips) while the exam contains two unseen Leafs with different winter drag/heater error (`-20%` low). Calibration cannot fix a regime that dev has only one unit of.

## 1. Point Accuracy: Physics-Trained GBDT vs Released Neural Incumbent

All GBDTs trained with zero real labels (`n_boot = 1,000` paired unit-grouped bootstrap; negative `Mono - Inc` means GBDT beats Incumbent).

| Platform | Exam units (rows) | Metric (lower better) | Sim-val: Inc / Plain / Mono | Real exam: Incumbent P | GBDT-Plain | GBDT-Mono | Mono − Inc [95% CI] | Parity vs LGB |
|---|---:|---|---|---|---|---|---|---:|
| **BuildingDetective (RECS 2020)** | 7,401 (7,401) | Floor-area MdAPE (%) | 59.13 / 29.72 / 30.58 | **33.62** | 45.30 | 44.96 | +11.34 [+10.07, +12.53] (loss) | `0.0e+00` |
| **AirfoilNet (UIUC LSAT drag)** | 71 (5,565) | `\|ln(Cd_pred/Cd_true)\|` MAE | 0.158 / 0.153 / 0.158 | **0.166** | 0.179 | 0.184 | +0.018 [+0.005, +0.030] (loss) | `0.0e+00` |
| **EnergyNet (VED real trips)** | 350 (25,793) | Mean MdAPE ICE/HEV/PHEV (%) | 11.49 / 15.36 / 16.04 | **12.97** | 15.41 | 15.68 | +2.71 [+0.42, +5.31] (loss) | `0.0e+00` |

### Where trees beat the released neural incumbents by stratum

| Platform | Stratum | Exam rows | Incumbent P | GBDT-Plain | GBDT-Mono | Margin (best GBDT vs P) |
|---|---|---:|---:|---:|---:|---:|
| BuildingDetective | `cold_elec` (floor-area MdAPE %) | 298 | 63.38% | **33.57%** | 34.78% | **-29.81 pp** |
| BuildingDetective | `mixed_elec` (floor-area MdAPE %) | 652 | 47.12% | 31.57% | **30.47%** | **-16.65 pp** |
| BuildingDetective | `warm_elec` (floor-area MdAPE %) | 742 | 37.61% | 29.64% | **28.27%** | **-9.34 pp** |
| EnergyNet | `EV` (trip energy MdAPE %) | 167 | 23.22% | **17.65%** | 19.43% | **-5.57 pp** |
| AirfoilNet | `lowRe_highA` (`\|ln Cd\|` MAE) | 718 | 0.290 | **0.261** | 0.296 | **-0.028** |

## 2. Uncertainty Calibration: Pooled Widen-Only vs Mondrian Conformal (Nominal 68.3%)

All calibrators fitted strictly on real dev residuals with `widen_only=True` (`>= 1.0`), graded once across held-out exam strata. Worst gap is `max(0, 0.6827 - cov_s)` across strata (lower is better).

| Platform & Learner | Raw cov (worst gap) | Pooled widen cov (worst gap) | Mondrian raw cov (worst gap) | Mondrian shrunk cov (worst gap) | Mean width: Pooled -> Shrunk |
|---|---|---|---|---|---|
| **BuildingDetective** (Incumbent P) | 64.8% (`-17.7 pp`) | 66.8% (`-15.3 pp`) | 67.5% (`-5.1 pp`) | **67.4% (`-5.7 pp`)** | 0.2179 -> 0.2214 |
| **BuildingDetective** (GBDT-Mono) | 35.1% (`-51.0 pp`) | 66.1% (`-16.3 pp`) | 66.6% (`-6.8 pp`) | **67.0% (`-5.9 pp`)** | 0.3675 -> 0.3464 |
| **AirfoilNet** (Incumbent P) | 41.2% (`-48.2 pp`) | 73.4% (`-21.6 pp`) | 75.0% (`-5.1 pp`) | **74.7% (`-4.7 pp`)** | 0.2542 -> 0.2695 |
| **AirfoilNet** (GBDT-Mono) | 49.8% (`-24.1 pp`) | **63.0% (`-9.8 pp`)** | 64.0% (`-11.5 pp`) | 64.0% (`-11.1 pp`) | 0.2029 -> 0.2069 |
| **EnergyNet** (Incumbent P) | 46.0% (`-43.7 pp`) | 64.4% (`-31.1 pp`) | 59.1% (`-26.4 pp`) | **59.4% (`-27.0 pp`)** | 0.1830 -> 0.1605 |
| **EnergyNet** (GBDT-Mono) | 46.6% (`-26.2 pp`) | 59.4% (`-14.4 pp`) | 62.6% (`-6.2 pp`) | **62.5% (`-5.8 pp`)** | 0.1782 -> 0.1819 |

Empirical-Bayes shrinkage (`shrink_k=10`) specifically helped on AirfoilNet's small `lowRe_highA` dev stratum (`n_dev = 52`), trimming `mondrian_raw`'s overshoot (`84.5%` coverage, `max_abs_gap = 16.3 pp`) down to `79.8%` (`max_abs_gap = 11.9 pp`) while improving worst under-coverage from `-5.1 pp` to `-4.7 pp`.

## 3. Exact TreeSHAP Twin-Test Diagnostics (Real Dev vs Simulator)

### BuildingDetective (RECS 2020) — zero-label paired input shift (`300` BuildingSim twins, `% floor-area shift`)

Without using a single dev floor-area label, `shap_twin_diff` compares `GBDT-Mono`'s exact TreeSHAP vectors on real RECS annual meters `X_real` against each home's simulated BuildingSim twin meters `X_twin`:

| Rank | Physical feature group | Mean twin | Mean real | Mean real − twin | Mean \|real − twin\| | Gap share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Electricity (kWh/yr)** | -2.57% | -49.10% | **-46.53%** | **52.16%** | **64.6%** |
| 2 | Fuel gas/oil (kWh/yr) | +9.36% | +8.63% | -0.73% | 16.33% | 20.2% |
| 3 | Heating degree days | -12.11% | -6.83% | +5.27% | 9.74% | 12.1% |
| 4 | Cooling degree days | +0.28% | -0.12% | -0.40% | 2.50% | 3.1% |

Uncalibrated BuildingSim predicts substantially higher residential electricity use per m² than real RECS 2020 homes, driving a `-46.5%` floor-area under-read on real meters (`64.6%` of the entire twin gap). That single diagnostic explains why raw-BuildingSim GBDTs lost `+11.3 pp` to BuildingDetective's residential-calibrated MLP.

### EnergyNet (VED dev) — forward residual gap (`1,496` trips across `16` cars, `% energy correction`)

| Rank | Physical feature group | Mean twin | Mean real | Mean \|real − twin\| | Gap share |
|---:|---|---:|---:|---:|---:|
| 1 | **Model year** | +9.09% | +9.09% | **3.94%** | **28.2%** |
| 2 | **Ambient temperature** | -2.93% | -2.81% | **3.04%** | **21.8%** |
| 3 | **Accessory load** | +0.88% | +0.79% | **2.34%** | **16.7%** |
| 4 | Stop density | -3.54% | -3.53% | 1.95% | 14.0% |
| 5 | Rolling work | -0.84% | -0.91% | 0.93% | 6.7% |
| 6 | Mass | +7.47% | +7.47% | 0.67% | 4.8% |

On the three spotlight regimes now shown live on the Model families page (`site/src/pages/model-families.astro`):
- **Cold-weather commute (`T < 0 C`, 327 trips, 15 cars):** real cars burn `+9.59%` more energy than RoadSim predicts; exact TreeSHAP isolates `+6.38%` to `Ambient temperature` (`+7.84%` real vs `+1.46%` twin).
- **City hybrid (`HEV` stop-and-go, 138 trips, 5 cars):** net error is `-0.85%`, but TreeSHAP uncovers opposing gaps (`Stop density` `+0.77%`, `Ambient temperature` `+0.67%`, masked by `Model year` `-1.30%`).
- **Mild highway ICE (`T >= 10 C`, 12 trips, 3 cars):** net error is `+0.89%`; aero (`-0.55%`), rolling (`+0.46%`), and grade (`-0.09%`) agree closely while `Model year` (`+4.19%`) carries the residual vehicle offset.

### AirfoilNet (UIUC dev) — forward residual gap (`429` points across `7` dev airfoils, `% drag shift`)

| Rank | Physical feature group | Mean twin | Mean real | Mean \|real − twin\| | Gap share |
|---:|---|---:|---:|---:|---:|
| 1 | **Angle of attack (alpha)** | +13.24% | +13.19% | **10.98%** | **38.3%** |
| 2 | **Mid-chord thickness (25–75% c)** | -1.61% | -1.58% | **8.68%** | **30.2%** |
| 3 | Mid/aft camber | +5.27% | +5.42% | 4.66% | 16.2% |
| 4 | Reynolds number (`log10 Re`) | -9.68% | -9.75% | 2.92% | 10.2% |

## 4. Label-Free Cross-Family Disagreement (`|GBDT-Mono - Incumbent|`)

| Platform | Exam rows | Spearman r vs Incumbent \|err\| | p-value | Top-10% disagreement error enrichment | Spearman r vs GBDT-Mono \|err\| |
|---|---:|---:|---:|---:|---:|
| **BuildingDetective (RECS 2020)** | 7,401 | +0.081 | `< 1e-11` | 1.09× | +0.374 |
| **AirfoilNet (UIUC LSAT drag)** | 5,565 | +0.229 | `< 1e-60` | **1.90×** | +0.303 |
| **EnergyNet (VED real trips)** | 25,793 | +0.157 | `< 1e-140` | **2.11×** | +0.256 |

Even where the GBDT loses the pooled headline to the neural incumbent, keeping a lightweight pure-numpy forest beside the neural network gives an informative, label-free out-of-distribution / sim-to-real flag (`1.9–2.1x` error enrichment on the top decile of EnergyNet and AirfoilNet exam rows).
