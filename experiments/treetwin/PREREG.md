# Pre-Registration: TreeTwin + ConformalRatchet (Method Experiment M2)

**Date:** 2026-11-24  
**Status:** Frozen via `freeze.py` (`prereg_log.json`) **before** any exam split is opened.

## 1. Questions

1. **Monotone physics-trained GBDTs (`gbm`):** On tabular physics-feature problems in ML Ratchet, when gradient-boosted decision trees are trained strictly on physics simulations (zero real training labels), constrained by known physical monotonicities, and exported to pure numpy (`mlratchet.gbm.TreeEnsemble`), how do they compare against each platform's released neural network on held-out real exam units?
2. **TreeSHAP twin-test diagnostics (`mlratchet.twintest`):** Does exact path-dependent TreeSHAP (Lundberg et al. 2020 Algorithm 2) on real dev units vs simulated twins recover a ranked, physically meaningful list of simulator gaps — replacing the placeholder chart on `site/src/pages/model-families.astro` with real measurements?
3. **Mondrian conformal calibration (`conformal`):** Across real exam strata with heterogeneous sim-to-real gaps (powertrains, climate-fuel regimes, Reynolds-angle regimes), does dev-fitted per-stratum Mondrian conformal calibration with empirical-Bayes shrinkage (`mlratchet.calibrate.MondrianConformal`, `shrink_k=10, min_n=5, widen_only=True`) reduce worst-stratum under-coverage relative to ML Ratchet's current single pooled widening factor (`widen_factor`), without inflating mean interval width unnecessarily?
4. **Cross-family disagreement (`mlratchet.twintest.disagreement_detector`):** Without labels at inference, does `|GBDT-Mono - Incumbent MLP|` correlate positively with real exam error?

## 2. Iron Rules

- **Zero real training labels:** Every GBDT trains strictly on physics simulation banks (`sim_train`) and tunes tree complexity (`num_leaves in {15, 31, 63}`, `min_child_samples in {20, 30, 50}`) strictly on the frozen simulated validation split (`sim_val`).
- **Dev only for twin diagnostics & calibration:** Real dev splits are used strictly for (a) TreeSHAP real-vs-twin diagnostic reports and (b) fitting pooled widen-only, raw Mondrian (`shrink_k=0`), and shrunk Mondrian (`shrink_k=10`) calibration factors. The shallow diagnostic residual tree in `shap_residual_gap` is strictly a dev-only lens and never produces exam predictions.
- **One-shot real exam:** `harness.py <platform> exam` verifies the SHA-256 of this file against `prereg_log.json`, scores every arm once on the exact same exam rows with unit-grouped paired 95% bootstrap CIs (`n_boot=1000`, seed `20261124`), and refuses to overwrite an existing score file.

## 3. Platforms, Targets, Monotone Constraints & Strata

| Platform | Simulator (zero real labels) | Features | Target & Headline Metric (lower is better) | Monotone Physics Constraints (`GBDT-Mono`) | Real Dev / Exam Splits (frozen) | Conformal Strata |
|---|---|---|---|---|---|---|
| **BuildingDetective (RECS 2020)** | BuildingSim 5R1C + HVAC driven by 10 NASA POWER 2020 US climate stations (`1,200` train / `300` val) | 12 annual meter & climate features (`feat_annual`) | `log10(floor_area_m2)` -> **Floor-area MdAPE (%)** | `+1` on `log10(kWh)` for all 4 meter channels (elec, fuel, dheat, dcool) | `shared/building/data/proc/splits.json`: `1,848` dev homes (`300` paired BuildingSim twins) / `7,401` exam homes | Climate band (`cold`/`mixed`/`warm`) × fuel (`fuel`/`elec`) |
| **AirfoilNet (UIUC LSAT drag)** | `mfoil_fast` boundary-layer solver on non-tested UIUC & synthetic CST shapes (`360` train / `70` val sweeps, `ncrit ~ U(5,11)`) | 53 tabular shape + Re + alpha features | `ln(Cd)` -> **`|ln(Cd_pred/Cd_true)|` MAE** | `-1` on `log10(Re)` (skin friction & bubble drag drop with Re); `+1` on `tmax` (form drag rises with thickness) | `models/airfoilnet/training/splits.json`: 7 dev airfoils (4 families) / 75 unseen exam airfoils | Reynolds (`<1.5e5` vs `>=1.5e5`) × `|alpha|` (`<6°` vs `>=6°`) |
| **EnergyNet (VED real trips)** | RoadSim physics v2 (`800` train / `200` val generated vehicles on VED `train_noise` routes + dyno cycles) | 151 tabular route + physics + vehicle features | `log(E / E_phys)` -> **Mean MdAPE (%) over ICE, HEV, PHEV** (EV reported alongside) | `-1` on `T_mean`, `T_min` (cold weather raises consumption vs room-T `E_phys`); `+1` on EPA fuel & electric window-sticker ratios | `shared/vehicle/data/proc/splits.json`: 16 dev cars / ~350 unseen exam cars | Powertrain (`ICE`, `HEV`, `PHEV`, `EV`) |

## 4. Arms Compared on Every Exam

1. **Incumbent `P`:** Each model's shipped neural network weights (`models/<id>/`), evaluated on the exact same exam rows.
2. **`GBDT-Plain`:** 3-seed LightGBM mean ensemble + 16th/84th quantile forests, unconstrained, exported to pure numpy (`mlratchet.gbm.TreeEnsemble`).
3. **`GBDT-Mono`:** Same as `GBDT-Plain` with the table's physically justified monotone constraints enforced.
4. **Uncertainty calibration (on both `Incumbent P` and `GBDT-Mono`, nominal 68.27%):** `raw`, `pooled_widen`, `mondrian_raw` (`shrink_k=0`), `mondrian_shrunk` (`shrink_k=10`).
