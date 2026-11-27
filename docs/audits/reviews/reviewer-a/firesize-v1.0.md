# FireSize v1.0 (`r7` + widen) | Zero-variance Slash-Blowdown feature (`sd = 1e-6`) collapses escape risk to `0.000`

**First pass (`reviewer-a`). Verdict: CRITICAL DEFECT.** Audited against model base `61c70bf` across `firesize_weights.npz`, [`firesize.py`](../../../../models/firesize/firesize.py), 8 shipped FPA-FOD exam examples (`demo_examples.csv`), trainer seeding (`training/src/train.py`), and `training/ratchet.json` following [`SKILL.md`](../../SKILL.md).

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Given a US wildfire ignition record (cause, agency, region, time), 13×13 LANDFIRE 250 m fuel/topography summaries, Rothermel ROS features, and discovery-day/prior-day gridMET weather, predict `P(escape >= 100 acres)` and a 12-class final fire-size distribution.
- **Information boundary:** Trained exclusively on 8,500 simulated FireSim ignitions (`v1` physics + dev-calibrated two-mode initial attack). Dev calibration (`911` twins + `temp = 1.4586`) and the dev gate (`4,429` ignitions) split the 8,913 dev ignitions into disjoint halves; exam is 655,869 FPA-FOD ignitions (2012–2020).

## 2. Findings

### Finding A — Zero-variance Slash-Blowdown fuel feature `pt_SB` (`sd = 1.00e-6`) with unclipped standardization and large weights (`0.76–0.79`) collapses `P(escape)` to `0.0000` on every SB ignition [`CRITICAL DEFECT`]
- **What failed?** Feature `15` (`pt_SB`, binary indicator for LANDFIRE Slash-Blowdown fuel codes 201–204 at the ignition cell) had zero positive occurrences in the 8,500-ignition simulated training bank (`mu[15] = 0.0`, **`sd[15] = 1.00e-6`**), while the neighboring patch-fraction features `f5_SB` (`sd[16] = 0.0040`) and `f13_SB` (`sd[17] = 0.0029`) were also near-zero. Across all three ensemble members, the first-layer input column norms on `pt_SB` remain large (**`0.764`, `0.764`, `0.790`**), and [`firesize.py:55`](../../../../models/firesize/firesize.py#L55) computes `z = (Z - W['mu']) / W['sd']` with **no clipping**.
- **What changed when tested?**
  - Setting `pt_SB = 1.0` (clearing the previous `pt_*` fuel one-hot, producing **`z[15] = +1,000,000`**) on all 8 shipped exam ignitions—whose nominal `P(escape)` ranges from `0.0052` to `0.3117` (`North Moore`, `11.02` median acres)—overwhelmed all 74 other weather, fuel, and regional features and collapsed **all 8 fires to identical `P(escape) = 0.000000` and identical `median_acres = 1.7263`**.
  - Even with `pt_SB = 0.0`, placing a realistic nearby Slash-Blowdown patch (`f5_SB = 0.25`, `f13_SB = 0.10`, yielding `z[16] = +62.2` and `z[17] = +34.5`) crushed `P(escape)` across all 8 fires from max **`0.3117 → 0.0009`** (`> 99%` drop)—the opposite of physical fire behavior in heavy timber slash.
- **What consequence remains unmeasured?** How many of the 655,869 real FPA-FOD exam ignitions fell on or near LANDFIRE Slash-Blowdown (`201–204`) cells and were silently forced to `P(escape) ≈ 0` remains unmeasured without the external FPA-FOD feature table.

### Finding B — All-NaN input silently returns median-imputed forecast; tight `r7` gate margin & parameter bounds [`MINOR HYGIENE` / `DOCUMENTED LIMIT`]
- **What failed?** [`firesize.py:54`](../../../../models/firesize/firesize.py#L54) replaces every non-finite value with `W['fill']` without checking whether an entire ignition record is missing, so an all-`NaN` row silently returns **`P(escape) = 0.0795`**, `median_acres = 0.81`.
- **Documented limits verified:** `ratchet.yaml` honestly notes that `r7` cleared the dev-gate AUC floor by **`+0.0004`** (`0.7561` vs `0.7557`), that the `v1` bank stopped at 8,500 ignitions, and that 10/31 suppression parameters sit within `0.5%` of bounds.

## 3. Clean Checks (Items 2, 3, 7, 8, 10)
- **Weights, dead units & ensemble diversity (CLEAN):** All arrays finite, **`0` dead rows** (`min_norm = 0.5191`), genuinely independent 3-seed `L0_w` cosines (**`0.0303, 0.0424, 0.0456`**), `temp = 1.4586` (strictly widen-only).
- **3-member PyTorch ↔ NumPy export parity (CLEAN):** Reconstructing the 3-member ReLU MLP + softmax in PyTorch (float32) matched `FireSize.probs()` to **`1.70e-7`** on real + random inputs; all 8 shipped examples match `exam_p_escape` within `1e-4`.
- **Seeding & ratchet replay (CLEAN):** `training/src/train.py:80,95,146` cleanly seeds both NumPy (`default_rng`) and PyTorch (`seed + 100 * e`). All **`9/9`** rungs (`r0..r7`) in `training/ratchet.json` match the published promotion rule (`rpss > inc + 0.01`, `auc >= inc - 0.01`, `mae <= 1.05 * inc`).

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firesize_v1_0_probe.py
```

Model snapshot `61c70bf`, audit checkout `69873b7`. Uses NumPy and CPU PyTorch. [Reproduction probe](firesize_v1_0_probe.py)
