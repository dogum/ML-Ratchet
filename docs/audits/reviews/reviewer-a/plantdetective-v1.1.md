# PlantDetective v1.1 (`R6_cs`) | Clean core ensemble/parity, unsorted-timestamp sensitivity, and silent all-NaN inference

**First pass (`reviewer-a`). Verdict: MINOR HYGIENE.** Audited against model base `61c70bf` across the shipped 3-member ensemble (`plantdetective_weights.npz`), feature/inference pipeline, 3 shipped EPA CAMPD exam units, trainer seeding, and both `v1.0`/`v1.1` ratchet logs following [`SKILL.md`](../../SKILL.md). [Model code](../../../../models/plantdetective/plantdetective.py) · [Core features](../../../../models/plantdetective/pdcore.py) · [Trainer](../../../../models/plantdetective/training/model.py)

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Infer a thermal power plant unit's technology, operating flexibility class, minimum stable load, ramp limits, start frequency, run/down times, heat-rate curve, and capacity factor from `>= 30` days of hourly gross MW output (with optional hourly power/gas prices and EIA-860/923 context).
- **Information boundary:** Trained exclusively on GridSim + self-committed DP simulations; real dev units calibrate twin parameters, `sigma_factors_v1_1.json`, and the dev gate. All 3 shipped examples (`55086_101`, `6179_3`, `8063_CT2`) are EPA CAMPD exam units.

## 2. Findings

### Finding A — `Detective.infer()` does not sort by timestamp `t`, silently corrupting estimates on reversed/unsorted rows [`MINOR HYGIENE`]
- **What failed?** [`plantdetective.py`](../../../../models/plantdetective/plantdetective.py) passes `x`, `t`, `price`, and `gas` directly into `pdcore.features` and run-length / start-ramp counters without sorting by `t` or rejecting non-monotonic timestamps.
- **What changed when tested?** Passing the exact same observations reversed (`newest-first`) shifted estimates on all three shipped exam units:

| Shipped exam unit | Original (`pmin` / `starts_yr` / `cf`) | Reversed rows (`pmin` / `starts_yr` / `cf`) | Largest relative shift |
|---|---|---|---|
| `55086_101` (`gas_cc`) | `0.488` / `79.7` / `0.136` | **`0.435`** / **`84.3`** / **`0.106`** | **`−22.1%` on `cf`** (`hr_full 11.12 → 11.80`) |
| `6179_3` (`coal_steam`) | `0.369` / `4.9` / `0.693` | `0.361` / `5.0` / `0.681` | `−2.2%` on `pmin` |
| `8063_CT2` (`gas_ct`) | `0.625` / `64.0` / `0.031` | `0.635` / `62.6` / `0.030` | `−2.2%` on `starts_yr` |

- **What consequence remains unmeasured?** Archived CAMPD exam series are chronologically sorted, so this affects operational CSV/API callers rather than published exam scores.

### Finding B — All-NaN and all-zero generation series silently return full unit diagnoses [`MINOR HYGIENE`]
- **What failed?** While `Detective.infer()` properly rejects `< 30` days (`short_168h` raises `ValueError: need at least 30 days of hourly output`), it performs no minimum-finite-sample or non-zero-output check on `x`.
- **What changed when tested?** On `55086_101`, replacing all `70,122` hourly MW readings with `NaN` silently returned `tech = gas_cc (1.00)`, `flex = peaking`, **`pmin_frac = 0.586`**, `starts_yr = 1.3`, **`cf = 0.066`**, **`hr_full = 14.81`**. Replacing all MW with `0.0` silently returned `gas_cc`, `peaking`, `pmin_frac = 0.401`, `cf = 0.040`.
- **What consequence remains unmeasured?** Exam units are screened for valid CAMPD operating hours, so impact is confined to operational callers supplying blank or retired-unit telemetry.

### Finding C — Trainer seeds PyTorch only; minibatch shuffling (`np.random.shuffle`) is unseeded [`MINOR HYGIENE`]
- **What failed?** [`training/model.py:55,90`](../../../../models/plantdetective/training/model.py#L55) calls `torch.manual_seed(seed)` but never seeds `np.random` before calling `np.random.shuffle(idx_tr)` every epoch.
- **What consequence remains unmeasured?** Exact single-seed replay variance from unseeded epoch shuffling was not re-run.

## 3. Clean Checks (Items 2, 3, 8, 10)
- **Weights, normalization, dead units & ensemble diversity (CLEAN):** All 3 members (`m0..m2`) have finite weights, **`0` dead rows** (`min_row_norm = 0.1850`), healthy feature `sd` in **`[0.01798, 7.38200]`** (171 features), real `Z` in `[-4.075, 9.343]`, and genuinely independent 3-seed first-layer `l0w` cosines (**`0.0089, 0.0133, 0.0170`**).
- **3-member PyTorch ↔ NumPy export parity (CLEAN):** Reconstructing the LeakyReLU(`0.01`) MLP + `count_offsets` in PyTorch (float32) across all 3 members on real + random inputs matched `np_forward` to **`prob = 4.80e-7`**, **`mu = 1.09e-6`**, **`sd = 3.51e-7`**.
- **Calibration & ratchet replay (CLEAN):** All `13/13` dev widen factors are strictly `>= 1.0` (`1.0000–2.0705`). All `5/5` v1.0 rungs (`ratchet.json`) and `3/3` v1.1 rungs (`ratchet_v1_1.json`, including `R5` rejected on dev `−0.0058 < −0.005` and `R6_cs` kept) match their published rules.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/plantdetective_v1_1_probe.py
```

Model snapshot `61c70bf`, audit checkout `759c345`. Uses NumPy and CPU PyTorch. [Reproduction probe](plantdetective_v1_1_probe.py)
