# FirePilot v1.0 (`r1`) | Zero-variance crew-clock feature (`sd = 1e-6`) causes unclipped value-network explosion off afternoon detections

**First pass (`reviewer-a`). Verdict: CRITICAL DEFECT.** Audited against model base `61c70bf` across shipped weights (`firepilot_weights.npz`), planner & feature code, 3 shipped real-fire examples, trainer, and 5-rung ratchet log following [`SKILL.md`](../../SKILL.md). [Model code](../../../../models/firepilot/firepilot.py) · [Value trainer](../../../../models/firepilot/training/train_value.py) · [Scenario generator](../../../../models/firepilot/training/fp/scenarios.py)

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Given a wildfire detection perimeter, landscape fuels/topography, forecast burn arrival/flame-length rasters, crew working clock (`work_clock`), and crew line-building rate (`m/h`), select a containment-line plan maximizing the fraction of 72-hour no-action growth stopped, with a 1-sigma uncertainty interval.
- **Information boundary:** Trained exclusively on 297 synthetic FireSim scenarios (`p1`, dev-calibrated ROS from 28 dev fires, excluding 12 gate fires and 48 exam fires). Every training scenario (`_pm_start`, [`fp/scenarios.py:80`](../../../../models/firepilot/training/fp/scenarios.py#L80)) and every exam fire starts at an afternoon (`~13:00–14:00` local solar time) satellite detection with fixed `horizon_h = 72.0` and `mobilize_h = 2.0`.

## 2. Findings

### Finding A — Zero-variance feature `feat[6]` (`W[-1] / 36.0`) has `sd = 1.00e-6` with unclipped standardization and non-zero weights, saturating the network on any non-afternoon detection or custom horizon [`CRITICAL DEFECT`]
- **What failed?** Feature `6` of the 31-input scenario vector (`_scn[6] = W[-1] / 36.0`, cumulative crew working hours over the horizon divided by 36) is strictly constant (`34.0 / 36.0 = 0.9444444444`) across every afternoon-detection 72-hour scenario in the training bank. [`train_value.py:79`](../../../../models/firepilot/training/train_value.py#L79) computes `sd = X[~is_sel].std(0) + 1e-6`, yielding **`sd[6] = 1.00e-6`** while the first-layer weights on feature `6` remain non-zero (**`||W0[6]||_2 = 1.394e-4`**). At inference time, [`ValueNet.__call__`](../../../../models/firepilot/firepilot.py#L296) divides `(X - mu) / sd` with **no `z`-score clipping**.
- **What changed when tested?**
  - Evaluating `fp.work_clock(start, lon=-110.0, horizon_h=72.0)` across all 24 integer start hours returns `W[-1]` spanning **`34.00 h` to `36.00 h`** (`36.00 h` for morning/overnight detections). Even a `+0.05 h` (3-minute) shift produces `z[6] = +1,389`, and `W[-1] = 36.0 h` produces **`z[6] = +55,556`** (multiplying `W0[6]` by `55,556`, saturating the `tanh` hidden layer). A 48-hour planning horizon (`W[-1] = 22.0 h`) produces **`z[6] = -333,333`**.
  - On all three shipped real-fire examples (`rate = 300 m/h`), shifting **only `ctx._scn[6]`** (holding the actual forecast, fire geometry, and crew clock fixed) systematically corrupts both predicted saved fractions and CEM-chosen containment plans:

| Shipped exam fire | Nominal `W=34h` (`z=0.00`) pred / proxy / `d_h` | Morning/overnight `W=36h` (`z=+55,556`) pred / proxy / `d_h` | 48-h horizon `W=22h` (`z=-333,333`) pred / proxy / `d_h` |
|---|---|---|---|
| `AZ3395311201220190903` | `0.7134` / `0.7376` / `707 m` | **`0.8926`** / `0.7259` / `679 m` | **`0.0872`** / `0.7125` / `561 m` |
| `MT4707111382020220904` | `0.7782` / `0.7208` / `154 m` | **`0.8829`** / `0.8205` / `113 m` | **`0.0949`** / **`0.1510`** / `104 m` |
| `OK3464409522620180309` | `0.7859` / `0.9595` / `90 m` | **`0.8924`** / **`0.8446`** / `185 m` | **`0.1012`** / `0.9459` / `160 m` |

  On `OK3464409522620180309`, the `W = 36 h` morning shift pushes the chosen head-line distance from `90 m` to `185 m` and drops proxy saved fraction from **`0.9595` to `0.8446`** (`-0.115`); on `MT4707111382020220904`, the `W = 22 h` horizon shift collapses proxy saved fraction from **`0.7208` to `0.1510`** (`-0.570`).
- **What consequence remains unmeasured?** Because all 48 archived exam fires and all 3 shipped examples use afternoon PM FEDS detections with 72-hour horizon (`W[-1] = 34.0 h`, `z[6] = 0.00`), the published 48-fire exam table was shielded from this defect—except for any exam fire whose exact local detection minute shifts `work_clock[-1]` away from `34.0 h` (unchecked without external FEDS timestamps). Any public API caller using morning/night detections or non-72-hour horizons hits immediate network saturation.

### Finding B — Epoch minibatch permutation ignores `--seed` [`MINOR HYGIENE`]
- **What failed?** [`train_value.py:90`](../../../../models/firepilot/training/train_value.py#L90) seeds per-epoch batch shuffling with `ep` (`np.random.default_rng(ep)`) rather than combining `a.seed` and `ep`.
- **What changed when tested?** Static inspection confirmed all multi-seed runs share identical epoch batch orderings regardless of `--seed` (only PyTorch weight initialization varies with `--seed`).
- **What consequence remains unmeasured?** Whether coupling batch order to `--seed` changes the small single-network variance across training seeds remains unmeasured.

## 3. Clean Checks (Items 2, 3, 8, 10)
- **Weight health & dead units (CLEAN):** All 9 arrays in `firepilot_weights.npz` are finite, with **`0` dead hidden/output units** (`min_out_norm`: `W0 = 0.3082`, `W1 = 0.3473`, `W2 = 0.8250`), and `sig_scale = 1.0000` (widen-only compliant).
- **PyTorch ↔ NumPy export parity (CLEAN):** Reconstructing `train_value.Net` from `firepilot_weights.npz` matched `ValueNet.__call__` to **`3.23e-8`** (mean saved fraction) and **`5.09e-8`** (sigma) on 64 random valid inputs.
- **Ratchet arithmetic replay (CLEAN):** All **`5/5`** logged rungs (`r0..r4`) in [`training/ratchet.json`](../../../../models/firepilot/training/ratchet.json) match the published rule (`val > inc + 0.01` and `gate >= 0.95 * inc`), with `r1` kept and `r2..r4` rejected.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/firepilot_v1_0_probe.py
```

Model snapshot `61c70bf`, audit checkout `db285c4`. Uses NumPy, pandas, SciPy, and CPU PyTorch. [Reproduction probe](firepilot_v1_0_probe.py)
