# BasinDetective v1.0 (`R5`) | Clean core ML/parity, API guardrail crashes, and hardcoded ERA5 backward-drift prior

**First pass (`reviewer-a`). Verdict: MINOR HYGIENE & DOCUMENTED DIRECTIONAL PRIOR.** Audited against model base `61c70bf` across the shipped 5-member ensemble (`weights/basindetective_v1.npz`), feature pipeline, training/export scripts, 4 shipped CAMELS-US examples, and 7-rung ratchet log following [`SKILL.md`](../../SKILL.md). [Model code](../../../../models/basindetective/basindetective/model.py) · [Features](../../../../models/basindetective/basindetective/features.py) · [Trainer](../../../../models/basindetective/training/train.py)

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Given ~9 years of daily meteorological forcing (`P`, `T_mean`, `PET`) and observed streamflow (`Q_mm`) for a river basin plus a second decade's forcing, predict 6 inter-decadal flow-signature changes, upstream reservoir regulation probability, and 4 latent hydrologic properties with 1-sigma intervals.
- **Information boundary:** All 4 shipped examples (`camels_04185000`, `camels_08165300`, `camels_12388400`, `camels_14020000`) belong to the 619-basin exam set (`training/exam_per_basin.csv`). Real dev calibration (gauge noise, precipitation bias, ERA5-Land decadal drift, annual error, widen-only `sig_scale`, and `cls_temp`) used only the 52 dev basins.

## 2. Findings

### Finding A — Hardcoded ERA5-Land backward-decade precipitation-drift prior violates identical-climate zero-change identity [`UNDOCUMENTED OPERATIONAL RISK` / `DOCUMENTED EXAM PRIOR`]
- **What failed?** Physics rungs `v3`/`v4` ([`MODEL_CARD.md:73-76`](../../../../models/basindetective/MODEL_CARD.md#L73)) bake an asymmetric decadal ERA5-Land precipitation-drift correction (`TRUE target-decade P = SEEN / exp(delta)` with `delta ~ N(+0.029, 0.038)`) directly into the simulated training bank to match the backward `1999-2008 -> 1989-99` CAMELS-US exam. The public `bd.predict(det, cal, qcal, tgt)` API has no flag to disable this drift prior or specify forward vs. backward time direction.
- **What changed when tested?** Passing **identical** target and calibration forcing (`tgt == cal`, where true physical change is identically `0.0`) systematically predicts negative mean-flow and runoff-ratio changes across all four shipped CAMELS-US exam basins:

| Exam basin | `d_logq` (`tgt == cal`) | Implied `%` flow shift | `d_rr` (`tgt == cal`) | `d_logq5` (`tgt == cal`) | `d_hfd` (`tgt == cal`) |
|---|---:|---:|---:|---:|---:|
| `camels_04185000` | **`-0.1304`** | **`-12.2%`** | `-0.0416` | `-0.1055` | `+2.56 d` |
| `camels_08165300` | `-0.0020` | `-0.2%` | `-0.0017` | **`-0.2327`** | **`+8.14 d`** |
| `camels_12388400` | `-0.0451` | `-4.4%` | `-0.0182` | `-0.0575` | `-0.32 d` |
| `camels_14020000` | `-0.0322` | `-3.2%` | `-0.0238` | `-0.0049` | `+0.39 d` |

  By contrast, physically perturbing target precipitation (`+20% P` on `camels_04185000`) moves `d_logq` cleanly in the right direction (`-0.0150 -> +0.4117`), and flattening calibration streamflow to a constant raises regulation probability (`p_reg: 0.0281 -> 0.5152`).
- **What consequence remains unmeasured?** Applying the released weights to forward climate projections (`2000s -> 2050s`) or drift-free gauge/Daymet forcing will inherit a systematic negative flow bias (especially on high-elasticity basins); the magnitude of this bias on forward CAMELS splits (`1989-99 -> 1999-2008`) was not re-evaluated here.

### Finding B — Public API crashes with unhandled `KeyError: 'bfi_lh'` on empty/short records and silently predicts on all-zero streamflow [`MINOR HYGIENE`]
- **What failed?** In [`basindetective/features.py:120`](../../../../models/basindetective/basindetective/features.py#L120), `sigs()` returns an empty dict `{}` whenever fewer than `365` finite streamflow days are present, and [`features.py:227`](../../../../models/basindetective/basindetective/features.py#L227) unconditionally indexes `s['bfi_lh']` without validation (while `_climate` emits `RuntimeWarning: Mean of empty slice` when `< 12` calendar months appear). Conversely, an all-zero streamflow series (`Q_mm == 0` everywhere, e.g., dead/zero-filled gauge) passes `qq.size >= 365` and proceeds silently into the network.
- **What changed when tested?** On `camels_04185000`:
  - `all-NaN Q_mm`, `30 d` record, and `180 d` record all crash with unhandled **`KeyError: 'bfi_lh'`** (instead of an actionable `ValueError`).
  - `365 d` record and `all-zero Q_mm` silently return finite predictions (`all-zero Q_mm`: **`d_logq = +0.5007 ± 0.5182`** (`+65%`), `p_reg = 0.2487`; `365 d`: `d_logq = +0.1740 ± 0.2158`).
- **What consequence remains unmeasured?** All 619 CAMELS-US exam basins have multi-year gauged records, so this affects operational/API error handling rather than published exam metrics.

## 3. Clean Checks (Items 2, 3, 7, 8, 9, 10)
- **Weights, normalization, and dead units (CLEAN):** All 60 numeric weight arrays across all 5 ensemble members are finite with **`0` dead rows** (`min_row_norm = 0.7101`). Feature scales `x_iqr` span **`[0.01886, 10.65425]`** (no poisoned or zero-variance scales). On all 4 shipped CAMELS-US examples, normalized `Z` spans **`[-4.029, 3.743]`** with **`0/600`** values hitting the `±6` clip bound.
- **Ensemble diversity (CLEAN):** Pairwise first-layer `w0` cosine similarities across all 10 pairs of the 5 members lie in **`[0.0007, 0.0411]`**—confirming genuinely independent 5-seed training.
- **PyTorch ↔ NumPy export parity (CLEAN):** Reconstructing `train.MLP` across all 5 members on real + random inputs gives max raw output diff **`1.78e-14`** in float64 (matching `gelu_exact` via `scipy.special.erf` to machine precision) and **`3.75e-6`** in float32. All 4 shipped examples match `examples/expected.json` to **`0.00e+00`**.
- **Seeding, resume & calibration (CLEAN):** [`training/train.py:118-119,152-153`](../../../../models/basindetective/training/train.py#L118) seeds both PyTorch and NumPy and persists/restores `rs` in resume checkpoints. All 11 `sig_scale` entries satisfy widen-only compliance (`1.0000–1.7291`), and `cls_temp = 1.3` (`> 1.0`).
- **Ratchet arithmetic replay (CLEAN):** All **`7/7`** rungs (`R1`, `R1b`, `R1e`, `R2`, `R3`, `R4`, `R5`) match the documented promotion rule (`sim_skill > inc + 0.01` on current physics and `dev_composite >= inc - 0.02`).

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/basindetective_v1_0_probe.py
```

Model snapshot `61c70bf`, audit checkout `db285c4`. Uses NumPy, pandas, SciPy, and CPU PyTorch. [Reproduction probe](basindetective_v1_0_probe.py)
