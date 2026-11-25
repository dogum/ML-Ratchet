# CellSense v1.3 (LFP + NMC811) | Shared-seed NMC811 ensemble collinearity, silent all-NaN outputs, and `endrule` discontinuity

**First pass (`reviewer-a`). Verdict: MINOR HYGIENE / ENSEMBLE & API DEFECTS.** Audited against model base `61c70bf` across all seven shipped `.npz` archives (3 LFP networks: health, lifetime, partial; 4 NMC811 ensemble members) following the 10-point protocol in [`SKILL.md`](../../SKILL.md). [Model code](../../../../models/cellsense/cellsense.py) · [NMC811 trainer](../../../../models/cellsense/training/nmc_train.py) · [NMC811 exporter](../../../../models/cellsense/training/nmc_export.py)

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Diagnose capacity (`q_total`), resistance, degradation modes (`lli`, `lam_ne`, `lam_pe`, `fade`), and (on LFP) remaining cycle life from a single constant-current discharge curve `Q(V)` (plus `T(V)` on LFP) and an optional fresh reference curve from the same cell.
- **Information boundary:** Dev calibration uses 6 LFP cells (2 per batch) and 2 NMC811 cells (`W8`, `W10`), excluded from all exam cells. All networks are trained exclusively on PyBaMM-verified single-particle simulations.

## 2. Findings

### Finding A — NMC811 4-member ensemble shares default `--seed 0` initialization (collinear weights) [`MINOR HYGIENE` / `UNDOCUMENTED LIMIT`]
- **What failed?** Rung `n12` in [`training/nmc_ratchet.json`](../../../../models/cellsense/training/nmc_ratchet.json) packs four prior checkpoints (`n8_ocpj.pt | n9_ocpj2.pt | n10_ocpd.pt | n11_ocpd_mix.pt`) into the shipped 4-member NMC811 ensemble (`cellsense_nmc811_{weights,m1,m2,m3}.npz`). [`nmc_train.py:52,56`](../../../../models/cellsense/training/nmc_train.py#L52) defaults to `--seed 0` for both `torch.manual_seed` and `np.random.default_rng`, so all four members started from the exact same random weight initialization and differed only in training-bank perturbation mixes (`ocpj`, `ocpj2`, `ocpd`, `ocpd_mix`).
- **What changed when tested?** Computing pairwise weight cosine similarities across all 6 member pairs reveals strong collinearity rather than independent-seed ensemble diversity (`~0.00–0.05`):

| Layer | Min pairwise cosine (6 pairs) | Max pairwise cosine (6 pairs) |
|---|---:|---:|
| `stem_w` | **`0.8542`** | **`0.9870`** |
| `b0c1_w` | **`0.8985`** | **`0.9657`** |
| `b2c1_w` | `0.5906` | `0.7154` |
| `h0_w` | **`0.7219`** | **`0.7403`** |
| `h1_w` | `0.6055` | `0.6576` |

  In addition, `n12` passed the frozen-sim ratchet margin (`> +0.01` on `ocpd`) by **`+0.0002`** (`0.7235` vs `0.7133`, `Δ = +0.0102`), while improving dev partial MAE (`126.1 -> 92.5 mAh`).
- **What consequence remains unmeasured?** How much additional accuracy and epistemic uncertainty coverage a genuinely distinct-seed 4-member NMC811 ensemble would achieve on the held-out NMC811 exam cells remains unmeasured without retraining.

### Finding B — Empty / all-NaN and all-zero curves silently return plausible health estimates [`MINOR HYGIENE`]
- **What failed?** [`cellsense.py`](../../../../models/cellsense/cellsense.py) has no minimum valid-sample or non-zero-charge guard in `CellSense.diagnose()` for either LFP or NMC811. Missing voltages are zero/constant-filled into the CNN channels and evaluated normally.
- **What changed when tested?** Passing completely missing (`250` `NaN`s) or flat-zero curves silently returned finite, normal-looking battery diagnoses without warning or error:
  - LFP `diagnose(all-NaN, all-NaN)`: **`q_total = 0.8168 ± 0.1074 Ah`**, `lli = 0.1777`.
  - LFP `diagnose(zeros(250), 30 °C)`: **`q_total = 0.9970 ± 0.0371 Ah`**.
  - NMC811 `diagnose(all-NaN)`: **`q_total = 4.0251 ± 0.3786 Ah`**, `lli = 0.1552`.
- **What consequence remains unmeasured?** Exam curves are pre-screened for valid discharges, so this affects operational / API callers feeding corrupted or blank telemetry rather than published exam scores.

### Finding C — NMC811 `endrule` sharp discontinuity at `2.50 V` (`index 249`) [`MINOR HYGIENE`]
- **What failed?** On NMC811 (`endrule=True`), any curve that reaches the final `2.50 V` grid point bypasses the CNN for `q_total` and returns direct coulomb counting (`q_last`) with a tiny `±0.0020 Ah` floor; dropping even that single endpoint reverts to the neural predictor.
- **What changed when tested?** On packaged exam cell `m50t_W9[5]` (with fresh ref `m50t_W9[0]`), setting only the final voltage bin (`index 249`, `2.50 V`) to `NaN` jumped `q_total` from **`4.4864 ± 0.0020 Ah`** to **`4.4915 ± 0.0215 Ah`** (**`+5.12 mAh`**, **`10.75×`** sigma jump).
- **What consequence remains unmeasured?** Whether near-complete real discharges stopping at `2.51–2.55 V` exhibit systematic bias across the full NMC811 exam population beyond the published partial-discharge slices remains unmeasured.

## 3. Clean Checks (Items 2, 3, 7, 8, 10)
- **Weight health & dead units:** All 7 `.npz` archives contain 100% finite weights and **`0` dead output filters/rows** (`norm < 1e-4`).
- **Folded PyTorch ↔ NumPy export parity:** Reconstructing the folded 1-D ResNet in PyTorch (`torch.nn.functional.conv1d`/`gelu`/`linear`) matched `CellSenseNet.forward` within **`1.31e-6` to `1.00e-5`** across all 7 `.npz` files (`6.68e-6` LFP health, `9.54e-6` LFP life, `1.00e-5` LFP partial, `1.31e-6–4.17e-6` NMC811 members). Replacing `cellsense._gelu`'s Abramowitz-Stegun polynomial `erf` (`4.77e-7` single-op max error vs `torch.nn.functional.gelu`) with exact `scipy.special.erf` gave essentially identical float32 accumulation error (`1.67e-6–1.23e-5`).
- **Seeding & resume:** Both [`train.py:47`](../../../../models/cellsense/training/train.py#L47) and [`nmc_train.py:56`](../../../../models/cellsense/training/nmc_train.py#L56) seed both PyTorch and NumPy; `nmc_train.py:66` deterministically offsets `rng = np.random.default_rng(a.seed + start)` on resume.
- **Calibration widen factors:** NMC811 widen factors are strictly widen-only (`q_total: 1.0`, `fade: 1.5223`).
- **Ratchet arithmetic replay:** All **`9/9`** LFP rungs (`training/ratchet.json`) and all **`13/13`** NMC811 rungs (`training/nmc_ratchet.json`) match their documented promotion rules.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/cellsense_v1_3_probe.py
```

Model snapshot `61c70bf`, audit checkout `db285c4`. Uses NumPy, SciPy, and CPU PyTorch. [Reproduction probe](cellsense_v1_3_probe.py)
