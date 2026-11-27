# ML & Physics Audit: EnergyNet v2.1

- **Date:** 2026-11-25 (cross-checked by two independent reviewers)
- **Base commit:** `61c70bf` (`origin/main`)
- **Mode:** Read-only forensic audit (no model code or weights modified)
- **Shipped weights audited:** `models/energynet/weights/r15_s0.npz` .. `r15_s4.npz` (5-member MLP ensemble)
- **Files inspected:**
  - `models/energynet/training/features.py`
  - `models/energynet/training/dataset.py`
  - `models/energynet/training/realfeat.py`
  - `models/energynet/training/train.py`
  - `models/energynet/training/model_np.py`
  - `models/energynet/training/release.py`
  - `models/energynet/energynet.py`
  - `models/energynet/examples/examples.json` + `.npz` trips
  - `models/energynet/sigma_factors.json`
  - `models/energynet/ratchet.yaml`
- **Overall verdict:** **`CRITICAL DEFECT`** (1 verified high-impact silent normalization bug + 1 minor loss-masking observation; PyTorch↔NumPy parity, ensemble diversity, feature alignment, and NLL math are all clean).

---

## Checklist Summary

| # | Check | Status | Summary |
|---|---|---|---|
| 1 | Weight, normalization & sensitivity | **CRITICAL DEFECT** | Zero NaN/Inf, zero dead rows, healthy 5-seed diversity (`cos 0.11–0.14`), **but** 4 of 5 physics work-fraction features are crushed to near-constants (`std = 4e-6 .. 4.5e-5`) by outlier-poisoned `x_mu`/`x_sd` (`sd` up to `53,606`); counterfactual ablation moves predictions `<= 0.00039%` vs `6.08%` control. |
| 2 | PyTorch ↔ NumPy parity | **CLEAN** | Reconstructed `train.Net` from all 5 shipped `.npz` files; max abs diff vs `ENet.forward`: `mu <= 2.69e-6`, `sig <= 1.49e-5`, `comp <= 1.20e-5`. |
| 3 | Train-vs-real feature alignment | **CLEAN** | `dataset.py`, `realfeat.py`, and `energynet.py` call the same `features.py` helpers with matching units, `PHYS_K`, and tier masks. |
| 4 | Loss, gradient & optimization | **MINOR HYGIENE** | Exact Gaussian NLL (`ls + 0.5 * ((y - m)/exp(ls))^2`), AdamW + OneCycleLR (60 ep), grad clip `5.0`. Auxiliary component MSE (`train.py:91`) is not masked by `w = (yv > 0.02)`, unlike NLL (see caveat below). |
| 5 | Twin-test triage | **DOCUMENTED** | Known `-4.5%` FTP twin read is logged in `ratchet.yaml` / `MODEL_CARD.md`; whether restoring the 4 crushed `phys_vec` ratios closes part of that gap is testable in a matched-seed repair rung. |
| 6 | Uncertainty calibration | **MINOR HYGIENE** | Per-fleet dev factors in `sigma_factors.json` are `>= 1.0`; simulation-side `SIM_conv = 0.9656` is `< 1.0` (effective `SIM_conv * VED_ICE = 1.45 > 1.0`). |
| 7 | Undocumented limits | **CLEAN** | Final-epoch checkpointing (no best-val early stop) and fixed normalization seed `12345` across all 5 ensemble members are transparent in `train.py`. |

---

## Detailed Findings

### Finding 1 (`CRITICAL DEFECT`): Four core physics work-fraction features crushed by 20,000×–250,000×

- **What failed?**
  - In `models/energynet/training/features.py:136-143` (`phys_vec`), aerodynamic (`ph[2]`), rolling (`ph[3]`), downhill-grade (`ph[5]`), and positive-inertia (`ph[6]`) work are divided by `Ep = max(ph[0], 1e-6)`.
  - On a **moving downhill trip** (note: a fully stationary trip gives `0` for all five ratios), net positive wheel work `ph[0]` can be `0.0` while aero drag, rolling resistance, and gravity still do substantial work. Independently reproduced on a synthetic `-8%` grade trip at constant `15 m/s` (`300 s`): `ph[0] = 0.0`, `aero/Ep = +377,943`, `roll/Ep = +654,192`, `grade_dn/Ep = -5,808,582`.
  - In `models/energynet/training/train.py:59-61`, raw sample `mu = X0.mean(0); sd = X0.std(0) + 1e-3` is computed before clipping, using fixed seed `12345` across all 5 ensemble seeds, poisoning `x_mu` and `x_sd` in all 5 shipped `.npz` files.

- **What changed when we tested it?**
  1. **Real-trip post-normalization spread (1,496 real VED dev trips, shipped `r15_s0.npz`):**

     | Col | Feature | Raw real-trip p5..p95 | Shipped `x_mu` | Shipped `x_sd` | Normalized `(X - mu)/sd` p5..p95 | Normalized real-trip `std` |
     |---:|---|---:|---:|---:|---:|---:|
     | 90 | `log(E)` (control) | `-0.366 .. 3.096` | `3.10` | `1.58` | `-2.1925 .. -0.0050` | `0.676188` |
     | 93 | **`aero / Ep`** | `0.072 .. 0.418` | `8.42` | **`2,385.44`** | `-0.003500 .. -0.003355` | **`0.000045`** |
     | 94 | **`rolling / Ep`** | `0.238 .. 0.585` | `16.23` | **`4,639.52`** | `-0.003447 .. -0.003372` | **`0.000024`** |
     | 95 | `grade_up / Ep` (control) | `0.077 .. 0.427` | `0.19` | `0.15` | `-0.7748 .. +1.5674` | `0.713481` |
     | 96 | **`grade_dn / Ep`** | `-0.602 .. -0.060` | `-183.11` | **`53,606.09`** | `+0.003405 .. +0.003415` | **`0.000004`** |
     | 97 | **`inertia_pos / Ep`** | `0.373 .. 0.840` | `30.71` | **`8,865.52`** | `-0.003422 .. -0.003369` | **`0.000016`** |

  2. **Counterfactual feature-sensitivity ablation (5 shipped real VED examples in `models/energynet/examples/`, holding suspect inputs constant at their median across the 5 trips while keeping all other inputs unchanged):**
     - Holding all four columns `[93, 94, 96, 97]` constant together: max prediction change **`0.00039%`**.
     - Holding healthy control column `95` (`grade_up / Ep`) constant alone: max prediction change **`6.077%`**.

- **What consequence remains unmeasured?**
  - **Predictive accuracy cost is still unmeasured.** Other channels (`log(E)`, `log(Ep/E)`, `log(En/E)`, speed×accel histogram, grade histogram) carry correlated information. Measuring whether bounding these work ratios improves total-energy, component, or downhill-trip accuracy requires a matched-seed repair rung on the same simulation bank judged strictly on the existing dev gate (leaving exam untouched).
  - Identifying the exact historical bank rows that dominated each `x_sd` requires the original `train2di+train2bdi+train2cdi` bank files.

### Finding 2 (`MINOR HYGIENE`): Auxiliary component loss unmasked on near-zero net-energy trips

- **What failed?** `models/energynet/training/train.py:90-91` masks `nll` by `w = (yv > 0.02)`, whereas `a.wc * ((c - Ct[idx]) ** 2).mean()` averages over all rows including `yv <= 0.02`.
- **What changed when we tested it?** Not replayed; `comp_targets` clips components to `>= 1e-3 * E` (`E >= 1e-3`), preventing `NaN`/`Inf`.
- **What consequence remains unmeasured?** A small net-energy target (e.g., regenerative downhill EV/HEV trip) can still have meaningful positive component work (`aero`, `rolling`, `aux`); inspect those rows' component targets before deciding whether to mask them or keep them unmasked.

---

## Clean Checks (Verified Measurements)

- **PyTorch ↔ NumPy parity (64 random inputs, `float32` vs `float64`):**
  - `r15_s0.npz`: `mu=2.29e-6`, `sig=6.81e-6`, `comp=9.14e-6`
  - `r15_s1.npz`: `mu=2.54e-6`, `sig=6.24e-6`, `comp=8.07e-6`
  - `r15_s2.npz`: `mu=1.28e-6`, `sig=1.57e-6`, `comp=1.20e-5`
  - `r15_s3.npz`: `mu=2.69e-6`, `sig=1.49e-5`, `comp=5.29e-6`
  - `r15_s4.npz`: `mu=1.69e-6`, `sig=3.76e-6`, `comp=6.34e-6`
- **Ensemble diversity (`body_0_weight` pairwise cosine similarity across `s0..s4`):** off-diagonal range `0.114 .. 0.143` (five genuinely independent seeds).
- **Weight health:** Zero non-finite values, zero near-zero rows (`norm < 1e-4`) across all layers of all 5 members.
