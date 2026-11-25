# ML & Physics Audit: EnergyNet v2.1

- **Date:** 2026-11-25
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
  - `models/energynet/sigma_factors.json`
  - `models/energynet/ratchet.yaml`
- **Overall verdict:** **`CRITICAL DEFECT`** (1 high-impact silent normalization bug + 1 minor loss-masking hygiene item; PyTorch↔NumPy parity, ensemble diversity, feature alignment, and NLL math are all clean).

---

## 7-Point Checklist Summary

| # | Check | Status | Summary |
|---|---|---|---|
| 1 | Weight & normalization health | **CRITICAL DEFECT** | Zero NaN/Inf, zero dead rows, healthy 5-seed diversity (`cos 0.11–0.14`), **but** 4 of 5 physics work-fraction features are crushed to near-constants (`std = 4e-6 .. 4.5e-5`) by outlier-poisoned `x_mu`/`x_sd` (`sd` up to `53,606`). |
| 2 | PyTorch ↔ NumPy parity | **CLEAN** | Reconstructed `train.Net` from all 5 shipped `.npz` files; max abs diff vs `ENet.forward`: `mu <= 2.69e-6`, `sig <= 1.49e-5`, `comp <= 1.20e-5`. |
| 3 | Train-vs-real feature alignment | **CLEAN** | `dataset.py`, `realfeat.py`, and `energynet.py` call the same `features.py` helpers with matching units, `PHYS_K`, and tier masks. |
| 4 | Loss, gradient & optimization | **MINOR HYGIENE** | Exact Gaussian NLL (`ls + 0.5 * ((y - m)/exp(ls))^2`), AdamW + OneCycleLR (60 ep), grad clip `5.0`. Minor: auxiliary component MSE (`train.py:91`) is not masked by `w = (yv > 0.02)`, unlike NLL. |
| 5 | Twin-test triage | **DOCUMENTED** | Known `-4.5%` FTP twin read is logged in `ratchet.yaml` / `MODEL_CARD.md`; fixing the crushed `phys_vec` ratios is a concrete hypothesis for a future rung. |
| 6 | Uncertainty calibration | **MINOR HYGIENE** | Per-fleet dev factors in `sigma_factors.json` are `>= 1.0`; simulation-side `SIM_conv = 0.9656` is `< 1.0` (product `SIM_conv * VED_ICE = 1.45 > 1.0`, so effective VED ICE factor is widen-only). |
| 7 | Undocumented limits | **CLEAN** | Final-epoch checkpointing (no best-val early stop) and fixed normalization seed `12345` across all 5 seeds are transparent in `train.py`. |

---

## Detailed Findings

### Finding 1 (`CRITICAL DEFECT`): Four core physics work-fraction features crushed by 20,000×–250,000×

**Where:**
- `models/energynet/training/features.py:136-143` (`phys_vec`):
  ```python
  Ep = max(ph[0], 1e-6)
  F[3] = ph[2] / Ep; F[4] = ph[3] / Ep; F[5] = ph[4] / Ep; F[6] = ph[5] / Ep; F[7] = ph[6] / Ep
  ```
- `models/energynet/training/train.py:59-61`:
  ```python
  T0 = tiers(np.random.default_rng(12345))
  X0, E0 = DS.assemble(B, T0)
  mu = X0.mean(0); sd = X0.std(0) + 1e-3
  ```

**Mechanism:**
On standstill or pure-downhill simulated bank trips where positive wheel work `ph[0] ≈ 0`, `Ep` hits the floor `1e-6` while `ph[2]` (aero), `ph[3]` (rolling), `ph[5]` (downhill grade), and `ph[6]` (positive inertia) remain non-zero or tiny negative/positive residuals, sending the ratios `ph[k] / Ep` to `~10^7`. Because `train.py` computes raw sample mean and standard deviation across `X0` without clipping or robust quantiles — and uses the same fixed seed `12345` for all 5 ensemble members — all 5 shipped `.npz` files store poisoned `x_mu` and `x_sd` values.

**Measured impact across 1,496 real VED dev trips (shipped `r15_s0.npz`):**

| Col | Feature | Raw real-trip p5..p95 | Shipped `x_mu` | Shipped `x_sd` | Normalized `(X - mu)/sd` p5..p95 | Normalized real-trip `std` |
|---:|---|---:|---:|---:|---:|---:|
| 90 | `log(E)` (control) | `-0.366 .. 3.096` | `3.10` | `1.58` | `-2.1925 .. -0.0050` | `0.676188` |
| 93 | **`aero / Ep`** | `0.072 .. 0.418` | `8.42` | **`2,385.44`** | `-0.003500 .. -0.003355` | **`0.000045`** |
| 94 | **`rolling / Ep`** | `0.238 .. 0.585` | `16.23` | **`4,639.52`** | `-0.003447 .. -0.003372` | **`0.000024`** |
| 95 | `grade_up / Ep` (control) | `0.077 .. 0.427` | `0.19` | `0.15` | `-0.7748 .. +1.5674` | `0.713481` |
| 96 | **`grade_dn / Ep`** | `-0.602 .. -0.060` | `-183.11` | **`53,606.09`** | `+0.003405 .. +0.003415` | **`0.000004`** |
| 97 | **`inertia_pos / Ep`** | `0.373 .. 0.840` | `30.71` | **`8,865.52`** | `-0.003422 .. -0.003369` | **`0.000016`** |

For every normal real trip, EnergyNet v2.1 receives essentially constant values (`std ~ 1e-5`) for the aerodynamic, rolling, downhill-grade, and acceleration-inertia fractions of wheel work.

**Why existing tests did not catch it:**
- `mlratchet validate` checks schema and golden smoke outputs.
- `site/tests/parity.test.ts` checks NumPy ↔ TypeScript using the same poisoned `x_mu`/`x_sd`.
- `train.py` still converges smoothly because `log(E)`, `log(Ep/E)`, `log(En/E)`, and the 56-bin speed×accel histogram carry correlated signal.

**Candidate fix (for a future ratchet rung, not applied here):**
Clamp `phys_vec` ratios cleanly (e.g., divide by `max(ph[0], 0.05)` or clip `F[3:8]` to `[-5, 5]`) and/or compute `x_mu`/`x_sd` with robust quantiles or on `okm = yv > 0.02` rows before standardizing, then judge on the existing frozen `val2` + dev `D2` gate.

### Finding 2 (`MINOR HYGIENE`): Auxiliary component loss unmasked on near-zero trips
- `models/energynet/training/train.py:90-91`: `nll` is masked by `w = (yv > 0.02)`, whereas `a.wc * ((c - Ct[idx]) ** 2).mean()` averages over all rows including `yv <= 0.02`. Bounded impact because `comp_targets` clips components to `1e-3 * E` (`E >= 1e-3`), preventing NaN/Inf.

### Clean Checks (Verified Measurements)
- **PyTorch ↔ NumPy parity (64 random inputs, `float32` vs `float64`):**
  - `r15_s0.npz`: `mu=2.29e-6`, `sig=6.81e-6`, `comp=9.14e-6`
  - `r15_s1.npz`: `mu=2.54e-6`, `sig=6.24e-6`, `comp=8.07e-6`
  - `r15_s2.npz`: `mu=1.28e-6`, `sig=1.57e-6`, `comp=1.20e-5`
  - `r15_s3.npz`: `mu=2.69e-6`, `sig=1.49e-5`, `comp=5.29e-6`
  - `r15_s4.npz`: `mu=1.69e-6`, `sig=3.76e-6`, `comp=6.34e-6`
- **Ensemble diversity (`body_0_weight` pairwise cosine similarity across `s0..s4`):** off-diagonal range `0.114 .. 0.143` (five genuinely independent seeds).
- **Weight health:** Zero non-finite values, zero near-zero rows (`norm < 1e-4`) across all layers of all 5 members.
