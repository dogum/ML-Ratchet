# ML & Physics Audit: FlowCast v1.0

- **Date:** 2026-11-25
- **Base commit:** `61c70bf` (`origin/main`)
- **Mode:** Read-only forensic audit (no model code or weights modified)
- **Shipped weights audited:** `models/flowcast/flowcast_weights.npz` (ungauged `r1b`), `models/flowcast/flowcast_nowcast_weights.npz` (gauged `n1`)
- **Files inspected:**
  - `models/flowcast/training/src/common.py`
  - `models/flowcast/training/src/simbank.py`
  - `models/flowcast/training/src/bankdata.py`
  - `models/flowcast/training/src/model.py`
  - `models/flowcast/training/src/train.py`
  - `models/flowcast/training/src/evalreal.py`
  - `models/flowcast/training/src/export.py`
  - `models/flowcast/training/src/calib_sigma.py`
  - `models/flowcast/flowcast.py`
  - `models/flowcast/ratchet.yaml`
  - `models/flowcast/training/configs/r1b.json`, `r5.json`
  - `models/flowcast/training/results/{export_check,sigma_r1b,twin_r0_p0_summary,twin_r1b_p1_summary,rescore_main_summary_2026-11-18,nowcast_summary}.json`
- **Overall verdict:** **`CLEAN`** (zero silent bugs found; performance limits are transparent, documented CPU-budget and simulator choices).

---

## 7-Point Checklist Summary

| # | Check | Status | Summary |
|---|---|---|---|
| 1 | Weight & normalization health | **CLEAN** | All weights finite; all 18 static `st_mu`/`st_sd` well-scaled (`sd` in `[0.0096, 0.8979]`); all 29 LSTM input column norms balanced (`1.12–3.69`); LSTM gate biases and all 6 head rows healthy. |
| 2 | PyTorch ↔ NumPy parity | **CLEAN** | Reconstructed `model.FlowNet` from both shipped `.npz` files; max abs diff vs `FlowCast.forward`: `mean=2.67e-7, q=6.99e-7` (ungauged) and `mean=3.94e-7, q=5.97e-7` (nowcast). Matches shipped `export_check.json` (`1.69e-7` rel err on 4 dev basins). |
| 3 | Train-vs-real feature alignment | **CLEAN** | Exact match across `simbank.py`, `bankdata.py`, `evalreal.py`, and `flowcast.py` for dynamic feature scaling (`dyn_features`), static vector construction, DOY formula `(doy[a] - 1 + arange(L)) % 365 + 1`, 365-day warm-up blocks, and nowcast lag channels. |
| 4 | Loss, gradient & optimization | **CLEAN** | NSE* mean loss + monotone cumulative-softplus pinball quantiles (`model.py:27-43`), gradient clipping `1.0`, cosine LR schedule (`2e-3 -> 1e-4`). Subtle `target_std` loss bias was caught by its own twin test and fixed in rung `r0b` (`weight: "budyko"`). |
| 5 | Twin-test triage | **CLEAN** | `twin_r0_p0_summary.json` -> `twin_r1b_p1_summary.json` honestly documents the `r0 -> r0b/r1b` improvement (`ctwin NSE 0.520 -> 0.595`, `real dev NSE 0.390 -> 0.425`, `flash_ratio 0.54 -> 0.79`). |
| 6 | Uncertainty calibration | **CLEAN** | `calib_sigma.py` fits widen-only `k >= 1.0` on 52 dev basins (`k50 = 1.00`, `k90 = 1.08`). The `0.863 -> 0.953` jump at `k90 = 1.08` is expected because clamping `q05` at `0.0` catches near-zero low-flow days. |
| 7 | Undocumented limits | **DOCUMENTED** | Single-seed, single-layer 64-unit LSTM (`24,710` params, no weight decay, no ensemble, `3,000` steps, last-step checkpoint rather than best-on-sim-val). All explicitly stated in `NOTEBOOK.md` and `ratchet.yaml` as single-core CPU budget choices. |

---

## Key Measurements

### Static Input Normalization & First-Layer Norms (`flowcast_weights.npz`)
- `st_sd` range across all 18 static features: `min = 0.0096` (`hp_freq`, naturally in `[0, 0.1]`), `max = 0.8979` (`elev_range_km`). Zero outlier-poisoned channels.
- `w_ih` column norms:
  - 11 dynamic forcing channels: `1.72 .. 3.69` (highest on `P/10` = `3.69` and `log1p(P)` = `2.57`, physically sensible).
  - 18 static basin channels: `1.12 .. 2.80` (highest on `log_area` = `2.80`, `frac_snow` = `1.84`, `reservoir_frac` = `1.80`, `log_p_mean` = `1.80`).
- LSTM gate bias means `(i, f, g, o)`: `[-0.024, +0.139, -0.034, +0.052]` (healthy, no saturated/dead gates).
- Output head row norms `(mean, q05, q25, q50, q75, q95)`: `[1.902, 1.769, 1.596, 1.425, 1.441, 2.019]`.

### PyTorch ↔ NumPy Exact Parity (4 × 60 random sequences)
- `flowcast_weights.npz` (`n_dyn=11, n_st=18, H=64`): `max |mean_np - mean_torch| = 2.666e-7`, `max |q_np - q_torch| = 6.994e-7`.
- `flowcast_nowcast_weights.npz` (`n_dyn=14, n_st=18, H=64`): `max |mean_np - mean_torch| = 3.943e-7`, `max |q_np - q_torch| = 5.969e-7`.

### Honest Non-Bug Notes for Future Work
- **Capacity & ensembling:** FlowCast v1.0 is a single-seed, single-layer `H=64` LSTM (`24.7k` weights) trained for 3,000 steps (`batch=32`) with no weight decay and last-step checkpointing. (Note: `r5` already tested +5,000 extra steps on the same `H=64` network and saw no frozen-set gain `-0.006`, while the River Bake-off independently confirmed that 5 families converge near the same WaterSim p1 ceiling and that multi-member ensembling + refit water-yield prior are the real levers.)
