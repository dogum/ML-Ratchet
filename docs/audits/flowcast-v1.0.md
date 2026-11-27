# ML & Physics Audit: FlowCast v1.0

- **Date:** 2026-11-25 (cross-checked by two independent reviewers)
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
  - `models/flowcast/training/src/rescore_main.py`
  - `models/flowcast/flowcast.py`
  - `models/flowcast/examples/examples.json` + forcing/obs CSVs
  - `models/flowcast/ratchet.yaml`
  - `models/flowcast/training/configs/r1b.json`, `r5.json`
  - `models/flowcast/training/results/{export_check,sigma_r1b,twin_r0_p0_summary,twin_r1b_p1_summary,rescore_main_summary_2026-11-18,nowcast_summary}.json`
- **Overall verdict:** **`MINOR HYGIENE`** (released forward pass, Torch↔NumPy parity, static normalization, and published exam lag paths all pass cleanly; two concrete hygiene/reproducibility issues surfaced on second-reviewer cross-check and independently verified).

---

## Checklist Summary

| # | Check | Status | Summary |
|---|---|---|---|
| 1 | Weight & normalization health | **CLEAN** | All weights finite; all 18 static `st_mu`/`st_sd` well-scaled (`sd` in `[0.0096, 0.8979]`); across all 3 shipped example basins, `max \|norm_st\|` is `1.47`, `2.01`, `1.73`; all 29 LSTM input column norms balanced (`1.12–3.69`); LSTM gate biases and all 6 head rows healthy. |
| 2 | PyTorch ↔ NumPy parity | **CLEAN** | Reconstructed `model.FlowNet` from both shipped `.npz` files; max abs diff vs `FlowCast.forward`: `mean=2.67e-7, q=6.99e-7` (ungauged) and `mean=3.94e-7, q=5.97e-7` (nowcast). Matches shipped `export_check.json` (`1.69e-7` rel err on 4 dev basins). |
| 3 | Causal / API guardrail check | **MINOR HYGIENE** | Published exam leads `(1, 3, 7)` are strictly causal (`0.0` same-day shift at `lead=1`), **but** `flowcast.py:148-155, 168-191` silently accepts `lead=0`, leaking same-day observed flow (`0.654 mm/d` shift from `+10 mm/d` perturbation). |
| 4 | Train-vs-real feature alignment | **CLEAN** | Exact match across `simbank.py`, `bankdata.py`, `evalreal.py`, and `flowcast.py` for dynamic feature scaling (`dyn_features`), static vector construction, DOY formula `(doy[a] - 1 + arange(L)) % 365 + 1`, 365-day warm-up blocks, and nowcast lag channels. |
| 5 | Loss, gradient & seed reproducibility | **MINOR HYGIENE** | NSE* mean loss + monotone cumulative-softplus pinball quantiles (`model.py:27-43`), gradient clipping `1.0`, cosine LR schedule (`2e-3 -> 1e-4`), and rung `r0b` fixed `target_std` loss bias. **However**, `train.py:52-79` seeds NumPy only (`rng`) and never calls `torch.manual_seed`, and resume does not save/restore RNG state. |
| 6 | Twin-test triage | **CLEAN** | `twin_r0_p0_summary.json` -> `twin_r1b_p1_summary.json` honestly documents the `r0 -> r0b/r1b` improvement (`ctwin NSE 0.520 -> 0.595`, `real dev NSE 0.390 -> 0.425`, `flash_ratio 0.54 -> 0.79`). |
| 7 | Uncertainty calibration | **CLEAN** | `calib_sigma.py` fits widen-only `k >= 1.0` on 52 dev basins (`k50 = 1.00`, `k90 = 1.08`). The `0.863 -> 0.953` jump at `k90 = 1.08` is expected because clamping `q05` at `0.0` catches near-zero low-flow days. |
| 8 | Undocumented limits | **DOCUMENTED** | Single-seed, single-layer 64-unit LSTM (`24,710` params, no weight decay, no ensemble, `3,000` steps, last-step checkpoint rather than best-on-sim-val). All explicitly stated in `NOTEBOOK.md` and `ratchet.yaml` as single-core CPU budget choices. |

---

## Detailed Findings

### Finding 1 (`MINOR HYGIENE`): Training seed does not reproduce PyTorch init, dropout, or resumed runs

- **What failed?** `models/flowcast/training/src/train.py:58,74-79` initializes `FlowNet` before seeding PyTorch (no `torch.manual_seed` call anywhere in `train.py`), seeds only NumPy's batch sampler (`rng = np.random.default_rng(cfg.get('seed', 0))`), and on resume constructs a fresh NumPy RNG (`default_rng([seed, step])`) without saving or restoring either NumPy or PyTorch generator state.
- **What changed when we tested it?** Building the same configuration twice with the same `seed` produces different initial PyTorch weights (and non-reproducible dropout masks during training).
- **What consequence remains unmeasured?** Whether any historical close rung decision around the `0.01` NSE promotion margin would flip under a re-run with a different random seed has not been measured (note `ratchet.yaml:12-22` already flags that the `0.01` margin sits inside the 800-basin paired bootstrap noise).

### Finding 2 (`MINOR HYGIENE`): Nowcast API silently permits `lead=0` (same-day truth leakage)

- **What failed?** `models/flowcast/flowcast.py:148-155, 168-191` does not validate `lead >= 1`. When `lead=0`, `lag[0:] = qw[:L]` feeds the target day's own observed streamflow directly into the LSTM input channel.
- **What changed when we tested it?** On shipped example basin `camels_01137500`, perturbing a single day's observed flow `Q_obs[t]` by `+10 mm/d`:
  - At `lead=0`: `Q_pred[t]` shifted by **`0.6541 mm/d`** on the exact same day `t`.
  - At `lead=1`: `Q_pred[t]` shifted by **`0.0000e+00`** on day `t`, and first shifted on day `t+1` by **`0.5522 mm/d`** (confirming published leads `1, 3, 7` in `rescore_main.py:33` are strictly causal).
- **What consequence remains unmeasured?** Published nowcast exam scores are unaffected (`rescore_main.py` explicitly passes `(1, 3, 7)`), so the risk is strictly to future callers or scripts accidentally passing `lead=0`.

---

## Clean Checks (Verified Measurements)

### Static Input Normalization & First-Layer Norms (`flowcast_weights.npz`)
- `st_sd` range across all 18 static features: `min = 0.0096` (`hp_freq`), `max = 0.8979` (`elev_range_km`). Zero outlier-poisoned channels.
- Shipped example basins (`models/flowcast/examples/examples.json`, list manifest):
  - `camels_01137500`: `max |norm_st| = 1.47` (`log_ksat_cmh`)
  - `camels_07315700`: `max |norm_st| = 2.01` (`forest_frac`)
  - `camels_12095000`: `max |norm_st| = 1.73` (`log_aridity`)
- `w_ih` column norms: dynamic channels `1.72 .. 3.69` (highest on `P/10` = `3.69`), static channels `1.12 .. 2.80` (highest on `log_area` = `2.80`).
- LSTM gate bias means `(i, f, g, o)`: `[-0.024, +0.139, -0.034, +0.052]`; head row norms `[1.902, 1.769, 1.596, 1.425, 1.441, 2.019]`.

### PyTorch ↔ NumPy Exact Parity (4 × 60 random sequences)
- `flowcast_weights.npz` (`n_dyn=11, n_st=18, H=64`): `max |mean_np - mean_torch| = 2.666e-7`, `max |q_np - q_torch| = 6.994e-7`.
- `flowcast_nowcast_weights.npz` (`n_dyn=14, n_st=18, H=64`): `max |mean_np - mean_torch| = 3.943e-7`, `max |q_np - q_torch| = 5.969e-7`.
