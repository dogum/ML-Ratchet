# SnowSense v1.0 | Bidirectional gap interpolation leaks future temperature, unseeded Torch init

**Reviewer A, first pass.** Model base `61c70bf`, audit checkout `d903cb6`, released rung `R6`. **Verdict: CRITICAL DEFECT (causal gap-interpolation leakage in real-time nowcasting) + MINOR HYGIENE (unseeded PyTorch init).** SnowSense v1.0 is a 2-layer causal GRU (`50,949` parameters) that maps a station's daily precipitation and temperature to daily SWE quantiles. When temperatures are complete, the forward GRU is strictly causal (`0.0e+00` shift from future perturbations) and matches PyTorch `SnowNet` within `2.52e-6`. However, **missing temperatures are filled by bidirectional linear interpolation across the entire record** in inference, exam evaluation, and simulated training alike—leaking post-gap temperatures into earlier SWE estimates whenever a temperature gap straddles or precedes the current day.

## 1. Bidirectional temperature interpolation leaks future weather across gaps

- **What failed?** [`snowsense.py:61-62`](../../../../models/snowsense/snowsense.py#L61-L62) (`clean`) and [`snowsense.py:122-124`](../../../../models/snowsense/snowsense.py#L122-L124) (`windows`) run `interpolate(limit_direction='both')` on `T_mean`, `T_min`, and `T_max` across the whole DataFrame before feeding the causal GRU. The exact same bidirectional fill runs in the exam loader ([`training/common.py:53-54,95-97`](../../../../models/snowsense/training/common.py#L53-L54)) and in simulated training gap injection ([`training/simbank.py:131-134`](../../../../models/snowsense/training/simbank.py#L131-L134)). This matches a retrospective reanalysis workflow (full completed water-year archive), but violates operational causality for the stated daily question (*"How much water is stored as snow at this mountain station today?"*) whenever a thermometer gap is open or straddles the forecast date.
- **What changed when tested?** Across all three packaged SNOTEL exam stations in [`models/snowsense/examples/`](../../../../models/snowsense/examples):
  - **Complete-prefix control** (perturb `P +50 mm` and `T +15 °C` strictly after `2022-03-01`): **`0.0e+00` mm** change on `<= 2022-03-01` at all three stations.
  - **Synthetic 2-day `T` gap straddling `2022-03-01`** (`Feb 28–Mar 1` set `NaN`, `+15 °C` applied **only** from `2022-03-02` onward):
    - `584_OR_SNTL`: Mar 1 SWE moved **`410.36 -> 340.76 mm` (`-69.60 mm`)**.
    - `497_WY_SNTL`: Mar 1 SWE moved **`251.87 -> 247.83 mm` (`-4.04 mm`)**.
    - `349_MT_SNTL`: Mar 1 SWE moved **`193.96 -> 173.28 mm` (`-20.68 mm`)**.
  - **Real pre-existing gap in `497_WY_SNTL` (`2022-01-19`, next observed `T` on `2022-01-21`):** perturbing only `>= 2022-01-21` by `+15 °C` shifted `2022-01-19` SWE from **`226.87 -> 224.73 mm` (`-2.14 mm`)**.
  - **Causal forward-fill (`ffill`) replay on the three packaged exam stations** (preserving the exact `mT` missingness flags):

| Packaged exam station | Real `T` NaN days | Bidirectional interp NSE | Causal `ffill` NSE | Max daily SWE diff |
|---|---:|---:|---:|---:|
| `584_OR_SNTL` | 72 | 0.81991 | 0.81958 | 5.16 mm |
| `497_WY_SNTL` | 10 | 0.98040 | 0.98040 | 1.04 mm |
| `349_MT_SNTL` | 24 | 0.82207 | 0.82203 | 0.25 mm |

- **What consequence remains unmeasured?** On these three 10-year packaged stations (which have short gaps), replacing bidirectional interpolation with causal `ffill` moved NSE by at most `-0.00033`. Across the full 621-station SNOTEL exam (not stored in the repository; rebuilt via `mlratchet data water`), the count of multi-week thermometer outages and the full-exam NSE/peak/melt shift under causal gap filling remain unmeasured.

## 2. Unseeded PyTorch initialization in `training/train.py`

- **What failed?** [`training/train.py:64,100`](../../../../models/snowsense/training/train.py#L64) seeds NumPy batch sampling with `np.random.default_rng(SEED + step)`, **but never calls `torch.manual_seed`**. Initial `SnowNet` linear and GRU weights are therefore drawn from an unseeded Torch RNG state on every fresh run, and resume checkpoints (`train.py:123`) omit RNG state (restarting NumPy at `SEED + step`).
- **What changed when tested?** Verified directly by source inspection of `training/train.py`; re-running full multi-bank training across fresh seeds was outside this read-only pass.
- **What consequence remains unmeasured?** Whether seed-to-seed variance spans the `0.01` NSE promotion margin (`R6` beat `R4` on `val_p4` by `+0.0164`, while `R5` missed by `-0.0016` at `+0.0084`) remains unmeasured without multi-seed retraining.

## Physical ablations, weights, parity, calibration, and ratchet checks

- **Physical ablations:** Setting precipitation `P = 0` collapsed mean annual peak SWE from `682.7 / 426.2 / 239.8 mm` to **`0.8 / 8.3 / 5.1 mm`**, and adding `+15 °C` warming collapsed peaks to **`0.0 / 6.3 / 9.2 mm`**. The small positive residuals (`0.8–8.3 mm` under zero precipitation) reflect the lack of a hard mass-conservation floor (`physics_anchor: false`).
- **Weights & calibration:** All arrays in `snowsense_weights.npz` are finite with **0 dead rows or columns**. Stored widen factors `k = [1.0, 1.0, 1.0]` match [`training/k_r6.json`](../../../../models/snowsense/training/k_r6.json), where raw dev 68% coverages (`0.722` daily, `0.763` peak, `0.876` melt) already exceeded `0.68`, clamping widen-only multipliers at `1.0`.
- **Export parity:** Reconstructing PyTorch `SnowNet` ([`training/model.py:19`](../../../../models/snowsense/training/model.py#L19)) from `snowsense_weights.npz` matched `SnowSense.forward` across all 30 packaged station-years within **`2.52e-6`** in quantile space.
- **Ratchet replay:** All **6 rungs (`R1–R6`)** in [`training/ratchet.json`](../../../../models/snowsense/training/ratchet.json) match the published `val_p4` rule (`sim > inc + 0.01` and `(1 - NSE_dev) <= 1.05 * (1 - NSE_inc)`), rejecting `R5` (`+0.0084`) and keeping `R6` (`+0.0164` over `R4`). Note that `R6` explicitly records being fine-tuned from the rejected `R5` checkpoint with a second `p4` bank before passing against incumbent `R4`.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/snowsense_v1_0_probe.py
```

Requires NumPy, pandas, and CPU PyTorch. [Probe script](snowsense_v1_0_probe.py)
