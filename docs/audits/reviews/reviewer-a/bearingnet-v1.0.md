# BearingNet v1.0 (`r2`) | Single-NaN silent 'Healthy' bug, dead-sensor false OR alarms, and healthy-class miscalibration

**First pass (`reviewer-a`). Verdict: CRITICAL DEFECT (public-API NaN safety) & MINOR HYGIENE.** Audited against model base `61c70bf` across `bearingnet_weights.npz`, [`bearingnet.py`](../../../../models/bearingnet/bearingnet.py), [`pipeline.py`](../../../../models/bearingnet/pipeline.py), all 14 shipped CWRU/Paderborn exam examples, trainer seeding, and [`training/ratchet.json`](../../../../models/bearingnet/training/ratchet.json) following [`SKILL.md`](../../SKILL.md).

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** Classify a rolling-element bearing from a few seconds of 1-D accelerometer vibration, bearing geometry, and nominal shaft RPM (`±5%`) into Healthy (`N`), Outer-Race (`OR`), Inner-Race (`IR`), or Ball (`B`), while refining shaft speed and estimating defect size.
- **Information boundary:** Trained exclusively on simulated fault impulses injected into real healthy-machine backgrounds (`CWRU 98` + `PB K001–K003`); real dev recordings calibrated physics v2 and post-hoc `calib` (`speed = 2.578`, `size = 2.079`, `temp = 1.5`, `nbias = 0.5`).

## 2. Findings

### Finding A — A single `NaN` sample poisons the FFT and silently reports `condition = 'N'` (Healthy) on a real bearing fault [`CRITICAL DEFECT`]
- **What failed?** [`pipeline.py`](../../../../models/bearingnet/pipeline.py) runs `np.fft.rfft(x)` with no `NaN` validation or sanitization. A single `NaN` propagates through every envelope strip and network logit, producing `p = [nan, nan, nan, nan]`. Then [`bearingnet.py:70`](../../../../models/bearingnet/bearingnet.py#L70) selects `m['classes'][int(p.argmax())]`: because `np.argmax` on all-`NaN` returns index `0`, **any recording with even one dropped/NaN sample silently outputs `condition = 'N'` (Healthy)** alongside `NaN` probabilities and `NaN` RPM.
- **What changed when tested?** On packaged outer-race fault `03_CWRU_135_OR.npz` (`60,885` samples, base prediction `OR` at `p = 0.846`, `1,802 rpm`):
  - Setting **just 1 sample (`x[100] = np.nan`)** flipped the returned diagnosis to **`condition = 'N'` (Healthy)** with `p = {'N': nan, 'OR': nan, 'IR': nan, 'B': nan}` and `rpm = nan`—with zero warning or exception.
  - Passing an **all-`NaN`** array produced the exact same silent `'N'` diagnosis.
- **What consequence remains unmeasured?** Clean benchmark recordings in CWRU/Paderborn have no `NaN`s, so this defect affects real industrial DAQ streams where a single dropped sample silently masks a fault as healthy.

### Finding B — Disconnected/zero accelerometer and Gaussian white noise silently trigger Outer-Race (`OR`) fault alarms, and 16-sample fragments predict `'N'` [`MINOR HYGIENE`]
- **What failed?** `BearingNet.read()` performs no minimum-amplitude, zero-variance, or minimum-duration check (`pad_short` mirror-pads arbitrarily short arrays up to `NWIN = 32,768`).
- **What changed when tested?** Across five tested exam bearings (`01_CWRU_100_N`, `03_CWRU_135_OR`, `05_CWRU_107_IR`, `07_CWRU_3006_B`, `11_PB_KA04_OR`):
  - **All-zero signal (dead sensor):** silently returned **`condition = 'OR'` (`p = 0.60`)** on **5/5** bearings.
  - **Pure Gaussian white noise:** silently returned **`'OR'` (`p = 0.44`)** on all 4 CWRU bearings and **`'B'` (`p = 0.37`)** on Paderborn.
  - **16-sample fragment (`1.3 ms`):** mirror-padded and silently returned **`'N'` (`p = 0.805`)**.
- **What consequence remains unmeasured?** Impact is restricted to unscreened operational feeds lacking sensor-health checks.

### Finding C — All `4/4` shipped healthy exam recordings are misclassified as `OR` (`5/14` demo accuracy) and survive phase-scrambling [`UNDOCUMENTED / KNOWN MODEL LIMIT`]
- **What failed?** On the 14 packaged exam examples in `demo.py`, accuracy is **`5 / 14`**, with **all four healthy (`N`) recordings misclassified as `OR`** (`01_CWRU_100_N: 0.804`, `02_CWRU_97_N: 0.462`, `09_PB_K005_N: 0.900`, `10_PB_K005_N: 0.734`) despite post-hoc `nbias = +0.5`.
- **What changed when tested?** Phase-scrambling the FFT (preserving power spectrum while destroying periodic fault-impulse envelope coherence) still returned **`OR` (`0.83`)** on `01_CWRU_100_N` and **`OR` (`0.91`)** on `11_PB_KA04_OR`, whereas shifting `rpm_nominal` by `+30%` flipped `03_CWRU_135_OR` from `OR (0.85)` to `N (0.56)` and pulled predicted RPM to `2,252` (confirming the documented `±5%` nominal-speed window).
- **What consequence remains unmeasured?** Full-exam confusion matrices across all CWRU/Paderborn bearings are documented in `MODEL_CARD.md`; whether phase-scrambled false-OR triggers correlate with specific spectral-kurtosis bands remains unmeasured.

## 3. Clean Checks (Items 2, 3, 7, 8, 10)
- **Weights, dead filters & calibration (CLEAN):** All arrays in `bearingnet_weights.npz` are finite with **`0` dead output filters/rows** (`min_norm = 0.3652`); all scale calibrations are strictly widen-only (`speed = 2.578`, `size = 2.079`, `temp = 1.5`).
- **Folded PyTorch ↔ NumPy export parity (CLEAN):** Reconstructing the folded 4-block 1-D ResNet (`stem` stride 2) + 3-layer MLP in PyTorch matched `BearingNet.forward` within **`2.38e-6`** on all 14 real examples and **`2.68e-6`** on 8 random inputs.
- **Seeding, resume & ratchet replay (CLEAN):** [`training/bn_train.py:24,31`](../../../../models/bearingnet/training/bn_train.py#L24) seeds both PyTorch and NumPy and persists/restores `rng.bit_generator.state` on resume. All **`4/4`** logged rungs (`r1..r4`) in `training/ratchet.json` match their documented promotion rule.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/bearingnet_v1_0_probe.py
```

Model snapshot `61c70bf`, audit checkout `759c345`. Uses NumPy, SciPy, and CPU PyTorch. [Reproduction probe](bearingnet_v1_0_probe.py)
