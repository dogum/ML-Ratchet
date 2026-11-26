# QuakeNet v1 (`q7_zrt`) | Clean single-station rotational physics, missing empty/noise/NaN API guardrails

**First pass (`reviewer-a`). Verdict: MINOR HYGIENE.** Audited against model base `61c70bf` across `quakenet_weights.npz`, [`quakenet.py`](../../../../models/quakenet/quakenet.py), [`pipeline.py`](../../../../models/quakenet/pipeline.py), all 4 shipped single-station exam records, [`training/train.py`](../../../../models/quakenet/training/train.py), and [`ratchet.yaml`](../../../../models/quakenet/ratchet.yaml) following [`SKILL.md`](../../SKILL.md). (Note: multi-station `QuakeNet v2` has a separate reconciled report in [quakenet-v2.md](../../quakenet-v2.md).)

## 1. Intended-Use Contract & Information Boundary
- **Intended use:** From 41 minutes (`−60 s` to `+2,400 s` around a P-wave pick) of 3-component (`Z, N, E`) 1 sps ground velocity at a single seismic station, estimate earthquake moment magnitude (`Mw`), epicentral distance (`25–95°`), depth (`km`), and back-azimuth (`baz`) with 68% error bars.
- **Information boundary:** Unlike `QuakeNet v2`'s multi-station timestamp triangulation shortcut, `QuakeNet v1` takes a single station's waveform and a single `p_guess` scalar (no station coordinates or multi-station arrival differences); `p_guess` enters only as the center of a `±30 s` STA/LTA window search.

## 2. Findings

### Finding A — Missing empty-input, pure-noise, and single-NaN API guardrails [`MINOR HYGIENE`]
- **What failed?** [`pipeline.py:16-28`](../../../../models/quakenet/pipeline.py#L16) applies `sosfiltfilt`, STA/LTA picking, and `normalize()` (`max(std, 1e-12)`) with no check for finite samples, zero amplitude, or minimum STA/LTA trigger SNR.
- **What changed when tested?** On `us10008hu2_IU_PAB.npz`:
  - **All-zero trace:** silently returned physically impossible **`Mw = −9.84 ± 13.26`** and **`dist = −127.8°`**, `baz = 354.7°`.
  - **Pure Gaussian white noise (`1e-6` scale):** silently returned a confident phantom earthquake: **`Mw = 6.22 ± 0.26`**, **`dist = 80.2°`**, **`depth = 163 km`**, `baz = 136.7°`.
  - **Single `NaN` sample (`X[0, 100] = np.nan`):** poisoned `sosfiltfilt` and silently returned `NaN` for `Mw`, `dist`, `depth`, and `baz` with no exception.
- **What consequence remains unmeasured?** Exam waveforms are pre-selected triggered earthquakes without NaNs, so impact is confined to unscreened public API / continuous-stream callers.

### Finding B — Folded Torch↔NumPy raw parity slightly above `1e-5` on real waveforms (`3.05e-5`) and missing raw `ratchet.json` [`MINOR HYGIENE`]
- **What failed?** Reconstructing the 7-block folded 1-D ResNet in PyTorch (float32) gave max raw output diff **`3.05e-5`** on the 4 real examples (`4.20e-5` even with exact `scipy.special.erf` replacing Abramowitz-Stegun `_gelu`; `3.04e-6` on random inputs)—float32 convolution accumulation across 7 deep residual blocks. Also, `training/` does not archive the raw `ratchet.json` file (only the normalized summary ladder in `ratchet.yaml` is committed).
- **What consequence remains unmeasured?** `3.05e-5` in standardized head units is negligible (`< 0.001°` / `< 0.0001 Mw`), while raw per-rung dev-gate numbers cannot be independently re-verified without the external work folder's `ratchet.json`.

## 3. Clean Checks (Items 2, 5, 7, 10)
- **Weights, dead filters & calibration (CLEAN):** All 48 weight arrays are finite with **`0` dead output filters/rows** (`min_norm = 0.0676`). All 4 `sigma_factor` values are strictly widen-only (`mw = 1.000`, `dist = 1.640`, `logz = 2.416`, `baz = 1.140`).
- **Physical rotational equivariance & waveform reliance (CLEAN):** Rotating horizontal components 90° clockwise `(N, E) → (−E, N)` shifted predicted back-azimuth by **`+90°`** within **`−3.8°` to `+1.3°`** across all 4 shipped earthquakes (`0.0°` residual on `II.KAPI`), confirming genuine 3-component P/surface-wave polarization reading. Shifting `p_guess` by `+60 s` (outside the `±30 s` STA/LTA search window) degraded `Mw` by only `0.00–0.11` and distance by `1.0–5.5°`.
- **Seeding & resume (CLEAN):** [`training/train.py:27,34`](../../../../models/quakenet/training/train.py#L27) seeds both PyTorch and NumPy and persists/restores `rng.bit_generator.state` on resume.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/quakenet_v1_probe.py
```

Model snapshot `61c70bf`, audit checkout `69873b7`. Uses NumPy, SciPy, and CPU PyTorch. [Reproduction probe](quakenet_v1_probe.py)
