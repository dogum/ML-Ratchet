# GridPulse v1.0 | Live-cutoff interpolation leakage and prefix requirement

**Cross-checked critical live-inference defect.** Reviewer A independently reran reviewer B's `dc30c02` probe against model base `61c70bf`, verified the exam evaluator's interpolation call path, and tested live prefix length and pre-event frequency normalization. [Original report and probe](reviews/reviewer-b/gridpulse-v1.0.md) · [Release API](../../models/gridpulse/gridpulse.py#L25) · [GB evaluator](../../models/gridpulse/training/evaluate.py#L55)

**What failed and reproduced.** `window(f, i0)` interpolates missing readings over the entire `[-20, +60] s` window before `_features` masks samples beyond the requested horizon `N`. On `examples/trace_example.csv` with fixed onset at second 80, removing `+9 s` and `+10 s` and raising only `+11 s` by `0.050 Hz` moved the `+10 s` loss estimate from **1,245.82 to 1,346.42 MW** (`+100.6 MW`), inertia from **175.78 to 183.54 GW s**, and settling forecast from **−0.1707 to −0.1309 Hz**; the complete-prefix control moved zero outputs. In addition, [`window`](../../models/gridpulse/gridpulse.py#L28) enforces `i0 + 60 < len(f)` regardless of `N`, raising `ValueError: need >= 60 s before and 60 s after the onset at 1 s resolution` when passed a true live `+10 s` prefix (`len = 91, onset_index = 80`).

**Exam path & independent checks.** Both weight archives (`gridpulse_weights.npz`, `ercot_head.npz`) are finite with zero dead rows and well-conditioned scales (`xs ∈ [0.1174, 2.1090]` waveform; `[0.2008, 0.6797]` ERCOT heads). Waveform and ERCOT-head Torch↔NumPy parity reproduced within `8.3e-7` and `2.2e-7`, zeroing the waveform degrades MAE across all six shipped examples, and all logged `p2s` / `p2r` rung decisions reproduced. In `read()`, discarding `window()`'s measured `f0` (`49.9937 Hz`) in favor of nominal `50.0 Hz` shifts raw outputs by at most `2.98e-4` (matching `training/evaluate.py:88`).

**What remains open (handoff check).** [`training/exam.py:17`](../../models/gridpulse/training/exam.py#L17) calls [`evaluate.py:55-62`](../../models/gridpulse/training/evaluate.py#L55-L62), which runs `pd.Series(x[0]).interpolate(limit_direction='both')` across the full trace before windowing and horizon masking. However, the raw NESO 1 s `griddata` archive is not stored in this repository (`examples/gb_exam_examples.npz` ships six pre-windowed, NaN-free 81 s rows and `examples/trace_example.csv` has zero NaNs). Counting cutoff-straddling gaps across the 412 GB exam events and replaying them with prefix-only causal imputation therefore requires running against the external `griddata` archive.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/gridpulse_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_dc30c02_probe.py gridpulse
```

The second-pass probe uses NumPy, SciPy, pandas, and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_dc30c02_probe.py)
