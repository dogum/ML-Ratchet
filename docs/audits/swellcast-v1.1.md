# SwellCast v1.1 | Historical NDBC 99-sentinel parser defect and false swell detection on all-NaN windows

**Cross-checked preprocessing defect and empty-input false-detection guardrail gap.** Reviewer A independently reran reviewer B's `759c345` probes against model base `61c70bf`, audited field-specific sentinel erasure across all three packaged NDBC 51004 (2018) raw files (`swden`, `swdir`, `swr1`), and tested all-NaN / all-zero spectrograms through `SwellCast.read()`. [Original report and probe](reviews/reviewer-b/swellcast-v1.1.md) · [Historical parser](../../models/swellcast/pipeline.py#L46) · [Public reader](../../models/swellcast/swellcast.py#L85)

**What failed and reproduced.**
- **Generic `99` missing-value rule erases valid compass bearings:** [`pipeline.py:46`](../../models/swellcast/pipeline.py#L46) (and [`training/fetch.py:35`](../../models/swellcast/training/fetch.py#L35)) masks `[98.9, 99.01]` as `NaN` on every historical NDBC file regardless of physical field. In packaged buoy `51004d2018.txt.gz` (`swdir`), this erased **`3,445`** valid `99°` bearings (`52` inside Hurricane Olivia's 240-hour window). Restoring `99°` in `swdir` shifted Olivia's estimate from `1,947.71 → 1,951.75 km`, bearing `50.22° → 49.76°` (`−0.46°`), peak time `−37.38 → −37.46 h`, and `p_swell 0.9791 → 0.9774` (detection preserved).
- **Split reconstruction & exam scope:** Confirmed exam arrivals reproduced at **`182 / 288`** (`63.2%`) across `107` exam storms with `0` overlap against the `10` recoverable dev storm IDs (`8` original dev storm IDs without confirmed arrivals are documented as lost in `dev_storms_v1.json`). Rotating directional moments by `90°` preserved detection on all `5` packaged windows with bearing residuals `−3.34°`, `+42.63°`, `+17.86°` on detected cases.
- **Weights, parity, and ratchet:** `0/745` near-zero rows; folded PyTorch↔NumPy max raw-head diffs **`3.05e-5`** (real, matching archived `1.72e-5` range) and **`1.91e-6`** (random); widen-only multipliers `1.00–1.45`; all `5/5` network promotion decisions replayed.

**Second-pass field-specific sentinel audit & all-NaN false-detection guardrail.**
- **All three packaged 2018 NDBC 51004 files (`409,417` cells each):**
  - `51004w2018.txt.gz` (`swden`, energy density): `0` cells in `[98.9, 99.01]`, max `86.270`.
  - `51004d2018.txt.gz` (`swdir`, compass bearing `0–359°`): **`3,445`** valid `99°` cells erased.
  - `51004j2018.txt.gz` (`swr1`, directional coherence stored `1..100` then divided by `100` at [`pipeline.py:55`](../../models/swellcast/pipeline.py#L55)): `0` cells in `[98.9, 99.01]` in this buoy-year because the observed max was `98.0` (`r1 = 0.98`), though the same line would also silently erase any `r1 = 0.99` (highest-coherence swell) cell on buoy-years where it occurs.
- **All-NaN spectrogram triggers a false storm-swell detection (`detected = 1.0`):** Calling `SwellCast.read()` on an all-`NaN` `(3, 80, 34)` window silently returns **`detected = 1.0`** (`p_swell = 0.5746`, `D_km = 2,878.2 km`, `hs = 5.32 m`, `dir = 255.3°`), whereas an all-zero window returns `detected = 0.0` (`p_swell = 0.0009`, `D_km = 2,908.1 km`, `hs = 3.38 m`).

**What remains open.** Re-scoring the full multi-year NDBC exam archive with field-specific sentinel parsing (`swdir`, `swr1`, `swden`), recovering the `8` unrecorded non-confirmed dev storm IDs, and evaluating exposure-adjusted false alarms outside the designated 600 quiet windows remain open.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/swellcast_v1_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave5_export_probe.py swellcast
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_759c345_probe.py swellcast
```

Uses NumPy and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_759c345_probe.py)

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| Generic 99 sentinel erases valid bearings | Yes | **Opt-in fix, re-score queued**: `read_ndbc(..., field_sentinels=True)` keeps 99 in `swdir` and `swr1` (999 stays missing). Default unchanged because the exam (and the demo) were parsed the old way. | Audit: 3,445 bearings in the packaged 51004 2018 file; Olivia −0.46°. |
| Follow-up (27 Nov 2026): re-score with field sentinels | Run | **Dated re-score, then default (v1.1.1).** 422 buoy-years' direction files re-downloaded and re-parsed through `training/fetch.py` (`training/rescore_fs.py`); the old parse reproduced every stored grid. Confirmed exam: distance 29.32 → 29.27%, time 26.82 → 26.86 h, direction 19.23 → 19.31°, detected 63.2% both. No verdict changes. `read_ndbc(field_sentinels=True)` is now the default. | `training/exam_v111_fs.json`; NOTEBOOK 27 Nov 2026. |
| All-NaN window reads as a storm swell | Yes | **Fixed**: `read` raises when the window has no finite energy. | Demo unchanged. |
