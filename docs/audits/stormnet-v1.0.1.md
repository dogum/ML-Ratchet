# StormNet v1.0.1 | Causal forecast windows, calibration boundary labels, and solar-wind counterfactuals

**Cross-checked clean causal pipeline with minor calibration-label mismatch and documented freshness sensitivity.** Reviewer A independently reran reviewer B's `db285c4` probe against model base `61c70bf`, tested all-NaN / missing-input guardrails on the public `StormNet.predict` API, and ran solar-wind vs. `Dst`-persistence counterfactual ablations across all 721 matched gallery issues. [Original report and probe](reviews/reviewer-b/stormnet-v1.0.1.md) · [Exam window slicer](../../models/stormnet/training/split.py#L45) · [Public reader](../../models/stormnet/stormnet.py#L74)

**What passed and what failed.**
- **Causal windowing & gap handling (CLEAN):** Passing all six gallery series through the exam slicer with a 2-hour `Bz` gap and perturbing strictly post-cutoff `Bz` (`+20 nT`) and `Dst` (`+30 nT`) left issued medians and half-widths identical (`0.0` shift) while moving the verification target by exactly `+30 nT`. A 3-hour gap cleanly excluded the issue on all six series.
- **Calibration edge-label mismatch (MINOR HYGIENE):** Fitting ([`calib_sigma.py:33`](../../models/stormnet/training/calib_sigma.py#L33)), Python inference ([`stormnet.py:74`](../../models/stormnet/stormnet.py#L74)), and browser inference consistently place `Dst == -50 nT` in the middle bin (`1.2317`) and `Dst == -30 nT` in the upper bin (`1.0915`), whereas [`calibration_sigma.json`](../../models/stormnet/training/calibration_sigma.json) prose describes inclusive lower bounds (`1.5752` and `1.2317`). Across `6` and `5` boundary gallery issues, prose-implied factors shift half-widths by up to **`2.43 nT`** with zero point-forecast or coverage changes.
- **Freshness sensitivity (DOCUMENTED OPERATIONAL LIMIT):** Substituting one-hour-stale `Dst` across all 721 matched gallery issues raised MAE from **`8.97 -> 11.21 nT`** overall, **`15.54 -> 19.37 nT`** for storms (`<= -50 nT`, `n=307`), and **`39.12 -> 46.85 nT`** for extremes (`<= -150 nT`, `n=67`), with max single-issue shift **`96.79 nT`**.
- **Weights, parity, splits, and ratchet (CLEAN):** All arrays finite, `0` dead units across all 48 hidden random-feature blocks, gallery normalized feature `std` in `1.20–2.01` (`70` clipped values), exact numerical reproduction from exported ridge blocks, disjoint year splits, and all `2/2` original promotions + `5/5` release-ladder entries replay cleanly.

**Second-pass missing-input guardrails & solar-wind counterfactual ablations.**
- **Missing-input API guardrails (CLEAN):** `StormNet.predict` properly rejects all-NaN inputs, all-NaN wind with valid `Dst = -100 nT`, and quiet wind with `NaN` `Dst`, raising `ValueError: Supply finite observations with positive speed and dynamic pressure` in all three cases (no silent hallucinated forecast).
- **Solar-wind and persistence comparison (721 issues).** Replacing all 36 hours of observed wind with fixed inputs (Bz 0 nT, speed 400 km/s, dynamic pressure 5 nPa) increases MAE in all three groups. The fixed-input run does worse than persistence overall and on storms. It does better than persistence on extremes, 45.38 versus 46.46 nT. The full model has the lowest MAE in each group. Zeroing current Dst also increases MAE in every group.

| Stratum (observed 3-h min `Dst`) | Issues | Full StormNet MAE | `Dst` persistence MAE | Quiet solar wind (`Bz=0, v=400, pressure=5`) MAE | Zero `Dst` (`Dst=0`) MAE |
|---|---:|---:|---:|---:|---:|
| All matched gallery issues | `721` | **`8.97 nT`** | `10.60 nT` | `11.18 nT` | `56.20 nT` |
| Storms (`<= -50 nT`) | `307` | **`15.54 nT`** | `18.46 nT` | `18.79 nT` | `98.82 nT` |
| Extremes (`<= -150 nT`) | `67` | **`39.12 nT`** | `46.46 nT` | `45.38 nT` | `189.23 nT` |

**What remains open.** Archived OMNI files omit real-time publication timestamps and provisional-to-final `Dst` revisions, so live operational latency and revision error remain unmeasured beyond the fixed one-hour stress test.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/stormnet_v1_0_1_probe.py
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_db285c4_probe.py stormnet
```

Uses NumPy and PyYAML. [Second-pass probe](reviews/reviewer-a/peer_db285c4_probe.py)
