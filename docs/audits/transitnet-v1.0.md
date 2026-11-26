# TransitNet v1.0 | Missing transit coverage, partial-coverage SNR cliff, and constant-flux NaN guardrail

**Cross-checked input guardrails, with positive evidence that classifications depend on measured transit photometry.** Reviewer A independently reran reviewer B's `69873b7` probes against model base `61c70bf`, swept partial in-transit excision fractions across all three packaged TESS candidates to pinpoint when finite posteriors decouple from usable SNR, and tested all-NaN flux guardrails. [Original report and probe](reviews/reviewer-b/transitnet-v1.0.md) · [Reader](../../models/transitnet/transitnet.py#L76) · [Scalar construction](../../models/transitnet/pipeline.py#L114)

**What failed and reproduced.**
- **Unobserved transit labelled `"nothing"` with `snr = -inf`:** Excising `|phase| <= duration` around predicted transits preserved metadata and out-of-transit baselines while making `"nothing"` the top class on all 3 candidates (**`0.9913`** Pi Men c, **`0.9268`** TOI-270 c, **`0.8128`** TOI-617) alongside **`snr = -inf`**. Permuting flux within sectors also moved all 3 to `"nothing"` (`0.7305–0.9828`), confirming genuine reliance on phase-folded transit shape.
- **Constant positive flux & reversed-row guardrails:** Unit constant flux with no centroids produced zero binned noise → `-inf` log-noise scalar → **`NaN`** in all 4 class probabilities and radius ratio `k`. Reversing rows inside a sector raised a negative-bin `ValueError`, whereas reversing sector order, shifting `epoch` by `+P`/`+2P`, or omitting centroids matched original outputs (`<= 1e-5`).
- **Controls:** `0/1,388` near-zero rows; folded PyTorch↔NumPy parity **`3.34e-6`** (real) / **`1.91e-6`** (random); all `4/4` logged network decisions matched; TIC-grouped split excludes shared dev TICs.

**Second-pass partial in-transit coverage sweep & all-NaN guardrail.**
Excising `|phase| < frac * duration` around transit centre on all 3 packaged candidates brackets the transition from valid vetting to unflagged missing-data `"nothing"` calls (while `all_nan_flux` cleanly raises `ValueError`):

| Excision `abs(phase) < frac × dur` | Pi Men c (`top` / `P(nothing)` / `SNR`) | TOI-270 c (`top` / `P(nothing)` / `SNR`) | TOI-617 (`top` / `P(nothing)` / `SNR`) |
|---:|---|---|---|
| `0.00` (full) | `planet` (`0.9999`) / `0.0000` / `58.94` | `planet` (`0.9999`) / `0.0000` / `53.98` | `blend` (`0.9786`) / `0.0000` / `105.02` |
| `0.25` (half core left) | `planet` (`0.9990`) / `0.0001` / `35.58` | `planet` (`0.9993`) / `0.0000` / `29.99` | `blend` (`0.5906`) / `0.0001` / `45.41` |
| `0.50` (core gone, shoulders left) | **`nothing` (`0.9539`)** / `0.9539` / `2.50` | **`nothing` (`0.5705`)** / `0.5705` / `1.43` | **`nothing` (`0.6189`)** / `0.6189` / `0.50` |
| `0.75` | `nothing` (`0.9898`) / `0.9898` / `−0.84` | `nothing` (`0.7700`) / `0.7700` / `0.12` | `nothing` (`0.6865`) / `0.6865` / `1.65` |
| `0.90` | `nothing` (`0.9823`) / `0.9823` / **`−inf`** | `nothing` (`0.8912`) / `0.8912` / **`−inf`** | `nothing` (`0.5629`) / `0.5629` / **`−inf`** |
| `1.00` | `nothing` (`0.9913`) / `0.9913` / **`−inf`** | `nothing` (`0.9268`) / `0.9268` / **`−inf`** | `nothing` (`0.8128`) / `0.8128` / **`−inf`** |

Notice the two-stage ambiguity: once the transit core (`|ph| < 0.50 dur`) is missing, the model already flips to `"nothing"` with finite low SNR (`0.50–2.50`), and by `0.90 dur` SNR drops to `-inf` while high-confidence `"nothing"` posteriors persist without an abstention flag.

**What remains open.** Prevalence of missing-core or constant-flux light curves in the full TESS exam roster and the missing raw polish-trial log remain outside the repository.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/transitnet_v1_0_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave7_export_probe.py transitnet
OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_69873b7_probe.py transitnet
```

Uses NumPy and CPU PyTorch. [Second-pass probe](reviews/reviewer-a/peer_69873b7_probe.py)
