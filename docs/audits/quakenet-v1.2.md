# QuakeNet v1.2 · Single-station input coverage and event splits

**Cross-checked input guardrail gaps, with positive rotation and split controls.** The single-station `q7_zrt` release is registered as v1.2. Its contract calls for a known or expected P arrival and 40 minutes of post-arrival motion. The public reader accepts a truncated trace, pads the missing tail with zeros and still supplies calibrated-looking magnitudes. On the IU.POHA example (catalogue Mw 7.5), the estimate falls from **7.54 to 5.68 ± 0.14** with data ending 60 seconds after the original arrival guess. The P pick stayed fixed. Model base `61c70bf`, closing review at `abd0b51`. [Model card](../../models/quakenet/MODEL_CARD.md) · [Window and picker](../../models/quakenet/pipeline.py#L16) · [First pass](reviews/reviewer-a/quakenet-v1.md)

| Station | Mw on full trace | Mw with shortened trace | Azimuth residual after 180° horizontal rotation |
|---|---:|---:|---:|
| IU.PAB | 6.27 | 5.506 ± 0.035 | −1.35° |
| IU.POHA | 7.54 | 5.68 ± 0.14 | −1.33° |
| IU.KNTN | 6.09 | 5.524 ± 0.049 | −0.39° |
| II.KAPI | 6.29 | 5.51 ± 0.05 | −2.34° |

**Quiet, missing and rotated inputs.** All four traces kept the same pick under truncation. Reversing both horizontal channels while preserving vertical polarity rotated the inferred back-azimuth by approximately 180° on all four. One NaN on the vertical component at the original pick made every physical point estimate nonfinite on all four. Unit-variance Gaussian noise scaled by 10⁻⁶ m/s produced Mw 6.13–6.37. Feeding the six packaged real quiet-station windows to `read` with an assumed P at sample 60 produced **Mw 5.44–5.67** with reported sigmas of **0.043–0.149**. Four of the six had picker strength above 3. These controls exercise a magnitude estimator conditional on a supplied event. They do not estimate a continuous earthquake detector's false-alarm rate. The exam builder discards nonfinite and constant-component records. [Quiet-window export](../../models/quakenet/training/webdata.py) · [Reader](../../models/quakenet/quakenet.py) · [Closing probe](reviews/reviewer-b/quakenet_v1_closing_probe.py)

**Event split verified from a second archive.** The v2 package preserves the v1.2 event IDs and original station memberships from `real_meta.json`. Replaying the v1 selection rule (40 from sorted distinct event IDs, seed 2026) exactly recovered the archived dev IDs. Dev has **40 events / 74 records**, exam **1,734 events / 3,105 records**, with **zero shared event IDs**. The four packaged examples belong to the exam. Station reuse across events is part of this benchmark. [Original split code](../../models/quakenet/training/realbank.py#L14) · [Archive provenance](../../models/quakenet-v2/training/make_splits.py) · [Frozen IDs](../../models/quakenet-v2/splits.json)

**Promotion and export.** The original raw rung ledger is absent from this checkout, but the shipped lab page embeds the frozen simulation objectives, four dev errors and decisions rounded to three decimals. Applying the published two gates reproduces **7/7 decisions at displayed precision**. Rung 5 improves 0.011 over a strict 0.010 gate, close enough that the last digit matters. The dev ratios on all six comparisons clear 1.05 even at their upper rounding bounds. The training gate uses simulation and dev scores, and the script also computes and prints exam scores per rung. The code cannot establish who consulted the printed results. The folded Torch/NumPy probe reproduced a maximum raw head difference of 3.81 × 10⁻⁵ on the four real examples and 5.25 × 10⁻⁶ on random input here. Widening factors, seeding and saved NumPy sampler state matched the first review. [Gate logic](../../models/quakenet/training/ratchet.py#L10) · [Published ladder](../../models/quakenet/lab/index.html) · [Export-parity probe](reviews/reviewer-a/quakenet_v1_probe.py)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-a/quakenet_v1_probe.py
python3 -B docs/audits/reviews/reviewer-b/quakenet_v1_closing_probe.py
```
