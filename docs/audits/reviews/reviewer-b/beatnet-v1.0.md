# BeatNet v1.0 · Subject overlap and missing-sample handling

**First pass · one subject crosses the dev/exam boundary, and a single missing sample can invalidate an entire strip.** These are separate findings. The first affects the interpretation of the inter-patient benchmark. The second affects the public research classifier. Model snapshot `61c70bf`, audit checkout `73ee7e6`.

## The held-out unit

The six recorded dev IDs and 22 DS2 exam IDs have zero intersection by record number. Dev includes **201** and DS2 includes **202**. PhysioNet identifies these as recordings from the same person. Grouping that pair produces **one shared subject**. Record 201 also belongs to the pool from which the training pipeline extracts background ECG residuals for simulated beats. The released 202 example confirms the other recording reached the exam. This establishes a boundary crossing in calibration and background selection. The evidence neither measures how much it helped nor attributes any exam prediction to memorization. A subject-grouped replay would need to hold this person out of every dev-derived input and report results on the same exam population. [PhysioNet database directory](https://physionet.mit.edu/physiobank/database/html/mitdbdir/intro.htm) · [Dev IDs](../../../../models/beatnet/training/calib_v1.json) · [DS2 list](../../../../models/beatnet/training/common.py) · [Training background extraction](../../../../models/beatnet/training/quietnoise.py)

## One missing sample

Reviewer B changed one measured ECG sample to NaN at the midpoint of each of the four packaged strips. The reference beat positions stayed fixed. Full-strip filtering and feature extraction propagated the missing value into all **326/326 scored probability vectors**. Every returned label was `N`, including **35 beats labelled S, V or F** by the reference. Interpolating only that one sample from its neighbours restored **326/326 original labels**. On a constant trace the optional detector produced no peaks and the classifier raised a generic concatenation error. Missing-sample prevalence in the historical exam is unknown. [Public classifier](../../../../models/beatnet/beatnet.py) · [Filtering and feature extraction](../../../../models/beatnet/pipeline.py#L21)

## Reference timing versus automatic detection

The model reads a retrospective ECG strip and uses neighbouring beats. On the same displayed intervals, automatic detections were matched one-to-one to reference beats within 150 ms. Unmatched reference beats count as misses.

| MIT-BIH record | Reference beats | Automatic detections matched | Correct class with reference positions | Correct detection and class with automatic positions |
|---|---:|---:|---:|---:|
| 123 | 53 | 53 | 53 | 53 |
| 202 | 93 | 91 | 77 | 76 |
| 213 | 108 | 107 | 97 | 97 |
| 221 | 72 | 71 | 61 | 60 |
| Total | 326 | 322 | 288 | 286 |

The four displayed intervals had no unmatched automatic detections. The full detector exam separately archives 1,008 missed beats and 44 extra detections against 49,705 annotations. The example table is descriptive of these four strips. [Full exam results](../../../../models/beatnet/training/exam.json)

**Controls.** All 751 shipped kernel rows exceeded the near-zero threshold. The training network reconstructed from the folded weights agreed with NumPy to `3.82e-6` on real inputs and `5.73e-6` on random inputs. All five network-rung decisions matched the frozen-score and dev gates. The release’s QRS calibration applies a 5.06× sigma multiplier. Reseeding, epoch-level replay and recalibration after excluding the shared subject remain open. [Promotion log](../../../../models/beatnet/training/ratchet.json) · [Model card](../../../../models/beatnet/MODEL_CARD.md)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-b/beatnet_v1_0_probe.py
python3 -B docs/audits/reviews/reviewer-b/wave7_export_probe.py beatnet
```
