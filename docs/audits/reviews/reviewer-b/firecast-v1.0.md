# FireCast v1.0 | How much of the 10 km scoring area is visible?

**First pass, reviewer B. Verified evaluation limit on two of three packaged fire-days.** Audited released rung r2 at `61c70bf` from audit checkout `4820cf3`. For a forecast issued at the afternoon overpass, the current fire perimeter is known. Its eventual perimeter and the following day's observed weather become available later.

## The scored area depends on the archived grid

The score selects cells within 10 km of the issue-time burned area. The archive constructs the underlying grid from the **final** fire perimeter plus padding. The real-day builder bounds its issue-time crop by that grid. To measure the intersection, the probe adds 112 empty 90 m cells on each side of the shipped issue mask, recomputes its distance window, and counts candidate cells. The calculation uses only the issue mask and estimates geometric support. It assigns no burn outcomes to the added cells. [Grid provenance](../../../../models/firecast/examples/README.md) · [Real-day crop](../../../../models/firecast/training/src/realdays.py#L77) · [Scoring rule](../../../../models/firecast/firecast.py#L345)

| Fire-day | Unburned candidates scored | Candidates in nominal 10 km window | Outside supplied grid |
|---|---:|---:|---:|
| Hwy 113 | 22,853 | 41,375 | 44.8% |
| Coon | 20,228 | 49,375 | 59.0% |
| Pole Creek | 162,129 | 162,129 | 0.0% |

All three archived AP scores reproduced within `8e-8`, and all three F1 scores matched. The reported metrics describe the available area on growth days. These counts establish neither the sign nor the size of a change in AP or F1 on a prospectively drawn grid. They also leave open whether the network itself exploits the boundary. The decisive comparison would use one issue-defined grid per fire-day, source the inputs and outcomes on that grid, then score the model and baselines together.

## Weather at forecast issue

The published exam uses daily gridMET observations for the issue day and following day. Holding the issue-day inputs fixed and copying their eight weather features into day 1 gives a reproducible sensitivity check. The changed winds also flow through the directional spatial features. This comparison uses a persistence assumption and does not estimate skill with archived issue-time weather forecasts. [Weather contract](../../../../models/firecast/MODEL_CARD.md#L10) · [Feature construction](../../../../models/firecast/firecast.py#L158)

| Fire-day | AP with archived weather → repeated day 0 | F1 with archived weather → repeated day 0 | Candidate cells crossing the 0.10 threshold |
|---|---:|---:|---:|
| Hwy 113 | 0.2963 → 0.2838 | 0.1852 → 0.1994 | 135 |
| Coon | 0.0556 → 0.0192 | 0 → 0 | 0 |
| Pole Creek | 0.5752 → 0.5764 | 0.4297 → 0.4354 | 219 |

Complementing the entire next-day truth mask and keeping all predictive inputs fixed changed **zero probabilities** on all three days.

## Other checks

- All exported arrays were finite. Running the training U-Net graph with shipped, BatchNorm-folded convolutions on a 64 × 64 feature crop from each fire reproduced NumPy logits within `1.91e-6`. This covers FiLM, skips, pooling, decoder and head arithmetic. Independent verification of the original BatchNorm statistics and export fold requires the pre-export checkpoint.
- The dev calibration file matches the shipped temperature, bias and threshold. Its 0.7–1.0 bin contains 1,489 sampled cells, mean predicted probability 0.795 and observed fraction 0.611. Residual overconfidence is documented. The trainer uses masked weighted binary cross-entropy, seeds both generators, and saves the NumPy sampling state. This pass did not refit calibration or retrain. [Dev calibration](../../../../models/firecast/training/results/cal_r2_F.json) · [Trainer](../../../../models/firecast/training/src/train.py#L28)
- On frozen p1, candidate AP gains were +0.00919, +0.01368 and +0.00142. All passed the real-dev gate. Only r2 cleared the required +0.01 simulation gain. The r0 comparison used its p1 rescore. The archived paired prior-predictive twin comparison covers 126 dev fire-days. It supplies no measurement for omitted exterior cells. [Ratchet log](../../../../models/firecast/training/ratchet.json) · [Twin comparison](../../../../models/firecast/training/results/twin_compare_p0_p1.json)

Run from the repository root with NumPy, SciPy and CPU PyTorch.

```sh
OPENBLAS_NUM_THREADS=1 python -B docs/audits/reviews/reviewer-b/firecast_v1_0_probe.py
```
