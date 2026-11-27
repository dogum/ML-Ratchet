# PlantDetective v1.1 · Missing operating hours become starts

**Cross-checked material defect in the annual starts estimate.** The public input contract represents a missing hour by NaN. The feature builder substitutes zero and identifies operating runs from the resulting series. An isolated missing hour inside a continuous run can therefore look like a shutdown followed by a restart. The annualized count enters the R6 regression head directly. Model snapshot `61c70bf`, review at `73ee7e6`. [Missing-hour and run logic](../../models/plantdetective/pdcore.py#L134) · [Count contribution to output](../../models/plantdetective/plantdetective.py#L38) · [First pass](reviews/reviewer-a/plantdetective-v1.1.md)

Reviewer B selected the most complete full calendar year for each packaged unit. All three selected years were 2019 with 100% measured output. On each eligible day, the test removed the first interior operating hour whose immediate neighbours also exceeded 0.5 MW. Timestamps, prices, context and neighbours stayed fixed. The original record supplies the control for what actually happened during each removed hour.

| Unit | Original predicted starts/year | After one missing hour | Hours removed in daily test | Predicted starts/year in daily test |
|---|---:|---:|---:|---:|
| 55086_101 | 85.35 | 86.30 | 99 | 152.36 |
| 6179_3 | 2.76 | 4.32 | 362 | 107.58 |
| 8063_CT2 | 27.11 | 28.13 | 19 | 47.34 |

On 6179_3, hiding 4.13% of the year increased the explicit log-count contribution from 1.099 to 5.939. The learned correction reduced the apparent count but still returned 107.58 starts/year. Estimated capacity factor went from .6995 to .4617. The stress test targets running hours. Exam-average error under naturally occurring outages and the score of a missing-aware replacement remain open.

**Ordering.** The full reversal in the first review reproduced. Moving just day zero to the end within each complete 2019 record had a much smaller effect. The maximum absolute starts change was below 0.001/year. Sorting restored all original estimates tested. The larger missing-hour result is the priority for a repair experiment.

**Promotion and reproducibility.** Reviewer A’s script counted five v1.0 rows but did not recalculate their decisions. The independent probe applied both gates to every row, including the R2 frozen-set change (rescored incumbent 0.265846). All 5 v1.0 and 3 v1.1 decisions match. The original three-member forward check and widen-only checks passed. The training code sets the Torch seed while using NumPy’s global shuffle without a corresponding seed in that routine. The resulting training-score variance remains unmeasured. [v1.0 ledger](../../models/plantdetective/training/ratchet.json) · [v1.1 ledger](../../models/plantdetective/training/ratchet_v1_1.json) · [Trainer](../../models/plantdetective/training/model.py#L55)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/plantdetective_v1_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/peer_73ee7e6_probe.py plantdetective
```
