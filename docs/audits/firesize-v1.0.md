# FireSize v1.0 · Escape probability collapses on slash–blowdown ignitions

**Cross-checked critical defect.** On each of eight packaged exam landscapes, changing only the ignition cell to any of the four valid LANDFIRE slash–blowdown codes (201–204) gave exactly zero probability of a fire reaching 100 acres. All 32 runs gave the same 1.7263-acre median. This is a reproducible loss of discrimination for a fuel class explicitly accepted by the release. The test supplies no observed final sizes for those counterfactual ignitions. Model base `61c70bf`, closing review at `abd0b51`. [Model card](../../models/firesize/MODEL_CARD.md) · [First pass](reviews/reviewer-a/firesize-v1.0.md)

**Raw-input check.** Reviewer B joined the published 13 × 13 fuel patches and the exam feature rows by FOD_ID. Weather, cause classification, discovery time and other metadata came from the original row and date. Canopy and slope grids were reconstructed to preserve their published summaries. The public `make_features` reproduced all 75 inputs within numerical tolerance and all eight archived escape probabilities within 0.00001. Replacing a fuel code then regenerated the point indicator, both neighbourhood fractions and the dependent Rothermel spread-rate features together. [Raw patches and data provenance](../../models/firesize/examples/README.md) · [Feature builder](../../models/firesize/firesize.py#L128)

| Change to the original fuel grid | Escape probability across eight examples |
|---|---:|
| Original patch | 0.00524–0.31170 |
| One code-201 cell immediately east of ignition | 0.01409–0.28895 |
| One code-201 cell at ignition | 0 on all eight |
| Adjoining 7 × 2 block of code 201, original ignition intact | 0.000000080–0.00001422 |

The larger adjoining patch filled 10/25 inner cells and 14/169 outer cells. One neighbouring cell increased the score on five examples and decreased it on three. The large-patch test reproduces a distinct suppression from the neighbourhood inputs under physically realizable fractions. The first pass also isolated the point indicator by changing only the fuel one-hot vector. [First-pass probe](reviews/reviewer-a/firesize_v1_0_probe.py) · [Raw-patch probe](reviews/reviewer-b/firesize_closing_probe.py)

**Mechanism and limits.** Of 75 shipped standard deviations, only `pt_SB` is at or below 0.001. Its stored mean is zero and its standard deviation equals the training floor of 0.000001. On a slash–blowdown ignition its standardized value reaches one million with nonzero weights in each of the three members. The rerun reproduced the reported rungs (9/9), widening temperature (1.4586), and Torch/NumPy probability agreement (maximum difference 2.14 × 10⁻⁷ here). Frequency and observed escape rates on that fuel stratum in the full 655,869-ignition exam, and the effect of a retrained model, remain unmeasured. [Training scaler](../../models/firesize/training/src/train.py#L87) · [Released inference](../../models/firesize/firesize.py#L50)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-a/firesize_v1_0_probe.py
python3 -B docs/audits/reviews/reviewer-b/firesize_closing_probe.py
```

## Resolution (27 Nov 2026)

| Finding | Real? | Action | Evidence |
|---|---|---|---|
| `pt_SB` (sd 1e-6) collapses escape probability on slash–blowdown ignitions | Yes, and it reached the exam | **Warned, opt-in clip, repair and re-score queued.** `probs()` warns on any input beyond 8 sd; `FireSize.z_clip = 8` clips. Default unchanged so the card's exam numbers describe v1.0. | Counted 27 Nov 2026 in `work/results/exam_preds.parquet`: **91 of 655,869 exam ignitions** have escape probability exactly 0 and the identical 0.6986 ha median, the collapse's signature. That answers the report's open question about frequency; the pooled scores cannot move much (0.014 % of rows), the SB stratum's can. |
