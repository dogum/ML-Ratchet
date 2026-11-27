# TransitNet v1.0 · Missing transit coverage and constant photometry

**First pass · input guardrails, with positive evidence that the model uses the transit.** This release vets a supplied candidate with a period, epoch and approximate duration. Its real exam is conditional on candidate selection. We tested the public reader on both packaged planets and the packaged background-binary candidate. Model snapshot `61c70bf`, audit checkout `73ee7e6`. [Reader](../../../../models/transitnet/transitnet.py#L76) · [Exam and selection](../../../../models/transitnet/MODEL_CARD.md#L77)

| Candidate | Original highest class | P(nothing) after excluding transit-region rows | P(nothing) after permuting flux within each sector |
|---|---:|---:|---:|
| Pi Men c | planet .99994 | .99132 | .98284 |
| TOI-270 c | planet .99985 | .92684 | .73053 |
| TOI-617 | blend .97863 | .81282 | .85047 |

Excluding every measurement within one supplied duration on either side of each predicted centre preserved the catalogue metadata and the out-of-transit observations. The public output then reported `snr = -inf` for all three candidates alongside finite class probabilities and planet parameters. This is an important ambiguity for a caller: a high probability of “nothing” can accompany a light curve with **no observation of the putative dip**. Permuting flux retained the sampling times and amplitude distribution and also made “nothing” the most probable class in all three cases. These controls show that the original classifications depend on the measured phase structure. [Box score and refinement](../../../../models/transitnet/pipeline.py#L64)

**Two input boundaries.** Unit constant positive flux at the real timestamps, with centroids absent, produced NaNs in all four probabilities and the radius ratio on every example. The estimated binned noise is zero and a log-noise scalar becomes nonfinite. Reversing rows within each otherwise complete sector raises an internal negative-bin `ValueError` during detrending. Reversing sector order, moving the epoch by one or two whole periods, or omitting only the centroids preserved the reported outputs to displayed precision. The released four-branch network has no centroid branch. That candidate failed the documented promotion margin. The prevalence of these input conditions in the original exam remains unknown. [Scalar construction](../../../../models/transitnet/pipeline.py#L114) · [Promotion history](../../../../models/transitnet/ratchet.yaml)

**Controls and limits.** The shipped kernels were finite with zero near-zero rows among 1,388. The reconstructed training graph agreed with NumPy to `3.82e-6` on the real TOI-617 inputs and on a seeded random batch. All four raw network decisions matched. The later polish trial was rejected on the rebuilt validation set (reported 0.841 versus 0.863 for the incumbent). Its full raw log is absent from the four-row ledger. The archived release exam reports 69.96% one-sigma containment for radius ratio. The split builder excludes all candidates sharing a dev TIC. The full real roster, stratified coverage and retraining were outside this replay. [Raw scores](../../../../models/transitnet/training/ratchet.json) · [TIC split](../../../../models/transitnet/training/split.py)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B docs/audits/reviews/reviewer-b/transitnet_v1_0_probe.py
python3 -B docs/audits/reviews/reviewer-b/wave7_export_probe.py transitnet
```
