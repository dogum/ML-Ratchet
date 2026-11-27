# SwellCast v1.1 · Historical direction parsing and event selection

**First pass · confirmed preprocessing defect with a small measured effect on one storm.** The historical buoy-file parser applies the same near-99 missing-value rule to energy and compass bearings. It erases 3,445 readings of exactly 99° in the packaged NOAA 51004 direction file for 2018, including 52 within Hurricane Olivia’s 240-hour window. The training fetcher contains the same filter. Model snapshot `61c70bf`, audit checkout `ea2bfde`. [Public parser](../../../../models/swellcast/pipeline.py#L25) · [Training fetcher](../../../../models/swellcast/training/fetch.py#L35)

Reviewer B restored only the 99° cells in the raw direction table, then ran the complete hourly interpolation, three-hour binning, trained model and release prior. Energy was bitwise identical, including its missing-data mask.

| Olivia, buoy 51004 | Released parser | 99° restored |
|---|---:|---:|
| Storm distance | 1,947.71 km | 1,951.75 km |
| Bearing | 50.22° | 49.76° |
| Peak time relative to window centre | −37.38 h | −37.46 h |
| Swell probability | .9791 | .9774 |

Olivia remains detected. Its reference bearing is 82.76°, so restoring these cells slightly increases its bearing error. The parser demonstrably loses observations. A full matched exam replay would be needed to measure the direction and size of the resulting accuracy change.

**What the exam measures.** Catalogue storm time and distance determine candidate window centres with jitter. Coverage, crowding, measured energy and direction along the truth-based ridge determine eligibility for the confirmed subset. The heading “arrivals detected” applies to that population. A separate scan placed windows at seven-day intervals across the same buoy-year using only calendar time and the exam’s energy-coverage criterion. The release flagged 13 of 51 eligible windows. Without independent storm labels this is a trigger count. The published false-alarm test uses a different set of 600 designated quiet windows. [Exam construction](../../../../models/swellcast/training/events.py#L42) · [Archived quiet-control results](../../../../models/swellcast/training/final.json)

**Scope of split reconstruction.** The raw archived predictions give 182 detections among 288 confirmed exam arrivals (63.2%). Their 107 storm IDs have zero overlap with the ten frozen dev IDs. The manifest explicitly records that these ten are the recoverable dev storms with confirmed arrivals. The other eight original dev IDs were lost. A rebuild assigns every unlisted storm to exam, so the manifest can reproduce the confirmed split while leaving original all-pair and visible-subset membership unresolved. [Frozen IDs and recovery note](../../../../models/swellcast/training/dev_storms_v1.json)

**Directional sensitivity.** Rotating both directional moments by 90° at fixed energy preserved detection status on all five packaged windows. The three windows detected before and after rotation missed the expected bearing rotation by −3.34°, +42.63° and +17.86°. The largest residual overall (+135.37°) was on a window below threshold in both runs. This synthetic rotation measures orientation sensitivity. Physical storm geometry and the real distribution of other swells constrain its interpretation.

**Model controls.** No near-zero rows among 745 kernel outputs, all weights finite, and training/inference feature tensors matched exactly on the five shipped windows. On raw head outputs, PyTorch versus NumPy had maximum absolute errors of `1.91e-5` on those windows and `2.48e-5` on random features, above the protocol’s `1e-5` target and comparable with the archived `1.72e-5` check. The four shipped sigma multipliers range from 1.00 to 1.45. All five network promotion decisions reproduced from the raw logs. The later distance/time prior was selected on real dev storms using leave-one-storm-out scores. Its published rung table records the r8 distance guard rejecting a 29% → 32% median error. A per-storm refit, recovery of the eight missing IDs and full exposure-adjusted detection evaluation remain open. [Export check](../../../../models/swellcast/training/release_checks.json) · [Promotion history](../../../../models/swellcast/ratchet.yaml) · [Prior evaluation](../../../../models/swellcast/training/prior_dev.py)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/swellcast_v1_1_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-b/wave5_export_probe.py swellcast
```
